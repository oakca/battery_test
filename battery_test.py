"""
Battery Test Model
oakca
"""

# imports
from oemof.network import Node
import oemof.solph as solph
import oemof.outputlib as outputlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# timesteps
timesteps = 60*24

# time index
date_time_index = pd.date_range('1/1/2019', periods=timesteps,
                                freq='min')

# import data
demand_data = pd.read_csv('demand.csv', sep=';').drop(['datetime'], axis=1)
solar_data = pd.read_csv('solar.csv', sep=';')

# fix for 1 day
demand_data = demand_data.truncate(after=60*24-1)
solar_data = solar_data.truncate(after=60*24-1)['Globalstrahlung horizontal ']
solar_data[0] = 0

import pdb; pdb.set_trace()

# create energy system
es = solph.EnergySystem(timeindex=date_time_index)
Node.registry = es

# create buses
b_pv = solph.Bus(label='bus_pv')
b_elec = solph.Bus(label='bus_elec')
b_grid = solph.Bus(label='bus_grid')

# create pv-systems
pv = solph.Source(label='pv',
                  outputs={b_pv: solph.Flow(
                      actual_value=0.5,
                      fixed=True,
                      investment=solph.Investment(
                          ep_costs=10,
                          maximum=1750,
                          existing=0),
                      variable_costs=0)})

# create grid imports
grid_imp = solph.Source(label='grid_imp',
                        outputs={b_elec: solph.Flow(
                            variable_costs=20)})

# create battery
battery = solph.components.GenericStorage(
              label='battery',
              inputs={b_elec: solph.Flow(
                  variable_costs=0)},
              outputs={b_elec: solph.Flow(
                  variable_costs=0)},
              investment=solph.Investment(
                 ep_costs=10,
                 maximum=1000,
                 existing=0),
              initial_capacity=1,
              inflow_conversion_factor=0.98,
              outflow_conversion_factor=0.98,
              invest_relation_input_capacity=1,
              invest_relation_output_capacity=1,
              capacity_loss=0)

# create transformers
pv_to_elec = solph.Transformer(label='pte',
                               inputs={b_pv: solph.Flow()},
                               outputs={b_elec: solph.Flow(
                                   variable_costs=-24)},
                               conversion_factors={b_pv: 1, b_elec: 1})


# create demand
demand = solph.Sink(label='e_demand',
                    inputs={b_elec: solph.Flow(
                        actual_value=demand_data['P_el'],
                        fixed=True,
                        nominal_value=1)})

# create grid exports (value = max pv capacity)
grid_exp = solph.Sink(label='grid_exp',
                      inputs={b_pv: solph.Flow(
                          fixed=False,
                          nominal_value=1000,
                          variable_costs=-9)})

# create model
model = solph.Model(es)
model.name = 'Battery Test'

# add constant cost of battery
model.objective.expr.__add__(3000*100)

# solve model
model.solve(solver='cbc',
            solve_kwargs={'logfile': 'log.txt', 'tee': True})

# print objective
print(model.objective()/100)

# write LP file
filename = os.path.join(os.path.dirname(__file__), 'battery_test.lp')
model.write(filename, io_options={'symbolic_solver_labels': True})

# get results
es.results['main'] = outputlib.processing.results(model)

# visualisation data
elec_flow = outputlib.views.node(es.results['main'], 'bus_elec')['sequences']
storage_content = outputlib.views.node(es.results['main'], 'None')['sequences']
storage_power = outputlib.views.node(es.results['main'], 'bus_elec')['scalars']
storage_capacity = outputlib.views.node(es.results['main'], 'None')['scalars']

pv_flow = outputlib.views.node(es.results['main'], 'bus_pv')['sequences']
pv_invest = outputlib.views.node(es.results['main'], 'bus_pv')['scalars']

demand = outputlib.views.node(es.results['main'], 'e_demand')['sequences']
grid_export = outputlib.views.node(es.results['main'], 'grid_exp')['sequences'] # -9cent
grid_import = outputlib.views.node(es.results['main'], 'grid_imp')['sequences'] # +20cent

pte = outputlib.views.node(es.results['main'], 'pte')['sequences'] # -24cent

# plot
fig = plt.figure()

# x-Axis (timesteps)
iterations = [i for i in range(1, len(es.timeindex))]
x = np.array(iterations)

# demand
values=[]
for i in range(1, len(es.timeindex)):
    values.append(demand[(('bus_elec', 'e_demand'), 'flow')][(i-1)])
y = np.array(values)
plt.plot(x, y, label='demand', linestyle='-')

# pv to elec
values=[]
for i in range(1, len(es.timeindex)):
    values.append(pv_flow[(('bus_pv', 'pte'), 'flow')][(i-1)])
y = np.array(values)
plt.plot(x, y, label='pv_to_elec', linestyle='-')

# grid_exp
values=[]
for i in range(1, len(es.timeindex)):
    values.append(pv_flow[(('bus_pv', 'grid_exp'), 'flow')][(i-1)])
y = np.array(values)
plt.plot(x, y, label='pv_to_grid', linestyle='-')

# grid import
values=[]
for i in range(1, len(es.timeindex)):
    values.append(grid_import[(('grid_imp', 'bus_elec'), 'flow')][(i-1)])
y = np.array(values)
plt.plot(x, y, label='elec_from_grid', linestyle='-')

# to battery
values=[]
for i in range(1, len(es.timeindex)):
    values.append(elec_flow[(('bus_elec', 'battery'), 'flow')][(i-1)])
y = np.array(values)
plt.plot(x, y, label='elec_to_battery', linestyle='-')

# battery content
values=[]
for i in range(1, len(es.timeindex)):
    values.append(storage_content[(('battery', 'None'), 'capacity')][(i-1)])
y = np.array(values)
plt.bar(x, y, label='storage_content', align='center', alpha=0.75, width=0.2)

plt.grid(True)
plt.legend()
plt.show()
