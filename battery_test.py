"""
Battery Test Model
oakca
"""

# imports
from oemof.network import Node
import oemof.solph as solph
import oemof.outputlib as outputlib
import pandas as pd
import os


# timesteps
timesteps = 1

# time index
date_time_index = pd.date_range('1/1/2019', periods=timesteps,
                                freq='H')

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
                      actual_value=1,
                      fixed=True,
                      investment=solph.Investment(
                          ep_costs=50,
                          maximum=1000,
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
                 ep_costs=300*100,
                 maximum=1000,
                 existing=0),
              initial_capacity=1,
              nominal_storage_capacity=1000,
              inflow_conversion_factor=0.98,
              outflow_conversion_factor=0.98,
              invest_relation_input_capacity=1,
              invest_relation_output_capacity=1,
              capacity_loss=0,
              balanced=True)

# create transformers
pv_to_elec = solph.Transformer(label='pte',
                               inputs={b_pv: solph.Flow()},
                               outputs={b_elec: solph.Flow(
                                   variable_costs=-24)},
                               conversion_factors={b_pv: 1, b_elec: 1})


# create demand
demand = solph.Sink(label='e_demand',
                    inputs={b_elec: solph.Flow(
                        actual_value=1000,
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
model.solve(solver='glpk',
            solve_kwargs={'logfile': 'log.txt', 'tee': True})

# print objective
print(model.objective()/100)

# write LP file
filename = os.path.join(os.path.dirname(__file__), 'battery_test.lp')
model.write(filename, io_options={'symbolic_solver_labels': True})

# get results
es.results['main'] = outputlib.processing.results(model)
es.dump(dpath=None, filename=None)
