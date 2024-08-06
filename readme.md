A model of a combined cycle gas turbine, with state, energy and exergy analysis.

![image](https://github.com/user-attachments/assets/8003bdcb-7411-4448-86c1-e8962ac5c84e)

The gas cycle side runs an air-standard Brayton cycle, with an isobaric combustor.
The steam cycle side runs a superheated Rankine cycle, with a heat recovery steam generator (HRSG) exchanging heat from the gas to the steam.
The gas outlet of the HRSG is exhausted to the atmosphere, and the heat rejected from the steam condenser is lost.

CoolProp is used to model accurate fluid properties at varying temperatures, pressures and states.

## Figures

![image](Figures/Fig1_energy_exergy_balances.svg)
![image](Figures/Fig2_TS_diagrams.svg)

## Features

Done:

- [x] Solve all states and properties of CCGT
- [x] Compute efficiencies and show energy and exergy balances
- [x] Show the system on a T-s diagram
- [x] Calculate from the F-factor and overall heat transfer coefficient of the HRSG
- [x] Show the T-X diagram of the heat exchanger, compute pinch point
- [x] Warn the user about turbine inlet temperatures, supercritical states, 
  sub-triple point states, wet steam turbine outlets.

To do:

- [ ] Account for fuel in the gas cycle, model the combustion accurately
- [ ] Show the energy and exergy flows as a Sankey diagram instead of a pie chart
- [ ] Add reheat stages in the gas and/or steam cycle
- [ ] Add recuperation from the gas exhaust
- [ ] Add a solid oxide cell (electrolysis and fuel cell)
- [ ] Add a heat output as a combined heat and power (CHP) scheme
- [ ] Add a CO2 capture and storage (CCS) scheme
- [ ] Add a three-phase AC generator at the turbine shaft

#### Ambitious end goal

Turn it into a dashboard with interactive sliders for the inputs and outputs.

#### Possible extensions 

- calculate optimal operating parameters to maximise efficiency
- use this as a model for a control system, e.g. MPC.
