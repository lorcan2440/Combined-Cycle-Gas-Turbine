A model of a combined cycle gas turbine, complete with energy and exergy analysis.

![image](https://github.com/user-attachments/assets/8003bdcb-7411-4448-86c1-e8962ac5c84e)

The gas cycle side runs an air-standard Brayton cycle, with an isobaric combustor.
The steam cycle side runs a superheated Rankine cycle, with a heat recovery steam generator (HRSG) exchanging heat from the gas to the steam.
The gas outlet of the HRSG is exhausted to the atmosphere, and the heat rejected from the steam condenser is lost.

CoolProp is used to model accurate fluid properties at varying temperatures, pressures and states.

![image](https://github.com/user-attachments/assets/ff905b89-21bb-4f00-8a7d-7bcaa73fafb0)

Done:

- [x] Solve all states and properties of CCGT
- [x] Compute efficiencies and show energy and exergy balances

To do:

- [ ] Show the system on a T-s diagram
- [ ] Show the T-x diagram of the heat exchanger, compute pinch point
- [ ] Calculate the F-factor, LMTD and overall heat transfer coefficient of the HRSG
- [ ] Account for fuel in the gas cycle, model the combustion accurately
- [ ] Allow reheat stages in the gas and/or steam cycle
- [ ] Allow recuperation from the gas exhaust
