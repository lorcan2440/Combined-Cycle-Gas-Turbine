from matplotlib import pyplot as plt
import numpy as np
import CoolProp as CP
from CoolProp.CoolProp import PropsSI
from CoolPlot.Plot import PropertyPlot

import warnings
import re
import logging

plt.style.use(r"C:\LibsAndApps\Python config files\proplot_style.mplstyle")

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    filename="logger.log",
    filemode="a",
    encoding="utf-8",
    level=logging.INFO,
)


class CombinedCycleGasTurbine:

    # default constants - can be overridden by passing with kwargs
    # pass with lower case letters - e.g. {p_5: 1.1e5}
    # except for temperature and heat input - e.g. {T_5: 300, Q_67: 1e9}
    _DEF_GAS_FLUID = "air"
    _DEF_STEAM_FLUID = "water"
    _DEF_P_5 = 101325
    _DEF_T_5 = 298.15
    _DEF_R_P_COMP = 23
    _DEF_N_C_GAS = 0.85
    _DEF_M_DOT_GAS = 1.0559e3
    _DEF_Q_67 = 1.0453e9
    _DEF_R_P_TURB = 20.9186
    _DEF_N_T_GAS = 0.85
    _DEF_T_3 = 873
    _DEF_M_DOT_STEAM = 150.3
    _DEF_P_1 = 0.04 * 101325
    _DEF_R_P_PUMP = 1000
    _DEF_N_C_STEAM = 0.85
    _DEF_T_1 = 297.9
    _DEF_N_T_STEAM = 0.8

    def attrs_from_kwargs_or_default(self, **kwargs: dict) -> None:
        """
        Populate the attributes of the class, taking values from the kwargs dictionary,
        or using the default values defined in the class if keys are unspecified.

        ### Arguments
        #### Optional
        - `kwargs` (dict, default = {}): dictionary of {attribute name: attribute value} to set
        """

        def_items = self.__class__.__dict__.items()
        def_items = filter(lambda item: item[0].startswith("_DEF_"), def_items)
        for default_key, default_value in def_items:
            key = default_key.removeprefix("_DEF_")
            # T_n and Q_mn are not converted to lowercase
            key = (key if re.match(r"(T|Q)_\d+", key) else key.lower())
            if key in kwargs:
                logging.info(f"Setting attribute {key} to specified value {kwargs[key]}")
                setattr(self, key, kwargs[key])
            else:
                logging.info(f"Setting attribute {key} to default value {default_value}")
                setattr(self, key, default_value)

    def __init__(self, **kwargs):
        """
        A Combined Cycle Gas Turbine model, featuring an air-standard Brayton cycle with isobaric combustor,
        for the gas turbine and a superheated Rankine cycle for the steam turbine, with a heat recovery
        steam generator (HRSG) in between.

        ### Key Points in the Model
        #### Steam cycle (Rankine cycle)
        1. Steam cycle pump inlet / Steam cycle condenser outlet
        2. Steam cycle HRSG inlet / Steam cycle pump outlet
        3. Steam cycle turbine inlet / Steam cycle HRSG outlet
        4. Steam cycle condenser inlet / Steam cycle turbine outlet
        #### Gas cycle (Brayton cycle)
        5. Gas cycle compressor inlet
        6. Gas cycle combustor inlet / Gas cycle compressor outlet
        7. Gas cycle turbine inlet / Gas cycle combustor outlet
        8. Gas cycle HRSG inlet / Gas cycle turbine outlet
        9. Gas cycle HRSG outlet

        ### Arguments
        #### Optional
        - `gas_fluid` (str, default = "air"): the gas fluid used in the gas turbine
        - `steam_fluid` (str, default = "water"): the steam fluid used in the steam turbine
        - `p_5` (float, default = 101325): the pressure at point 5 (gas cycle compressor inlet), in Pa
        - `T_5` (float, default = 293.15): the temperature at point 5 (gas cycle compressor inlet), in K
        - `r_p_comp` (float, default = 23): the pressure ratio of the gas cycle compressor
        - `n_c_gas` (float, default = 0.85): the isentropic efficiency of the gas cycle compressor
        - `m_dot_gas` (float, default = 1.0559e3): the mass flow rate in the gas cycle, in kg/s
        - `Q_67` (float, default = 1.0453e9): the heat input in the gas cycle combustor, in W
        - `r_p_turb` (float, default = 20.9186): the pressure ratio of the gas cycle turbine
        - `n_t_gas` (float, default = 0.85): the isentropic efficiency of the gas cycle turbine
        - `T_3` (float, default = 873): the temperature at point 3 (steam cycle turbine inlet), in K
        - `m_dot_steam` (float, default = 150.3): the mass flow rate in the steam cycle, in kg/s
        - `p_1` (float, default = 0.04 * 101325): the pressure at point 1 (steam cycle pump inlet), in Pa
        - `r_p_pump` (float, default = 1000): the pressure ratio of the steam cycle pump
        - `n_c_steam` (float, default = 0.85): the isentropic efficiency of the steam cycle pump
        - `T_1` (float, default = 280): the temperature at point 1 (steam cycle pump inlet), in K
        - `n_t_steam` (float, default = 0.8): the isentropic efficiency of the steam cycle turbine
        """
        self.attrs_from_kwargs_or_default(**kwargs)

    def calc_states(self):
        """
        Calculates the states at each key point around the CCGT cycles.
        All fluid data is taken from CoolProp.
        Available attributes are:

        - `p`: pressure, in Pa
        - `T`: temperature, in K
        - `h`: specific enthalpy, in J/kg
        - `s`: specific entropy, in J/K/kg
        - `x`: dryness fraction / quality (only for steam cycle - points 1, 2, 3, 4)
        - `ex`: specific exergy, in J/kg

        The results are stored as attributes of the class in the form "symbol_n",
        where "symbol" is the symbol in the above list and "n" is the point number,
        e.g. `h_6` for the specific enthalpy at point 6 (gas cycle pump outlet).

        The results are also stored for the 'ideal' states (isentropic processes) of the compressor/turbines/pump.
        The available ideal states are at 6s, 8s, 2s and 4s.
        e.g. `h_6s` for the specific enthalpy at point 6s (gas cycle pump outlet if it were reversible).

        Additional useful attributes:

        - `lmtd_hrsg`: the log mean temperature difference in the HRSG, in K
        - `Q_23` = `-Q_89`: the heat transfer through the HRSG, in W
        - `Q_14`: the heat rejected from the steam cycle condenser, in W
        """

        # calculate gas compressor outlet conditions
        self.p_6 = self.p_5 * self.r_p_comp
        self.h_5 = PropsSI("H", "P", self.p_5, "T", self.T_5, self.gas_fluid)
        self.s_5 = PropsSI("S", "P", self.p_5, "T", self.T_5, self.gas_fluid)
        self.ex_5 = self.specific_exergy_at_point(5)
        self.h_6s = PropsSI("H", "P", self.p_6, "S", self.s_5, self.gas_fluid)
        self.T_6s = PropsSI("T", "P", self.p_6, "S", self.s_5, self.gas_fluid)
        self.h_6 = self.h_5 + (self.h_6s - self.h_5) / self.n_c_gas
        self.s_6 = PropsSI("S", "P", self.p_6, "H", self.h_6, self.gas_fluid)
        self.T_6 = PropsSI("T", "P", self.p_6, "H", self.h_6, self.gas_fluid)
        self.ex_6 = self.specific_exergy_at_point(6)

        # calculate combustion chamber outlet conditions
        self.p_7 = self.p_6  # assume isobaric combustion
        self.h_7 = self.h_6 + self.Q_67 / self.m_dot_gas
        self.T_7 = PropsSI("T", "P", self.p_7, "H", self.h_7, self.gas_fluid)
        self.s_7 = PropsSI("S", "P", self.p_7, "H", self.h_7, self.gas_fluid)
        self.ex_7 = self.specific_exergy_at_point(7)

        # calculate gas turbine outlet conditions
        self.p_8 = self.p_7 / self.r_p_turb
        self.h_8s = PropsSI("H", "P", self.p_8, "S", self.s_7, self.gas_fluid)
        self.T_8s = PropsSI("T", "P", self.p_8, "S", self.s_7, self.gas_fluid)
        self.h_8 = self.h_7 - (self.h_7 - self.h_8s) * self.n_t_gas
        self.s_8 = PropsSI("S", "P", self.p_8, "H", self.h_8, self.gas_fluid)
        self.T_8 = PropsSI("T", "P", self.p_8, "H", self.h_8, self.gas_fluid)
        self.ex_8 = self.specific_exergy_at_point(8)

        # calculate steam pump outlet conditions
        self.h_1 = PropsSI("H", "P", self.p_1, "T", self.T_1, self.steam_fluid)
        self.s_1 = PropsSI("S", "P", self.p_1, "T", self.T_1, self.steam_fluid)
        self.x_1 = PropsSI("Q", "P", self.p_1, "T", self.T_1, self.steam_fluid)
        self.ex_1 = self.specific_exergy_at_point(1)
        self.p_2 = self.p_1 * self.r_p_pump
        self.h_2s = PropsSI("H", "P", self.p_2, "S", self.s_1, self.steam_fluid)
        self.T_2s = PropsSI("T", "P", self.p_2, "S", self.s_1, self.steam_fluid)
        self.h_2 = self.h_1 + (self.h_2s - self.h_1) / self.n_c_steam
        self.s_2 = PropsSI("S", "P", self.p_2, "H", self.h_2, self.steam_fluid)
        self.T_2 = PropsSI("T", "P", self.p_2, "H", self.h_2, self.steam_fluid)
        self.x_2 = PropsSI("Q", "P", self.p_2, "H", self.h_2, self.steam_fluid)
        self.ex_2 = self.specific_exergy_at_point(2)

        # calculate HRSG steam side outlet conditions
        self.p_3 = self.p_2
        self.h_3 = PropsSI("H", "P", self.p_3, "T", self.T_3, self.steam_fluid)
        self.s_3 = PropsSI("S", "P", self.p_3, "T", self.T_3, self.steam_fluid)
        self.x_3 = PropsSI("Q", "P", self.p_3, "T", self.T_3, self.steam_fluid)
        self.ex_3 = self.specific_exergy_at_point(3)
        self.Q_23 = self.m_dot_steam * (self.h_3 - self.h_2)
        self.Q_89 = -1 * self.Q_23
        self.h_9 = self.h_8 - self.Q_23 / self.m_dot_gas
        self.T_9 = PropsSI("T", "P", self.p_8, "H", self.h_9, self.gas_fluid)
        self.s_9 = PropsSI("S", "P", self.p_8, "H", self.h_9, self.gas_fluid)
        self.ex_9 = self.specific_exergy_at_point(9)

        # calculate HRSG LMTD
        self.dT_hot_hrsg = self.T_8 - self.T_3
        self.dT_cold_hrsg = self.T_9 - self.T_2
        self.lmtd_hrsg = (self.dT_hot_hrsg - self.dT_cold_hrsg) / np.log(
            self.dT_hot_hrsg / self.dT_cold_hrsg
        )

        # calculate steam turbine outlet conditions
        self.p_4 = self.p_1
        self.h_4s = PropsSI("H", "P", self.p_4, "S", self.s_3, self.steam_fluid)
        self.T_4s = PropsSI("T", "P", self.p_4, "S", self.s_3, self.steam_fluid)
        self.h_4 = self.h_3 - (self.h_3 - self.h_4s) * self.n_t_steam
        self.s_4 = PropsSI("S", "P", self.p_4, "H", self.h_4, self.steam_fluid)
        self.T_4 = PropsSI("T", "P", self.p_4, "H", self.h_4, self.steam_fluid)
        self.x_4 = PropsSI("Q", "P", self.p_4, "H", self.h_4, self.steam_fluid)
        self.ex_4 = self.specific_exergy_at_point(4)

        # calculate condenser heat transfer
        self.Q_14 = self.m_dot_steam * (self.h_4 - self.h_1)

    def calc_energy_exergy_balances(self):

        '''
        Calculates metrics relating to the power inputs/outputs, thermal efficiencies,
        maximum thermal efficiencies, exergy efficiencies and losses of available power.

        Plots a pie chart showing the energy and exergy balances in the CCGT, showing the
        proportions of their losses by the components.

        Some of the useful attributes are:

        #### Power
        - `W_56`: power input to gas compressor, in W
        - `W_78`: power output from gas turbine, in W
        - `W_12`: power input to steam pump, in W
        - `W_34`: power output from steam turbine, in W
        - `W_gas`: net power output from gas cycle, in W
        - `W_steam`: net power output from steam cycle, in W
        - `W_total`: total net CCGT power output, in W
        #### Exergy
        - `Ex_67`: exergy input to gas combustor, in W
        - `W_56_loss`: exergy destruction rate in gas compressor, in W
        - `W_78_loss`: exergy destruction rate in gas turbine, in W
        - `hrsg_loss`: exergy destruction rate in HRSG, in W
        - `gas_exhaust_loss`: exergy rejected from gas exhaust, in W
        - `W_12_loss`: exergy destruction rate in steam pump, in W
        - `W_34_loss`: exergy destruction rate in steam turbine, in W
        - `Ex_14`: exergy rejected from steam condenser, in W
        #### Efficiency
        - `eta_gas_th`: thermal efficiency of gas cycle
        - `eta_steam_th`: thermal efficiency of steam cycle
        - `eta_th`: overall CCGT thermal efficiency
        - `eta_ex`: exergy efficiency of CCGT
        - `eta_th_max`: maximum possible thermal efficiency of CCGT
        '''        

        # actual energy balance
        # power input to gas compressor
        self.W_56 = self.m_dot_gas * (self.h_6 - self.h_5)
        # power output from gas turbine
        self.W_78 = self.m_dot_gas * (self.h_7 - self.h_8)
        # power input to steam pump
        self.W_12 = self.m_dot_steam * (self.h_2 - self.h_1)
        # power output from steam turbine
        self.W_34 = self.m_dot_steam * (self.h_3 - self.h_4)

        # powers
        self.W_gas = self.W_78 - self.W_56
        self.W_steam = self.W_34 - self.W_12
        self.W_total = self.W_gas + self.W_steam

        # exergies
        # gas combustor exergy input
        self.Ex_67 = self.m_dot_gas * (self.ex_7 - self.ex_6)
        # steam condenser exergy output
        self.Ex_14 = self.m_dot_steam * (self.ex_4 - self.ex_1)

        # efficiencies
        self.eta_gas_th = self.W_gas / self.Q_67
        self.eta_steam_th = self.W_steam / self.Q_23
        self.eta_th = self.W_total / self.Q_67

        self.eta_ex = self.W_total / self.Ex_67
        self.eta_th_max = self.Ex_67 / self.Q_67

        # losses of available power
        # gas compressor
        self.W_56_loss = self.m_dot_gas * 298 * (self.s_6 - self.s_5)
        # gas turbine
        self.W_78_loss = self.m_dot_gas * 298 * (self.s_8 - self.s_7)
        # HRSG
        self.hrsg_loss = 298 * (
            self.m_dot_steam * (self.s_3 - self.s_2)
            + self.m_dot_gas * (self.s_9 - self.s_8)
        )
        # gas exhaust
        self.gas_exhaust_loss = self.m_dot_gas * (
            self.ex_9 - self.specific_exergy_at_state(101325, self.T_9, self.gas_fluid)
        )
        # steam pump
        self.W_12_loss = self.m_dot_steam * 298 * (self.s_2 - self.s_1)
        # steam turbine
        self.W_34_loss = self.m_dot_steam * 298 * (self.s_4 - self.s_3)

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(12, 6), subplot_kw=dict(aspect="equal")
        )

        ax1.pie(
            [self.W_total, self.m_dot_gas * (self.h_9 - self.h_5), self.Q_14],
            labels=[
                f"Net power: \n{self.W_total / 1e6 :.2f} MW",
                f"Exhaust enthalpy: \n{self.m_dot_gas * (self.h_9 - self.h_5) / 1e6 :.2f} MW",
                f"Condenser heat rejection: \n{self.Q_14 / 1e6 :.2f} MW",
            ],
            startangle=90,
            autopct="%1.1f%%",
            explode=(0.1, 0, 0),
            colors=["#4fc26e", "#f2b134", "#d15454"],
        )
        ax1.set_title(
            f"Total energy balance: from heat input rate {self.Q_67 / 1e6 :.2f} MW"
        )

        ax2.pie(
            [
                self.W_total,
                self.m_dot_gas * (self.ex_9 - self.ex_5),
                self.Ex_14,
                self.W_56_loss,
                self.W_78_loss,
                self.hrsg_loss,
                self.W_12_loss,
                self.W_34_loss,
            ],
            labels=[
                f"Net power: \n{(self.W_total) / 1e6 :.2f} MW",
                f"Exhaust exergy: \n{(self.m_dot_gas * (self.ex_9 - self.ex_5)) / 1e6 :.2f} MW",
                f"Condenser exergy rejection: \n{(self.Ex_14) / 1e6 :.2f} MW",
                f"Gas compressor loss: \n{(self.W_56_loss) / 1e6 :.2f} MW",
                f"Gas turbine loss: \n{(self.W_78_loss) / 1e6 :.2f} MW",
                f"HRSG loss: \n{(self.hrsg_loss) / 1e6 :.2f} MW",
                f"Steam pump loss: \n{(self.W_12_loss) / 1e6 :.2f} MW",
                f"Steam turbine loss: \n{(self.W_34_loss) / 1e6 :.2f} MW",
            ],
            startangle=90,
            autopct="%1.1f%%",
            explode=(0.1, 0, 0, 0, 0, 0, 0, 0),
            colors=["#4fc26e", "#f2b134", "#ebbe42", "#f0643a", "#e85238", "#ed3c2f", "#cf1f1f", "#a12323"],
        )
        ax2.set_title(
            f"Total exergy balance: from heat input rate {self.Ex_67 / 1e6 :.2f} MW"
        )

        fig.suptitle(
            f"Combined cycle gas turbine energy and exergy balances\nThermal efficiency: "
            f"{self.eta_th :.2%}\nMaximum possible efficiency: {self.eta_th_max :.2%}\n"
            f"Exergy efficiency: {self.eta_ex :.2%}"
        )
        fig.tight_layout()
        fig.savefig("Figures/Fig1_energy_exergy_balances.svg", dpi=300)
        plt.show()

    def plot_Ts_diagram(self):

        '''
        Plots the T-s diagrams for the gas and steam cycles, showing the processes
        and states of the CCGT. The steam cycle is plotted using CoolProp's PropertyPlot,
        showing the phase boundaries of the two-phase region, with isobars and 
        iso-dryness fraction contours.
        '''        

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # ax1: T-s diagram (gas turbine)
        s_range = np.linspace(self.s_5, self.s_8, 100)
        # plot compressor
        ax1.plot(
            [self.s_5, self.s_6], [self.T_5, self.T_6], "r", marker="o", label="Process"
        )
        ax1.plot(self.s_5, self.T_6s, "ko")
        ax1.plot(
            [self.s_5, self.s_5],
            [self.T_5, self.T_6s],
            "k--",
            alpha=0.5,
            label="Isentropic Process",
        )
        ax1.plot(
            s_range,
            PropsSI("T", "P", self.p_5, "S", s_range, self.gas_fluid),
            "green",
            alpha=0.25,
        )
        ax1.plot(
            s_range,
            PropsSI("T", "P", self.p_6, "S", s_range, self.gas_fluid),
            "green",
            alpha=0.25,
        )
        ax1.annotate(
            "5",
            (self.s_5, self.T_5),
            textcoords="offset points",
            xytext=(-10, 0),
            ha="center",
        )
        ax1.annotate(
            "6",
            (self.s_6, self.T_6),
            textcoords="offset points",
            xytext=(-10, 0),
            ha="center",
        )
        # plot combustor
        s_range_comb = np.linspace(self.s_6, self.s_7, 100)
        p_range_comb = np.linspace(self.p_6, self.p_7, 100)
        ax1.plot(
            s_range,
            PropsSI("T", "P", self.p_7, "S", s_range, self.gas_fluid),
            "green",
            alpha=0.25,
            label="Isobars",
        )
        ax1.plot(
            s_range_comb,
            PropsSI("T", "P", p_range_comb, "S", s_range_comb, self.gas_fluid),
            "r",
        )
        ax1.annotate(
            "7",
            (self.s_7, self.T_7),
            textcoords="offset points",
            xytext=(10, 0),
            ha="center",
        )
        # plot turbine
        ax1.plot([self.s_7, self.s_8], [self.T_7, self.T_8], "r", marker="o")
        ax1.plot(self.s_7, self.T_8s, "ko")
        ax1.plot([self.s_7, self.s_7], [self.T_7, self.T_8s], "k--", alpha=0.5)
        ax1.plot(
            s_range,
            PropsSI("T", "P", self.p_8, "S", s_range, self.gas_fluid),
            "green",
            alpha=0.25,
        )
        ax1.annotate(
            "8",
            (self.s_8, self.T_8),
            textcoords="offset points",
            xytext=(10, 0),
            ha="center",
        )
        # plot hrsg
        s_range_hrsg_gas = np.linspace(self.s_9, self.s_8, 100)
        ax1.plot(
            s_range_hrsg_gas,
            PropsSI("T", "P", self.p_8, "S", s_range_hrsg_gas, self.gas_fluid),
            "r",
        )
        ax1.plot(self.s_9, self.T_9, "ro")
        ax1.annotate(
            "9",
            (self.s_9, self.T_9),
            textcoords="offset points",
            xytext=(-10, 0),
            ha="center",
        )

        ax1.set_xlabel("Specific entropy s, J / K / kg")
        ax1.set_ylabel("Temperature T, K")
        ax1.legend()
        ax1.set_title("Gas Cycle: $ \eta = $ " + f"{self.eta_gas_th :.2%}")

        # ax2: T-s diagram (steam turbine)
        s_range_steam = np.linspace(self.s_1, self.s_3, 100)
        # plot pump
        ax2.plot(
            [self.s_1, self.s_2], [self.T_1, self.T_2], "r", marker="o", label="Process"
        )
        ax2.plot(self.s_1, self.T_2s, "ko")
        ax2.plot(
            [self.s_1, self.s_1],
            [self.T_1, self.T_2s],
            "k--",
            alpha=0.5,
            label="Isentropic Process",
        )
        ax2.plot(
            s_range_steam,
            PropsSI("T", "P", self.p_1, "S", s_range_steam, self.steam_fluid),
            "green",
            alpha=0.25,
            label="Isentropic Process",
        )
        ax2.plot(
            s_range_steam,
            PropsSI("T", "P", self.p_2, "S", s_range_steam, self.steam_fluid),
            "green",
            alpha=0.25,
        )
        ax2.annotate(
            "1",
            (self.s_1, self.T_1),
            textcoords="offset points",
            xytext=(-10, 0),
            ha="center",
        )
        ax2.annotate(
            "2",
            (self.s_2, self.T_2),
            textcoords="offset points",
            xytext=(-10, 0),
            ha="center",
        )
        # plot hrsg
        s_range_hrsg_steam = np.linspace(self.s_2, self.s_3, 100)
        ax2.plot(
            s_range_hrsg_steam,
            PropsSI("T", "P", self.p_2, "S", s_range_hrsg_steam, self.steam_fluid),
            "r",
        )
        ax2.plot(self.s_3, self.T_3, "ro")
        ax2.annotate(
            "3",
            (self.s_3, self.T_3),
            textcoords="offset points",
            xytext=(10, 0),
            ha="center",
        )
        # plot turbine
        ax2.plot([self.s_3, self.s_4], [self.T_3, self.T_4], "r", marker="o")
        ax2.plot(self.s_3, self.T_4s, "ko")
        ax2.plot([self.s_3, self.s_3], [self.T_3, self.T_4s], "k--", alpha=0.5)
        ax2.plot(
            s_range_steam,
            PropsSI("T", "P", self.p_4, "S", s_range_steam, self.steam_fluid),
            "green",
            alpha=0.25,
        )
        ax2.annotate(
            "4",
            (self.s_4, self.T_4),
            textcoords="offset points",
            xytext=(10, 0),
            ha="center",
        )
        # plot condenser
        ax2.plot([self.s_4, self.s_1], [self.T_4, self.T_1], "r")
        ax2.annotate(
            "1",
            (self.s_1, self.T_1),
            textcoords="offset points",
            xytext=(-10, 0),
            ha="center",
        )

        # plot T-s diagram of steam using CoolProp
        pp = PropertyPlot("water", "TS", axis=ax2, unit_system="SI")
        pp.calc_isolines(CP.iP, iso_range=[1e3, 1e7], num=10)  # isobars
        pp.calc_isolines(
            CP.iQ, iso_range=[0, 1], num=6
        )  # equal dryness fraction (quality) lines
        pp.draw()

        ax2.set_xlim(0, self.s_4 * 1.1)
        ax2.set_ylim(self.T_1 - 50, self.T_3 + 50)
        ax2.set_xlabel("Specific entropy s, J / K / kg")
        ax2.set_ylabel("Temperature T, K")
        ax2.legend()
        ax2.set_title("Steam Cycle: $ \eta = $ " + f"{self.eta_steam_th :.2%}")

        fig.suptitle("Combined cycle gas turbine T-s diagrams")
        fig.tight_layout()
        fig.savefig("Figures/Fig2_TS_diagrams.svg", dpi=300)
        plt.show()


    def specific_exergy_at_point(
        self, n: int, p_0: float = 101325, T_0: float = 298.15
    ) -> float:
        '''
        Calculates the specific exergy of the flow at a given key point in the CCGT.
        
        ### Arguments
        #### Required
        - `n` (int): numbered point from 1-9 to measure at
        #### Optional
        - `p_0` (float, default = 101325): dead state pressure, Pa
        - `T_0` (float, default = 298.15): dead state temperature, K
        
        ### Returns
        - `float`: specific exergy, J/kg
        '''        

        if n in (5, 6, 7, 8, 9):
            fluid = self.gas_fluid
        elif n in (1, 2, 3, 4):
            fluid = self.steam_fluid
        # fluid state
        h = getattr(self, f"h_{n}")
        s = getattr(self, f"s_{n}")
        # dead state
        h_0 = PropsSI("H", "T", T_0, "P", p_0, fluid)
        s_0 = PropsSI("S", "T", T_0, "P", p_0, fluid)
        # specific steady flow availability function
        b = h - T_0 * s
        b_0 = h_0 - T_0 * s_0
        # specific exergy
        ex = b - b_0
        return ex

    def specific_exergy_at_state(
        self, p: float, T: float, fluid: str, p_0: float = 101325, T_0: float = 298.15
    ) -> float:
        '''
        Calculates the specific exergy of an arbitrary flow state.
        
        ### Arguments
        #### Required
        - `p` (float): pressure of the fluid, Pa
        - `T` (float): temperature of the fluid, K
        - `fluid` (str): fluid name
        #### Optional
        - `p_0` (float, default = 101325): dead state pressure, Pa
        - `T_0` (float, default = 298.15): dead state temperature, K
        
        ### Returns
        - `float`: specific exergy, J/kg
        '''     

        # fluid state
        h = PropsSI("H", "T", T, "P", p, fluid)
        s = PropsSI("S", "T", T, "P", p, fluid)
        # dead state
        h_0 = PropsSI("H", "T", T_0, "P", p_0, fluid)
        s_0 = PropsSI("S", "T", T_0, "P", p_0, fluid)
        # specific steady flow availability function
        b = h - T * s
        b_0 = h_0 - T_0 * s_0
        # specific exergy
        ex = b - b_0
        return ex

    def __str__(self) -> str:
        '''
        Returns a readable summary of the CCGT model, after all computations.
        
        ### Returns
        - `str`: string summary.
        '''
        try:
            return (
                f"Gas turbine: \n"
                f"\tCompressor power input: {self.W_56 / 1e6 :.2f} MW\n"
                f"\tCombustion heat input: {self.Q_67 / 1e6 :.2f} MW\n"
                f"\tTurbine power output: {self.W_78 / 1e6 :.2f} MW\n"
                f"Steam turbine: \n"
                f"\tPump power input: {self.W_12 / 1e6 :.2f} MW\n"
                f"\tHRSG heat transfer: {self.Q_23 / 1e6 :.2f} MW\n"
                f"\tTurbine power output: {self.W_34 / 1e6 :.2f} MW\n"
                f"Efficiencies: \n"
                f"\tGas turbine thermal efficiency: {self.eta_gas_th :.2%}\n"
                f"\tSteam turbine thermal efficiency: {self.eta_steam_th :.2%}\n"
                f"\tOverall thermal efficiency: {self.eta_th :.2%}\n"
                f"\tMaximum possible thermal efficiency: {self.eta_th_max :.2%}\n"
                f"\tOverall exergy efficiency: {self.eta_ex :.2%}\n"
            )
        except AttributeError:
            return "Model has not been computed yet. Run `calc_states()` and `calc_energy_exergy_balances()` first."


ccgt = CombinedCycleGasTurbine()

ccgt.calc_states()
ccgt.calc_energy_exergy_balances()
ccgt.plot_Ts_diagram()
print(ccgt)
