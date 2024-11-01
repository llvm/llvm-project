//===-- RISCVProcessors.td - RISCV Processors --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RISC-V processors supported.
//===----------------------------------------------------------------------===//

class RISCVProcessorModel<string n,
                          SchedMachineModel m,
                          list<SubtargetFeature> f,
                          list<SubtargetFeature> tunef = [],
                          string default_march = "">
      :  ProcessorModel<n, m, f, tunef> {
  string DefaultMarch = default_march;
}

class RISCVTuneProcessorModel<string n,
                              SchedMachineModel m,
                              list<SubtargetFeature> tunef = [],
                              list<SubtargetFeature> f = []>
      : ProcessorModel<n, m, f,tunef>;

def GENERIC_RV32 : RISCVProcessorModel<"generic-rv32",
                                       NoSchedModel,
                                       [Feature32Bit]>;
def GENERIC_RV64 : RISCVProcessorModel<"generic-rv64",
                                       NoSchedModel,
                                       [Feature64Bit]>;
// Support generic for compatibility with other targets. The triple will be used
// to change to the appropriate rv32/rv64 version.
def : ProcessorModel<"generic", NoSchedModel, []>;

def ROCKET_RV32 : RISCVProcessorModel<"rocket-rv32",
                                      RocketModel,
                                      [Feature32Bit]>;
def ROCKET_RV64 : RISCVProcessorModel<"rocket-rv64",
                                      RocketModel,
                                      [Feature64Bit]>;
def ROCKET : RISCVTuneProcessorModel<"rocket",
                                     RocketModel>;

def SIFIVE_7 : RISCVTuneProcessorModel<"sifive-7-series",
                                       SiFive7Model,
                                       [TuneSiFive7]>;

def SIFIVE_E20 : RISCVProcessorModel<"sifive-e20",
                                     RocketModel,
                                     [Feature32Bit,
                                      FeatureStdExtM,
                                      FeatureStdExtC]>;

def SIFIVE_E21 : RISCVProcessorModel<"sifive-e21",
                                     RocketModel,
                                     [Feature32Bit,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtC]>;

def SIFIVE_E24 : RISCVProcessorModel<"sifive-e24",
                                     RocketModel,
                                     [Feature32Bit,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtF,
                                      FeatureStdExtC]>;

def SIFIVE_E31 : RISCVProcessorModel<"sifive-e31",
                                     RocketModel,
                                     [Feature32Bit,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtC]>;

def SIFIVE_E34 : RISCVProcessorModel<"sifive-e34",
                                     RocketModel,
                                     [Feature32Bit,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtF,
                                      FeatureStdExtC]>;

def SIFIVE_E76 : RISCVProcessorModel<"sifive-e76",
                                     SiFive7Model,
                                     [Feature32Bit,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtF,
                                      FeatureStdExtC],
                                     [TuneSiFive7]>;

def SIFIVE_S21 : RISCVProcessorModel<"sifive-s21",
                                     RocketModel,
                                     [Feature64Bit,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtC]>;

def SIFIVE_S51 : RISCVProcessorModel<"sifive-s51",
                                     RocketModel,
                                     [Feature64Bit,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtC]>;

def SIFIVE_S54 : RISCVProcessorModel<"sifive-s54",
                                      RocketModel,
                                      [Feature64Bit,
                                       FeatureStdExtM,
                                       FeatureStdExtA,
                                       FeatureStdExtF,
                                       FeatureStdExtD,
                                       FeatureStdExtC]>;

def SIFIVE_S76 : RISCVProcessorModel<"sifive-s76",
                                     SiFive7Model,
                                     [Feature64Bit,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtF,
                                      FeatureStdExtD,
                                      FeatureStdExtC],
                                     [TuneSiFive7]>;

def SIFIVE_U54 : RISCVProcessorModel<"sifive-u54",
                                     RocketModel,
                                     [Feature64Bit,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtF,
                                      FeatureStdExtD,
                                      FeatureStdExtC]>;

def SIFIVE_U74 : RISCVProcessorModel<"sifive-u74",
                                     SiFive7Model,
                                     [Feature64Bit,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtF,
                                      FeatureStdExtD,
                                      FeatureStdExtC],
                                     [TuneSiFive7]>;

def SYNTACORE_SCR1_BASE : RISCVProcessorModel<"syntacore-scr1-base",
                                              SyntacoreSCR1Model,
                                              [Feature32Bit,
                                               FeatureStdExtC],
                                              [TuneNoDefaultUnroll]>;

def SYNTACORE_SCR1_MAX : RISCVProcessorModel<"syntacore-scr1-max",
                                             SyntacoreSCR1Model,
                                             [Feature32Bit,
                                              FeatureStdExtM,
                                              FeatureStdExtC],
                                             [TuneNoDefaultUnroll]>;
