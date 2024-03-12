//===-- RISCVProcessors.td - RISC-V Processors -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RISC-V processors supported.
//===----------------------------------------------------------------------===//

defvar RVA20U64Features = [Feature64Bit,
                           FeatureStdExtM,
                           FeatureStdExtA,
                           FeatureStdExtF,
                           FeatureStdExtD,
                           FeatureStdExtC,
                           FeatureStdExtZicntr,
                           FeatureStdExtZiccif,
                           FeatureStdExtZiccrse,
                           FeatureStdExtZiccamoa,
                           FeatureStdExtZa128rs,
                           FeatureStdExtZicclsm];

defvar RVA20S64Features = !listconcat(RVA20U64Features,
                                      [FeatureStdExtZifencei,
                                       FeatureStdExtSvbare,
                                       FeatureStdExtSvade,
                                       FeatureStdExtSsccptr,
                                       FeatureStdExtSstvecd,
                                       FeatureStdExtSstvala]);

defvar RVA22U64Features = [Feature64Bit,
                           FeatureStdExtM,
                           FeatureStdExtA,
                           FeatureStdExtF,
                           FeatureStdExtD,
                           FeatureStdExtC,
                           FeatureStdExtZicntr,
                           FeatureStdExtZiccif,
                           FeatureStdExtZiccrse,
                           FeatureStdExtZiccamoa,
                           FeatureStdExtZicclsm,
                           FeatureStdExtZa64rs,
                           FeatureStdExtZihpm,
                           FeatureStdExtZihintpause,
                           FeatureStdExtZba,
                           FeatureStdExtZbb,
                           FeatureStdExtZbs,
                           FeatureStdExtZic64b,
                           FeatureStdExtZicbom,
                           FeatureStdExtZicbop,
                           FeatureStdExtZicboz,
                           FeatureStdExtZfhmin,
                           FeatureStdExtZkt];

defvar RVA22S64Features = !listconcat(RVA22U64Features,
                                      [FeatureStdExtZifencei,
                                       FeatureStdExtSvbare,
                                       FeatureStdExtSvade,
                                       FeatureStdExtSsccptr,
                                       FeatureStdExtSstvecd,
                                       FeatureStdExtSstvala,
                                       FeatureStdExtSscounterenw,
                                       FeatureStdExtSvpbmt,
                                       FeatureStdExtSvinval]);

defvar RVA23U64Features = [Feature64Bit,
                           FeatureStdExtM,
                           FeatureStdExtA,
                           FeatureStdExtF,
                           FeatureStdExtD,
                           FeatureStdExtC,
                           FeatureStdExtZicntr,
                           FeatureStdExtZihpm,
                           FeatureStdExtZiccif,
                           FeatureStdExtZiccrse,
                           FeatureStdExtZiccamoa,
                           FeatureStdExtZicclsm,
                           FeatureStdExtZa64rs,
                           FeatureStdExtZihintpause,
                           FeatureStdExtZba,
                           FeatureStdExtZbb,
                           FeatureStdExtZbs,
                           FeatureStdExtZic64b,
                           FeatureStdExtZicbom,
                           FeatureStdExtZicbop,
                           FeatureStdExtZicboz,
                           FeatureStdExtZfhmin,
                           FeatureStdExtZkt,
                           FeatureStdExtV,
                           FeatureStdExtZvfhmin,
                           FeatureStdExtZvbb,
                           FeatureStdExtZvkt,
                           FeatureStdExtZihintntl,
                           FeatureStdExtZicond,
                           FeatureStdExtZimop,
                           FeatureStdExtZcmop,
                           FeatureStdExtZcb,
                           FeatureStdExtZfa,
                           FeatureStdExtZawrs];

defvar RVA23S64Features = !listconcat(RVA23U64Features,
                                      [FeatureStdExtZifencei,
                                       FeatureStdExtSvbare,
                                       FeatureStdExtSvade,
                                       FeatureStdExtSsccptr,
                                       FeatureStdExtSstvecd,
                                       FeatureStdExtSstvala,
                                       FeatureStdExtSscounterenw,
                                       FeatureStdExtSvpbmt,
                                       FeatureStdExtSvinval,
                                       FeatureStdExtSvnapot,
                                       FeatureStdExtSstc,
                                       FeatureStdExtSscofpmf,
                                       FeatureStdExtSsnpm,
                                       FeatureStdExtSsu64xl,
                                       FeatureStdExtH,
                                       FeatureStdExtSsstateen,
                                       FeatureStdExtShcounterenw,
                                       FeatureStdExtShvstvala,
                                       FeatureStdExtShtvala,
                                       FeatureStdExtShvstvecd,
                                       FeatureStdExtShvsatpa,
                                       FeatureStdExtShgatpa]);

defvar RVB23U64Features = [Feature64Bit,
                           FeatureStdExtM,
                           FeatureStdExtA,
                           FeatureStdExtF,
                           FeatureStdExtD,
                           FeatureStdExtC,
                           FeatureStdExtZicntr,
                           FeatureStdExtZihpm,
                           FeatureStdExtZiccif,
                           FeatureStdExtZiccrse,
                           FeatureStdExtZiccamoa,
                           FeatureStdExtZicclsm,
                           FeatureStdExtZa64rs,
                           FeatureStdExtZihintpause,
                           FeatureStdExtZba,
                           FeatureStdExtZbb,
                           FeatureStdExtZbs,
                           FeatureStdExtZic64b,
                           FeatureStdExtZicbom,
                           FeatureStdExtZicbop,
                           FeatureStdExtZicboz,
                           FeatureStdExtZkt,
                           FeatureStdExtZihintntl,
                           FeatureStdExtZicond,
                           FeatureStdExtZimop,
                           FeatureStdExtZcmop,
                           FeatureStdExtZcb,
                           FeatureStdExtZfa,
                           FeatureStdExtZawrs];

defvar RVB23S64Features = !listconcat(RVB23U64Features,
                                      [FeatureStdExtZifencei,
                                       FeatureStdExtSvnapot,
                                       FeatureStdExtSvbare,
                                       FeatureStdExtSvade,
                                       FeatureStdExtSsccptr,
                                       FeatureStdExtSstvecd,
                                       FeatureStdExtSstvala,
                                       FeatureStdExtSscounterenw,
                                       FeatureStdExtSvpbmt,
                                       FeatureStdExtSvinval,
                                       FeatureStdExtSstc,
                                       FeatureStdExtSscofpmf,
                                       FeatureStdExtSsu64xl]);

defvar RVM23U32Features = [Feature32Bit,
                           FeatureStdExtM,
                           FeatureStdExtZba,
                           FeatureStdExtZbb,
                           FeatureStdExtZbs,
                           FeatureStdExtZicond,
                           FeatureStdExtZihintpause,
                           FeatureStdExtZihintntl,
                           FeatureStdExtZce,
                           FeatureStdExtZicbop,
                           FeatureStdExtZimop,
                           FeatureStdExtZcmop];

class RISCVTuneInfo {
  bits<8> PrefFunctionAlignment = 1;
  bits<8> PrefLoopAlignment = 1;

  // Information needed by LoopDataPrefetch.
  bits<16> CacheLineSize = 0;
  bits<16> PrefetchDistance = 0;
  bits<16> MinPrefetchStride = 1;
  bits<32> MaxPrefetchIterationsAhead = -1;

  bits<32> MinimumJumpTableEntries = 5;
}

def RISCVTuneInfoTable : GenericTable {
  let FilterClass = "RISCVTuneInfo";
  let CppTypeName = "RISCVTuneInfo";
  let Fields = ["Name", "PrefFunctionAlignment", "PrefLoopAlignment",
                "CacheLineSize", "PrefetchDistance",
                "MinPrefetchStride", "MaxPrefetchIterationsAhead",
                "MinimumJumpTableEntries"];
}

def getRISCVTuneInfo : SearchIndex {
  let Table = RISCVTuneInfoTable;
  let Key = ["Name"];
}

class GenericTuneInfo: RISCVTuneInfo;

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

class RISCVGenericProcessorModel<string n, list<SubtargetFeature> f>
    : RISCVProcessorModel<n, NoSchedModel, f>,
      GenericTuneInfo;

def GENERIC_RV32 : RISCVGenericProcessorModel<"generic-rv32",
                                              [Feature32Bit]>;

def GENERIC_RV64 : RISCVGenericProcessorModel<"generic-rv64",
                                              [Feature64Bit]>;

def GENERIC_RVI20U32 : RISCVGenericProcessorModel<"generic-rvi20u32",
                                                  [Feature32Bit]>;

def GENERIC_RVI20U64 : RISCVGenericProcessorModel<"generic-rvi20u64",
                                                  [Feature64Bit]>;

def GENERIC_RVA20U64 : RISCVGenericProcessorModel<"generic-rva20u64",
                                                  RVA20U64Features>;

def GENERIC_RVA20S64 : RISCVGenericProcessorModel<"generic-rva20s64",
                                                  RVA20S64Features>;

def GENERIC_RVA22U64 : RISCVGenericProcessorModel<"generic-rva22u64",
                                                  RVA22U64Features>;

def GENERIC_RVA22S64 : RISCVGenericProcessorModel<"generic-rva22s64", 
                                                  RVA22S64Features>;

def GENERIC_RVA23U64 : RISCVGenericProcessorModel<"generic-rva23u64", 
                                                  RVA23U64Features>;

def GENERIC_RVA23S64 : RISCVGenericProcessorModel<"generic-rva23s64", 
                                                  RVA23S64Features>;

def GENERIC_RVB23U64 : RISCVGenericProcessorModel<"generic-rvb23u64",
                                                  RVB23U64Features>;

def GENERIC_RVB23S64 : RISCVGenericProcessorModel<"generic-rvb23s64",
                                                  RVB23S64Features>;

def GENERIC_RVM23U32 : RISCVGenericProcessorModel<"generic-rvm23u32",
                                                  RVM23U32Features>;

// Support generic for compatibility with other targets. The triple will be used
// to change to the appropriate rv32/rv64 version.
def : ProcessorModel<"generic", NoSchedModel, []>, GenericTuneInfo;

def ROCKET_RV32 : RISCVProcessorModel<"rocket-rv32",
                                      RocketModel,
                                      [Feature32Bit,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtZicsr]>;
def ROCKET_RV64 : RISCVProcessorModel<"rocket-rv64",
                                      RocketModel,
                                      [Feature64Bit,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtZicsr]>;
def ROCKET : RISCVTuneProcessorModel<"rocket",
                                     RocketModel>;

def SIFIVE_7 : RISCVTuneProcessorModel<"sifive-7-series",
                                       SiFive7Model,
                                       [TuneSiFive7]>;

def SIFIVE_E20 : RISCVProcessorModel<"sifive-e20",
                                     RocketModel,
                                     [Feature32Bit,
                                      FeatureStdExtZicsr,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtC]>;

def SIFIVE_E21 : RISCVProcessorModel<"sifive-e21",
                                     RocketModel,
                                     [Feature32Bit,
                                      FeatureStdExtZicsr,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtC]>;

def SIFIVE_E24 : RISCVProcessorModel<"sifive-e24",
                                     RocketModel,
                                     [Feature32Bit,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtF,
                                      FeatureStdExtC]>;

def SIFIVE_E31 : RISCVProcessorModel<"sifive-e31",
                                     RocketModel,
                                     [Feature32Bit,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtZicsr,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtC]>;

def SIFIVE_E34 : RISCVProcessorModel<"sifive-e34",
                                     RocketModel,
                                     [Feature32Bit,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtF,
                                      FeatureStdExtC]>;

def SIFIVE_E76 : RISCVProcessorModel<"sifive-e76",
                                     SiFive7Model,
                                     [Feature32Bit,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtF,
                                      FeatureStdExtC],
                                     [TuneSiFive7]>;

def SIFIVE_S21 : RISCVProcessorModel<"sifive-s21",
                                     RocketModel,
                                     [Feature64Bit,
                                      FeatureStdExtZicsr,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtC]>;

def SIFIVE_S51 : RISCVProcessorModel<"sifive-s51",
                                     RocketModel,
                                     [Feature64Bit,
                                      FeatureStdExtZicsr,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtC]>;

def SIFIVE_S54 : RISCVProcessorModel<"sifive-s54",
                                      RocketModel,
                                      [Feature64Bit,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtM,
                                       FeatureStdExtA,
                                       FeatureStdExtF,
                                       FeatureStdExtD,
                                       FeatureStdExtC]>;

def SIFIVE_S76 : RISCVProcessorModel<"sifive-s76",
                                     SiFive7Model,
                                     [Feature64Bit,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtF,
                                      FeatureStdExtD,
                                      FeatureStdExtC,
                                      FeatureStdExtZihintpause],
                                     [TuneSiFive7]>;

def SIFIVE_U54 : RISCVProcessorModel<"sifive-u54",
                                     RocketModel,
                                     [Feature64Bit,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtF,
                                      FeatureStdExtD,
                                      FeatureStdExtC]>;

def SIFIVE_U74 : RISCVProcessorModel<"sifive-u74",
                                     SiFive7Model,
                                     [Feature64Bit,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtF,
                                      FeatureStdExtD,
                                      FeatureStdExtC],
                                     [TuneSiFive7]>;

def SIFIVE_X280 : RISCVProcessorModel<"sifive-x280", SiFive7Model,
                                      [Feature64Bit,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtM,
                                       FeatureStdExtA,
                                       FeatureStdExtF,
                                       FeatureStdExtD,
                                       FeatureStdExtC,
                                       FeatureStdExtV,
                                       FeatureStdExtZvl512b,
                                       FeatureStdExtZfh,
                                       FeatureStdExtZvfh,
                                       FeatureStdExtZba,
                                       FeatureStdExtZbb],
                                      [TuneSiFive7,
                                       TuneDLenFactor2]>;

def SIFIVE_P450 : RISCVProcessorModel<"sifive-p450", SiFiveP400Model,
                                      [Feature64Bit,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtM,
                                       FeatureStdExtA,
                                       FeatureStdExtF,
                                       FeatureStdExtD,
                                       FeatureStdExtC,
                                       FeatureStdExtZa64rs,
                                       FeatureStdExtZic64b,
                                       FeatureStdExtZicbop,
                                       FeatureStdExtZicbom,
                                       FeatureStdExtZicboz,
                                       FeatureStdExtZiccamoa,
                                       FeatureStdExtZiccif,
                                       FeatureStdExtZicclsm,
                                       FeatureStdExtZiccrse,
                                       FeatureStdExtZihintntl,
                                       FeatureStdExtZihintpause,
                                       FeatureStdExtZihpm,
                                       FeatureStdExtZba,
                                       FeatureStdExtZbb,
                                       FeatureStdExtZbs,
                                       FeatureStdExtZfhmin,
                                       FeatureFastUnalignedAccess],
                                      [TuneNoDefaultUnroll,
                                       TuneConditionalCompressedMoveFusion,
                                       TuneLUIADDIFusion,
                                       TuneAUIPCADDIFusion]>;

def SIFIVE_P670 : RISCVProcessorModel<"sifive-p670", NoSchedModel,
                                      [Feature64Bit,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtM,
                                       FeatureStdExtA,
                                       FeatureStdExtF,
                                       FeatureStdExtD,
                                       FeatureStdExtC,
                                       FeatureStdExtZa64rs,
                                       FeatureStdExtZic64b,
                                       FeatureStdExtZicbop,
                                       FeatureStdExtZicbom,
                                       FeatureStdExtZicboz,
                                       FeatureStdExtZiccamoa,
                                       FeatureStdExtZiccif,
                                       FeatureStdExtZicclsm,
                                       FeatureStdExtZiccrse,
                                       FeatureStdExtZihintntl,
                                       FeatureStdExtZihintpause,
                                       FeatureStdExtZihpm,
                                       FeatureStdExtZba,
                                       FeatureStdExtZbb,
                                       FeatureStdExtZbs,
                                       FeatureStdExtZfhmin,
                                       FeatureStdExtV,
                                       FeatureStdExtZvl128b,
                                       FeatureStdExtZvbb,
                                       FeatureStdExtZvknc,
                                       FeatureStdExtZvkng,
                                       FeatureStdExtZvksc,
                                       FeatureStdExtZvksg,
                                       FeatureFastUnalignedAccess],
                                      [TuneNoDefaultUnroll,
                                       TuneConditionalCompressedMoveFusion,
                                       TuneLUIADDIFusion,
                                       TuneAUIPCADDIFusion,
                                       TuneNoSinkSplatOperands]>;

def SYNTACORE_SCR1_BASE : RISCVProcessorModel<"syntacore-scr1-base",
                                              SyntacoreSCR1Model,
                                              [Feature32Bit,
                                               FeatureStdExtZicsr,
                                               FeatureStdExtZifencei,
                                               FeatureStdExtC],
                                              [TuneNoDefaultUnroll]>;

def SYNTACORE_SCR1_MAX : RISCVProcessorModel<"syntacore-scr1-max",
                                             SyntacoreSCR1Model,
                                             [Feature32Bit,
                                              FeatureStdExtZicsr,
                                              FeatureStdExtZifencei,
                                              FeatureStdExtM,
                                              FeatureStdExtC],
                                             [TuneNoDefaultUnroll]>;

def VENTANA_VEYRON_V1 : RISCVProcessorModel<"veyron-v1",
                                            NoSchedModel,
                                            [Feature64Bit,
                                             FeatureStdExtZifencei,
                                             FeatureStdExtZicsr,
                                             FeatureStdExtZicntr,
                                             FeatureStdExtZihpm,
                                             FeatureStdExtZihintpause,
                                             FeatureStdExtM,
                                             FeatureStdExtA,
                                             FeatureStdExtF,
                                             FeatureStdExtD,
                                             FeatureStdExtC,
                                             FeatureStdExtZba,
                                             FeatureStdExtZbb,
                                             FeatureStdExtZbc,
                                             FeatureStdExtZbs,
                                             FeatureStdExtZicbom,
                                             FeatureStdExtZicbop,
                                             FeatureStdExtZicboz,
                                             FeatureVendorXVentanaCondOps],
                                             [TuneVentanaVeyron,
                                              TuneLUIADDIFusion,
                                              TuneAUIPCADDIFusion,
                                              TuneZExtHFusion,
                                              TuneZExtWFusion,
                                              TuneShiftedZExtWFusion,
                                              TuneLDADDFusion]>;

def XIANGSHAN_NANHU : RISCVProcessorModel<"xiangshan-nanhu",
                                          XiangShanNanHuModel,
                                          [Feature64Bit,
                                           FeatureStdExtZicsr,
                                           FeatureStdExtZifencei,
                                           FeatureStdExtM,
                                           FeatureStdExtA,
                                           FeatureStdExtF,
                                           FeatureStdExtD,
                                           FeatureStdExtC,
                                           FeatureStdExtZba,
                                           FeatureStdExtZbb,
                                           FeatureStdExtZbc,
                                           FeatureStdExtZbs,
                                           FeatureStdExtZkn,
                                           FeatureStdExtZksed,
                                           FeatureStdExtZksh,
                                           FeatureStdExtSvinval,
                                           FeatureStdExtZicbom,
                                           FeatureStdExtZicboz],
                                           [TuneNoDefaultUnroll,
                                            TuneZExtHFusion,
                                            TuneZExtWFusion,
                                            TuneShiftedZExtWFusion]>;
