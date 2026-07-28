//===-- GCNProcessors.td - GCN Processor definitions ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The code produced for "generic" is only useful for tests and cannot
// reasonably be expected to execute on any particular target.
def : ProcessorModel<"generic", NoSchedModel,
  []
>;

def : ProcessorModel<"generic-hsa", NoSchedModel,
  [FeatureFlatAddressSpace]
>;

//===------------------------------------------------------------===//
// GCN GFX6 (Southern Islands (SI)).
//===------------------------------------------------------------===//

def : ProcessorModel<"gfx600", SIFullSpeedModel,
  FeatureISAVersion6_0_0.Features
>;

def : ProcessorAlias<"tahiti", "gfx600">;

def : ProcessorModel<"gfx601", SIQuarterSpeedModel,
  FeatureISAVersion6_0_1.Features
>;

def : ProcessorAlias<"pitcairn", "gfx601">;
def : ProcessorAlias<"verde", "gfx601">;

def : ProcessorModel<"gfx602", SIQuarterSpeedModel,
  FeatureISAVersion6_0_2.Features
>;

def : ProcessorAlias<"hainan", "gfx602">;
def : ProcessorAlias<"oland", "gfx602">;

//===------------------------------------------------------------===//
// GCN GFX7 (Sea Islands (CI)).
//===------------------------------------------------------------===//

def : ProcessorModel<"gfx700", SIQuarterSpeedModel,
  FeatureISAVersion7_0_0.Features
>;

def : ProcessorAlias<"kaveri", "gfx700">;

def : ProcessorModel<"gfx701", SIFullSpeedModel,
  FeatureISAVersion7_0_1.Features
>;

def : ProcessorAlias<"hawaii", "gfx701">;

def : ProcessorModel<"gfx702", SIQuarterSpeedModel,
  FeatureISAVersion7_0_2.Features
>;

def : ProcessorModel<"gfx703", SIQuarterSpeedModel,
  FeatureISAVersion7_0_3.Features
>;

def : ProcessorAlias<"kabini", "gfx703">;
def : ProcessorAlias<"mullins", "gfx703">;

def : ProcessorModel<"gfx704", SIQuarterSpeedModel,
  FeatureISAVersion7_0_4.Features
>;

def : ProcessorAlias<"bonaire", "gfx704">;

def : ProcessorModel<"gfx705", SIQuarterSpeedModel,
  FeatureISAVersion7_0_5.Features
>;

//===------------------------------------------------------------===//
// GCN GFX8 (Volcanic Islands (VI)).
//===------------------------------------------------------------===//

def : ProcessorModel<"gfx801", SIQuarterSpeedModel,
  FeatureISAVersion8_0_1.Features
>;

def : ProcessorAlias<"carrizo", "gfx801">;

def : ProcessorModel<"gfx802", SIQuarterSpeedModel,
  FeatureISAVersion8_0_2.Features
>;

def : ProcessorAlias<"iceland", "gfx802">;
def : ProcessorAlias<"tonga", "gfx802">;

def : ProcessorModel<"gfx803", SIQuarterSpeedModel,
  FeatureISAVersion8_0_3.Features
>;

def : ProcessorAlias<"fiji", "gfx803">;
def : ProcessorAlias<"polaris10", "gfx803">;
def : ProcessorAlias<"polaris11", "gfx803">;

def : ProcessorModel<"gfx805", SIQuarterSpeedModel,
  FeatureISAVersion8_0_5.Features
>;

def : ProcessorAlias<"tongapro", "gfx805">;

def : ProcessorModel<"gfx810", SIQuarterSpeedModel,
  FeatureISAVersion8_1_0.Features
>;

def : ProcessorAlias<"stoney", "gfx810">;

//===------------------------------------------------------------===//
// GCN GFX9.
//===------------------------------------------------------------===//

def : ProcessorModel<"gfx900", SIQuarterSpeedModel,
  FeatureISAVersion9_0_0.Features
>;

def : ProcessorModel<"gfx902", SIQuarterSpeedModel,
  FeatureISAVersion9_0_2.Features
>;

def : ProcessorModel<"gfx904", SIQuarterSpeedModel,
  FeatureISAVersion9_0_4.Features
>;

def : ProcessorModel<"gfx906", SIQuarterSpeedModel,
  FeatureISAVersion9_0_6.Features
>;

def : ProcessorModel<"gfx908", SIQuarterSpeedModel,
  FeatureISAVersion9_0_8.Features
>;

def : ProcessorModel<"gfx909", SIQuarterSpeedModel,
  FeatureISAVersion9_0_9.Features
>;

def : ProcessorModel<"gfx90a", SIDPFullSpeedModel,
  FeatureISAVersion9_0_A.Features
>;

def : ProcessorModel<"gfx90c", SIQuarterSpeedModel,
  FeatureISAVersion9_0_C.Features
>;

def : ProcessorModel<"gfx942", SIDPGFX942FullSpeedModel,
  FeatureISAVersion9_4_2.Features
>;

def : ProcessorModel<"gfx950", SIDPGFX950FullSpeedModel,
  FeatureISAVersion9_5_0.Features
>;

// [gfx900, gfx902, gfx904, gfx906, gfx909, gfx90c]
def : ProcessorModel<"gfx9-generic", SIQuarterSpeedModel,
  FeatureISAVersion9_Generic.Features
>;

// [gfx942]
def : ProcessorModel<"gfx9-4-generic", SIDPGFX942FullSpeedModel,
  FeatureISAVersion9_4_Generic.Features
>;

//===----------------------------------------------------------------------===//
// GCN GFX10.
//===----------------------------------------------------------------------===//

def : ProcessorModel<"gfx1010", GFX10SpeedModel,
  FeatureISAVersion10_1_0.Features
>;

def : ProcessorModel<"gfx1011", GFX10SpeedModel,
  FeatureISAVersion10_1_1.Features
>;

def : ProcessorModel<"gfx1012", GFX10SpeedModel,
  FeatureISAVersion10_1_2.Features
>;

def : ProcessorModel<"gfx1013", GFX10SpeedModel,
  FeatureISAVersion10_1_3.Features
>;

def : ProcessorModel<"gfx1030", GFX10SpeedModel,
  FeatureISAVersion10_3_0.Features
>;

def : ProcessorModel<"gfx1031", GFX10SpeedModel,
  FeatureISAVersion10_3_0.Features
>;

def : ProcessorModel<"gfx1032", GFX10SpeedModel,
  FeatureISAVersion10_3_0.Features
>;

def : ProcessorModel<"gfx1033", GFX10SpeedModel,
  FeatureISAVersion10_3_0.Features
>;

def : ProcessorModel<"gfx1034", GFX10SpeedModel,
  FeatureISAVersion10_3_0.Features
>;

def : ProcessorModel<"gfx1035", GFX10SpeedModel,
  FeatureISAVersion10_3_0.Features
>;

def : ProcessorModel<"gfx1036", GFX10SpeedModel,
  FeatureISAVersion10_3_0.Features
>;

// [gfx1010, gfx1011, gfx1012, gfx1013]
def : ProcessorModel<"gfx10-1-generic", GFX10SpeedModel,
  FeatureISAVersion10_1_Generic.Features
>;

// [gfx1030, gfx1031, gfx1032, gfx1033, gfx1034, gfx1035, gfx1036]
def : ProcessorModel<"gfx10-3-generic", GFX10SpeedModel,
  FeatureISAVersion10_3_Generic.Features
>;

//===----------------------------------------------------------------------===//
// GCN GFX11.
//===----------------------------------------------------------------------===//

def : ProcessorModel<"gfx1100", GFX11SpeedModel,
  FeatureISAVersion11_0_0.Features
>;

def : ProcessorModel<"gfx1101", GFX11SpeedModel,
  FeatureISAVersion11_0_1.Features
>;

def : ProcessorModel<"gfx1102", GFX11SpeedModel,
  FeatureISAVersion11_0_2.Features
>;

def : ProcessorModel<"gfx1103", GFX11SpeedModel,
  FeatureISAVersion11_0_3.Features
>;

def : ProcessorModel<"gfx1150", GFX11SpeedModel,
  FeatureISAVersion11_5_0.Features
>;

def : ProcessorModel<"gfx1151", GFX11SpeedModel,
  FeatureISAVersion11_5_1.Features
>;

def : ProcessorModel<"gfx1152", GFX11SpeedModel,
  FeatureISAVersion11_5_2.Features
>;

def : ProcessorModel<"gfx1153", GFX11SpeedModel,
  FeatureISAVersion11_5_Common.Features
>;

def : ProcessorModel<"gfx1154", GFX11SpeedModel,
  FeatureISAVersion11_5_Common.Features
>;

def : ProcessorModel<"gfx1170", GFX11SpeedModel,
  FeatureISAVersion11_7_Common.Features
>;

def : ProcessorModel<"gfx1171", GFX11SpeedModel,
  FeatureISAVersion11_7_Common.Features
>;

def : ProcessorModel<"gfx1172", GFX11SpeedModel,
  FeatureISAVersion11_7_Common.Features
>;

// [gfx1100, gfx1101, gfx1102, gfx1103,
//  gfx1150, gfx1151, gfx1152, gfx1153, gfx1154]
def : ProcessorModel<"gfx11-generic", GFX11SpeedModel,
  FeatureISAVersion11_Generic.Features
>;

// [gfx1170, gfx1171, gfx1172]
def : ProcessorModel<"gfx11-7-generic", GFX11SpeedModel,
  FeatureISAVersion11_7_Generic.Features
>;

//===----------------------------------------------------------------------===//
// GCN GFX12.
//===----------------------------------------------------------------------===//

def : ProcessorModel<"gfx1200", GFX12SpeedModel,
  FeatureISAVersion12.Features
>;

def : ProcessorModel<"gfx1201", GFX12SpeedModel,
  FeatureISAVersion12.Features
>;

// [gfx1200, gfx1201]
def : ProcessorModel<"gfx12-generic", GFX12SpeedModel,
  FeatureISAVersion12_Generic.Features
>;

def : ProcessorModel<"gfx1250", GFX1250SpeedModel,
  FeatureISAVersion12_50.Features
>;

def : ProcessorModel<"gfx1251", GFX1251SpeedModel,
  FeatureISAVersion12_51.Features
>;

// [gfx1250, gfx1251]
def : ProcessorModel<"gfx12-5-generic", GFX125xGenericSpeedModel,
  FeatureISAVersion12_5_Generic.Features
>;

//===----------------------------------------------------------------------===//
// GCN GFX13.
//===----------------------------------------------------------------------===//

def : ProcessorModel<"gfx1310", GFX12SpeedModel,
  FeatureISAVersion13.Features
>;

// [gfx1310]
def : ProcessorModel<"gfx13-generic", GFX12SpeedModel,
  FeatureISAVersion13_Generic.Features
>;
