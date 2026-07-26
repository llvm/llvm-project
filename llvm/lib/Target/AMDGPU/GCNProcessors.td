//===-- GCNProcessors.td - GCN Processor definitions ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Shared ArchFeature lists. Grouped by wavefront mode and the XNACK/SRAMECC/WGP
// features that recur across many GPUs in a generation.
defvar ArchFeaturesW64Xnack = [FEATURE_FAST_FMA_F32, FEATURE_FAST_DENORMAL_F32,
                              FEATURE_XNACK, FEATURE_XNACK_ON_OFF_MODES];
defvar ArchFeaturesW64XnackSramEcc = ArchFeaturesW64Xnack # [FEATURE_SRAMECC];
defvar ArchFeaturesW32XnackWgp = [FEATURE_FAST_FMA_F32, FEATURE_FAST_DENORMAL_F32,
                                 FEATURE_WAVE32, FEATURE_XNACK,
                                 FEATURE_XNACK_ON_OFF_MODES, FEATURE_WGP];
defvar ArchFeaturesW32Wgp = [FEATURE_FAST_FMA_F32, FEATURE_FAST_DENORMAL_F32,
                            FEATURE_WAVE32, FEATURE_WGP];

// The code produced for "generic" is only useful for tests and cannot
// be expected to execute on any target.
def : AMDGPUProcessorModel<"generic", NoSchedModel, [], [6, 0, 0]> {
  let IsPseudoTarget = true;
}

def : AMDGPUProcessorModel<"generic-hsa", NoSchedModel,
  [FeatureFlatAddressSpace], [7, 0, 0]> {
  let IsPseudoTarget = true;
}

//===------------------------------------------------------------===//
// GCN GFX6 (Southern Islands (SI)).
//===------------------------------------------------------------===//

def GFX600 : AMDGPUProcessorModel<"gfx600", SIFullSpeedModel,
  FeatureISAVersion6_0_0.Features, [6, 0, 0]> {
  let ArchFeatures = [FEATURE_FAST_FMA_F32];
}

def : ProcessorAlias<"tahiti", "gfx600">;

def GFX601 : AMDGPUProcessorModel<"gfx601", SIQuarterSpeedModel,
  FeatureISAVersion6_0_1.Features, [6, 0, 1]
>;

def : ProcessorAlias<"pitcairn", "gfx601">;
def : ProcessorAlias<"verde", "gfx601">;

def GFX602 : AMDGPUProcessorModel<"gfx602", SIQuarterSpeedModel,
  FeatureISAVersion6_0_2.Features, [6, 0, 2]
>;

def : ProcessorAlias<"hainan", "gfx602">;
def : ProcessorAlias<"oland", "gfx602">;

//===------------------------------------------------------------===//
// GCN GFX7 (Sea Islands (CI)).
//===------------------------------------------------------------===//

def GFX700 : AMDGPUProcessorModel<"gfx700", SIQuarterSpeedModel,
  FeatureISAVersion7_0_0.Features, [7, 0, 0]
>;

def : ProcessorAlias<"kaveri", "gfx700">;

def GFX701 : AMDGPUProcessorModel<"gfx701", SIFullSpeedModel,
  FeatureISAVersion7_0_1.Features, [7, 0, 1]> {
  let ArchFeatures = [FEATURE_FAST_FMA_F32];
}

def : ProcessorAlias<"hawaii", "gfx701">;

def GFX702 : AMDGPUProcessorModel<"gfx702", SIQuarterSpeedModel,
  FeatureISAVersion7_0_2.Features, [7, 0, 2]> {
  let ArchFeatures = [FEATURE_FAST_FMA_F32];
}

def GFX703 : AMDGPUProcessorModel<"gfx703", SIQuarterSpeedModel,
  FeatureISAVersion7_0_3.Features, [7, 0, 3]
>;

def : ProcessorAlias<"kabini", "gfx703">;
def : ProcessorAlias<"mullins", "gfx703">;

def GFX704 : AMDGPUProcessorModel<"gfx704", SIQuarterSpeedModel,
  FeatureISAVersion7_0_4.Features, [7, 0, 4]
>;

def : ProcessorAlias<"bonaire", "gfx704">;

def GFX705 : AMDGPUProcessorModel<"gfx705", SIQuarterSpeedModel,
  FeatureISAVersion7_0_5.Features, [7, 0, 5]
>;

//===------------------------------------------------------------===//
// GCN GFX8 (Volcanic Islands (VI)).
//===------------------------------------------------------------===//

def GFX801 : AMDGPUProcessorModel<"gfx801", SIQuarterSpeedModel,
  FeatureISAVersion8_0_1.Features, [8, 0, 1]> {
  let ArchFeatures = ArchFeaturesW64Xnack;
}

def : ProcessorAlias<"carrizo", "gfx801">;

def GFX802 : AMDGPUProcessorModel<"gfx802", SIQuarterSpeedModel,
  FeatureISAVersion8_0_2.Features, [8, 0, 2]> {
  let ArchFeatures = [FEATURE_FAST_DENORMAL_F32, FEATURE_SGPR_INIT_BUG];
}

def : ProcessorAlias<"iceland", "gfx802">;
def : ProcessorAlias<"tonga", "gfx802">;

def GFX803 : AMDGPUProcessorModel<"gfx803", SIQuarterSpeedModel,
  FeatureISAVersion8_0_3.Features, [8, 0, 3]> {
  let ArchFeatures = [FEATURE_FAST_DENORMAL_F32];
}

def : ProcessorAlias<"fiji", "gfx803">;
def : ProcessorAlias<"polaris10", "gfx803">;
def : ProcessorAlias<"polaris11", "gfx803">;

def GFX805 : AMDGPUProcessorModel<"gfx805", SIQuarterSpeedModel,
  FeatureISAVersion8_0_5.Features, [8, 0, 5]> {
  let ArchFeatures = [FEATURE_FAST_DENORMAL_F32, FEATURE_SGPR_INIT_BUG];
}

def : ProcessorAlias<"tongapro", "gfx805">;

def GFX810 : AMDGPUProcessorModel<"gfx810", SIQuarterSpeedModel,
  FeatureISAVersion8_1_0.Features, [8, 1, 0]> {
  let ArchFeatures = [FEATURE_FAST_DENORMAL_F32, FEATURE_XNACK, FEATURE_XNACK_ON_OFF_MODES];
}

def : ProcessorAlias<"stoney", "gfx810">;

//===------------------------------------------------------------===//
// GCN GFX9.
//===------------------------------------------------------------===//

def GFX900 : AMDGPUProcessorModel<"gfx900", SIQuarterSpeedModel,
  FeatureISAVersion9_0_0.Features, [9, 0, 0]> {
  let ArchFeatures = ArchFeaturesW64Xnack;
}

def GFX902 : AMDGPUProcessorModel<"gfx902", SIQuarterSpeedModel,
  FeatureISAVersion9_0_2.Features, [9, 0, 2]> {
  let ArchFeatures = ArchFeaturesW64Xnack;
}

def GFX904 : AMDGPUProcessorModel<"gfx904", SIQuarterSpeedModel,
  FeatureISAVersion9_0_4.Features, [9, 0, 4]> {
  let ArchFeatures = ArchFeaturesW64Xnack;
}

def GFX906 : AMDGPUProcessorModel<"gfx906", SIQuarterSpeedModel,
  FeatureISAVersion9_0_6.Features, [9, 0, 6]> {
  let ArchFeatures = ArchFeaturesW64XnackSramEcc;
}

def GFX908 : AMDGPUProcessorModel<"gfx908", SIQuarterSpeedModel,
  FeatureISAVersion9_0_8.Features, [9, 0, 8]> {
  let ArchFeatures = ArchFeaturesW64XnackSramEcc;
}

def GFX909 : AMDGPUProcessorModel<"gfx909", SIQuarterSpeedModel,
  FeatureISAVersion9_0_9.Features, [9, 0, 9]> {
  let ArchFeatures = ArchFeaturesW64Xnack;
}

def GFX90A : AMDGPUProcessorModel<"gfx90a", SIDPFullSpeedModel,
  FeatureISAVersion9_0_A.Features, [9, 0, 0xa]> {
  let ArchFeatures = ArchFeaturesW64XnackSramEcc;
}

def GFX90C : AMDGPUProcessorModel<"gfx90c", SIQuarterSpeedModel,
  FeatureISAVersion9_0_C.Features, [9, 0, 0xc]> {
  let ArchFeatures = ArchFeaturesW64Xnack;
}

def GFX942 : AMDGPUProcessorModel<"gfx942", SIDPGFX942FullSpeedModel,
  FeatureISAVersion9_4_2.Features, [9, 4, 2]> {
  let ArchFeatures = ArchFeaturesW64XnackSramEcc;
}

def GFX950 : AMDGPUProcessorModel<"gfx950", SIDPGFX950FullSpeedModel,
  FeatureISAVersion9_5_0.Features, [9, 5, 0]> {
  let ArchFeatures = ArchFeaturesW64XnackSramEcc;
}

def GFX9_GENERIC : AMDGPUProcessorModel<"gfx9-generic", SIQuarterSpeedModel,
  FeatureISAVersion9_Generic.Features, [9, 0, 0]> {
  let ArchFeatures = ArchFeaturesW64Xnack;
  let CoveredGPUs = [GFX900, GFX902, GFX904, GFX906, GFX909, GFX90C];
}

def GFX9_4_GENERIC : AMDGPUProcessorModel<"gfx9-4-generic", SIDPGFX942FullSpeedModel,
  FeatureISAVersion9_4_Generic.Features, [9, 4, 0]> {
  let ArchFeatures = ArchFeaturesW64XnackSramEcc;
  let CoveredGPUs = [GFX942, GFX950];
}

//===----------------------------------------------------------------------===//
// GCN GFX10.
//===----------------------------------------------------------------------===//

def GFX1010 : AMDGPUProcessorModel<"gfx1010", GFX10SpeedModel,
  FeatureISAVersion10_1_0.Features, [10, 1, 0]> {
  let ArchFeatures = ArchFeaturesW32XnackWgp;
}

def GFX1011 : AMDGPUProcessorModel<"gfx1011", GFX10SpeedModel,
  FeatureISAVersion10_1_1.Features, [10, 1, 1]> {
  let ArchFeatures = ArchFeaturesW32XnackWgp;
}

def GFX1012 : AMDGPUProcessorModel<"gfx1012", GFX10SpeedModel,
  FeatureISAVersion10_1_2.Features, [10, 1, 2]> {
  let ArchFeatures = ArchFeaturesW32XnackWgp;
}

def GFX1013 : AMDGPUProcessorModel<"gfx1013", GFX10SpeedModel,
  FeatureISAVersion10_1_3.Features, [10, 1, 3]> {
  let ArchFeatures = ArchFeaturesW32XnackWgp;
}

def GFX1030 : AMDGPUProcessorModel<"gfx1030", GFX10SpeedModel,
  FeatureISAVersion10_3_0.Features, [10, 3, 0]> {
  let ArchFeatures = ArchFeaturesW32Wgp;
}

def GFX1031 : AMDGPUProcessorModel<"gfx1031", GFX10SpeedModel,
  FeatureISAVersion10_3_0.Features, [10, 3, 1]> {
  let ArchFeatures = ArchFeaturesW32Wgp;
}

def GFX1032 : AMDGPUProcessorModel<"gfx1032", GFX10SpeedModel,
  FeatureISAVersion10_3_0.Features, [10, 3, 2]> {
  let ArchFeatures = ArchFeaturesW32Wgp;
}

def GFX1033 : AMDGPUProcessorModel<"gfx1033", GFX10SpeedModel,
  FeatureISAVersion10_3_0.Features, [10, 3, 3]> {
  let ArchFeatures = ArchFeaturesW32Wgp;
}

def GFX1034 : AMDGPUProcessorModel<"gfx1034", GFX10SpeedModel,
  FeatureISAVersion10_3_0.Features, [10, 3, 4]> {
  let ArchFeatures = ArchFeaturesW32Wgp;
}

def GFX1035 : AMDGPUProcessorModel<"gfx1035", GFX10SpeedModel,
  FeatureISAVersion10_3_0.Features, [10, 3, 5]> {
  let ArchFeatures = ArchFeaturesW32Wgp;
}

def GFX1036 : AMDGPUProcessorModel<"gfx1036", GFX10SpeedModel,
  FeatureISAVersion10_3_0.Features, [10, 3, 6]> {
  let ArchFeatures = ArchFeaturesW32Wgp;
}

def GFX10_1_GENERIC : AMDGPUProcessorModel<"gfx10-1-generic", GFX10SpeedModel,
  FeatureISAVersion10_1_Generic.Features, [10, 1, 0]> {
  let ArchFeatures = ArchFeaturesW32XnackWgp;
  let CoveredGPUs = [GFX1010, GFX1011, GFX1012, GFX1013];
}

def GFX10_3_GENERIC : AMDGPUProcessorModel<"gfx10-3-generic", GFX10SpeedModel,
  FeatureISAVersion10_3_Generic.Features, [10, 3, 0]> {
  let ArchFeatures = ArchFeaturesW32Wgp;
  let CoveredGPUs = [GFX1030, GFX1031, GFX1032, GFX1033, GFX1034, GFX1035, GFX1036];
}

//===----------------------------------------------------------------------===//
// GCN GFX11.
//===----------------------------------------------------------------------===//

def GFX1100 : AMDGPUProcessorModel<"gfx1100", GFX11SpeedModel,
  FeatureISAVersion11_0_0.Features, [11, 0, 0]> {
  let ArchFeatures = ArchFeaturesW32Wgp;
}

def GFX1101 : AMDGPUProcessorModel<"gfx1101", GFX11SpeedModel,
  FeatureISAVersion11_0_1.Features, [11, 0, 1]> {
  let ArchFeatures = ArchFeaturesW32Wgp;
}

def GFX1102 : AMDGPUProcessorModel<"gfx1102", GFX11SpeedModel,
  FeatureISAVersion11_0_2.Features, [11, 0, 2]> {
  let ArchFeatures = ArchFeaturesW32Wgp;
}

def GFX1103 : AMDGPUProcessorModel<"gfx1103", GFX11SpeedModel,
  FeatureISAVersion11_0_3.Features, [11, 0, 3]> {
  let ArchFeatures = ArchFeaturesW32Wgp;
}

def GFX1150 : AMDGPUProcessorModel<"gfx1150", GFX11SpeedModel,
  FeatureISAVersion11_5_0.Features, [11, 5, 0]> {
  let ArchFeatures = ArchFeaturesW32Wgp;
}

def GFX1151 : AMDGPUProcessorModel<"gfx1151", GFX11SpeedModel,
  FeatureISAVersion11_5_1.Features, [11, 5, 1]> {
  let ArchFeatures = ArchFeaturesW32Wgp;
}

def GFX1152 : AMDGPUProcessorModel<"gfx1152", GFX11SpeedModel,
  FeatureISAVersion11_5_2.Features, [11, 5, 2]> {
  let ArchFeatures = ArchFeaturesW32Wgp;
}

def GFX1153 : AMDGPUProcessorModel<"gfx1153", GFX11SpeedModel,
  FeatureISAVersion11_5_Common.Features, [11, 5, 3]> {
  let ArchFeatures = ArchFeaturesW32Wgp;
}

def GFX1154 : AMDGPUProcessorModel<"gfx1154", GFX11SpeedModel,
  FeatureISAVersion11_5_Common.Features, [11, 5, 4]> {
  let ArchFeatures = ArchFeaturesW32Wgp;
}

def GFX1170 : AMDGPUProcessorModel<"gfx1170", GFX11SpeedModel,
  FeatureISAVersion11_7_Common.Features, [11, 7, 0]> {
  let ArchFeatures = ArchFeaturesW32Wgp;
}

def GFX1171 : AMDGPUProcessorModel<"gfx1171", GFX11SpeedModel,
  FeatureISAVersion11_7_Common.Features, [11, 7, 1]> {
  let ArchFeatures = ArchFeaturesW32Wgp;
}

def GFX1172 : AMDGPUProcessorModel<"gfx1172", GFX11SpeedModel,
  FeatureISAVersion11_7_Common.Features, [11, 7, 2]> {
  let ArchFeatures = ArchFeaturesW32Wgp;
}

def GFX11_GENERIC : AMDGPUProcessorModel<"gfx11-generic", GFX11SpeedModel,
  FeatureISAVersion11_Generic.Features, [11, 0, 3]> {
  let ArchFeatures = ArchFeaturesW32Wgp;
  let CoveredGPUs = [GFX1100, GFX1101, GFX1102, GFX1103, GFX1150, GFX1151, GFX1152, GFX1153,
                     GFX1154];
}

def GFX11_7_GENERIC : AMDGPUProcessorModel<"gfx11-7-generic", GFX11SpeedModel,
  FeatureISAVersion11_7_Generic.Features, [11, 7, 0]> {
  let ArchFeatures = ArchFeaturesW32Wgp;
  let CoveredGPUs = [GFX1170, GFX1171, GFX1172];
}

//===----------------------------------------------------------------------===//
// GCN GFX12.
//===----------------------------------------------------------------------===//

def GFX1200 : AMDGPUProcessorModel<"gfx1200", GFX12SpeedModel,
  FeatureISAVersion12.Features, [12, 0, 0]> {
  let ArchFeatures = ArchFeaturesW32Wgp;
}

def GFX1201 : AMDGPUProcessorModel<"gfx1201", GFX12SpeedModel,
  FeatureISAVersion12.Features, [12, 0, 1]> {
  let ArchFeatures = ArchFeaturesW32Wgp;
}

def GFX12_GENERIC : AMDGPUProcessorModel<"gfx12-generic", GFX12SpeedModel,
  FeatureISAVersion12_Generic.Features, [12, 0, 0]> {
  let ArchFeatures = ArchFeaturesW32Wgp;
  let CoveredGPUs = [GFX1200, GFX1201];
}

def GFX1250 : AMDGPUProcessorModel<"gfx1250", GFX1250SpeedModel,
  FeatureISAVersion12_50.Features, [12, 5, 0]> {
  let ArchFeatures = [FEATURE_FAST_FMA_F32, FEATURE_FAST_DENORMAL_F32, FEATURE_WAVE32, FEATURE_XNACK, FEATURE_SRAMECC];
}

def GFX1251 : AMDGPUProcessorModel<"gfx1251", GFX1251SpeedModel,
  FeatureISAVersion12_51.Features, [12, 5, 1]> {
  let ArchFeatures = [FEATURE_FAST_FMA_F32, FEATURE_FAST_DENORMAL_F32, FEATURE_WAVE32, FEATURE_XNACK, FEATURE_SRAMECC];
}

def GFX12_5_GENERIC : AMDGPUProcessorModel<"gfx12-5-generic", GFX125xGenericSpeedModel,
  FeatureISAVersion12_5_Generic.Features, [12, 5, 0]> {
  let ArchFeatures = [FEATURE_FAST_FMA_F32, FEATURE_FAST_DENORMAL_F32, FEATURE_WAVE32, FEATURE_XNACK, FEATURE_SRAMECC];
  let CoveredGPUs = [GFX1250, GFX1251];
}

//===----------------------------------------------------------------------===//
// GCN GFX13.
//===----------------------------------------------------------------------===//

def GFX1310 : AMDGPUProcessorModel<"gfx1310", GFX12SpeedModel,
  FeatureISAVersion13.Features, [13, 1, 0]> {
  let ArchFeatures = ArchFeaturesW32Wgp;
}

def GFX13_GENERIC : AMDGPUProcessorModel<"gfx13-generic", GFX12SpeedModel,
  FeatureISAVersion13_Generic.Features, [13, 1, 0]> {
  let ArchFeatures = ArchFeaturesW32Wgp;
  let CoveredGPUs = [GFX1310];
}
