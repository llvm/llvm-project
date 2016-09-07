//===-- Processors.td - R600 Processor definitions ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

class Proc<string Name, ProcessorItineraries itin, list<SubtargetFeature> Features>
: Processor<Name, itin, Features>;

//===----------------------------------------------------------------------===//
// R600
//===----------------------------------------------------------------------===//
def : Proc<"r600",       R600_VLIW5_Itin,
    [FeatureR600, FeatureVertexCache, FeatureWavefrontSize64]>;

def : Proc<"r630",       R600_VLIW5_Itin,
    [FeatureR600, FeatureVertexCache, FeatureWavefrontSize32]>;

def : Proc<"rs880",      R600_VLIW5_Itin,
    [FeatureR600, FeatureWavefrontSize16]>;

def : Proc<"rv670",      R600_VLIW5_Itin,
    [FeatureR600, FeatureFP64, FeatureVertexCache, FeatureWavefrontSize64]>;

//===----------------------------------------------------------------------===//
// R700
//===----------------------------------------------------------------------===//

def : Proc<"rv710",      R600_VLIW5_Itin,
    [FeatureR700, FeatureVertexCache, FeatureWavefrontSize32]>;

def : Proc<"rv730",      R600_VLIW5_Itin,
    [FeatureR700, FeatureVertexCache, FeatureWavefrontSize32]>;

def : Proc<"rv770",      R600_VLIW5_Itin,
    [FeatureR700, FeatureFP64, FeatureVertexCache, FeatureWavefrontSize64]>;

//===----------------------------------------------------------------------===//
// Evergreen
//===----------------------------------------------------------------------===//

def : Proc<"cedar",      R600_VLIW5_Itin,
    [FeatureEvergreen, FeatureVertexCache, FeatureWavefrontSize32,
     FeatureCFALUBug]>;

def : Proc<"redwood",    R600_VLIW5_Itin,
    [FeatureEvergreen, FeatureVertexCache, FeatureWavefrontSize64,
     FeatureCFALUBug]>;

def : Proc<"sumo",       R600_VLIW5_Itin,
    [FeatureEvergreen, FeatureWavefrontSize64, FeatureCFALUBug]>;

def : Proc<"juniper",    R600_VLIW5_Itin,
    [FeatureEvergreen, FeatureVertexCache, FeatureWavefrontSize64]>;

def : Proc<"cypress",    R600_VLIW5_Itin,
    [FeatureEvergreen, FeatureFP64, FeatureVertexCache,
     FeatureWavefrontSize64]>;

//===----------------------------------------------------------------------===//
// Northern Islands
//===----------------------------------------------------------------------===//

def : Proc<"barts",      R600_VLIW5_Itin,
    [FeatureNorthernIslands, FeatureVertexCache, FeatureCFALUBug]>;

def : Proc<"turks",      R600_VLIW5_Itin,
    [FeatureNorthernIslands, FeatureVertexCache, FeatureCFALUBug]>;

def : Proc<"caicos",     R600_VLIW5_Itin,
    [FeatureNorthernIslands, FeatureCFALUBug]>;

def : Proc<"cayman",     R600_VLIW4_Itin,
    [FeatureNorthernIslands, FeatureFP64, FeatureCaymanISA]>;

//===----------------------------------------------------------------------===//
// Southern Islands
//===----------------------------------------------------------------------===//

def : ProcessorModel<"SI", SIFullSpeedModel,
  [FeatureSouthernIslands, FeatureFastFMAF32, HalfRate64Ops]
>;

def : ProcessorModel<"tahiti", SIFullSpeedModel,
  [FeatureSouthernIslands, FeatureFastFMAF32, HalfRate64Ops]
>;

def : ProcessorModel<"pitcairn", SIQuarterSpeedModel, [FeatureSouthernIslands]>;

def : ProcessorModel<"verde",    SIQuarterSpeedModel, [FeatureSouthernIslands]>;

def : ProcessorModel<"oland",    SIQuarterSpeedModel, [FeatureSouthernIslands]>;

def : ProcessorModel<"hainan",   SIQuarterSpeedModel, [FeatureSouthernIslands]>;

//===----------------------------------------------------------------------===//
// Sea Islands
//===----------------------------------------------------------------------===//

def : ProcessorModel<"bonaire",    SIQuarterSpeedModel,
  [FeatureSeaIslands, FeatureLDSBankCount32, FeatureISAVersion7_0_0]
>;

def : ProcessorModel<"kabini",     SIQuarterSpeedModel,
  [FeatureSeaIslands, FeatureLDSBankCount16]
>;

def : ProcessorModel<"kaveri",     SIQuarterSpeedModel,
  [FeatureSeaIslands, FeatureLDSBankCount32, FeatureISAVersion7_0_0]
>;

def : ProcessorModel<"hawaii", SIFullSpeedModel,
  [FeatureSeaIslands, FeatureFastFMAF32, HalfRate64Ops,
   FeatureLDSBankCount32, FeatureISAVersion7_0_1]
>;

def : ProcessorModel<"mullins",    SIQuarterSpeedModel,
  [FeatureSeaIslands, FeatureLDSBankCount16]>;

//===----------------------------------------------------------------------===//
// Volcanic Islands
//===----------------------------------------------------------------------===//

def : ProcessorModel<"tonga",   SIQuarterSpeedModel,
  [FeatureVolcanicIslands, FeatureSGPRInitBug, FeatureISAVersion8_0_0,
   FeatureLDSBankCount32]
>;

def : ProcessorModel<"iceland", SIQuarterSpeedModel,
  [FeatureVolcanicIslands, FeatureSGPRInitBug, FeatureISAVersion8_0_0,
   FeatureLDSBankCount32]
>;

def : ProcessorModel<"carrizo", SIQuarterSpeedModel,
  [FeatureVolcanicIslands, FeatureISAVersion8_0_1, FeatureLDSBankCount32, FeatureXNACK]
>;

def : ProcessorModel<"fiji", SIQuarterSpeedModel,
  [FeatureVolcanicIslands, FeatureISAVersion8_0_3, FeatureLDSBankCount32]
>;

def : ProcessorModel<"stoney", SIQuarterSpeedModel,
  [FeatureVolcanicIslands, FeatureISAVersion8_0_1, FeatureLDSBankCount16, FeatureXNACK]
>;

def : ProcessorModel<"polaris10", SIQuarterSpeedModel,
  [FeatureVolcanicIslands, FeatureISAVersion8_0_1, FeatureLDSBankCount32]
>;

def : ProcessorModel<"polaris11", SIQuarterSpeedModel,
  [FeatureVolcanicIslands, FeatureISAVersion8_0_1, FeatureLDSBankCount32]
>;
