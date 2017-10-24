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

// The code produced for "generic" is only useful for tests and cannot
// reasonably be expected to execute on any particular target.
def : ProcessorModel<"generic", NoSchedModel, [
  FeatureGCN, FeatureWavefrontSize64
]>;

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

def : ProcessorModel<"gfx600",     SIFullSpeedModel,
  [FeatureISAVersion6_0_0]>;

def : ProcessorModel<"tahiti",     SIFullSpeedModel,
  [FeatureISAVersion6_0_0]
>;

def : ProcessorModel<"gfx601",     SIQuarterSpeedModel,
  [FeatureISAVersion6_0_1]
>;

def : ProcessorModel<"pitcairn",   SIQuarterSpeedModel,
  [FeatureISAVersion6_0_1]>;

def : ProcessorModel<"verde",      SIQuarterSpeedModel,
  [FeatureISAVersion6_0_1]>;

def : ProcessorModel<"oland",      SIQuarterSpeedModel,
  [FeatureISAVersion6_0_1]>;

def : ProcessorModel<"hainan",     SIQuarterSpeedModel, [FeatureISAVersion6_0_1]>;

//===----------------------------------------------------------------------===//
// Sea Islands
//===----------------------------------------------------------------------===//

def : ProcessorModel<"gfx700",     SIQuarterSpeedModel,
  [FeatureISAVersion7_0_0]
>;

def : ProcessorModel<"bonaire",    SIQuarterSpeedModel,
  [FeatureISAVersion7_0_0]
>;

def : ProcessorModel<"kaveri",     SIQuarterSpeedModel,
  [FeatureISAVersion7_0_0]
>;

def : ProcessorModel<"gfx701",     SIFullSpeedModel,
  [FeatureISAVersion7_0_1]
>;

def : ProcessorModel<"hawaii",     SIFullSpeedModel,
  [FeatureISAVersion7_0_1]
>;

def : ProcessorModel<"gfx702",     SIQuarterSpeedModel,
  [FeatureISAVersion7_0_2]
>;

def : ProcessorModel<"gfx703",     SIQuarterSpeedModel,
  [FeatureISAVersion7_0_3]
>;

def : ProcessorModel<"kabini",     SIQuarterSpeedModel,
  [FeatureISAVersion7_0_3]
>;

def : ProcessorModel<"mullins",    SIQuarterSpeedModel,
  [FeatureISAVersion7_0_3]>;

//===----------------------------------------------------------------------===//
// Volcanic Islands
//===----------------------------------------------------------------------===//

def : ProcessorModel<"tonga",   SIQuarterSpeedModel,
  [FeatureISAVersion8_0_2]
>;

def : ProcessorModel<"iceland", SIQuarterSpeedModel,
  [FeatureISAVersion8_0_0]
>;

def : ProcessorModel<"carrizo", SIQuarterSpeedModel,
  [FeatureISAVersion8_0_1]
>;

def : ProcessorModel<"fiji",    SIQuarterSpeedModel,
  [FeatureISAVersion8_0_3]
>;

def : ProcessorModel<"stoney",  SIQuarterSpeedModel,
  [FeatureISAVersion8_1_0]
>;

def : ProcessorModel<"polaris10", SIQuarterSpeedModel,
  [FeatureISAVersion8_0_3]
>;

def : ProcessorModel<"polaris11", SIQuarterSpeedModel,
  [FeatureISAVersion8_0_3]
>;

def : ProcessorModel<"gfx800", SIQuarterSpeedModel,
  [FeatureISAVersion8_0_0]
>;

def : ProcessorModel<"gfx801", SIQuarterSpeedModel,
  [FeatureISAVersion8_0_1]
>;

def : ProcessorModel<"gfx802", SIQuarterSpeedModel,
  [FeatureISAVersion8_0_2]
>;

def : ProcessorModel<"gfx803", SIQuarterSpeedModel,
  [FeatureISAVersion8_0_3]
>;

def : ProcessorModel<"gfx804", SIQuarterSpeedModel,
  [FeatureISAVersion8_0_4]
>;

def : ProcessorModel<"gfx810", SIQuarterSpeedModel,
  [FeatureISAVersion8_1_0]
>;

//===----------------------------------------------------------------------===//
// GFX9
//===----------------------------------------------------------------------===//

def : ProcessorModel<"gfx900", SIQuarterSpeedModel,
  [FeatureISAVersion9_0_0]
>;

def : ProcessorModel<"gfx901", SIQuarterSpeedModel,
  [FeatureISAVersion9_0_1]
>;

def : ProcessorModel<"gfx902", SIQuarterSpeedModel,
  [FeatureISAVersion9_0_2]
>;

def : ProcessorModel<"gfx903", SIQuarterSpeedModel,
  [FeatureISAVersion9_0_3]
>;

