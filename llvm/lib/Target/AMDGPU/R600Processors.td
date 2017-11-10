//===-- R600Processors.td - R600 Processor definitions --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Radeon HD 2000/3000 Series (R600).
//===----------------------------------------------------------------------===//

def : Processor<"r600", R600_VLIW5_Itin,
  [FeatureR600, FeatureWavefrontSize64, FeatureVertexCache]
>;

def : Processor<"r630", R600_VLIW5_Itin,
  [FeatureR600, FeatureWavefrontSize32, FeatureVertexCache]
>;

def : Processor<"rs880", R600_VLIW5_Itin,
  [FeatureR600, FeatureWavefrontSize16]
>;

def : Processor<"rv670", R600_VLIW5_Itin,
  [FeatureR600, FeatureWavefrontSize64, FeatureVertexCache, FeatureFP64]
>;

//===----------------------------------------------------------------------===//
// Radeon HD 4000 Series (R700).
//===----------------------------------------------------------------------===//

def : Processor<"rv710", R600_VLIW5_Itin,
  [FeatureR700, FeatureWavefrontSize32, FeatureVertexCache]
>;

def : Processor<"rv730", R600_VLIW5_Itin,
  [FeatureR700, FeatureWavefrontSize32, FeatureVertexCache]
>;

def : Processor<"rv770", R600_VLIW5_Itin,
  [FeatureR700, FeatureWavefrontSize64, FeatureVertexCache, FeatureFP64]
>;

//===----------------------------------------------------------------------===//
// Radeon HD 5000 Series (Evergreen).
//===----------------------------------------------------------------------===//

def : Processor<"cedar", R600_VLIW5_Itin,
  [FeatureEvergreen, FeatureWavefrontSize32, FeatureVertexCache,
   FeatureCFALUBug]
>;

def : Processor<"cypress", R600_VLIW5_Itin,
  [FeatureEvergreen, FeatureWavefrontSize64, FeatureVertexCache, FeatureFP64]
>;

def : Processor<"juniper", R600_VLIW5_Itin,
  [FeatureEvergreen, FeatureWavefrontSize64, FeatureVertexCache]
>;

def : Processor<"redwood", R600_VLIW5_Itin,
  [FeatureEvergreen, FeatureWavefrontSize64, FeatureVertexCache,
   FeatureCFALUBug]
>;

def : Processor<"sumo", R600_VLIW5_Itin,
  [FeatureEvergreen, FeatureWavefrontSize64, FeatureCFALUBug]
>;

//===----------------------------------------------------------------------===//
// Radeon HD 6000 Series (Northern Islands).
//===----------------------------------------------------------------------===//

def : Processor<"barts", R600_VLIW5_Itin,
  [FeatureNorthernIslands, FeatureVertexCache, FeatureCFALUBug]
>;

def : Processor<"caicos", R600_VLIW5_Itin,
  [FeatureNorthernIslands, FeatureCFALUBug]
>;

def : Processor<"cayman", R600_VLIW4_Itin,
  [FeatureNorthernIslands, FeatureFP64, FeatureCaymanISA]
>;

def : Processor<"turks", R600_VLIW5_Itin,
  [FeatureNorthernIslands, FeatureVertexCache, FeatureCFALUBug]
>;
