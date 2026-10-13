//===-- R600Processors.td - R600 Processor definitions --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Radeon HD 2000/3000 Series (R600).
//===----------------------------------------------------------------------===//

def : R600ProcessorModel<"r600", R600_VLIW5_Itin,
  [FeatureR600, FeatureWavefrontSize64, FeatureVertexCache]
>;

def : ProcessorAlias<"rv630", "r600">;
def : ProcessorAlias<"rv635", "r600">;

def : R600ProcessorModel<"r630", R600_VLIW5_Itin,
  [FeatureR600, FeatureWavefrontSize32, FeatureVertexCache]
>;

def : ProcessorAlias<"rs780", "rs880">;

def : R600ProcessorModel<"rs880", R600_VLIW5_Itin,
  [FeatureR600, FeatureWavefrontSize16]
>;

def : ProcessorAlias<"rv610", "rs880">;
def : ProcessorAlias<"rv620", "rs880">;

def : R600ProcessorModel<"rv670", R600_VLIW5_Itin,
  [FeatureR600, FeatureWavefrontSize64, FeatureVertexCache]
>;

//===----------------------------------------------------------------------===//
// Radeon HD 4000 Series (R700).
//===----------------------------------------------------------------------===//

def : R600ProcessorModel<"rv710", R600_VLIW5_Itin,
  [FeatureR700, FeatureWavefrontSize32, FeatureVertexCache]
>;

def : R600ProcessorModel<"rv730", R600_VLIW5_Itin,
  [FeatureR700, FeatureWavefrontSize32, FeatureVertexCache]
>;

def : ProcessorAlias<"rv740", "rv770">;

def : R600ProcessorModel<"rv770", R600_VLIW5_Itin,
  [FeatureR700, FeatureWavefrontSize64, FeatureVertexCache]
>;

//===----------------------------------------------------------------------===//
// Radeon HD 5000 Series (Evergreen).
//===----------------------------------------------------------------------===//

def : R600ProcessorModel<"cedar", R600_VLIW5_Itin,
  [FeatureEvergreen, FeatureWavefrontSize32, FeatureVertexCache,
   FeatureCFALUBug]
>;

def : ProcessorAlias<"palm", "cedar">;

def : R600ProcessorModel<"cypress", R600_VLIW5_Itin,
  [FeatureEvergreen, FeatureWavefrontSize64, FeatureVertexCache, FeatureFMA]
>;

def : ProcessorAlias<"hemlock", "cypress">;

def : R600ProcessorModel<"juniper", R600_VLIW5_Itin,
  [FeatureEvergreen, FeatureWavefrontSize64, FeatureVertexCache]
>;

def : R600ProcessorModel<"redwood", R600_VLIW5_Itin,
  [FeatureEvergreen, FeatureWavefrontSize64, FeatureVertexCache,
   FeatureCFALUBug]
>;

def : R600ProcessorModel<"sumo", R600_VLIW5_Itin,
  [FeatureEvergreen, FeatureWavefrontSize64, FeatureCFALUBug]
>;

def : ProcessorAlias<"sumo2", "sumo">;

//===----------------------------------------------------------------------===//
// Radeon HD 6000 Series (Northern Islands).
//===----------------------------------------------------------------------===//

def : R600ProcessorModel<"barts", R600_VLIW5_Itin,
  [FeatureNorthernIslands, FeatureVertexCache, FeatureCFALUBug]
>;

def : R600ProcessorModel<"caicos", R600_VLIW5_Itin,
  [FeatureNorthernIslands, FeatureCFALUBug]
>;

def : ProcessorAlias<"aruba", "cayman">;

def : R600ProcessorModel<"cayman", R600_VLIW4_Itin,
  [FeatureNorthernIslands, FeatureCaymanISA, FeatureFMA]
>;

def : R600ProcessorModel<"turks", R600_VLIW5_Itin,
  [FeatureNorthernIslands, FeatureVertexCache, FeatureCFALUBug]
>;
