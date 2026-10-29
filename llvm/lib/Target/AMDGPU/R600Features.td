//===-- R600Features.td - R600 Feature Definitions ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

class SubtargetFeatureFetchLimit <string Value> :
                          SubtargetFeature <"fetch"#Value,
  "TexVTXClauseSize",
  Value,
  "Limit the maximum number of fetches in a clause to "#Value
>;

def FeatureR600ALUInst : SubtargetFeature<"R600ALUInst",
  "HasR600ALUInst",
  "false",
  "Older version of ALU instructions encoding"
>;

def FeatureFetchLimit8 : SubtargetFeatureFetchLimit <"8">;
def FeatureFetchLimit16 : SubtargetFeatureFetchLimit <"16">;

def FeatureVertexCache : SubtargetFeature<"HasVertexCache",
  "HasVertexCache",
  "true",
  "Specify use of dedicated vertex cache"
>;

def FeatureCaymanISA : SubtargetFeature<"caymanISA",
  "HasCaymanISA",
  "true",
  "Use Cayman ISA"
>;

def FeatureCFALUBug : SubtargetFeature<"cfalubug",
  "HasCFALUBug",
  "true",
  "GPU has CF_ALU bug"
>;

def FeatureMadMacF32Insts : SubtargetFeature<"mad-mac-f32-insts",
  "HasMadMacF32Insts",
  "true",
  "Has v_mad_f32/v_mac_f32/v_madak_f32/v_madmk_f32 instructions"
>;

class R600SubtargetFeatureGeneration <string Value, string FeatureName,
                                  list<SubtargetFeature> Implies> :
        SubtargetFeatureGeneration <Value, FeatureName, "R600Subtarget", Implies>;

def FeatureR600 : R600SubtargetFeatureGeneration<"R600", "r600",
  [FeatureR600ALUInst, FeatureFetchLimit8, FeatureMadMacF32Insts]
>;

def FeatureR700 : R600SubtargetFeatureGeneration<"R700", "r700",
  [FeatureFetchLimit16, FeatureMadMacF32Insts]
>;

def FeatureEvergreen : R600SubtargetFeatureGeneration<"EVERGREEN", "evergreen",
  [FeatureFetchLimit16, FeatureAddressableLocalMemorySize32768,
   FeatureLDSAllocGranularity256,
   FeatureLDSEncodingGranularity256,
   FeatureMadMacF32Insts]
>;

def FeatureNorthernIslands : R600SubtargetFeatureGeneration<"NORTHERN_ISLANDS",
  "northern-islands",
  [FeatureFetchLimit16, FeatureWavefrontSize64,
   FeatureAddressableLocalMemorySize32768,
   FeatureLDSAllocGranularity256,
   FeatureLDSEncodingGranularity256,
   FeatureMadMacF32Insts]
>;

// The subset of R600 SubtargetFeatures exposed outside of the backend.
def R600FrontendVisibleFeatures {
  list<SubtargetFeature> Features = [FeatureFMA];
}
