//===-- AMDGPUFeatures.td - AMDGPU Feature Definitions -----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

def FeatureFP64 : SubtargetFeature<"fp64",
  "FP64",
  "true",
  "Enable double precision operations"
>;

def FeatureFMA : SubtargetFeature<"fmaf",
  "FMA",
  "true",
  "Enable single precision FMA (not as fast as mul+add, but fused)"
>;

class SubtargetFeatureLocalMemorySize <int Value> : SubtargetFeature<
  "localmemorysize"#Value,
  "LocalMemorySize",
  !cast<string>(Value),
  "The size of local memory in bytes"
>;

def FeatureLocalMemorySize0 : SubtargetFeatureLocalMemorySize<0>;
def FeatureLocalMemorySize32768 : SubtargetFeatureLocalMemorySize<32768>;
def FeatureLocalMemorySize65536 : SubtargetFeatureLocalMemorySize<65536>;

class SubtargetFeatureWavefrontSize <int Value> : SubtargetFeature<
  "wavefrontsize"#Value,
  "WavefrontSize",
  !cast<string>(Value),
  "The number of threads per wavefront"
>;

def FeatureWavefrontSize16 : SubtargetFeatureWavefrontSize<16>;
def FeatureWavefrontSize32 : SubtargetFeatureWavefrontSize<32>;
def FeatureWavefrontSize64 : SubtargetFeatureWavefrontSize<64>;

class SubtargetFeatureGeneration <string Value, string Subtarget,
                                  list<SubtargetFeature> Implies> :
        SubtargetFeature <Value, "Gen", Subtarget#"::"#Value,
                          Value#" GPU generation", Implies>;

def FeatureDX10Clamp : SubtargetFeature<"dx10-clamp",
  "DX10Clamp",
  "true",
  "clamp modifier clamps NaNs to 0.0"
>;

def FeaturePromoteAlloca : SubtargetFeature <"promote-alloca",
  "EnablePromoteAlloca",
  "true",
  "Enable promote alloca pass"
>;

