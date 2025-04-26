//===-- AMDGPUFeatures.td - AMDGPU Feature Definitions -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

// Addressable local memory size is the maximum number of bytes of LDS that can
// be allocated to a single workgroup.
class SubtargetFeatureAddressableLocalMemorySize <int Value> : SubtargetFeature<
  "addressablelocalmemorysize"#Value,
  "AddressableLocalMemorySize",
  !cast<string>(Value),
  "The size of local memory in bytes"
>;

def FeatureAddressableLocalMemorySize32768 : SubtargetFeatureAddressableLocalMemorySize<32768>;
def FeatureAddressableLocalMemorySize65536 : SubtargetFeatureAddressableLocalMemorySize<65536>;
def FeatureAddressableLocalMemorySize163840 : SubtargetFeatureAddressableLocalMemorySize<163840>;

class SubtargetFeatureWavefrontSize <int ValueLog2> : SubtargetFeature<
  "wavefrontsize"#!shl(1, ValueLog2),
  "WavefrontSizeLog2",
  !cast<string>(ValueLog2),
  "The number of threads per wavefront"
>;

def FeatureWavefrontSize16 : SubtargetFeatureWavefrontSize<4>;
def FeatureWavefrontSize32 : SubtargetFeatureWavefrontSize<5>;
def FeatureWavefrontSize64 : SubtargetFeatureWavefrontSize<6>;

class SubtargetFeatureGeneration <string Value, string FeatureName,
                                 string Subtarget,
                                  list<SubtargetFeature> Implies> :
        SubtargetFeature <FeatureName, "Gen", Subtarget#"::"#Value,
                          Value#" GPU generation", Implies>;

def FeaturePromoteAlloca : SubtargetFeature <"promote-alloca",
  "EnablePromoteAlloca",
  "true",
  "Enable promote alloca pass"
>;

