//===-- AMDGPUFeatures.td - AMDGPU Feature Definitions -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "AMDGPUTargetParser.td"

def FeatureFP64 : SubtargetFeature<"fp64",
  "HasFP64",
  "true",
  "Enable double precision operations"
>;

def FeatureFMA : SubtargetFeature<"fmaf",
  "HasFMA",
  "true",
  "Enable single precision FMA (not as fast as mul+add, but fused)"
>;

class SubtargetFeatureFlatOffsetBits <int Value> : SubtargetFeature<
  "flat-offset-bits-" # Value,
  "FlatOffsetBitWidth",
  !cast<string>(Value),
  "Number of bits for flat offset encoding"
>;

def FeatureFlatOffsetBits12 : SubtargetFeatureFlatOffsetBits<12>;
def FeatureFlatOffsetBits24 : SubtargetFeatureFlatOffsetBits<24>;

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
def FeatureAddressableLocalMemorySize196608 : SubtargetFeatureAddressableLocalMemorySize<196608>;
def FeatureAddressableLocalMemorySize327680 : SubtargetFeatureAddressableLocalMemorySize<327680>;

// Total size of LDS on the whole block, in bytes. Set this only when it is
// larger than the addressable size, i.e. when one workgroup cannot use the whole
// block (e.g. gfx10/11/12: 128k physical block vs 64k addressable). If not set,
// it defaults to the addressable size.
class SubtargetFeatureLocalMemorySize <int Value> : SubtargetFeature<
  "localmemorysize"#Value,
  "LocalMemorySize",
  !cast<string>(Value),
  "The total size of local memory in bytes"
>;

def FeatureLocalMemorySize131072 : SubtargetFeatureLocalMemorySize<131072>;

// Whether each wavefront size mode is available on the hardware,
// independent of the active mode.
def FeatureSupportsWave32 : SubtargetFeature<"supports-wave32",
  "SupportsWave32",
  "true",
  "Hardware supports wave32 mode",
  [], InlineIgnore
>;

def FeatureSupportsWave64 : SubtargetFeature<"supports-wave64",
  "SupportsWave64",
  "true",
  "Hardware supports wave64 mode",
  [], InlineIgnore
>;

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

