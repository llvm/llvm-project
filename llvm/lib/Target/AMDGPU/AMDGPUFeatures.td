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
// be allocated to a single workgroup. A generic target's feature must be present
// on at least one of its covered GPUs.
class SubtargetFeatureAddressableLocalMemorySize <int Value> : AMDGPUGenericAnyFeature<
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

// A generic target uses the largest allocation granularity of its covered GPUs.
// Its feature must be present on at least one covered GPU.
class SubtargetFeatureLDSAllocGranularity <int Granularity> : AMDGPUGenericAnyFeature <
  "lds-alloc-granularity-"#Granularity,
  "LDSAllocationGranularity",
  !cast<string>(Granularity),
  "LDS allocation granularity in bytes.">;

def FeatureLDSAllocGranularity256 : SubtargetFeatureLDSAllocGranularity<256>;
def FeatureLDSAllocGranularity512 : SubtargetFeatureLDSAllocGranularity<512>;
def FeatureLDSAllocGranularity1024 : SubtargetFeatureLDSAllocGranularity<1024>;
def FeatureLDSAllocGranularity1280 : SubtargetFeatureLDSAllocGranularity<1280>;
def FeatureLDSAllocGranularity2048 : SubtargetFeatureLDSAllocGranularity<2048>;

// Units used to encode LDS size in program resource registers and metadata.
// These can differ from the hardware allocation granularity. A generic target's
// encoding granularity must be present on at least one covered GPU.
class SubtargetFeatureLDSEncodingGranularity <int Granularity> : AMDGPUGenericAnyFeature <
  "lds-encoding-granularity-"#Granularity,
  "LDSEncodingGranularity",
  !cast<string>(Granularity),
  "LDS encoding granularity in bytes.">;

def FeatureLDSEncodingGranularity256 : SubtargetFeatureLDSEncodingGranularity<256>;
def FeatureLDSEncodingGranularity512 : SubtargetFeatureLDSEncodingGranularity<512>;
def FeatureLDSEncodingGranularity1024 : SubtargetFeatureLDSEncodingGranularity<1024>;
def FeatureLDSEncodingGranularity1280 : SubtargetFeatureLDSEncodingGranularity<1280>;
def FeatureLDSEncodingGranularity2048 : SubtargetFeatureLDSEncodingGranularity<2048>;

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
