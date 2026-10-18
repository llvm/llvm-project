//===-- AGXFeatures.td - AGX feature definitions -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Subtarget features of the Apple AGX GPUs. Features fall in three groups:
//
//  * Generation features (FeatureG13 ... FeatureG17). Exactly one is set by
//    each processor. They select defaults, not legality.
//  * Encoding family features. They select which real (MC) instruction a
//    pseudo instruction is lowered to.
//  * Capability features. Every capability can be enabled or disabled on any
//    processor with -mattr, so code paths can be tried on generations where
//    the capability is not known to exist.
//
// The provenance of each capability is recorded in AGXUsage.rst.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Encoding families
//===----------------------------------------------------------------------===//

def FeatureAGX2Encoding : SubtargetFeature<"agx2-encoding",
  "HasAGX2Encoding", "true",
  "Uses the G13/G14 (\"AGX2\") instruction encoding"
>;

def FeatureApple9Encoding : SubtargetFeature<"apple9-encoding",
  "HasApple9Encoding", "true",
  "Uses the Apple9 (G16/G17) instruction encoding. Code generation for this "
  "encoding is experimental"
>;

//===----------------------------------------------------------------------===//
// Capability features
//===----------------------------------------------------------------------===//

def FeatureFP16Arith : SubtargetFeature<"fp16-arith",
  "HasFP16Arith", "true",
  "Has native half precision fadd/fmul/fmadd"
>;

def FeatureInt64Add : SubtargetFeature<"int64-add",
  "HasInt64Add", "true",
  "iadd and imadd accept 64-bit destinations and sources"
>;

def FeatureSIMDShuffle : SubtargetFeature<"simd-shuffle",
  "HasSIMDShuffle", "true",
  "Has simd_shuffle and quad_shuffle instructions"
>;

def FeatureSIMDReduce : SubtargetFeature<"simd-reduce",
  "HasSIMDReduce", "true",
  "Has simd/quad reduction and prefix instructions"
>;

def FeatureSIMDBallot : SubtargetFeature<"simd-ballot",
  "HasSIMDBallot", "true",
  "Has icmp_ballot and fcmp_ballot"
>;

def FeatureSIMDMatrix : SubtargetFeature<"simd-matrix",
  "HasSIMDMatrix", "true",
  "Has simdgroup 8x8 matrix multiply-accumulate"
>;

def FeatureQuadShuffleIndexMerge : SubtargetFeature<"quad-shuffle-index-merge",
  "HasQuadShuffleIndexMerge", "true",
  "Shuffles OR the low two index bits of the threads of each quad; "
  "divergent indices are expanded into four shuffles"
>;

def FeatureIEEEFP32 : SubtargetFeature<"ieee-fp32", "HasIEEEFP32", "true",
  "Implement IEEE 754 binary32 arithmetic in software where the hardware "
  "differs: denormals (in functions whose denormal mode for float is IEEE), "
  "and correctly rounded division and square root"
>;

def FeatureCanonicalNaN : SubtargetFeature<"canonical-nan", "HasCanonicalNaN",
  "true",
  "NaN results of floating point arithmetic are the canonical NaN of PTX "
  "(0x7fffffff, 0x7fff for half and bfloat)"
>;

def FeatureMultiDieCoherency : SubtargetFeature<"multi-die-coherency",
  "HasMultiDieCoherency", "true",
  "Device atomics must set the coherency bit (multi-die G13 parts)"
>;

def FeatureThreadgroupAtomics : SubtargetFeature<"threadgroup-atomics",
  "HasThreadgroupAtomics", "true",
  "Has threadgroup memory atomics"
>;

def FeatureDeviceAtomics : SubtargetFeature<"device-atomics",
  "HasDeviceAtomics", "true",
  "Has device memory atomics"
>;

def FeatureStackAccess : SubtargetFeature<"stack-access",
  "HasStackAccess", "true",
  "Has stack_load/stack_store for register spilling"
>;

def FeatureDynamicCaching : SubtargetFeature<"dynamic-caching",
  "HasDynamicCaching", "true",
  "Register file and threadgroup memory are allocated dynamically "
  "(no effect on code generation, reported for occupancy)"
>;

def FeatureRayTracing : SubtargetFeature<"ray-tracing",
  "HasRayTracing", "true",
  "Has hardware ray tracing instructions (not used by code generation)"
>;

def FeatureMeshShading : SubtargetFeature<"mesh-shading",
  "HasMeshShading", "true",
  "Has hardware mesh shading (not used by code generation)"
>;

def FeatureSourceLifetime : SubtargetFeature<"source-lifetime",
  "HasSourceLifetime", "true",
  "ALU sources carry release flags that free the register after the read; "
  "a released register reads as zero (Apple9)"
>;

//===----------------------------------------------------------------------===//
// Register file size
//===----------------------------------------------------------------------===//

class FeatureMaxGPRs<int Value> : SubtargetFeature<
  "max-gprs-"#Value,
  "MaxGPRs",
  !cast<string>(Value),
  "Up to "#Value#" addressable 32-bit general purpose registers"
>;

def FeatureMaxGPRs128 : FeatureMaxGPRs<128>;
def FeatureMaxGPRs64 : FeatureMaxGPRs<64>;

//===----------------------------------------------------------------------===//
// Generations
//===----------------------------------------------------------------------===//

class AGXSubtargetFeatureGeneration<string Value, string FeatureName,
                                    list<SubtargetFeature> Implies>
    : SubtargetFeature<FeatureName, "Gen", "AGXSubtarget::"#Value,
                       Value#" GPU generation", Implies>;

def FeatureG13 : AGXSubtargetFeatureGeneration<"G13", "g13",
  [FeatureAGX2Encoding, FeatureFP16Arith, FeatureInt64Add, FeatureSIMDShuffle,
   FeatureSIMDReduce, FeatureSIMDBallot, FeatureSIMDMatrix,
   FeatureQuadShuffleIndexMerge, FeatureThreadgroupAtomics,
   FeatureDeviceAtomics, FeatureStackAccess, FeatureMaxGPRs128]
>;

def FeatureG14 : AGXSubtargetFeatureGeneration<"G14", "g14",
  [FeatureAGX2Encoding, FeatureFP16Arith, FeatureInt64Add, FeatureSIMDShuffle,
   FeatureSIMDReduce, FeatureSIMDBallot, FeatureSIMDMatrix,
   FeatureQuadShuffleIndexMerge, FeatureThreadgroupAtomics,
   FeatureDeviceAtomics, FeatureStackAccess, FeatureMaxGPRs128]
>;

// No public instruction set documentation exists for G15. It is assumed to
// keep the G13/G14 encoding (see AGXUsage.rst, "Provenance").
def FeatureG15 : AGXSubtargetFeatureGeneration<"G15", "g15",
  [FeatureAGX2Encoding, FeatureFP16Arith, FeatureInt64Add, FeatureSIMDShuffle,
   FeatureSIMDReduce, FeatureSIMDBallot, FeatureSIMDMatrix,
   FeatureQuadShuffleIndexMerge, FeatureThreadgroupAtomics,
   FeatureDeviceAtomics, FeatureStackAccess, FeatureMaxGPRs128,
   FeatureDynamicCaching, FeatureRayTracing, FeatureMeshShading]
>;

def FeatureG16 : AGXSubtargetFeatureGeneration<"G16", "g16",
  [FeatureApple9Encoding, FeatureFP16Arith, FeatureSIMDShuffle,
   FeatureSIMDReduce, FeatureSIMDBallot, FeatureSIMDMatrix,
   FeatureThreadgroupAtomics, FeatureDeviceAtomics, FeatureMaxGPRs64,
   FeatureDynamicCaching, FeatureRayTracing, FeatureMeshShading,
   FeatureSourceLifetime]
>;

def FeatureG17 : AGXSubtargetFeatureGeneration<"G17", "g17",
  [FeatureApple9Encoding, FeatureFP16Arith, FeatureSIMDShuffle,
   FeatureSIMDReduce, FeatureSIMDBallot, FeatureSIMDMatrix,
   FeatureThreadgroupAtomics, FeatureDeviceAtomics, FeatureMaxGPRs64,
   FeatureDynamicCaching, FeatureRayTracing, FeatureMeshShading,
   FeatureSourceLifetime]
>;

//===----------------------------------------------------------------------===//
// Predicates
//===----------------------------------------------------------------------===//

def HasAGX2Encoding : Predicate<"Subtarget->hasAGX2Encoding()">,
                      AssemblerPredicate<(all_of FeatureAGX2Encoding)>;
def HasSIMDShuffle : Predicate<"Subtarget->hasSIMDShuffle()">;
def HasSIMDReduce : Predicate<"Subtarget->hasSIMDReduce()">;
def HasSIMDBallot : Predicate<"Subtarget->hasSIMDBallot()">;
def HasSIMDMatrix : Predicate<"Subtarget->hasSIMDMatrix()">;
def HasFP16Arith : Predicate<"Subtarget->hasFP16Arith()">;
def HasInt64Add : Predicate<"Subtarget->hasInt64Add()">;
def HasThreadgroupAtomics : Predicate<"Subtarget->hasThreadgroupAtomics()">;
def HasDeviceAtomics : Predicate<"Subtarget->hasDeviceAtomics()">;
