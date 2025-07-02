//===- HLSLRootSignatureUtils.h - HLSL Root Signature helpers -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains helper obejcts for working with HLSL Root
/// Signatures.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_HLSL_HLSLROOTSIGNATUREUTILS_H
#define LLVM_FRONTEND_HLSL_HLSLROOTSIGNATUREUTILS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/Frontend/HLSL/HLSLRootSignature.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
class LLVMContext;
class MDNode;
class Metadata;

namespace hlsl {
namespace rootsig {

LLVM_ABI raw_ostream &operator<<(raw_ostream &OS, const dxbc::RootFlags &Flags);

LLVM_ABI raw_ostream &operator<<(raw_ostream &OS,
                                 const RootConstants &Constants);

LLVM_ABI raw_ostream &operator<<(raw_ostream &OS,
                                 const DescriptorTableClause &Clause);

LLVM_ABI raw_ostream &operator<<(raw_ostream &OS, const DescriptorTable &Table);

LLVM_ABI raw_ostream &operator<<(raw_ostream &OS,
                                 const RootDescriptor &Descriptor);

LLVM_ABI raw_ostream &operator<<(raw_ostream &OS,
                                 const StaticSampler &StaticSampler);

LLVM_ABI raw_ostream &operator<<(raw_ostream &OS, const RootElement &Element);

LLVM_ABI void dumpRootElements(raw_ostream &OS, ArrayRef<RootElement> Elements);

class MetadataBuilder {
public:
  MetadataBuilder(llvm::LLVMContext &Ctx, ArrayRef<RootElement> Elements)
      : Ctx(Ctx), Elements(Elements) {}

  /// Iterates through the elements and dispatches onto the correct Build method
  ///
  /// Accumulates the root signature and returns the Metadata node that is just
  /// a list of all the elements
  LLVM_ABI MDNode *BuildRootSignature();

private:
  /// Define the various builders for the different metadata types
  MDNode *BuildRootFlags(const dxbc::RootFlags &Flags);
  MDNode *BuildRootConstants(const RootConstants &Constants);
  MDNode *BuildRootDescriptor(const RootDescriptor &Descriptor);
  MDNode *BuildDescriptorTable(const DescriptorTable &Table);
  MDNode *BuildDescriptorTableClause(const DescriptorTableClause &Clause);
  MDNode *BuildStaticSampler(const StaticSampler &Sampler);

  llvm::LLVMContext &Ctx;
  ArrayRef<RootElement> Elements;
  SmallVector<Metadata *> GeneratedMetadata;
};

struct RangeInfo {
  const static uint32_t Unbounded = ~0u;

  // Interval information
  uint32_t LowerBound;
  uint32_t UpperBound;

  // Information retained for diagnostics
  llvm::dxil::ResourceClass Class;
  uint32_t Space;
  dxbc::ShaderVisibility Visibility;
};

class ResourceRange {
public:
  using MapT = llvm::IntervalMap<uint32_t, const RangeInfo *, 16,
                                 llvm::IntervalMapInfo<uint32_t>>;

private:
  MapT Intervals;

public:
  ResourceRange(MapT::Allocator &Allocator) : Intervals(MapT(Allocator)) {}

  // Returns a reference to the first RangeInfo that overlaps with
  // [Info.LowerBound;Info.UpperBound], or, std::nullopt if there is no overlap
  LLVM_ABI std::optional<const RangeInfo *>
  getOverlapping(const RangeInfo &Info) const;

  // Return the mapped RangeInfo at X or nullptr if no mapping exists
  LLVM_ABI const RangeInfo *lookup(uint32_t X) const;

  // Removes all entries of the ResourceRange
  LLVM_ABI void clear();

  // Insert the required (sub-)intervals such that the interval of [a;b] =
  // [Info.LowerBound, Info.UpperBound] is covered and points to a valid
  // RangeInfo &.
  //
  // For instance consider the following chain of inserting RangeInfos with the
  // intervals denoting the Lower/Upper-bounds:
  //
  // A = [0;2]
  //   insert(A) -> false
  //   intervals: [0;2] -> &A
  // B = [5;7]
  //   insert(B) -> false
  //   intervals: [0;2] -> &A, [5;7] -> &B
  // C = [4;7]
  //   insert(C) -> true
  //   intervals: [0;2] -> &A, [4;7] -> &C
  // D = [1;5]
  //   insert(D) -> true
  //   intervals: [0;2] -> &A, [3;3] -> &D, [4;7] -> &C
  // E = [0;unbounded]
  //   insert(E) -> true
  //   intervals: [0;unbounded] -> E
  //
  // Returns a reference to the first RangeInfo that overlaps with
  // [Info.LowerBound;Info.UpperBound], or, std::nullopt if there is no overlap
  // (equivalent to getOverlapping)
  LLVM_ABI std::optional<const RangeInfo *> insert(const RangeInfo &Info);
};

} // namespace rootsig
} // namespace hlsl
} // namespace llvm

#endif // LLVM_FRONTEND_HLSL_HLSLROOTSIGNATUREUTILS_H
