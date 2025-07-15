//===- RootSignatureValidations.h - HLSL Root Signature helpers -----------===//
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

#ifndef LLVM_FRONTEND_HLSL_ROOTSIGNATUREVALIDATIONS_H
#define LLVM_FRONTEND_HLSL_ROOTSIGNATUREVALIDATIONS_H

#include "llvm/ADT/IntervalMap.h"
#include "llvm/Frontend/HLSL/HLSLRootSignature.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
namespace hlsl {
namespace rootsig {

// Basic verification of RootElements

LLVM_ABI bool verifyRootFlag(uint32_t Flags);
LLVM_ABI bool verifyVersion(uint32_t Version);
LLVM_ABI bool verifyRegisterValue(uint32_t RegisterValue);
LLVM_ABI bool verifyRegisterSpace(uint32_t RegisterSpace);
LLVM_ABI bool verifyRootDescriptorFlag(uint32_t Version, uint32_t FlagsVal);
LLVM_ABI bool verifyRangeType(uint32_t Type);
LLVM_ABI bool verifyDescriptorRangeFlag(uint32_t Version, uint32_t Type,
                                        uint32_t FlagsVal);
LLVM_ABI bool verifyNumDescriptors(uint32_t NumDescriptors);
LLVM_ABI bool verifySamplerFilter(uint32_t Value);
LLVM_ABI bool verifyAddress(uint32_t Address);
LLVM_ABI bool verifyMipLODBias(float MipLODBias);
LLVM_ABI bool verifyMaxAnisotropy(uint32_t MaxAnisotropy);
LLVM_ABI bool verifyComparisonFunc(uint32_t ComparisonFunc);
LLVM_ABI bool verifyBorderColor(uint32_t BorderColor);
LLVM_ABI bool verifyLOD(float LOD);

struct RangeInfo {
  const static uint32_t Unbounded = ~0u;

  // Interval information
  uint32_t LowerBound;
  uint32_t UpperBound;

  // Information retained for determining overlap
  llvm::dxil::ResourceClass Class;
  uint32_t Space;
  llvm::dxbc::ShaderVisibility Visibility;

  bool operator==(const RangeInfo &RHS) const {
    return std::tie(LowerBound, UpperBound, Class, Space, Visibility) ==
           std::tie(RHS.LowerBound, RHS.UpperBound, RHS.Class, RHS.Space,
                    RHS.Visibility);
  }

  bool operator<(const RangeInfo &RHS) const {
    return std::tie(Class, Space, LowerBound, UpperBound, Visibility) <
           std::tie(RHS.Class, RHS.Space, RHS.LowerBound, RHS.UpperBound,
                    RHS.Visibility);
  }
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

struct OverlappingRanges {
  const RangeInfo *A;
  const RangeInfo *B;

  OverlappingRanges(const RangeInfo *A, const RangeInfo *B) : A(A), B(B) {}
};

/// The following conducts analysis on resource ranges to detect and report
/// any overlaps in resource ranges.
///
/// A resource range overlaps with another resource range if they have:
/// - equivalent ResourceClass (SRV, UAV, CBuffer, Sampler)
/// - equivalent resource space
/// - overlapping visbility
///
/// The algorithm is implemented in the following steps:
///
/// 1. The user will collect RangeInfo from relevant RootElements:
///   - RangeInfo will retain the interval, ResourceClass, Space and Visibility
///   - It will also contain an index so that it can be associated to
/// additional diagnostic information
/// 2. The user is required to sort the RangeInfo's such that they are grouped
/// together by ResourceClass and Space
/// 3. Iterate through the collected RangeInfos by their groups
///   - For each group we will have a ResourceRange for each visibility
///   - As we iterate through we will:
///      A: Insert the current RangeInfo into the corresponding Visibility
///   ResourceRange
///      B: Check for overlap with any overlapping Visibility ResourceRange
LLVM_ABI llvm::SmallVector<OverlappingRanges>
findOverlappingRanges(ArrayRef<RangeInfo> Infos);

} // namespace rootsig
} // namespace hlsl
} // namespace llvm

#endif // LLVM_FRONTEND_HLSL_ROOTSIGNATUREVALIDATIONS_H
