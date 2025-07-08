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

namespace llvm {
namespace hlsl {
namespace rootsig {

// Basic verification of RootElements

bool verifyRootFlag(uint32_t Flags);
bool verifyVersion(uint32_t Version);
bool verifyRegisterValue(uint32_t RegisterValue);
bool verifyRegisterSpace(uint32_t RegisterSpace);
bool verifyDescriptorFlag(uint32_t Flags);
bool verifyRangeType(uint32_t Type);
bool verifyDescriptorRangeFlag(uint32_t Version, uint32_t Type,
                               uint32_t FlagsVal);
bool verifySamplerFilter(uint32_t Value);
bool verifyAddress(uint32_t Address);
bool verifyMipLODBias(float MipLODBias);
bool verifyMaxAnisotropy(uint32_t MaxAnisotropy);
bool verifyComparisonFunc(uint32_t ComparisonFunc);
bool verifyBorderColor(uint32_t BorderColor);
bool verifyLOD(float LOD);

struct RangeInfo {
  const static uint32_t Unbounded = ~0u;

  // Interval information
  uint32_t LowerBound;
  uint32_t UpperBound;

  // Information retained for diagnostics
  llvm::dxil::ResourceClass Class;
  uint32_t Space;
  llvm::dxbc::ShaderVisibility Visibility;
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

#endif // LLVM_FRONTEND_HLSL_ROOTSIGNATUREVALIDATIONS_H
