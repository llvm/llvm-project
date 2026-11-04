//===-- TargetParser - Parser for target features ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser to recognise hardware features such as
// FPU/CPU/ARCH names as well as specific support such as HDIV, etc.
//
//===----------------------------------------------------------------------===//

#include "llvm/TargetParser/TargetParser.h"
#include "llvm/ADT/ArrayRef.h"

using namespace llvm;

/// Find KV in array using binary search.
static const BasicSubtargetSubTypeKV *
find(StringRef S, ArrayRef<BasicSubtargetSubTypeKV> A) {
  // Binary search the array
  const BasicSubtargetSubTypeKV *F = llvm::lower_bound(A, S);
  // If not found then return NULL
  if (F == A.end() || StringRef(F->Key) != S)
    return nullptr;
  // Return the found array item
  return F;
}

/// For each feature that is (transitively) implied by this feature, set it.
static void setImpliedBits(FeatureBitset &Bits, const FeatureBitset &Implies,
                           ArrayRef<BasicSubtargetFeatureKV> FeatureTable) {
  // OR the Implies bits in outside the loop. This allows the Implies for CPUs
  // which might imply features not in FeatureTable to use this.
  Bits |= Implies;
  for (const auto &FE : FeatureTable)
    if (Implies.test(FE.Value))
      setImpliedBits(Bits, FE.Implies.getAsBitset(), FeatureTable);
}

std::optional<llvm::StringMap<bool>> llvm::getCPUDefaultTargetFeatures(
    StringRef CPU, ArrayRef<BasicSubtargetSubTypeKV> ProcDesc,
    ArrayRef<BasicSubtargetFeatureKV> ProcFeatures) {
  if (CPU.empty())
    return std::nullopt;

  const BasicSubtargetSubTypeKV *CPUEntry = ::find(CPU, ProcDesc);
  if (!CPUEntry)
    return std::nullopt;

  // Set the features implied by this CPU feature if there is a match.
  FeatureBitset Bits;
  llvm::StringMap<bool> DefaultFeatures;
  setImpliedBits(Bits, CPUEntry->Implies.getAsBitset(), ProcFeatures);

  [[maybe_unused]] unsigned BitSize = Bits.size();
  for (const BasicSubtargetFeatureKV &FE : ProcFeatures) {
    assert(FE.Value < BitSize && "Target Feature is out of range");
    if (Bits[FE.Value])
      DefaultFeatures[FE.Key] = true;
  }
  return DefaultFeatures;
}
