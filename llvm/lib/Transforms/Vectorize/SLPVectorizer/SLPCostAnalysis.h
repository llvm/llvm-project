//===- SLPCostAnalysis.h - SLP Vectorizer free cost helpers ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Internal header used by SLPVectorizer.cpp. It declares free cost helpers
// that do not depend on BoUpSLP or any other SLP-private type. The bulk of
// the SLP cost model still lives in SLPVectorizer.cpp because it references
// BoUpSLP internals.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPCOSTANALYSIS_H
#define LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPCOSTANALYSIS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Support/InstructionCost.h"

#include <utility>

namespace llvm {
class Type;
class Value;
class VectorType;
} // namespace llvm

namespace llvm::slpvectorizer {

/// Returns the cost of the shuffle instructions with the given \p Kind, vector
/// type \p Tp and optional \p Mask. Adds SLP-specific cost estimation for
/// insert subvector pattern.
InstructionCost getShuffleCost(const TargetTransformInfo &TTI,
                               TargetTransformInfo::ShuffleKind Kind,
                               VectorType *Tp, ArrayRef<int> Mask = {},
                               TargetTransformInfo::TargetCostKind CostKind =
                                   TargetTransformInfo::TCK_RecipThroughput,
                               int Index = 0, VectorType *SubTp = nullptr,
                               ArrayRef<const Value *> Args = {});

/// Calculate the scalar and the vector costs from vectorizing set of GEPs.
std::pair<InstructionCost, InstructionCost>
getGEPCosts(const TargetTransformInfo &TTI, ArrayRef<Value *> Ptrs,
            Value *BasePtr, unsigned Opcode,
            TargetTransformInfo::TargetCostKind CostKind, Type *ScalarTy,
            VectorType *VecTy);

/// Returns the cost of a BlendedLoadVectorize node loading \p VecTy: two masked
/// loads (one per candidate base), a xor to negate the false-lane mask and a
/// select. The blend mask is a separate operand node, so its cost is counted
/// there, not here.
InstructionCost
getBlendedLoadCost(const TargetTransformInfo &TTI, Type *VecTy, Align Alignment,
                   unsigned AddressSpace,
                   TargetTransformInfo::TargetCostKind CostKind);

} // namespace llvm::slpvectorizer

#endif // LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPCOSTANALYSIS_H
