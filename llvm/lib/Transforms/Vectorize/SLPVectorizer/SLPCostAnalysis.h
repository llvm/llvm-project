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
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/FMF.h"
#include "llvm/Support/InstructionCost.h"

#include <utility>

namespace llvm {
class FixedVectorType;
class Type;
class Value;
class VectorType;
} // namespace llvm

namespace llvm::slpvectorizer {

/// Returns the cost of the shuffle instructions with the given \p Kind, vector
/// type \p Tp and optional \p Mask. Adds SLP-specific cost estimation for
/// insert subvector pattern.
InstructionCost
getShuffleCost(const TargetTransformInfo &TTI,
               TargetTransformInfo::ShuffleKind Kind, VectorType *Tp,
               const TargetTransformInfo::TargetCostKind CostKind,
               ArrayRef<int> Mask = {}, int Index = 0,
               VectorType *SubTp = nullptr, ArrayRef<const Value *> Args = {});

/// Calculate the scalar and the vector costs from vectorizing set of GEPs.
std::pair<InstructionCost, InstructionCost>
getGEPCosts(const TargetTransformInfo &TTI, ArrayRef<Value *> Ptrs,
            Value *BasePtr, unsigned Opcode,
            const TargetTransformInfo::TargetCostKind CostKind, Type *ScalarTy,
            VectorType *VecTy);

/// Returns the cost of a BlendedLoadVectorize node loading \p VecTy: two masked
/// loads (one per candidate base), a xor to negate the false-lane mask and a
/// select. The blend mask is a separate operand node, so its cost is counted
/// there, not here.
InstructionCost
getBlendedLoadCost(const TargetTransformInfo &TTI, Type *VecTy, Align Alignment,
                   unsigned AddressSpace,
                   const TargetTransformInfo::TargetCostKind CostKind);

/// Returns the cost of materializing the identity element in the padding lanes
/// of a \p NumElts-wide reduction padded to \p PaddedVecTy, 0 if \p PaddedVecTy
/// is null. Targets reducing exactly the requested number of lanes need no
/// padding.
InstructionCost
getReductionPaddingCost(const TargetTransformInfo &TTI, VectorType *PaddedVecTy,
                        unsigned NumElts,
                        TargetTransformInfo::TargetCostKind CostKind);

/// Returns the cost of a \p NumElts-wide \p RdxKind reduction of \p ScalarTy,
/// padded to \p PaddedVecTy, plus the scalar reduction operations for the
/// \p NumTail values left out of the vector. Invalid for the reduction kinds
/// without a vector counterpart, and for revectorization, where the lanes are
/// combined with plain vector arithmetic, so the narrower width performs the
/// same operations and only adds subvector extracts.
InstructionCost getReductionWidthCost(const TargetTransformInfo &TTI,
                                      RecurKind RdxKind, Type *ScalarTy,
                                      unsigned NumElts, VectorType *PaddedVecTy,
                                      unsigned NumTail, FastMathFlags FMF);

/// Checks if a \p VecTy value, already padded to \p PaddedVecTy by a masked
/// operation, is cheaper to store with a single masked store than to narrow
/// back and store directly. Only worth it for an already padded value:
/// assembling the padded vector out of narrower pieces costs more than the
/// single store saves.
bool isMaskedStoreExpandProfitable(const TargetTransformInfo &TTI,
                                   FixedVectorType *VecTy,
                                   FixedVectorType *PaddedVecTy, unsigned AS,
                                   Align CommonAlignment);

} // namespace llvm::slpvectorizer

#endif // LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPCOSTANALYSIS_H
