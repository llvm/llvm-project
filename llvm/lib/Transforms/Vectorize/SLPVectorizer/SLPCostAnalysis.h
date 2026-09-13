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

#include "SLPUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Support/InstructionCost.h"

#include <tuple>
#include <utility>

namespace llvm {
class APInt;
class FastMathFlags;
class FixedVectorType;
class Instruction;
class TargetLibraryInfo;
class Type;
class User;
class Value;
class VectorType;
enum class RecurKind;
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

/// For a non-power-of-2 \p NumElts-wide integer div/rem \p Opcode, checks if
/// padding to a full register and using the masked div/rem intrinsic is
/// cheaper than the direct vector op. Returns the cost of the masked
/// alternative, or an invalid cost if it is not applicable or not cheaper.
InstructionCost
getMaskedDivRemCost(const TargetTransformInfo &TTI, bool ReVec, unsigned Opcode,
                    Type *ScalarTy, unsigned NumElts,
                    const TargetTransformInfo::TargetCostKind CostKind,
                    FixedVectorType **PaddedTy = nullptr);

/// Returns the cost of the booleanized logical and/or reduction of a vector
/// of type \p VecTy with the i1 root \p Root, emitted as the wide reduction
/// plus the result trunc.
InstructionCost
getBoolReduxWideRdxCost(const TargetTransformInfo &TTI, RecurKind RdxKind,
                        FixedVectorType *VecTy, const Value *Root,
                        FastMathFlags FMF,
                        TargetTransformInfo::TargetCostKind CostKind);

/// Returns the cost of the booleanized logical and/or reduction of a vector
/// of type \p VecTy with the i1 root \p Root, emitted as trunc+bitcast+cmp,
/// estimated in the context of the replaced cast chain \p ChainInsts.
InstructionCost
getBoolReduxBitcastCmpCost(const TargetTransformInfo &TTI, RecurKind RdxKind,
                           FixedVectorType *VecTy, const Value *Root,
                           ArrayRef<Instruction *> ChainInsts,
                           TargetTransformInfo::TargetCostKind CostKind);

/// This is similar to TargetTransformInfo::getScalarizationOverhead, but if
/// ScalarTy is a FixedVectorType, a vector will be inserted or extracted
/// instead of a scalar.
InstructionCost
getScalarizationOverhead(const TargetTransformInfo &TTI, bool ReVec,
                         Type *ScalarTy, VectorType *Ty,
                         const APInt &DemandedElts, bool Insert, bool Extract,
                         const TargetTransformInfo::TargetCostKind CostKind,
                         bool ForPoisonSrc = true, ArrayRef<Value *> VL = {},
                         TargetTransformInfo::VectorInstrContext VIC =
                             TargetTransformInfo::VectorInstrContext::None);

/// This is similar to TargetTransformInfo::getVectorInstrCost, but if ScalarTy
/// is a FixedVectorType, a vector will be extracted instead of a scalar.
InstructionCost
getVectorInstrCost(const TargetTransformInfo &TTI, bool ReVec, Type *ScalarTy,
                   unsigned Opcode, Type *Val,
                   const TargetTransformInfo::TargetCostKind CostKind,
                   unsigned Index, Value *Scalar,
                   ArrayRef<std::tuple<Value *, User *, int>> ScalarUserAndIdx);

/// This is similar to TargetTransformInfo::getExtractWithExtendCost, but if Dst
/// is a FixedVectorType, a vector will be extracted instead of a scalar.
InstructionCost
getExtractWithExtendCost(const TargetTransformInfo &TTI, bool ReVec,
                         unsigned Opcode, Type *Dst, VectorType *VecTy,
                         unsigned Index,
                         const TargetTransformInfo::TargetCostKind CostKind);

/// Returns the cost of the bitfield packing of \p SrcTy into \p ResultTy,
/// picking the cheapest shift width. The packing is a trunc, an lshr, a byte
/// shuffle and a bitcast. \p FreeByteTrunc marks the lanes as a zext from i8,
/// so compacting them to bytes is free.
InstructionCost getBitPackCost(const TargetTransformInfo &TTI,
                               FixedVectorType *SrcTy, Type *ResultTy,
                               const BitPackInfo &Info, bool FreeByteTrunc,
                               TargetTransformInfo::TargetCostKind CostKind,
                               const TargetLibraryInfo *TLI,
                               const Instruction *CxtI, unsigned &ShiftWidth);

} // namespace llvm::slpvectorizer

#endif // LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPCOSTANALYSIS_H
