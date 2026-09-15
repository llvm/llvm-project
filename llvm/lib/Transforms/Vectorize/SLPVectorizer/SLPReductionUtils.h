//===- SLPReductionUtils.h - SLP reduction match helpers -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Internal header used by SLPVectorizer.cpp. It declares free reduction
// pattern-match helpers that do not depend on BoUpSLP or any other SLP-private
// type.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPREDUCTIONUTILS_H
#define LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPREDUCTIONUTILS_H

#include "llvm/Analysis/TargetTransformInfo.h"

namespace llvm {
class FastMathFlags;
class IRBuilderBase;
class Instruction;
class PHINode;
class Type;
class Value;
enum class RecurKind;
} // namespace llvm

namespace llvm::slpvectorizer {

/// \returns the wide leaf type if the logical and/or reduction \p RdxKind
/// with the i1 root type \p RootTy and the leaf type \p LeafTy is a
/// booleanized reduction (performed in the wide leaf type, bit 0 of the
/// result is the final value), nullptr otherwise.
Type *getBoolReduxWideTy(RecurKind RdxKind, Type *RootTy, Type *LeafTy);

/// \returns the first operand of \p I that does not match \p Phi. If
/// the operand is not an instruction, returns nullptr.
Instruction *getNonPhiOperand(Instruction *I, PHINode *Phi);

/// \returns true if \p I is a candidate instruction for reduction
/// vectorization.
bool isReductionCandidate(Instruction *I);

/// Emits the booleanized logical and/or reduction of \p Vec with the i1 root
/// \p Root as trunc+bitcast+cmp (all-ones comparison for And, zero for Or) if
/// it is cheaper than the wide reduction plus the result trunc. \returns the
/// i1 result or nullptr if the wide reduction form is cheaper.
Value *tryEmitBoolReduxBitcastCmp(IRBuilderBase &Builder,
                                  const TargetTransformInfo &TTI,
                                  RecurKind RdxKind, Value *Vec,
                                  const Value *Root, FastMathFlags FMF,
                                  TargetTransformInfo::TargetCostKind CostKind);

} // namespace llvm::slpvectorizer

#endif // LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPREDUCTIONUTILS_H
