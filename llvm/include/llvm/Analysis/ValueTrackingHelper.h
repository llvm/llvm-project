//===- ValueTrackingHelper.h - Helper functions for ValueTracking ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_VALUETRACKINGHELPER_H
#define LLVM_ANALYSIS_VALUETRACKINGHELPER_H

#include "llvm/IR/BasicBlock.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class Type;
class Use;
class Value;
class Instruction;
class PHINode;
class BinaryOperator;
class IntrinsicInst;
class ShuffleVectorInst;
class Function;
class DataLayout;
class MDNode;
class ConstantRange;
class AssumptionCache;
class DominatorTree;
class CmpPredicate;
struct KnownBits;
struct SimplifyQuery;

/// Attempt to match a simple first order recurrence cycle of the form:
///   %iv = phi Ty [%Start, %Entry], [%Inc, %backedge]
///   %inc = binop %iv, %step
/// OR
///   %iv = phi Ty [%Start, %Entry], [%Inc, %backedge]
///   %inc = binop %step, %iv
///
/// A first order recurrence is a formula with the form: X_n = f(X_(n-1))
///
/// A couple of notes on subtleties in that definition:
/// * The Step does not have to be loop invariant.  In math terms, it can
///   be a free variable.  We allow recurrences with both constant and
///   variable coefficients. Callers may wish to filter cases where Step
///   does not dominate P.
/// * For non-commutative operators, we will match both forms.  This
///   results in some odd recurrence structures.  Callers may wish to filter
///   out recurrences where the phi is not the LHS of the returned operator.
/// * Because of the structure matched, the caller can assume as a post
///   condition of the match the presence of a Loop with P's parent as it's
///   header *except* in unreachable code.  (Dominance decays in unreachable
///   code.)
///
/// NOTE: This is intentional simple.  If you want the ability to analyze
/// non-trivial loop conditons, see ScalarEvolution instead.
LLVM_ABI bool matchSimpleRecurrence(const PHINode *P, BinaryOperator *&BO,
                                    Value *&Start, Value *&Step);

/// Analogous to the above, but starting from the binary operator
LLVM_ABI bool matchSimpleRecurrence(const BinaryOperator *I, PHINode *&P,
                                    Value *&Start, Value *&Step);

/// Determine the possible constant range of vscale with the given bit width,
/// based on the vscale_range function attribute.
LLVM_ABI ConstantRange getVScaleRange(const Function *F, unsigned BitWidth);

/// Return true if this function can prove that V does not have undef bits
/// and is never poison. If V is an aggregate value or vector, check whether
/// all elements (except padding) are not undef or poison.
/// Note that this is different from canCreateUndefOrPoison because the
/// function assumes Op's operands are not poison/undef.
///
/// If CtxI and DT are specified this method performs flow-sensitive analysis
/// and returns true if it is guaranteed to be never undef or poison
/// immediately before the CtxI.
LLVM_ABI bool
isGuaranteedNotToBeUndefOrPoison(const Value *V, AssumptionCache *AC = nullptr,
                                 const Instruction *CtxI = nullptr,
                                 const DominatorTree *DT = nullptr,
                                 unsigned Depth = 0);

/// Returns true if V cannot be poison, but may be undef.
LLVM_ABI bool isGuaranteedNotToBePoison(const Value *V,
                                        AssumptionCache *AC = nullptr,
                                        const Instruction *CtxI = nullptr,
                                        const DominatorTree *DT = nullptr,
                                        unsigned Depth = 0);

inline bool isGuaranteedNotToBePoison(const Value *V, AssumptionCache *AC,
                                      BasicBlock::iterator CtxI,
                                      const DominatorTree *DT = nullptr,
                                      unsigned Depth = 0) {
  // Takes an iterator as a position, passes down to Instruction *
  // implementation.
  return isGuaranteedNotToBePoison(V, AC, &*CtxI, DT, Depth);
}

/// Returns true if V cannot be undef, but may be poison.
LLVM_ABI bool isGuaranteedNotToBeUndef(const Value *V,
                                       AssumptionCache *AC = nullptr,
                                       const Instruction *CtxI = nullptr,
                                       const DominatorTree *DT = nullptr,
                                       unsigned Depth = 0);

/// Return the boolean condition value in the context of the given instruction
/// if it is known based on dominating conditions.
LLVM_ABI std::optional<bool>
isImpliedByDomCondition(const Value *Cond, const Instruction *ContextI,
                        const DataLayout &DL);
LLVM_ABI std::optional<bool>
isImpliedByDomCondition(CmpPredicate Pred, const Value *LHS, const Value *RHS,
                        const Instruction *ContextI, const DataLayout &DL);

/// Adjust \p Known for the given select \p Arm to include information from the
/// select \p Cond.
LLVM_ABI void adjustKnownBitsForSelectArm(KnownBits &Known, Value *Cond,
                                          Value *Arm, bool Invert,
                                          const SimplifyQuery &Q,
                                          unsigned Depth = 0);

/// Compute known bits from the range metadata.
/// \p KnownZero the set of bits that are known to be zero
/// \p KnownOne the set of bits that are known to be one
LLVM_ABI void computeKnownBitsFromRangeMetadata(const MDNode &Ranges,
                                                KnownBits &Known);

/// Merge bits known from context-dependent facts into Known.
LLVM_ABI void computeKnownBitsFromContext(const Value *V, KnownBits &Known,
                                          const SimplifyQuery &Q,
                                          unsigned Depth = 0);

namespace vthelper {
LLVM_ABI unsigned getBitWidth(Type *Ty, const DataLayout &DL);
LLVM_ABI const Instruction *safeCxtI(const Value *V, const Instruction *CxtI);
LLVM_ABI void breakSelfRecursivePHI(const Use *U, const PHINode *PHI,
                                    Value *&ValOut, Instruction *&CtxIOut,
                                    const PHINode **PhiOut = nullptr);
LLVM_ABI void unionWithMinMaxIntrinsicClamp(const IntrinsicInst *II,
                                            KnownBits &Known);
LLVM_ABI bool isKnownNonZero(const Value *V, const APInt &DemandedElts,
                             const SimplifyQuery &Q, unsigned Depth = 0);
LLVM_ABI bool getShuffleDemandedElts(const ShuffleVectorInst *Shuf,
                                     const APInt &DemandedElts,
                                     APInt &DemandedLHS, APInt &DemandedRHS);
LLVM_ABI void computeKnownBits(const Value *V, const APInt &DemandedElts,
                               KnownBits &Known, const SimplifyQuery &Q,
                               unsigned Depth);
} // namespace vthelper
} // namespace llvm

#endif
