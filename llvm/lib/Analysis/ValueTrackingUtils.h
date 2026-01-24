//===- ValueTrackingUtils.h - ValueTracking utilities ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANLAYSIS_VALUETRACKINGUTILS
#define LLVM_ANLAYSIS_VALUETRACKINGUTILS

#include "llvm/Analysis/SimplifyQuery.h"
#include "llvm/IR/Instructions.h"

namespace llvm {
enum class UndefPoisonKind {
  PoisonOnly = (1 << 0),
  UndefOnly = (1 << 1),
  UndefOrPoison = PoisonOnly | UndefOnly,
};
class Operator;
class AddOperator;
class AssumptionCache;
class DominatorTree;
class GEPOperator;
class WithOverflowInst;
struct KnownBits;
struct KnownFPClass;
class Loop;
class LoopInfo;
class MDNode;
class StringRef;
class TargetLibraryInfo;
class IntrinsicInst;
class ShuffleVectorInst;
} // namespace llvm

namespace llvm::vtutils {
unsigned getBitWidth(Type *Ty, const DataLayout &DL);
const Instruction *safeCxtI(const Value *V, const Instruction *CxtI);
bool getShuffleDemandedElts(const ShuffleVectorInst *Shuf,
                            const APInt &DemandedElts, APInt &DemandedLHS,
                            APInt &DemandedRHS);
bool isEphemeralValueOf(const Instruction *I, const Value *E);
bool programUndefinedIfUndefOrPoison(const Value *V, bool PoisonOnly);
bool canCreateUndefOrPoison(const Operator *Op, UndefPoisonKind Kind,
                            bool ConsiderFlagsAndMetadata);
bool includesPoison(UndefPoisonKind Kind);
bool includesUndef(UndefPoisonKind Kind);
bool cmpExcludesZero(CmpInst::Predicate Pred, const Value *RHS);
void breakSelfRecursivePHI(const Use *U, const PHINode *PHI, Value *&ValOut,
                           Instruction *&CtxIOut,
                           const PHINode **PhiOut = nullptr);
bool isSignedMinMaxClamp(const Value *Select, const Value *&In,
                         const APInt *&CLow, const APInt *&CHigh);
bool isSignedMinMaxIntrinsicClamp(const IntrinsicInst *II, const APInt *&CLow,
                                  const APInt *&CHigh);
void unionWithMinMaxIntrinsicClamp(const IntrinsicInst *II, KnownBits &Known);
bool isNonEqualPointersWithRecursiveGEP(const Value *A, const Value *B,
                                        const SimplifyQuery &Q);
bool isKnownNonNaN(const Value *V, FastMathFlags FMF);
bool isKnownNonZero(const Value *V);
Value *lookThroughCast(CmpInst *CmpI, Value *V1, Value *V2,
                       Instruction::CastOps *CastOp);
void setLimitForFPToI(const Instruction *I, APInt &Lower, APInt &Upper);
ConstantRange getRangeForIntrinsic(const IntrinsicInst &II, bool UseInstrInfo);
ConstantRange getRangeForSelectPattern(const SelectInst &SI,
                                       const InstrInfoQuery &IIQ);
void setLimitsForBinOp(const BinaryOperator &BO, APInt &Lower, APInt &Upper,
                       const InstrInfoQuery &IIQ, bool PreferSignedRange);
bool isTruePredicate(CmpInst::Predicate Pred, const Value *LHS,
                     const Value *RHS);
std::optional<std::pair<Value *, Value *>>
getInvertibleOperands(const Operator *Op1, const Operator *Op2);
} // namespace llvm::vtutils
#endif
