//===- llvm/Analysis/FloatingPointPredicateUtils.h ------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_FLOATINGPOINTPREDICATEUTILS_H
#define LLVM_ANALYSIS_FLOATINGPOINTPREDICATEUTILS_H

#include "llvm/ADT/GenericFloatingPointPredicateUtils.h"
#include "llvm/IR/SSAContext.h"

namespace llvm {

using FloatingPointPredicateUtils =
    GenericFloatingPointPredicateUtils<SSAContext>;

/// Returns a pair of values, which if passed to llvm.is.fpclass, returns the
/// same result as an fcmp with the given operands.
///
/// If \p LookThroughSrc is true, consider the input value when computing the
/// mask.
///
/// If \p LookThroughSrc is false, ignore the source value (i.e. the first pair
/// element will always be LHS.
inline std::pair<Value *, FPClassTest>
fcmpToClassTest(FCmpInst::Predicate Pred, const Function &F, Value *LHS,
                Value *RHS, bool LookThroughSrc = true) {
  return FloatingPointPredicateUtils::fcmpToClassTest(Pred, F, LHS, RHS,
                                                      LookThroughSrc = true);
}

/// Returns a pair of values, which if passed to llvm.is.fpclass, returns the
/// same result as an fcmp with the given operands.
///
/// If \p LookThroughSrc is true, consider the input value when computing the
/// mask.
///
/// If \p LookThroughSrc is false, ignore the source value (i.e. the first pair
/// element will always be LHS.
inline std::pair<Value *, FPClassTest>
fcmpToClassTest(FCmpInst::Predicate Pred, const Function &F, Value *LHS,
                const APFloat *ConstRHS, bool LookThroughSrc = true) {
  return FloatingPointPredicateUtils::fcmpToClassTest(Pred, F, LHS, *ConstRHS,
                                                      LookThroughSrc);
}

inline std::tuple<Value *, FPClassTest, FPClassTest>
fcmpImpliesClass(CmpInst::Predicate Pred, const Function &F, Value *LHS,
                 FPClassTest RHSClass, bool LookThroughSrc = true) {
  return FloatingPointPredicateUtils::fcmpImpliesClass(Pred, F, LHS, RHSClass,
                                                       LookThroughSrc);
}

inline std::tuple<Value *, FPClassTest, FPClassTest>
fcmpImpliesClass(CmpInst::Predicate Pred, const Function &F, Value *LHS,
                 const APFloat &ConstRHS, bool LookThroughSrc = true) {
  return FloatingPointPredicateUtils::fcmpImpliesClass(Pred, F, LHS, ConstRHS,
                                                       LookThroughSrc);
}

inline std::tuple<Value *, FPClassTest, FPClassTest>
fcmpImpliesClass(CmpInst::Predicate Pred, const Function &F, Value *LHS,
                 Value *RHS, bool LookThroughSrc = true) {
  return FloatingPointPredicateUtils::fcmpImpliesClass(Pred, F, LHS, RHS,
                                                       LookThroughSrc);
}

} // namespace llvm

#endif // LLVM_ANALYSIS_FLOATINGPOINTPREDICATEUTILS_H
