//===-- GuardUtils.h - Utils for work with guards ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Utils that are used to perform analyzes related to guards and their
// conditions.
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_GUARDUTILS_H
#define LLVM_ANALYSIS_GUARDUTILS_H

namespace llvm {

class BasicBlock;
class Use;
class User;
class Value;
template <typename T> class SmallVectorImpl;

/// Returns true iff \p U has semantics of a guard expressed in a form of call
/// of llvm.experimental.guard intrinsic.
bool isGuard(const User *U);

/// Returns true iff \p V has semantics of llvm.experimental.widenable.condition
/// call
bool isWidenableCondition(const Value *V);

/// Returns true iff \p U is a widenable branch (that is,
/// extractWidenableCondition returns widenable condition).
bool isWidenableBranch(const User *U);

/// Returns true iff \p U has semantics of a guard expressed in a form of a
/// widenable conditional branch to deopt block.
bool isGuardAsWidenableBranch(const User *U);

// The guard condition is expected to be in form of:
//   cond1 && cond2 && cond3 ...
// or in case of widenable branch:
//   cond1 && cond2 && cond3 && widenable_contidion ...
// Method collects the list of checks, but skips widenable_condition.
void parseWidenableGuard(const User *U, SmallVectorImpl<Value *> &Checks);

// Returns widenable_condition if it exists in the expression tree rooting from
// \p U and has only one use.
Value *extractWidenableCondition(const User *U);
} // llvm

#endif // LLVM_ANALYSIS_GUARDUTILS_H
