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

/// Returns true iff \p U has semantics of a guard expressed in a form of call
/// of llvm.experimental.guard intrinsic.
bool isGuard(const User *U);

/// Returns true iff \p U is a widenable branch (that is, parseWidenableBranch
/// returns true).
bool isWidenableBranch(const User *U);

/// Returns true iff \p U has semantics of a guard expressed in a form of a
/// widenable conditional branch to deopt block.
bool isGuardAsWidenableBranch(const User *U);

/// If U is widenable branch looking like:
///   %cond = ...
///   %wc = call i1 @llvm.experimental.widenable.condition()
///   %branch_cond = and i1 %cond, %wc
///   br i1 %branch_cond, label %if_true_bb, label %if_false_bb ; <--- U
/// The function returns true, and the values %cond and %wc and blocks
/// %if_true_bb, if_false_bb are returned in
/// the parameters (Condition, WidenableCondition, IfTrueBB and IfFalseFF)
/// respectively. If \p U does not match this pattern, return false.
bool parseWidenableBranch(const User *U, Value *&Condition,
                          Value *&WidenableCondition, BasicBlock *&IfTrueBB,
                          BasicBlock *&IfFalseBB);

/// Analgous to the above, but return the Uses so that that they can be
/// modified. Unlike previous version, Condition is optional and may be null.
bool parseWidenableBranch(User *U, Use *&Cond, Use *&WC, BasicBlock *&IfTrueBB,
                          BasicBlock *&IfFalseBB);
  
} // llvm

#endif // LLVM_ANALYSIS_GUARDUTILS_H
