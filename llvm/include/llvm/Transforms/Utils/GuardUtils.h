//===-- GuardUtils.h - Utils for work with guards ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Utils that are used to perform transformations related to guards and their
// conditions.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_GUARDUTILS_H
#define LLVM_TRANSFORMS_UTILS_GUARDUTILS_H

namespace llvm {

class CallInst;
class Function;

/// Splits control flow at point of \p Guard, replacing it with explicit branch
/// by the condition of guard's first argument. The taken branch then goes to
/// the block that contains  \p Guard's successors, and the non-taken branch
/// goes to a newly-created deopt block that contains a sole call of the
/// deoptimize function \p DeoptIntrinsic.
void makeGuardControlFlowExplicit(Function *DeoptIntrinsic, CallInst *Guard);

} // llvm

#endif // LLVM_TRANSFORMS_UTILS_GUARDUTILS_H
