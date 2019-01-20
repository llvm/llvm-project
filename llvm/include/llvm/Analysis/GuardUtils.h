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

class User;

/// Returns true iff \p U has semantics of a guard.
bool isGuard(const User *U);

} // llvm

#endif // LLVM_ANALYSIS_GUARDUTILS_H

