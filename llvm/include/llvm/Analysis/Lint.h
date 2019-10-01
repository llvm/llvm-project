//===-- llvm/Analysis/Lint.h - LLVM IR Lint ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines lint interfaces that can be used for some sanity checking
// of input to the system, and for checking that transformations
// haven't done something bad. In contrast to the Verifier, the Lint checker
// checks for undefined behavior or constructions with likely unintended
// behavior.
//
// To see what specifically is checked, look at Lint.cpp
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LINT_H
#define LLVM_ANALYSIS_LINT_H

namespace llvm {

class FunctionPass;
class Module;
class Function;

/// Create a lint pass.
///
/// Check a module or function.
FunctionPass *createLintPass();

/// Check a module.
///
/// This should only be used for debugging, because it plays games with
/// PassManagers and stuff.
void lintModule(
  const Module &M    ///< The module to be checked
);

// lintFunction - Check a function.
void lintFunction(
  const Function &F  ///< The function to be checked
);

} // End llvm namespace

#endif
