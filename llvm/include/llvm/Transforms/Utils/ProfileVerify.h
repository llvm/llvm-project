//===- ProfileVerify.h - Verify profile info for testing ----------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Inject profile information, as part of tests, to verify passes don't
// accidentally drop it.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_UTILS_PROFILEVERIFY_H
#define LLVM_TRANSFORMS_UTILS_PROFILEVERIFY_H

#include "llvm/IR/Analysis.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
/// Inject MD_prof metadata where it's missing. Used for testing that passes
/// don't accidentally drop this metadata.
class ProfileInjectorPass : public PassInfoMixin<ProfileInjectorPass> {
public:
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};

/// Checks that MD_prof is present on every instruction that supports it. Used
/// in conjunction with the ProfileInjectorPass. MD_prof "unknown" is considered
/// valid (i.e. !{!"unknown"})
class ProfileVerifierPass : public PassInfoMixin<ProfileVerifierPass> {
public:
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};

} // namespace llvm
#endif
