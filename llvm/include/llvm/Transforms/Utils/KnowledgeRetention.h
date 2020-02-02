//===- KnowledgeRetention.h - utilities to preserve informations *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contain tools to preserve informations. They should be used before
// performing a transformation that may move and delete instructions as those
// transformation may destroy or worsen information that can be derived from the
// IR.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_ASSUMEBUILDER_H
#define LLVM_TRANSFORMS_UTILS_ASSUMEBUILDER_H

#include "llvm/IR/Instruction.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// Build a call to llvm.assume to preserve informations that can be derived
/// from the given instruction.
/// If no information derived from \p I, this call returns null.
/// The returned instruction is not inserted anywhere.
CallInst *BuildAssumeFromInst(const Instruction *I, Module *M);
inline CallInst *BuildAssumeFromInst(Instruction *I) {
  return BuildAssumeFromInst(I, I->getModule());
}

/// This pass will try to build an llvm.assume for every instruction in the
/// function. Its main purpose is testing.
struct AssumeBuilderPass : public PassInfoMixin<AssumeBuilderPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif
