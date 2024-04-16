//===- ExpandVariadics.h - expand variadic functions ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_IPO_EXPANDVARIADICS_H
#define LLVM_TRANSFORMS_IPO_EXPANDVARIADICS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Module;
class ModulePass;
class OptimizationLevel;

enum class ExpandVariadicsMode {
  unspecified,
  disable,
  optimize,
  lowering,
};

class ExpandVariadicsPass : public PassInfoMixin<ExpandVariadicsPass> {
  const ExpandVariadicsMode ConstructedMode;

public:
  // Operates under passed mode unless overridden on commandline
  ExpandVariadicsPass(ExpandVariadicsMode ConstructedMode);

  // Chooses disable or optimize based on optimization level
  ExpandVariadicsPass(OptimizationLevel Level);

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

ModulePass *createExpandVariadicsPass(ExpandVariadicsMode);

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_EXPANDVARIADICS_H
