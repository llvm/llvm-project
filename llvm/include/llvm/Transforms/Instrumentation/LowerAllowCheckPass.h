//===- LowerAllowCheckPass.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file provides the interface for the pass responsible for removing
/// expensive ubsan checks.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_LOWERALLOWCHECKPASS_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_LOWERALLOWCHECKPASS_H

#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

// This pass is responsible for removing optional traps, like llvm.ubsantrap
// from the hot code.
class LowerAllowCheckPass : public PassInfoMixin<LowerAllowCheckPass> {
public:
  struct Options {
    std::vector<unsigned int> cutoffs;
    unsigned int runtime_check = 0;
  };

  explicit LowerAllowCheckPass(LowerAllowCheckPass::Options Opts)
      : Opts(std::move(Opts)) {};
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  LLVM_ABI static bool IsRequested();
  LLVM_ABI void
  printPipeline(raw_ostream &OS,
                function_ref<StringRef(StringRef)> MapClassName2PassName);

private:
  LowerAllowCheckPass::Options Opts;
};

} // namespace llvm

#endif
