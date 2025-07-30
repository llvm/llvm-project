//==- RegAllocGreedyPass.h --- greedy register allocator pass ------*-C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CODEGEN_REGALLOC_GREEDY_PASS_H
#define LLVM_CODEGEN_REGALLOC_GREEDY_PASS_H

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/RegAllocCommon.h"
#include "llvm/CodeGen/RegAllocFast.h"
#include "llvm/IR/PassManager.h"

using namespace llvm;

class RAGreedyPass : public PassInfoMixin<RAGreedyPass> {
public:
  struct Options {
    RegAllocFilterFunc Filter;
    StringRef FilterName;
    Options(RegAllocFilterFunc F = nullptr, StringRef FN = "all")
        : Filter(std::move(F)), FilterName(FN) {};
  };

  RAGreedyPass(Options Opts = Options()) : Opts(std::move(Opts)) {}
  PreservedAnalyses run(MachineFunction &F, MachineFunctionAnalysisManager &AM);

  MachineFunctionProperties getRequiredProperties() const {
    return MachineFunctionProperties().setNoPHIs();
  }

  MachineFunctionProperties getClearedProperties() const {
    return MachineFunctionProperties().setIsSSA();
  }

  void
  printPipeline(raw_ostream &OS,
                function_ref<StringRef(StringRef)> MapClassName2PassName) const;
  static bool isRequired() { return true; }

private:
  Options Opts;
};

#endif // LLVM_CODEGEN_REGALLOC_GREEDY_PASS_H
