//==- RegAllocGreedyPass.h --- greedy register allocator pass ------*-C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
        : Filter(F), FilterName(FN) {};
  };

  RAGreedyPass(Options Opts = Options()) : Opts(Opts) {}
  PreservedAnalyses run(MachineFunction &F, MachineFunctionAnalysisManager &AM);

  MachineFunctionProperties getRequiredProperties() const {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoPHIs);
  }

  MachineFunctionProperties getClearedProperties() const {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::IsSSA);
  }

  void printPipeline(raw_ostream &OS, function_ref<StringRef(StringRef)> MapClassName2PassName) const;
  static bool isRequired() { return true; }

private:
  Options Opts;
};
