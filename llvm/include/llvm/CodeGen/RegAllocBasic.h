//==- RegAllocBasic.h ----------- basic register allocator ---------*-C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REGALLOC_BASIC_H
#define LLVM_CODEGEN_REGALLOC_BASIC_H

#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/RegAllocCommon.h"

namespace llvm {

class RABasicPass : public RequiredPassInfoMixin<RABasicPass> {
public:
  struct Options {
    RegAllocFilterFunc Filter;
    StringRef FilterName;
    Options(RegAllocFilterFunc F = nullptr, StringRef FN = "all")
        : Filter(std::move(F)), FilterName(FN) {}
  };

  RABasicPass(Options Opts = Options()) : Opts(std::move(Opts)) {}

  LLVM_ABI PreservedAnalyses run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &MFAM);

  MachineFunctionProperties getRequiredProperties() const {
    return MachineFunctionProperties().setNoPHIs();
  }

  MachineFunctionProperties getClearedProperties() const {
    return MachineFunctionProperties().setIsSSA();
  }

private:
  Options Opts;
};

} // namespace llvm

#endif // LLVM_CODEGEN_REGALLOC_BASIC_H
