//==- RegAllocFast.h ----------- fast register allocator  ----------*-C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REGALLOCFAST_H
#define LLVM_CODEGEN_REGALLOCFAST_H

#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/RegAllocCommon.h"

namespace llvm {

class RegAllocFastPass : public PassInfoMixin<RegAllocFastPass> {
public:
  struct Options {
    RegAllocFilterFunc Filter;
    StringRef FilterName;
    bool ClearVRegs;
    Options(RegAllocFilterFunc F = nullptr, StringRef FN = "all",
            bool CV = true)
        : Filter(std::move(F)), FilterName(FN), ClearVRegs(CV) {}
  };

  RegAllocFastPass(Options Opts = Options()) : Opts(std::move(Opts)) {}

  MachineFunctionProperties getRequiredProperties() const {
    return MachineFunctionProperties().setNoPHIs();
  }

  MachineFunctionProperties getSetProperties() const {
    if (Opts.ClearVRegs) {
      return MachineFunctionProperties().setNoVRegs();
    }

    return MachineFunctionProperties();
  }

  MachineFunctionProperties getClearedProperties() const {
    return MachineFunctionProperties().setIsSSA();
  }

  PreservedAnalyses run(MachineFunction &MF, MachineFunctionAnalysisManager &);

  void printPipeline(raw_ostream &OS,
                     function_ref<StringRef(StringRef)> MapClassName2PassName);

  static bool isRequired() { return true; }

private:
  Options Opts;
};

} // namespace llvm

#endif // LLVM_CODEGEN_REGALLOCFAST_H
