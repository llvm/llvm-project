//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Holds an AsmPrinter instance so that state can be shared appropriately
// between the Module and MachineFunction portions of AsmPrinter.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_ASMPRINTERANALYSIS_H
#define LLVM_CODEGEN_ASMPRINTERANALYSIS_H

#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class AsmPrinterAnalysis : public AnalysisInfoMixin<AsmPrinterAnalysis> {
public:
  static AnalysisKey Key;
  std::unique_ptr<AsmPrinter> HeldPrinter;

  class Result {
    AsmPrinter &Printer;
    Result(AsmPrinter &Printer) : Printer(Printer) {}
    friend class AsmPrinterAnalysis;

  public:
    AsmPrinter &getPrinter() { return Printer; }

    bool invalidate(Module &, const PreservedAnalyses &,
                    ModuleAnalysisManager::Invalidator &) {
      return false;
    }
  };

  Result run(Module &M, ModuleAnalysisManager &) {
    return Result(*HeldPrinter);
  }

public:
  AsmPrinterAnalysis(std::unique_ptr<AsmPrinter> Printer)
      : HeldPrinter(std::move(Printer)) {}
};

} // namespace llvm

#endif //  LLVM_CODEGEN_ASMPRINTERANALYSIS_H
