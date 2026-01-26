//===-- CFGSCCPrinter.h ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CFGSCCPRINTER_H
#define LLVM_ANALYSIS_CFGSCCPRINTER_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class CFGSCCPrinterPass : public PassInfoMixin<CFGSCCPrinterPass> {
  raw_ostream &OS;

public:
  explicit CFGSCCPrinterPass(raw_ostream &OS) : OS(OS) {}
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};
} // namespace llvm

#endif
