//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_UNKNOWNINTRINSICPRINTER_H
#define LLVM_ANALYSIS_UNKNOWNINTRINSICPRINTER_H

#include "llvm/IR/PassManager.h"

namespace llvm {
class UnknownIntrinsicPrinterPass
    : public PassInfoMixin<UnknownIntrinsicPrinterPass> {
  raw_ostream &OS;

public:
  UnknownIntrinsicPrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};
} // namespace llvm

#endif // LLVM_ANALYSIS_UNKNOWNINTRINSICPRINTER_H
