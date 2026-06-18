//===- ModuleDebugInfoPrinter.h - -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MODULEDEBUGINFOPRINTER_H
#define LLVM_ANALYSIS_MODULEDEBUGINFOPRINTER_H

#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
class raw_ostream;

class ModuleDebugInfoPrinterPass
    : public RequiredPassInfoMixin<ModuleDebugInfoPrinterPass> {
  DebugInfoFinder Finder;
  raw_ostream &OS;

public:
  LLVM_ABI explicit ModuleDebugInfoPrinterPass(raw_ostream &OS);
  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};
} // end namespace llvm

#endif // LLVM_ANALYSIS_MODULEDEBUGINFOPRINTER_H
