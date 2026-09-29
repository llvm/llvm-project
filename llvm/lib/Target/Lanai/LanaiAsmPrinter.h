//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_LANAI_LANAIASMPRINTER_H
#define LLVM_LIB_TARGET_LANAI_LANAIASMPRINTER_H

#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class LanaiAsmPrinterBeginPass
    : public RequiredPassInfoMixin<LanaiAsmPrinterBeginPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MFAM);
};

class LanaiAsmPrinterPass : public RequiredPassInfoMixin<LanaiAsmPrinterPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class LanaiAsmPrinterEndPass
    : public RequiredPassInfoMixin<LanaiAsmPrinterEndPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MFAM);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_LANAI_LANAIASMPRINTER_H
