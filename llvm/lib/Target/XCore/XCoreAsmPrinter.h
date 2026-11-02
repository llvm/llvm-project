//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_XCORE_XCOREASMPRINTER_H
#define LLVM_LIB_TARGET_XCORE_XCOREASMPRINTER_H

#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class XCoreAsmPrinterBeginPass
    : public RequiredPassInfoMixin<XCoreAsmPrinterBeginPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MFAM);
};

class XCoreAsmPrinterPass : public RequiredPassInfoMixin<XCoreAsmPrinterPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class XCoreAsmPrinterEndPass
    : public RequiredPassInfoMixin<XCoreAsmPrinterEndPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MFAM);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_XCORE_XCOREASMPRINTER_H
