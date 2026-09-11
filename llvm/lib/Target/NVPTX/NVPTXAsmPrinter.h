//===-- NVPTXAsmPrinter.h - NVPTX LLVM assembly writer ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NVPTX_NVPTXASMPRINTER_H
#define LLVM_LIB_TARGET_NVPTX_NVPTXASMPRINTER_H

#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class NVPTXAsmPrinterBeginPass
    : public RequiredPassInfoMixin<NVPTXAsmPrinterBeginPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

class NVPTXAsmPrinterPass : public RequiredPassInfoMixin<NVPTXAsmPrinterPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class NVPTXAsmPrinterEndPass
    : public RequiredPassInfoMixin<NVPTXAsmPrinterEndPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_NVPTX_NVPTXASMPRINTER_H
