//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DIRECTX_DIRECTXASMPRINTER_H
#define LLVM_LIB_TARGET_DIRECTX_DIRECTXASMPRINTER_H

#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class DXILAsmPrinterBeginPass
    : public RequiredPassInfoMixin<DXILAsmPrinterBeginPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

class DXILAsmPrinterPass : public RequiredPassInfoMixin<DXILAsmPrinterPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class DXILAsmPrinterEndPass
    : public RequiredPassInfoMixin<DXILAsmPrinterEndPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_DIRECTX_DIRECTXASMPRINTER_H
