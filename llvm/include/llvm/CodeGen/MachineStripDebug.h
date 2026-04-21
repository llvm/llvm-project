//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This removes debug info from everything. It can be used to ensure tests can
// be debugified without affecting the output MIR.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINESTRIPDEBUG_H
#define LLVM_CODEGEN_MACHINESTRIPDEBUG_H

#include "llvm/IR/Analysis.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class StripDebugMachineModulePass
    : public PassInfoMixin<StripDebugMachineModulePass> {
public:
  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_CODEGEN_MACHINESTRIPDEBUG_H
