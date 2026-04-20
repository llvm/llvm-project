//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass attaches synthetic debug info to everything. It can be used to
// create targeted tests for debug info preservation, or test for CodeGen
// differences with vs. without debug info.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEDEBUGIFY_H_
#define LLVM_CODEGEN_MACHINEDEBUGIFY_H_

#include "llvm/IR/Analysis.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class DebugifyMachineModulePass
    : public PassInfoMixin<DebugifyMachineModulePass> {
public:
  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_CODEGEN_MACHINEDEBUGIFY_H_
