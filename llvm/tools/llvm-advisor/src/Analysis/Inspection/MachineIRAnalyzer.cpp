//===------------------- MachineIRAnalyzer.cpp - LLVM Advisor -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Inspection/MachineIRAnalyzer.h"
#include "Analysis/IR/IRAnalysisUtils.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
MachineIRAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  return withIRModule(Context, CapID, UnitID,
                      [&](LLVMContext &, Module &M) {
    int64_t Functions = 0;
    int64_t Instructions = 0;
    int64_t PhiNodes = 0;
    int64_t MemoryOps = 0;
    for (const Function &F : M) {
      if (F.isDeclaration())
        continue;
      ++Functions;
      for (const BasicBlock &BB : F) {
        for (const Instruction &I : BB) {
          ++Instructions;
          if (isa<PHINode>(I))
            ++PhiNodes;
          if (isa<LoadInst>(I) || isa<StoreInst>(I))
            ++MemoryOps;
        }
      }
    }

    return makeJSONResult(CapID, UnitID, json::Object{
        {"target_triple", M.getTargetTriple().str()},
        {"module_data_layout", M.getDataLayoutStr()},
        {"function_count", Functions},
        {"ir_instruction_count", Instructions},
        {"phi_count", PhiNodes},
        {"memory_op_count", MemoryOps},
        {"note", "MachineIR-relevant structural metrics extracted from LLVM IR"},
    });
  });
}
