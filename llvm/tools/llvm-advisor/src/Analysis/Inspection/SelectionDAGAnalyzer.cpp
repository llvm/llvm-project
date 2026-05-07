//===------------------- SelectionDAGAnalyzer.cpp - LLVM Advisor ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Inspection/SelectionDAGAnalyzer.h"
#include "Analysis/IR/IRAnalysisUtils.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
SelectionDAGAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  return withIRModule(Context, CapID, UnitID,
                      [&](LLVMContext &, Module &M) {
    int64_t Functions = 0;
    int64_t Calls = 0;
    int64_t Intrinsics = 0;
    for (const Function &F : M) {
      if (F.isDeclaration())
        continue;
      ++Functions;
      for (const BasicBlock &BB : F) {
        for (const Instruction &I : BB) {
          if (const auto *CB = dyn_cast<CallBase>(&I)) {
            ++Calls;
            if (const Function *Callee = CB->getCalledFunction())
              if (Callee->isIntrinsic())
                ++Intrinsics;
          }
        }
      }
    }

    return makeJSONResult(CapID, UnitID, json::Object{
        {"target_triple", M.getTargetTriple().str()},
        {"module_data_layout", M.getDataLayoutStr()},
        {"function_count", Functions},
        {"call_count", Calls},
        {"intrinsic_call_count", Intrinsics},
        {"note", "SelectionDAG-relevant lowering signals extracted from LLVM IR"},
    });
  });
}
