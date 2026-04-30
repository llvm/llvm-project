//===--- IRAnalyzer.cpp - LLVM Advisor -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/IR/IRAnalyzer.h"
#include "Analysis/IR/IRAnalysisUtils.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
IRAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  return withIRModule(Context, CapID, UnitID,
                      [&](LLVMContext &, Module &M) {
    int64_t FunctionCount = 0;
    int64_t InstructionCount = 0;
    for (const Function &F : M) {
      if (F.isDeclaration())
        continue;
      ++FunctionCount;
      InstructionCount += static_cast<int64_t>(F.getInstructionCount());
    }

    return makeJSONResult(CapID, UnitID, json::Object{
        {"module", M.getModuleIdentifier()},
        {"function_count", FunctionCount},
        {"instruction_count", InstructionCount},
        {"global_count", static_cast<int64_t>(M.global_size())}});
  });
}
