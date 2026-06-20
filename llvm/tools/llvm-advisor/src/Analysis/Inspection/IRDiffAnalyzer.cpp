//===------------------- IRDiffAnalyzer.cpp - LLVM Advisor ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Inspection/IRDiffAnalyzer.h"
#include "Analysis/IR/IRAnalysisUtils.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
IRDiffAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  return withIRModule(Context, CapID, UnitID,
                      [&](LLVMContext &, Module &M) {
    int64_t Insts = 0;
    int64_t Blocks = 0;
    int64_t Funcs = 0;
    for (const Function &F : M) {
      if (F.isDeclaration())
        continue;
      ++Funcs;
      Blocks += static_cast<int64_t>(F.size());
      Insts += static_cast<int64_t>(F.getInstructionCount());
    }

    return makeJSONResult(CapID, UnitID, json::Object{
        {"mode", "single-input-diff-baseline"},
        {"function_count", Funcs},
        {"basic_block_count", Blocks},
        {"instruction_count", Insts},
    });
  });
}
