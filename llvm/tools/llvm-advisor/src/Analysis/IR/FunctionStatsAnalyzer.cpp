//===--- FunctionStatsAnalyzer.cpp - LLVM Advisor ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/IR/FunctionStatsAnalyzer.h"
#include "Analysis/IR/IRAnalysisUtils.h"
#include "Utils/Hashing.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
FunctionStatsAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  return withIRModule(Context, CapID, UnitID, [&](LLVMContext &, Module &M) {
    json::Array Functions;
    for (const Function &F : M) {
      if (F.isDeclaration())
        continue;
      json::Object Func;
      Func["name"] = F.getName();
      Func["stable_key"] = hashString(F.getName());
      Func["basic_block_count"] = static_cast<int64_t>(F.size());
      Func["instruction_count"] = static_cast<int64_t>(F.getInstructionCount());
      Func["arg_count"] = static_cast<int64_t>(F.arg_size());
      if (F.hasSection())
        Func["section"] = F.getSection();
      Functions.push_back(std::move(Func));
    }

    return makeJSONResult(CapID, UnitID, json::Object{
        {"module", M.getModuleIdentifier()},
        {"function_count", static_cast<int64_t>(Functions.size())},
        {"functions", std::move(Functions)}});
  });
}
