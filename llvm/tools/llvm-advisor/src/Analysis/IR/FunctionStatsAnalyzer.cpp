//===------------------- FunctionStatsAnalyzer.cpp - LLVM Advisor
//----------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
// Uses LLVM IR libraries directly - no external process

#include "Analysis/IR/FunctionStatsAnalyzer.h"
#include "Utils/Hashing.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
FunctionStatsAnalyzer::run(const CapabilityContext &Context) {
  if (Context.IRPath.empty())
    return std::make_unique<JSONCapabilityResult>(
        json::Object{{"capability", getCapabilityID()},
                     {"unit_id", Context.Unit.ID},
                     {"available", false},
                     {"reason", "missing IR artifact"}});

  LLVMContext LLVMCtx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseIRFile(Context.IRPath, Err, LLVMCtx);
  if (!M)
    return createStringError(inconvertibleErrorCode(), "cannot parse IR: %s",
                             Context.IRPath.c_str());

  json::Array Functions;
  for (const Function &F : *M) {
    if (F.isDeclaration())
      continue;
    std::string StableKey = hashString(F.getName());
    json::Object Func;
    Func["name"] = F.getName().str();
    Func["stable_key"] = StableKey;
    Func["basic_blocks"] = static_cast<int64_t>(F.size());
    Func["instructions"] = static_cast<int64_t>(F.getInstructionCount());
    Func["arg_count"] = F.arg_size();
    if (F.hasSection())
      Func["section"] = F.getSection().str();
    Functions.push_back(std::move(Func));
  }

  return std::make_unique<JSONCapabilityResult>(
      json::Object{{"capability", getCapabilityID()},
                   {"unit_id", Context.Unit.ID},
                   {"module", M->getModuleIdentifier()},
                   {"function_count", static_cast<int64_t>(Functions.size())},
                   {"functions", std::move(Functions)}});
}
