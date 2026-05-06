//===--- PassTimingAnalyzer.cpp - LLVM Advisor ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/IR/PassTimingAnalyzer.h"
#include "Analysis/IR/IRAnalysisUtils.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include <chrono>

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
PassTimingAnalyzer::run(const CapabilityContext &Context) {
  // Intentionally not using withIRModule() because we need to measure and
  // record the IR parsing time (parse_time_us) in the result.
  if (Context.IRPath.empty())
    return makeUnavailableResult(getCapabilityID(), Context.Unit.ID,
                                 "missing IR artifact");

  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  LLVMContext LLVMCtx;
  std::unique_ptr<Module> M;
  int64_t ParseTimeUs = 0;
  {
    auto Start = std::chrono::steady_clock::now();
    auto MOrErr = parseIRModule(Context.IRPath, LLVMCtx);
    auto End = std::chrono::steady_clock::now();
    ParseTimeUs =
        std::chrono::duration_cast<std::chrono::microseconds>(End - Start)
            .count();
    if (!MOrErr)
      return MOrErr.takeError();
    M = std::move(*MOrErr);
  }

  int64_t FunctionCount = 0;
  int64_t InstructionCount = 0;
  int64_t AnalysisTimeUs = 0;
  {
    auto Start = std::chrono::steady_clock::now();
    for (const Function &F : *M) {
      if (F.isDeclaration())
        continue;
      ++FunctionCount;
      InstructionCount += static_cast<int64_t>(F.getInstructionCount());
    }
    auto End = std::chrono::steady_clock::now();
    AnalysisTimeUs =
        std::chrono::duration_cast<std::chrono::microseconds>(End - Start)
            .count();
  }

  return makeJSONResult(CapID, UnitID, json::Object{
      {"parse_time_us", ParseTimeUs},
      {"analysis_time_us", AnalysisTimeUs},
      {"function_count", FunctionCount},
      {"instruction_count", InstructionCount}});
}
