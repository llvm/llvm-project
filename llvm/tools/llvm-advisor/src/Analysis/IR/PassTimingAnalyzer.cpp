//===------------------- PassTimingAnalyzer.cpp - LLVM Advisor =============//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/IR/PassTimingAnalyzer.h"

#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"

#include <chrono>

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
PassTimingAnalyzer::run(const CapabilityContext &Context) {
  if (Context.IRPath.empty()) {
    return std::make_unique<JSONCapabilityResult>(json::Object{
        {"capability", getCapabilityID()},
        {"unit_id", Context.Unit.ID},
        {"available", false},
        {"reason", "missing IR artifact"},
    });
  }

  LLVMContext LLVMCtx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M;
  int64_t ParseTimeUs = 0;
  int64_t AnalysisTimeUs = 0;
  {
    auto Start = std::chrono::steady_clock::now();
    M = parseIRFile(Context.IRPath, Err, LLVMCtx);
    auto End = std::chrono::steady_clock::now();
    ParseTimeUs =
        std::chrono::duration_cast<std::chrono::microseconds>(End - Start)
            .count();
  }
  if (!M)
    return createStringError(inconvertibleErrorCode(), "cannot parse IR: %s",
                             Context.IRPath.c_str());

  int64_t Functions = 0;
  int64_t Instructions = 0;
  {
    auto Start = std::chrono::steady_clock::now();
    for (const Function &F : *M) {
      if (F.isDeclaration())
        continue;
      ++Functions;
      Instructions += static_cast<int64_t>(F.getInstructionCount());
    }
    auto End = std::chrono::steady_clock::now();
    AnalysisTimeUs =
        std::chrono::duration_cast<std::chrono::microseconds>(End - Start)
            .count();
  }

  return std::make_unique<JSONCapabilityResult>(json::Object{
      {"capability", getCapabilityID()},
      {"unit_id", Context.Unit.ID},
      {"parse_time_us", ParseTimeUs},
      {"analysis_time_us", AnalysisTimeUs},
      {"functions", Functions},
      {"instructions", Instructions},
  });
}
