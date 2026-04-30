//===------------------- PassStatsAnalyzer.cpp - LLVM Advisor -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/IR/PassStatsAnalyzer.h"

#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/SourceMgr.h"

using namespace llvm;
using namespace llvm::advisor;

namespace {
class CountPassesListener final : public PassRegistrationListener {
public:
  void passEnumerate(const PassInfo *PI) override {
    if (!PI)
      return;
    ++Count;
  }
  int64_t Count = 0;
};
} // namespace

Expected<std::unique_ptr<CapabilityResult>>
PassStatsAnalyzer::run(const CapabilityContext &Context) {
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
  std::unique_ptr<Module> M = parseIRFile(Context.IRPath, Err, LLVMCtx);
  if (!M)
    return createStringError(inconvertibleErrorCode(), "cannot parse IR: %s",
                             Context.IRPath.c_str());

  int64_t Funcs = 0;
  int64_t Blocks = 0;
  int64_t Insts = 0;
  for (const Function &F : *M) {
    if (F.isDeclaration())
      continue;
    ++Funcs;
    Blocks += static_cast<int64_t>(F.size());
    Insts += static_cast<int64_t>(F.getInstructionCount());
  }

  CountPassesListener Listener;
  Listener.enumeratePasses();

  return std::make_unique<JSONCapabilityResult>(json::Object{
      {"capability", getCapabilityID()},
      {"unit_id", Context.Unit.ID},
      {"registered_passes", Listener.Count},
      {"function_count", Funcs},
      {"basic_block_count", Blocks},
      {"instruction_count", Insts},
  });
}
