//===--- BitcodeAnalyzer.cpp - LLVM Advisor ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/IR/BitcodeAnalyzer.h"
#include "Analysis/IR/IRAnalysisUtils.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
BitcodeAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  return withBitcodeModule(Context, CapID, UnitID,
                           [&](LLVMContext &, Module &M) {
    return makeJSONResult(CapID, UnitID, json::Object{
        {"module", M.getModuleIdentifier()},
        {"function_count", static_cast<int64_t>(M.size())},
        {"global_count", static_cast<int64_t>(M.global_size())}});
  });
}
