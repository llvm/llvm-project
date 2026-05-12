//===--- RemarksMixAnalyzer.cpp - LLVM Advisor ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/IR/RemarksMixAnalyzer.h"
#include "Analysis/RemarksAnalysisUtils.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
RemarksMixAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  return withRemarksFile(
      Context, CapID, UnitID,
      [&](StringRef Path) -> Expected<std::unique_ptr<CapabilityResult>> {
        // Note: this counts frequencies of *all* argument keys across *all*
        // remark types, not strictly asm-printer InstructionMix remarks.
        int64_t TotalArgs = 0;
        json::Object Keys;
        if (Error E = foreachRemark(
                Path, [&](const remarks::Remark &R) -> Error {
                  for (const remarks::Argument &A : R.Args) {
                    ++TotalArgs;
                    // A.Key is a StringRef into the MemoryBuffer; copy to own
                    // the storage before the buffer is freed on return.
                    std::string KeyStr = A.Key.str();
                    json::Value &Val = Keys[KeyStr];
                    Val = Val.getAsInteger().value_or(0) + 1;
                  }
                  return Error::success();
                }))
          return std::move(E);

        return makeJSONResult(CapID, UnitID, json::Object{
            {"remarks_path", Path.str()},
            {"total_args", TotalArgs},
            {"instruction_mix", std::move(Keys)}});
      });
}
