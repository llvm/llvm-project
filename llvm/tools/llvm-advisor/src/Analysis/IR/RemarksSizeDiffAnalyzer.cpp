//===--- RemarksSizeDiffAnalyzer.cpp - LLVM Advisor ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/IR/RemarksSizeDiffAnalyzer.h"
#include "Analysis/RemarksAnalysisUtils.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
RemarksSizeDiffAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  return withRemarksFile(
      Context, CapID, UnitID,
      [&](StringRef Path) -> Expected<std::unique_ptr<CapabilityResult>> {
        // Heuristic: look for arguments whose key name contains "size" or
        // "delta" (case-insensitive) and treat their integer values as size
        // deltas. This is intentionally broad to catch varying remark formats.
        int64_t DeltaCount = 0;
        int64_t DeltaSum = 0;
        json::Array Entries;
        if (Error E = foreachRemark(
                Path, [&](const remarks::Remark &R) -> Error {
                  for (const remarks::Argument &A : R.Args) {
                    StringRef Key = A.Key;
                    if (!Key.contains_insensitive("size") &&
                        !Key.contains_insensitive("delta"))
                      continue;
                    std::optional<int64_t> V = A.getValAsInt<int64_t>();
                    if (!V)
                      continue;
                    ++DeltaCount;
                    DeltaSum += *V;
                    // StringRefs from remarks are into the MemoryBuffer;
                    // json::Value(StringRef) stores non-owning T_StringRef.
                    Entries.push_back(json::Object{
                        {"function", R.FunctionName.str()},
                        {"remark", R.RemarkName.str()},
                        {"key", Key.str()},
                        {"value", *V},
                    });
                  }
                  return Error::success();
                }))
          return std::move(E);

        return makeJSONResult(CapID, UnitID, json::Object{
            {"remarks_path", Path.str()},
            {"delta_count", DeltaCount},
            {"delta_sum", DeltaSum},
            {"entries", std::move(Entries)}});
      });
}
