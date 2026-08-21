//===--- RemarksDetailAnalyzer.cpp - LLVM Advisor ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Inspection/RemarksDetailAnalyzer.h"
#include "Analysis/RemarksAnalysisUtils.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
RemarksDetailAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  return withRemarksFile(
      Context, CapID, UnitID,
      [&](StringRef Path) -> Expected<std::unique_ptr<CapabilityResult>> {
        json::Array Items;
        bool Truncated = false;
        constexpr size_t Limit = 200;
        if (Error E = foreachRemark(
                Path, [&](const remarks::Remark &R) -> Error {
                  if (Items.size() >= Limit) {
                    Truncated = true;
                    return Error::success();
                  }
                  // All StringRef fields point into the MemoryBuffer; copy to
                  // std::string before storing as json::Value because
                  // json::Value(StringRef) stores T_StringRef (non-owning).
                  json::Object Item{
                      {"pass", R.PassName.str()},
                      {"name", R.RemarkName.str()},
                      {"type", remarks::typeToStr(R.RemarkType)},
                      {"function", R.FunctionName.str()},
                      {"message", R.getArgsAsMsg()},
                  };
                  if (R.Loc)
                    Item["location"] =
                        json::Object{{"file", R.Loc->SourceFilePath.str()},
                                     {"line", static_cast<int64_t>(R.Loc->SourceLine)},
                                     {"column", static_cast<int64_t>(R.Loc->SourceColumn)}};
                  if (R.Hotness)
                    Item["hotness"] = static_cast<int64_t>(*R.Hotness);
                  Items.push_back(std::move(Item));
                  return Error::success();
                }))
          return std::move(E);

        return makeJSONResult(CapID, UnitID, json::Object{
            {"remarks_path", Path.str()},
            {"count", static_cast<int64_t>(Items.size())},
            {"truncated", Truncated},
            {"remarks", std::move(Items)}});
      });
}
