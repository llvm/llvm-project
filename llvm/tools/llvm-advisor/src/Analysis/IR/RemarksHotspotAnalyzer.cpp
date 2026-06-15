//===--- RemarksHotspotAnalyzer.cpp - LLVM Advisor -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/IR/RemarksHotspotAnalyzer.h"
#include "Analysis/RemarksAnalysisUtils.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/JSON.h"

using namespace llvm;
using namespace llvm::advisor;

namespace {

struct HotspotEntry {
  std::string Function;
  std::string File;
  int64_t Line;
  int64_t Count;
  int64_t MaxHotness;
};

class HotspotBuilder {
public:
  void visit(const remarks::Remark &R) {
    std::string Key = (R.FunctionName.empty() ? "<unknown>" : R.FunctionName.str()) + ":" +
                      (R.Loc ? R.Loc->SourceFilePath.str() : "") + ":" +
                      (R.Loc ? std::to_string(R.Loc->SourceLine) : "0");
    
    auto &Entry = Hotspots[Key];
    if (Entry.Function.empty()) {
      if (R.FunctionName.empty()) {
        Entry.Function = "<unknown>";
      } else {
        Entry.Function = demangle(R.FunctionName);
      }
      Entry.File = R.Loc ? R.Loc->SourceFilePath.str() : "";
      Entry.Line = R.Loc ? static_cast<int64_t>(R.Loc->SourceLine) : 0;
      Entry.Count = 0;
      Entry.MaxHotness = -1;
    }
    Entry.Count++;
    if (R.Hotness && static_cast<int64_t>(*R.Hotness) > Entry.MaxHotness)
      Entry.MaxHotness = static_cast<int64_t>(*R.Hotness);
  }

  json::Object render(StringRef Path) {
    std::vector<HotspotEntry> Result;
    for (auto &KV : Hotspots)
      Result.push_back(KV.second);

    // Sort by count descending
    llvm::sort(Result, [](const HotspotEntry &A, const HotspotEntry &B) {
      return A.Count > B.Count;
    });

    json::Array HotspotArray;
    for (const auto &H : Result) {
      json::Object Obj;
      Obj["function"] = H.Function;
      Obj["file"] = H.File;
      Obj["line"] = H.Line;
      Obj["count"] = H.Count;
      Obj["max_hotness"] = H.MaxHotness;
      HotspotArray.push_back(std::move(Obj));
    }

    return json::Object{
        {"available", true},
        {"capability", "llvm.remarks.hotspot"},
        {"hotspots", std::move(HotspotArray)},
        {"count", static_cast<int64_t>(Result.size())},
        {"remarks_path", Path.str()},
    };
  }

private:
  llvm::StringMap<HotspotEntry> Hotspots;
};

} // namespace

Expected<std::unique_ptr<CapabilityResult>>
RemarksHotspotAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  return withRemarksFile(
      Context, CapID, UnitID,
      [&](StringRef Path) -> Expected<std::unique_ptr<CapabilityResult>> {
        HotspotBuilder Builder;
        if (Error E = foreachRemark(
                Path, [&](const remarks::Remark &R) -> Error {
                  Builder.visit(R);
                  return Error::success();
                }))
          return std::move(E);
        return makeJSONResult(CapID, UnitID, Builder.render(Path));
      });
}
