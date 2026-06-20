//===--- TemplateStatsAnalyzer.cpp - LLVM Advisor -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Analyzes Clang -ftime-trace output to identify the most expensive template
// instantiations. Groups events by name and sums their durations.
//
//===----------------------------------------------------------------------===//

#include "Analysis/Build/TemplateStatsAnalyzer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include <map>
#include <vector>

namespace llvm::advisor {

static const StringRef kTemplateEvents[] = {
    "InstantiateClass", "InstantiateFunction", "InstantiateVariable",
    "ParseTemplate",    "InstantiateRecord",
};

static bool isTemplateEvent(StringRef Name) {
  for (StringRef E : kTemplateEvents)
    if (Name.starts_with(E))
      return true;
  return false;
}

static std::string findTimeTracePath(const CapabilityContext &Ctx) {
  auto Try = [](const std::string &Base) -> std::string {
    std::string C = Base + ".time-trace";
    if (sys::fs::exists(C))
      return C;
    SmallString<256> Alt(sys::path::parent_path(Base));
    sys::path::append(Alt, sys::path::stem(Base));
    Alt += ".time-trace";
    if (sys::fs::exists(Alt.str()))
      return Alt.str().str();
    return {};
  };
  if (!Ctx.ObjectPath.empty()) {
    std::string P = Try(Ctx.ObjectPath);
    if (!P.empty())
      return P;
  }
  if (!Ctx.SourcePath.empty()) {
    std::string P = Try(Ctx.SourcePath);
    if (!P.empty())
      return P;
  }
  return {};
}

Expected<std::unique_ptr<CapabilityResult>>
TemplateStatsAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  std::string TracePath = findTimeTracePath(Context);
  if (TracePath.empty())
    return makeUnavailableResult(
        CapID, UnitID,
        "no .time-trace file — compile with -ftime-trace");

  ErrorOr<std::unique_ptr<MemoryBuffer>> Buf = MemoryBuffer::getFile(TracePath);
  if (!Buf)
    return createStringError(Buf.getError(), "cannot read %s",
                             TracePath.c_str());

  Expected<json::Value> Parsed = json::parse((*Buf)->getBuffer());
  if (!Parsed)
    return Parsed.takeError();

  const json::Object *Root = Parsed->getAsObject();
  const json::Array *Events = Root ? Root->getArray("traceEvents") : nullptr;
  if (!Events)
    return makeUnavailableResult(CapID, UnitID,
                                 "invalid Chrome tracing format");

  // key: template name, value: {total_us, count}
  std::map<std::string, std::pair<int64_t, int64_t>> Stats;
  int64_t TotalTemplateUs = 0;

  for (const json::Value &EV : *Events) {
    const json::Object *E = EV.getAsObject();
    if (!E)
      continue;
    const json::Value *NameV = E->get("name");
    if (!NameV)
      continue;
    StringRef EvName = NameV->getAsString().value_or("");
    if (!isTemplateEvent(EvName))
      continue;

    // Duration in microseconds (Chrome tracing "dur" field)
    int64_t Dur = 0;
    if (const json::Value *DurV = E->get("dur"))
      Dur = DurV->getAsInteger().value_or(0);

    // Template name is in args.detail
    std::string TplName;
    if (const json::Object *Args = E->getObject("args")) {
      if (const json::Value *Detail = Args->get("detail"))
        TplName = Detail->getAsString().value_or("").str();
    }
    if (TplName.empty())
      TplName = EvName.str();

    auto &Entry = Stats[TplName];
    Entry.first += Dur;
    Entry.second++;
    TotalTemplateUs += Dur;
  }

  // Build sorted list (top 50 by total duration)
  std::vector<std::pair<int64_t, std::pair<std::string, int64_t>>> Sorted;
  Sorted.reserve(Stats.size());
  for (auto &[Name, V] : Stats)
    Sorted.push_back({V.first, {Name, V.second}});
  llvm::sort(Sorted,
             [](const auto &A, const auto &B) { return A.first > B.first; });

  json::Array TopArr;
  for (size_t I = 0, E = std::min<size_t>(Sorted.size(), 50); I < E; ++I) {
    TopArr.push_back(json::Object{
        {"name", Sorted[I].second.first},
        {"total_us", Sorted[I].first},
        {"count", Sorted[I].second.second},
    });
  }

  return makeJSONResult(CapID, UnitID, json::Object{
      {"trace_path", TracePath},
      {"total_templates", static_cast<int64_t>(Stats.size())},
      {"total_template_us", TotalTemplateUs},
      {"top_templates", std::move(TopArr)},
  });
}

} // namespace llvm::advisor
