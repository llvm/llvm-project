//===--- CompilationFlow.cpp - LLVM Advisor ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Insights/CompilationFlow.h"
#include "llvm/Support/JSON.h"

using namespace llvm;
using namespace llvm::advisor;

namespace {

// Chrome tracing phase names produced by clang -ftime-trace.
// We map them to logical compilation stages.
StringRef classifyEvent(StringRef Name) {
  if (Name.starts_with("Frontend"))
    return "frontend";
  if (Name.starts_with("Backend") || Name.starts_with("CodeGen") ||
      Name == "EmitObj")
    return "codegen";
  if (Name.starts_with("Optimizer") || Name.starts_with("OptFunction") ||
      Name.starts_with("RunPass") || Name.starts_with("RunLoopPass"))
    return "optimizer";
  if (Name.starts_with("Source") || Name.starts_with("ParseDecl") ||
      Name.starts_with("ParseTemplate") || Name.starts_with("ParseClass") ||
      Name.starts_with("ParseFunctionLiteral"))
    return "parsing";
  if (Name.starts_with("InstantiateClass") ||
      Name.starts_with("InstantiateFunction") ||
      Name.starts_with("InstantiateTemplate"))
    return "instantiation";
  if (Name == "PerformPendingInstantiations")
    return "instantiation";
  if (Name.starts_with("DebugType") || Name.starts_with("DebugFunction") ||
      Name.starts_with("DebugGlobalVariable"))
    return "debug_info";
  return "other";
}

} // namespace

Expected<InsightOutput>
CompilationFlowInsight::analyze(const InsightInput &Input) const {
  const json::Object &D = *Input.PrimaryData;

  const json::Array *Events = D.getArray("traceEvents");
  if (!Events || Events->empty())
    return noDataError();

  // Accumulate duration (microseconds) per stage.
  StringMap<int64_t> StageDuration;

  for (const json::Value &V : *Events) {
    const json::Object *E = V.getAsObject();
    if (!E)
      continue;
    // Only complete events ("X") carry meaningful duration.
    if (E->getString("ph").value_or("") != "X")
      continue;
    StringRef Name = E->getString("name").value_or("");
    int64_t Dur = getInt(*E, "dur");
    if (Dur <= 0 || Name.empty())
      continue;
    StringRef Stage = classifyEvent(Name);
    StageDuration[Stage] += Dur;
  }

  // NOTE: TotalMicroseconds is the sum of classified stage durations. Because
  // trace events may be nested, this can exceed wall-clock time; percentages
  // are therefore relative to the classified total, not absolute time.
  int64_t TotalMicroseconds = 0;
  for (auto &KV : StageDuration)
    TotalMicroseconds += KV.second;

  // Build sorted stage breakdown (descending by duration).
  struct StageEntry {
    std::string Stage;
    int64_t DurationUs;
  };
  SmallVector<StageEntry, 8> Stages;
  for (auto &KV : StageDuration)
    Stages.push_back({KV.getKey().str(), KV.second});
  llvm::sort(Stages, [](const StageEntry &A, const StageEntry &B) {
    return A.DurationUs > B.DurationUs;
  });

  json::Array StageArray;
  for (const StageEntry &S : Stages) {
    double Pct = TotalMicroseconds > 0 ? 100.0 * S.DurationUs / TotalMicroseconds
                                         : 0.0;
    StageArray.push_back(json::Object{
        {"stage", S.Stage},
        {"duration_us", S.DurationUs},
        {"duration_ms", S.DurationUs / 1000},
        {"pct_of_total", roundToOneDecimal(Pct)},
    });
  }

  // Find the single most expensive named event.
  StringRef SlowestName;
  int64_t SlowestDuration = 0;
  for (const json::Value &V : *Events) {
    const json::Object *E = V.getAsObject();
    if (!E || E->getString("ph").value_or("") != "X")
      continue;
    int64_t Dur = getInt(*E, "dur");
    if (Dur > SlowestDuration) {
      SlowestDuration = Dur;
      SlowestName = E->getString("name").value_or("");
    }
  }

  InsightOutput Out;
  Out.Kind = getKind();
  Out.Name = getName().str();
  Out.Data = json::Object{
      {"total_duration_us", TotalMicroseconds},
      {"total_duration_ms", TotalMicroseconds / 1000},
      {"stages", std::move(StageArray)},
      {"slowest_event",
       json::Object{{"name", SlowestName}, {"duration_us", SlowestDuration}}},
  };
  return Out;
}
