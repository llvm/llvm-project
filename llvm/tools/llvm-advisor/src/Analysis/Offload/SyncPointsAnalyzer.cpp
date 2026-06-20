//===--- SyncPointsAnalyzer.cpp - LLVM Advisor ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Offload/SyncPointsAnalyzer.h"
#include "Analysis/Utils/SyncClassification.h"
#include "Analysis/Utils/TraceDiscovery.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

namespace llvm::advisor {

// ---------------------------------------------------------------------------
// Sync event record
// ---------------------------------------------------------------------------

struct SyncEvent {
  std::string Name;
  std::string Kind;
  int64_t DurationNs = 0; // Chrome trace dur is microseconds → convert
  int64_t TID = 0;
};

static void parseTraceForSyncEvents(StringRef Content,
                                    SmallVectorImpl<SyncEvent> &Out) {
  Expected<json::Value> Parsed = json::parse(Content);
  if (!Parsed)
    return;

  const json::Object *Root = Parsed->getAsObject();
  if (!Root)
    return;

  const json::Array *Events = Root->getArray("traceEvents");
  if (!Events)
    return;

  for (const json::Value &Ev : *Events) {
    const json::Object *E = Ev.getAsObject();
    if (!E)
      continue;

    std::optional<StringRef> Name = E->getString("name");
    if (!Name || !isSyncEvent(*Name))
      continue;

    SyncEvent SE;
    SE.Name = Name->str();
    SE.Kind = classifySyncKind(*Name).str();

    if (std::optional<int64_t> Dur = E->getInteger("dur"))
      SE.DurationNs = *Dur * 1000; // microseconds → nanoseconds

    if (std::optional<int64_t> TID = E->getInteger("tid"))
      SE.TID = *TID;

    Out.push_back(std::move(SE));
  }
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

Expected<std::unique_ptr<CapabilityResult>>
SyncPointsAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  std::string TracePath = findTraceJSON(Context.WorkingDirectory);
  if (TracePath.empty())
    return makeUnavailableResult(
        CapID, UnitID,
        "no rocprof trace JSON found (run with rocprof --hip-trace or "
        "--sys-trace)");

  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
      MemoryBuffer::getFile(TracePath);
  if (!BufOrErr)
    return createStringError(BufOrErr.getError(), "cannot read trace: %s",
                             TracePath.c_str());

  SmallVector<SyncEvent, 64> Events;
  parseTraceForSyncEvents((*BufOrErr)->getBuffer(), Events);

  if (Events.empty())
    return makeJSONResult(CapID, UnitID, json::Object{
        {"trace_path", TracePath},
        {"sync_count", 0},
        {"total_wait_ns", 0},
        {"by_kind", json::Object{}},
        {"top_stalls", json::Array{}},
    });

  // Aggregate by kind.
  StringMap<int64_t> CountByKind, TimeByKind;
  int64_t TotalWaitNs = 0;
  for (const SyncEvent &E : Events) {
    CountByKind[E.Kind]++;
    TimeByKind[E.Kind] += E.DurationNs;
    TotalWaitNs += E.DurationNs;
  }

  json::Object ByKind;
  for (const auto &KV : CountByKind)
    ByKind[KV.getKey()] = json::Object{
        {"count", KV.second},
        {"total_wait_ns", TimeByKind[KV.getKey()]},
    };

  // Top stalls: sort by duration descending, take top 20.
  SmallVector<const SyncEvent *, 64> Sorted;
  for (const SyncEvent &E : Events)
    Sorted.push_back(&E);
  llvm::sort(Sorted, [](const SyncEvent *A, const SyncEvent *B) {
    return A->DurationNs > B->DurationNs;
  });

  json::Array TopStalls;
  for (size_t I = 0, E = std::min<size_t>(Sorted.size(), 20); I < E; ++I) {
    const SyncEvent &SE = *Sorted[I];
    TopStalls.push_back(json::Object{
        {"name", SE.Name},
        {"kind", SE.Kind},
        {"duration_ns", SE.DurationNs},
        {"tid", SE.TID},
    });
  }

  return makeJSONResult(CapID, UnitID, json::Object{
      {"trace_path", TracePath},
      {"sync_count", static_cast<int64_t>(Events.size())},
      {"total_wait_ns", TotalWaitNs},
      {"by_kind", std::move(ByKind)},
      {"top_stalls", std::move(TopStalls)},
  });
}

} // namespace llvm::advisor
