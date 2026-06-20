//===--- SyncAnalysisAnalyzer.cpp - LLVM Advisor -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Offload/SyncAnalysisAnalyzer.h"
#include "Analysis/Utils/SyncClassification.h"
#include "Analysis/Utils/TraceDiscovery.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
SyncAnalysisAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  std::string Path = findTraceJSON(Context.WorkingDirectory);
  if (Path.empty())
    return makeUnavailableResult(CapID, UnitID, "no sync trace JSON found");

  ErrorOr<std::unique_ptr<MemoryBuffer>> MB = MemoryBuffer::getFile(Path);
  if (!MB)
    return createStringError(MB.getError(), "cannot read sync trace: %s",
                             Path.c_str());

  Expected<json::Value> Parsed = json::parse((*MB)->getBuffer());
  if (!Parsed)
    return Parsed.takeError();

  const json::Object *Root = Parsed->getAsObject();
  if (!Root)
    return createStringError(inconvertibleErrorCode(),
                             "trace root is not an object");

  const json::Array *Events = Root->getArray("traceEvents");
  if (!Events)
    return makeUnavailableResult(CapID, UnitID, "traceEvents array missing");

  struct SyncRecord {
    std::string Name;
    std::string Kind;
    int64_t DurationNs = 0;
  };
  SmallVector<SyncRecord, 64> Syncs;
  int64_t TotalSyncNs = 0;

  for (const json::Value &Ev : *Events) {
    const json::Object *E = Ev.getAsObject();
    if (!E)
      continue;
    std::optional<StringRef> Name = E->getString("name");
    if (!Name || !isSyncEvent(*Name))
      continue;
    SyncRecord R;
    R.Name = Name->str();
    R.Kind = classifySyncKind(*Name).str();
    if (std::optional<int64_t> Dur = E->getInteger("dur"))
      R.DurationNs = *Dur * 1000;
    Syncs.push_back(std::move(R));
    TotalSyncNs += R.DurationNs;
  }

  if (Syncs.empty())
    return makeUnavailableResult(CapID, UnitID,
                                 "no synchronization events found in trace");

  // Sort by duration descending to find bottlenecks.
  llvm::sort(Syncs, [](const SyncRecord &A, const SyncRecord &B) {
    return A.DurationNs > B.DurationNs;
  });

  json::Array Bottlenecks;
  for (size_t I = 0, E = std::min<size_t>(Syncs.size(), 20); I < E; ++I) {
    Bottlenecks.push_back(json::Object{
        {"name", Syncs[I].Name},
        {"kind", Syncs[I].Kind},
        {"duration_ns", Syncs[I].DurationNs},
    });
  }

  // Aggregate by kind.
  StringMap<int64_t> CountByKind, TimeByKind;
  for (const SyncRecord &R : Syncs) {
    CountByKind[R.Kind]++;
    TimeByKind[R.Kind] += R.DurationNs;
  }
  json::Object ByKind;
  for (const auto &KV : CountByKind) {
    ByKind[KV.getKey()] = json::Object{
        {"count", KV.second},
        {"total_wait_ns", TimeByKind[KV.getKey()]},
    };
  }

  return makeJSONResult(CapID, UnitID, json::Object{
      {"trace_path", Path},
      {"sync_count", static_cast<int64_t>(Syncs.size())},
      {"total_sync_time_ns", TotalSyncNs},
      {"by_kind", std::move(ByKind)},
      {"bottlenecks", std::move(Bottlenecks)},
  });
}
