//===--- CompareEngine.cpp - LLVM Advisor --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Compare/CompareEngine.h"
#include "Compare/MetricDiff.h"
#include "Compare/RegressionClassifier.h"

#include <cmath>

namespace llvm::advisor {

using CapAddrMap = StringMap<StringMap<std::string>>;

// Extract all top-level numeric fields from a JSON object into a map.
static void extractNumericFields(const json::Object &O,
                                 StringMap<double> &Out) {
  for (const auto &KV : O) {
    if (auto V = KV.getSecond().getAsNumber())
      Out[KV.getFirst()] = *V;
    else if (auto V = KV.getSecond().getAsInteger())
      Out[KV.getFirst()] = static_cast<double>(*V);
  }
}

// Fetch and parse a blob as a JSON object. Returns a null JSON value on error
// so the caller can decide how to report.
static json::Value fetchBlob(StorageManager &Storage,
                             StringRef ContentAddress) {
  Expected<std::string> Blob = Storage.blobs().get(ContentAddress);
  if (!Blob)
    return json::Value(nullptr);
  Expected<json::Value> V = json::parse(*Blob);
  if (!V)
    return json::Value(nullptr);
  return std::move(*V);
}

// Index capability addresses by unit for every entity in a snapshot.
// If FilterCapID is non-empty, only entities with that capability are kept.
static CapAddrMap indexCapabilities(StorageManager &Storage, StringRef SnapID,
                                    StringRef FilterCapID = StringRef()) {
  CapAddrMap Out;
  for (const EntityRecord &E :
       Storage.metadata().listEntities("representation", SnapID)) {
    auto CapID = E.Data.getString("capability_id");
    auto Addr = E.Data.getString("content_address");
    if (!CapID || E.UnitID.empty())
      continue;
    if (!FilterCapID.empty() && *CapID != FilterCapID)
      continue;
    Out[E.UnitID][*CapID] = Addr.value_or("");
  }
  return Out;
}

// Try to match a candidate unit against the base index.
// Returns the matched base unit (or nullptr) and records the match in Seen.
static const UnitRecord *matchUnit(
    const UnitRecord &CandUnit,
    const StringMap<const UnitRecord *> &BaseByID,
    const StringMap<const UnitRecord *> &BaseByPath,
    StringSet<> &Seen) {
  if (auto It = BaseByID.find(CandUnit.ID); It != BaseByID.end()) {
    Seen.insert(CandUnit.ID);
    return It->second;
  }
  if (auto It = BaseByPath.find(CandUnit.SourcePath);
      It != BaseByPath.end()) {
    Seen.insert(It->second->ID);
    return It->second;
  }
  return nullptr;
}

// Diff numeric metrics for two capability blobs (base vs candidate).
static json::Object diffCapabilityBlobs(StorageManager &Storage,
                                        StringRef BaseAddr, StringRef CandAddr,
                                        const RegressionClassifier &Cls) {
  json::Object Result;

  json::Value BaseBlob = fetchBlob(Storage, BaseAddr);
  json::Value CandBlob = fetchBlob(Storage, CandAddr);
  const json::Object *BaseObj = BaseBlob.getAsObject();
  const json::Object *CandObj = CandBlob.getAsObject();
  if (!BaseObj || !CandObj) {
    Result["error"] = "blob fetch or parse failed";
    return Result;
  }

  StringMap<double> BaseFields, CandFields;
  extractNumericFields(*BaseObj, BaseFields);
  extractNumericFields(*CandObj, CandFields);

  StringSet<> AllKeys;
  for (auto &KV : BaseFields)
    AllKeys.insert(KV.getKey());
  for (auto &KV : CandFields)
    AllKeys.insert(KV.getKey());

  json::Array Metrics;
  for (const auto &Key : AllKeys) {
    double Before = BaseFields.lookup(Key.getKey());
    double After = CandFields.lookup(Key.getKey());
    MetricDelta Delta = diffMetric(Before, After);
    StringRef Classification = Cls.classifyWithSeverity(Delta);
    if (Classification == "unchanged")
      continue;
    Metrics.push_back(json::Object{
        {"metric", Key.getKey()},
        {"before", Before},
        {"after", After},
        {"delta", Delta.Delta},
        {"pct_change", Before != 0.0
                           ? std::round(Delta.Delta / Before * 1000.0) / 10.0
                           : 0.0},
        {"classification", Classification},
    });
  }
  Result["changed_metrics"] = std::move(Metrics);
  return Result;
}

json::Value CompareEngine::compare(StringRef Before, StringRef After) const {
  SmallVector<UnitRecord, 64> BaseUnits = Storage.metadata().listUnits(Before);
  SmallVector<UnitRecord, 64> CandUnits = Storage.metadata().listUnits(After);

  StringMap<const UnitRecord *> BaseByID;
  StringMap<const UnitRecord *> BaseByPath;
  for (const UnitRecord &U : BaseUnits) {
    BaseByID[U.ID] = &U;
    BaseByPath[U.SourcePath] = &U;
  }

  CapAddrMap BaseCapAddrByUnit = indexCapabilities(Storage, Before);
  CapAddrMap CandCapAddrByUnit = indexCapabilities(Storage, After);

  RegressionClassifier Cls;
  StringSet<> SeenBase;
  uint64_t Matched = 0, Changed = 0, Added = 0;
  json::Array UnitChanges;

  for (const UnitRecord &CandUnit : CandUnits) {
    const UnitRecord *BaseUnit =
        matchUnit(CandUnit, BaseByID, BaseByPath, SeenBase);
    StringRef MatchType;
    if (BaseUnit && BaseUnit->ID == CandUnit.ID) {
      MatchType = "matched";
      ++Matched;
    } else if (BaseUnit) {
      MatchType = "changed";
      ++Changed;
    } else {
      MatchType = "added";
      ++Added;
    }

    json::Object Entry;
    Entry["match_type"] = MatchType;
    Entry["candidate_unit_id"] = CandUnit.ID;
    Entry["unit_name"] = CandUnit.SourcePath;
    if (BaseUnit)
      Entry["base_unit_id"] = BaseUnit->ID;

    if (BaseUnit && MatchType != "matched") {
      const auto &BaseCapMap = BaseCapAddrByUnit.lookup(BaseUnit->ID);
      const auto &CandCapMap = CandCapAddrByUnit.lookup(CandUnit.ID);

      json::Array CapDiffs;
      StringSet<> AllCaps;
      for (auto &KV : BaseCapMap)
        AllCaps.insert(KV.getKey());
      for (auto &KV : CandCapMap)
        AllCaps.insert(KV.getKey());

      for (const auto &Cap : AllCaps) {
        StringRef CID = Cap.getKey();
        auto BaseIt = BaseCapMap.find(CID);
        auto CandIt = CandCapMap.find(CID);

        if (BaseIt == BaseCapMap.end() || BaseIt->second.empty()) {
          CapDiffs.push_back(
              json::Object{{"capability", CID}, {"status", "added"}});
          continue;
        }
        if (CandIt == CandCapMap.end() || CandIt->second.empty()) {
          CapDiffs.push_back(
              json::Object{{"capability", CID}, {"status", "removed"}});
          continue;
        }

        json::Object Diff =
            diffCapabilityBlobs(Storage, BaseIt->second, CandIt->second, Cls);
        Diff["capability"] = CID;
        Diff["status"] = "diffed";
        CapDiffs.push_back(std::move(Diff));
      }

      if (!CapDiffs.empty())
        Entry["capability_diffs"] = std::move(CapDiffs);
    }

    UnitChanges.push_back(std::move(Entry));
  }

  uint64_t Removed = 0;
  for (const UnitRecord &BaseUnit : BaseUnits) {
    if (SeenBase.contains(BaseUnit.ID))
      continue;
    ++Removed;
    UnitChanges.push_back(json::Object{{"match_type", "removed"},
                                       {"base_unit_id", BaseUnit.ID},
                                       {"unit_name", BaseUnit.SourcePath}});
  }

  return json::Object{
      {"base_snapshot_id", Before},
      {"candidate_snapshot_id", After},
      {"match_summary",
       json::Object{{"matched", static_cast<int64_t>(Matched)},
                    {"changed", static_cast<int64_t>(Changed)},
                    {"added", static_cast<int64_t>(Added)},
                    {"removed", static_cast<int64_t>(Removed)}}},
      {"unit_changes", std::move(UnitChanges)},
  };
}

json::Value CompareEngine::compareCapability(StringRef Before, StringRef After,
                                             StringRef CapID) const {
  SmallVector<UnitRecord, 64> BaseUnits = Storage.metadata().listUnits(Before);
  SmallVector<UnitRecord, 64> CandUnits = Storage.metadata().listUnits(After);

  StringMap<const UnitRecord *> BaseByID;
  StringMap<const UnitRecord *> BaseByPath;
  for (const UnitRecord &U : BaseUnits) {
    BaseByID[U.ID] = &U;
    BaseByPath[U.SourcePath] = &U;
  }

  CapAddrMap BaseCapAddrByUnit = indexCapabilities(Storage, Before, CapID);
  CapAddrMap CandCapAddrByUnit = indexCapabilities(Storage, After, CapID);

  RegressionClassifier Cls;
  json::Array UnitDiffs;
  uint64_t Matched = 0, Added = 0, Removed = 0;
  StringSet<> SeenBase;

  for (const UnitRecord &CandUnit : CandUnits) {
    const UnitRecord *BaseUnit =
        matchUnit(CandUnit, BaseByID, BaseByPath, SeenBase);
    if (!BaseUnit) {
      ++Added;
      continue;
    }
    ++Matched;

    const auto &BaseCapMap = BaseCapAddrByUnit.lookup(BaseUnit->ID);
    const auto &CandCapMap = CandCapAddrByUnit.lookup(CandUnit.ID);
    auto BaseIt = BaseCapMap.find(CapID);
    auto CandIt = CandCapMap.find(CapID);

    json::Object Entry;
    Entry["unit_id"] = CandUnit.ID;
    Entry["unit_name"] = CandUnit.SourcePath;

    if (BaseIt == BaseCapMap.end() || BaseIt->second.empty()) {
      Entry["status"] = "missing_in_base";
    } else if (CandIt == CandCapMap.end() || CandIt->second.empty()) {
      Entry["status"] = "missing_in_candidate";
    } else {
      json::Object Diff =
          diffCapabilityBlobs(Storage, BaseIt->second, CandIt->second, Cls);
      Diff["status"] = "diffed";
      Entry = std::move(Diff);
      Entry["unit_id"] = CandUnit.ID;
      Entry["unit_name"] = CandUnit.SourcePath;
    }

    UnitDiffs.push_back(std::move(Entry));
  }

  for (const UnitRecord &BaseUnit : BaseUnits) {
    if (SeenBase.contains(BaseUnit.ID))
      continue;
    ++Removed;
  }

  return json::Object{
      {"base_snapshot_id", Before},
      {"candidate_snapshot_id", After},
      {"capability", CapID},
      {"match_summary",
       json::Object{{"matched", static_cast<int64_t>(Matched)},
                    {"added", static_cast<int64_t>(Added)},
                    {"removed", static_cast<int64_t>(Removed)}}},
      {"unit_diffs", std::move(UnitDiffs)},
  };
}

} // namespace llvm::advisor
