//===------------------- RuntimeCorrelator.cpp - LLVM Advisor --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of RuntimeCorrelator in Runtime
//
//===----------------------------------------------------------------------===//
#include "Runtime/RuntimeCorrelator.h"
#include "Utils/Hashing.h"
#include "Utils/JSON.h"
#include "llvm/Support/Path.h"

#include <optional>

using namespace llvm;
using namespace llvm::advisor;

static bool isRuntimeKind(StringRef Kind) {
  return Kind == "pgo-profile" || Kind == "coverage-mapping" ||
         Kind == "xray-trace" || Kind == "sanitizer-report" ||
         Kind == "sancov-points" || Kind == "offload-trace" ||
         Kind == "memprof-profile";
}

static bool mentionsUnit(const EntityRecord &Entity, const UnitRecord &Unit) {
  // Fast-path: check direct fields without serializing to JSON.
  if (Unit.SourcePath.empty())
    return false;
  StringRef Payload = Entity.Data.getString("path").value_or("");
  if (!Payload.empty() && (Payload == Unit.SourcePath ||
                            Payload.ends_with(sys::path::filename(Unit.SourcePath))))
    return true;
  // Fallback: if the entity carries a raw text payload, search it.
  if (std::optional<StringRef> Raw = Entity.Data.getString("raw"))
    return Raw->contains(Unit.SourcePath) ||
           Raw->contains(sys::path::filename(Unit.SourcePath));
  return false;
}

Expected<json::Value> RuntimeCorrelator::correlate(StorageManager &Storage,
                                                   StringRef SnapshotID) const {
  SmallVector<UnitRecord, 64> Units = Storage.metadata().listUnits(SnapshotID);
  if (Units.empty())
    return createStringError(inconvertibleErrorCode(),
                             "snapshot has no captured units: %s",
                             SnapshotID.data());

  SmallVector<EntityRecord, 64> Entities =
      Storage.metadata().listEntities(StringRef(), SnapshotID);
  json::Array Correlations;

  for (const EntityRecord &Entity : Entities) {
    StringRef PayloadKind;
    if (std::optional<StringRef> Kind = Entity.Data.getString("kind"))
      PayloadKind = *Kind;
    if (!isRuntimeKind(PayloadKind) && Entity.Kind != "finding")
      continue;

    for (const UnitRecord &Unit : Units) {
      if (!Entity.UnitID.empty() && Entity.UnitID != Unit.ID)
        continue;
      if (Entity.UnitID.empty() && !mentionsUnit(Entity, Unit))
        continue;

      SmallString<256> Key;
      Key += Entity.ID;
      Key += '\0';
      Key += Unit.ID;
      std::string MappingID = "map_" + hashString(Key);
      EntityRecord Mapping;
      Mapping.Kind = "mapping";
      Mapping.ID = MappingID;
      Mapping.SnapshotID = SnapshotID;
      Mapping.UnitID = Unit.ID;
      Mapping.OwnerID = Entity.ID;
      Mapping.Data = json::Object{
          {"kind", "source-to-profile"},
          {"mapping_id", MappingID},
          {"runtime_entity_id", Entity.ID},
          {"unit_id", Unit.ID},
          {"source_path", Unit.SourcePath},
          {"mapping_type", Entity.UnitID.empty() ? "approximate" : "exact"},
          {"confidence", Entity.UnitID.empty() ? "medium" : "high"},
          {"bidirectional", false},
          {"discriminator_aware", false}};
      if (Error Err = Storage.metadata().putEntity(Mapping))
        return std::move(Err);
      Correlations.push_back(toJSON(Mapping));
    }
  }

  return json::Object{{"snapshot_id", SnapshotID},
                      {"correlations", std::move(Correlations)}};
}
