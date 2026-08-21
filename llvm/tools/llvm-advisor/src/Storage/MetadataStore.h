//===------------------- MetadataStore.h - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of MetadataStore in Storage
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"
#include "Core/AdvisorTypes.h"
#include "Storage/BlobStore.h"

namespace llvm::advisor {

class MetadataStore {
public:
  MetadataStore(BlobStore &Blobs, std::string AnchorPath)
      : Blobs(Blobs), AnchorPath(std::move(AnchorPath)) {}

  Error load();
  Error flush();

  Expected<std::string> storeJSON(const json::Value &Value);

  Error putSnapshot(const SnapshotRecord &Snapshot);
  Error putUnit(const UnitRecord &Unit);
  Error putJob(const JobRecord &Job);
  Error putEntity(const EntityRecord &Entity);

  Expected<SnapshotRecord> getSnapshot(StringRef ID) const;
  Expected<UnitRecord> getUnit(StringRef ID) const;
  Expected<JobRecord> getJob(StringRef ID) const;
  Expected<EntityRecord> getEntity(StringRef ID) const;

  SmallVector<SnapshotRecord, 16> listSnapshots() const;
  SmallVector<UnitRecord, 64> listUnits(StringRef SnapshotID) const;
  SmallVector<JobRecord, 16> listJobs() const;
  SmallVector<EntityRecord, 64> listEntities(StringRef Kind,
                                             StringRef SnapshotID = {}) const;
  uint64_t snapshotCount() const { return Snapshots.size(); }
  uint64_t unitCount() const { return Units.size(); }

private:
  Error loadRecordMap(const json::Object &Root, StringRef Key);
  json::Object recordRefs() const;
  Error loadRef(StringRef Kind, StringRef ID, StringRef CASID);

  BlobStore &Blobs;
  std::string AnchorPath;
  StringMap<SnapshotRecord> Snapshots;
  StringMap<UnitRecord> Units;
  StringMap<JobRecord> Jobs;
  StringMap<EntityRecord> Entities;
  StringMap<std::string> SnapshotRefs;
  StringMap<std::string> UnitRefs;
  StringMap<std::string> JobRefs;
  StringMap<std::string> EntityRefs;
};

} // namespace llvm::advisor
