//===------------------- MetadataStore.cpp - LLVM Advisor ----------------===//
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
#include "Storage/MetadataStore.h"
#include "Utils/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

#include <optional>
#include <system_error>

using namespace llvm;
using namespace llvm::advisor;

static Error writeAnchor(StringRef Path, StringRef ID) {
  std::error_code EC;
  ToolOutputFile Out(Path, EC, sys::fs::OF_Text);
  if (EC)
    return createStringError(EC, "cannot write storage anchor '%s'",
                             Path.str().c_str());
  Out.os() << ID << '\n';
  Out.keep();
  return Error::success();
}

static Expected<std::string> readAnchor(StringRef Path) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer = MemoryBuffer::getFile(Path);
  if (!Buffer) {
    if (Buffer.getError() == std::errc::no_such_file_or_directory)
      return std::string();
    return createStringError(Buffer.getError(),
                             "cannot read storage anchor '%s'",
                             Path.str().c_str());
  }
  return (*Buffer)->getBuffer().trim().str();
}

Error MetadataStore::loadRef(StringRef Kind, StringRef ID, StringRef CASID) {
  Expected<std::string> Data = Blobs.get(CASID);
  if (!Data)
    return Data.takeError();
  Expected<json::Value> Value = json::parse(*Data);
  if (!Value)
    return Value.takeError();

  if (Kind == "snapshots") {
    Expected<SnapshotRecord> Snapshot = snapshotFromJSON(*Value);
    if (!Snapshot)
      return Snapshot.takeError();
    Snapshots[ID] = *Snapshot;
    SnapshotRefs[ID] = CASID.str();
    return Error::success();
  }
  if (Kind == "units") {
    Expected<UnitRecord> Unit = unitFromJSON(*Value);
    if (!Unit)
      return Unit.takeError();
    Units[ID] = *Unit;
    UnitRefs[ID] = CASID.str();
    return Error::success();
  }
  if (Kind == "jobs") {
    Expected<JobRecord> Job = jobFromJSON(*Value);
    if (!Job)
      return Job.takeError();
    Jobs[ID] = *Job;
    JobRefs[ID] = CASID.str();
    return Error::success();
  }
  if (Kind == "entities") {
    Expected<EntityRecord> Entity = entityFromJSON(*Value);
    if (!Entity)
      return Entity.takeError();
    Entities[ID] = *Entity;
    EntityRefs[ID] = CASID.str();
    return Error::success();
  }
  return createStringError(inconvertibleErrorCode(),
                           "unknown metadata kind: %s", Kind.str().c_str());
}

Error MetadataStore::loadRecordMap(const json::Object &Root, StringRef Key) {
  const json::Object *Map = Root.getObject(Key);
  if (!Map)
    return Error::success();

  for (const auto &Entry : *Map) {
    std::optional<StringRef> CASID = Entry.second.getAsString();
    if (!CASID)
      return createStringError(inconvertibleErrorCode(),
                               "metadata ref is not a string");
    if (Error Err = loadRef(Key, Entry.first, *CASID))
      return Err;
  }
  return Error::success();
}

Error MetadataStore::load() {
  Expected<std::string> RootID = readAnchor(AnchorPath);
  if (!RootID)
    return RootID.takeError();
  if (RootID->empty())
    return Error::success();

  Expected<std::string> Data = Blobs.get(*RootID);
  if (!Data)
    return Data.takeError();
  Expected<json::Value> Value = json::parse(*Data);
  if (!Value)
    return Value.takeError();
  const json::Object *Root = Value->getAsObject();
  if (!Root)
    return createStringError(inconvertibleErrorCode(),
                             "metadata root is not an object");

  if (Error Err = loadRecordMap(*Root, "snapshots"))
    return Err;
  if (Error Err = loadRecordMap(*Root, "units"))
    return Err;
  if (Error Err = loadRecordMap(*Root, "jobs"))
    return Err;
  return loadRecordMap(*Root, "entities");
}

json::Object MetadataStore::recordRefs() const {
  json::Object Root;
  json::Object SnapshotMap;
  json::Object UnitMap;
  json::Object JobMap;
  json::Object EntityMap;
  for (const StringMapEntry<std::string> &Entry : SnapshotRefs)
    SnapshotMap[Entry.first()] = Entry.second;
  for (const StringMapEntry<std::string> &Entry : UnitRefs)
    UnitMap[Entry.first()] = Entry.second;
  for (const StringMapEntry<std::string> &Entry : JobRefs)
    JobMap[Entry.first()] = Entry.second;
  for (const StringMapEntry<std::string> &Entry : EntityRefs)
    EntityMap[Entry.first()] = Entry.second;
  Root["snapshots"] = std::move(SnapshotMap);
  Root["units"] = std::move(UnitMap);
  Root["jobs"] = std::move(JobMap);
  Root["entities"] = std::move(EntityMap);
  return Root;
}

Error MetadataStore::flush() {
  Expected<std::string> RootID = storeJSON(recordRefs());
  if (!RootID)
    return RootID.takeError();
  return writeAnchor(AnchorPath, *RootID);
}

Expected<std::string> MetadataStore::storeJSON(const json::Value &Value) {
  return Blobs.put(stringifyJSON(Value));
}

Error MetadataStore::putSnapshot(const SnapshotRecord &Snapshot) {
  Expected<std::string> ID = storeJSON(toJSON(Snapshot));
  if (!ID)
    return ID.takeError();
  Snapshots[Snapshot.ID] = Snapshot;
  SnapshotRefs[Snapshot.ID] = *ID;
  return flush();
}

Error MetadataStore::putUnit(const UnitRecord &Unit) {
  Expected<std::string> ID = storeJSON(toJSON(Unit));
  if (!ID)
    return ID.takeError();
  Units[Unit.ID] = Unit;
  UnitRefs[Unit.ID] = *ID;
  return flush();
}

Error MetadataStore::putJob(const JobRecord &Job) {
  Expected<std::string> ID = storeJSON(toJSON(Job));
  if (!ID)
    return ID.takeError();
  Jobs[Job.ID] = Job;
  JobRefs[Job.ID] = *ID;
  return flush();
}

Error MetadataStore::putEntity(const EntityRecord &Entity) {
  Expected<std::string> ID = storeJSON(toJSON(Entity));
  if (!ID)
    return ID.takeError();
  Entities[Entity.ID] = Entity;
  EntityRefs[Entity.ID] = *ID;
  return flush();
}

Expected<SnapshotRecord> MetadataStore::getSnapshot(StringRef ID) const {
  StringMap<SnapshotRecord>::const_iterator I = Snapshots.find(ID);
  if (I == Snapshots.end())
    return createStringError(inconvertibleErrorCode(), "unknown snapshot: %s",
                             ID.str().c_str());
  return I->second;
}

Expected<UnitRecord> MetadataStore::getUnit(StringRef ID) const {
  StringMap<UnitRecord>::const_iterator I = Units.find(ID);
  if (I == Units.end())
    return createStringError(inconvertibleErrorCode(), "unknown unit: %s",
                             ID.str().c_str());
  return I->second;
}

Expected<JobRecord> MetadataStore::getJob(StringRef ID) const {
  StringMap<JobRecord>::const_iterator I = Jobs.find(ID);
  if (I == Jobs.end())
    return createStringError(inconvertibleErrorCode(), "unknown job: %s",
                             ID.str().c_str());
  return I->second;
}

Expected<EntityRecord> MetadataStore::getEntity(StringRef ID) const {
  StringMap<EntityRecord>::const_iterator I = Entities.find(ID);
  if (I == Entities.end())
    return createStringError(inconvertibleErrorCode(), "unknown entity: %s",
                             ID.str().c_str());
  return I->second;
}

SmallVector<SnapshotRecord, 16> MetadataStore::listSnapshots() const {
  SmallVector<SnapshotRecord, 16> Out;
  for (const StringMapEntry<SnapshotRecord> &Entry : Snapshots)
    Out.push_back(Entry.second);
  return Out;
}

SmallVector<UnitRecord, 64>
MetadataStore::listUnits(StringRef SnapshotID) const {
  SmallVector<UnitRecord, 64> Out;
  for (const StringMapEntry<UnitRecord> &Entry : Units) {
    if (Entry.second.SnapshotID == SnapshotID)
      Out.push_back(Entry.second);
  }
  return Out;
}

SmallVector<JobRecord, 16> MetadataStore::listJobs() const {
  SmallVector<JobRecord, 16> Out;
  for (const StringMapEntry<JobRecord> &Entry : Jobs)
    Out.push_back(Entry.second);
  return Out;
}

SmallVector<EntityRecord, 64>
MetadataStore::listEntities(StringRef Kind, StringRef SnapshotID) const {
  SmallVector<EntityRecord, 64> Out;
  for (const StringMapEntry<EntityRecord> &Entry : Entities) {
    if (!Kind.empty() && Entry.second.Kind != Kind)
      continue;
    if (!SnapshotID.empty() && Entry.second.SnapshotID != SnapshotID)
      continue;
    Out.push_back(Entry.second);
  }
  return Out;
}
