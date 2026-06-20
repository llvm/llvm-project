//===------------------- AdvisorTypes.cpp - LLVM Advisor
//-------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Data types for compilation commands, snapshots, and unit records.
// Used throughout the system for representing build entities.
//
//===----------------------------------------------------------------------===//

#include "Core/AdvisorTypes.h"

using namespace llvm;
using namespace llvm::advisor;

static Expected<std::string> readString(const json::Object &Object,
                                        StringRef Key) {
  std::optional<StringRef> Value = Object.getString(Key);
  if (!Value)
    return createStringError(inconvertibleErrorCode(), "missing '%s'",
                             Key.str().c_str());
  return Value->str();
}

static json::Array stringsToJSON(ArrayRef<std::string> Values) {
  json::Array Out;
  for (const std::string &Value : Values)
    Out.push_back(Value);
  return Out;
}

static SmallVector<std::string, 16> stringsFromJSON(const json::Object &Object,
                                                    StringRef Key) {
  SmallVector<std::string, 16> Out;
  const json::Array *Array = Object.getArray(Key);
  if (!Array)
    return Out;
  for (const json::Value &Value : *Array) {
    if (std::optional<StringRef> String = Value.getAsString())
      Out.push_back(String->str());
    // Non-string elements are silently skipped (lenient parsing).
  }
  return Out;
}

static StringRef jobStateToString(JobRecord::State S) {
  switch (S) {
  case JobRecord::Queued:    return "queued";
  case JobRecord::Running:   return "running";
  case JobRecord::Succeeded: return "succeeded";
  case JobRecord::Failed:    return "failed";
  case JobRecord::Cancelled: return "cancelled";
  }
  return "queued"; // Unreachable.
}

static Expected<JobRecord::State> jobStateFromString(StringRef State) {
  if (State == "queued") return JobRecord::Queued;
  if (State == "running") return JobRecord::Running;
  if (State == "succeeded") return JobRecord::Succeeded;
  if (State == "failed") return JobRecord::Failed;
  if (State == "cancelled") return JobRecord::Cancelled;
  return createStringError(inconvertibleErrorCode(), "unknown job state '%s'",
                           State.str().c_str());
}

json::Value llvm::advisor::toJSON(const CompileCommand &Command) {
  return json::Object{{"directory", Command.Directory},
                      {"file", Command.File},
                      {"arguments", stringsToJSON(Command.Arguments)}};
}

json::Value llvm::advisor::toJSON(const SnapshotRecord &Snapshot) {
  return json::Object{
      {"id", Snapshot.ID},
      {"source_root", Snapshot.SourceRoot},
      {"build_root", Snapshot.BuildRoot},
      {"parent_id", Snapshot.ParentID},
      {"created_unix", static_cast<int64_t>(Snapshot.CreatedUnix)}};
}

json::Value llvm::advisor::toJSON(const UnitRecord &Unit) {
  return json::Object{{"id", Unit.ID},
                      {"snapshot_id", Unit.SnapshotID},
                      {"source_path", Unit.SourcePath},
                      {"directory", Unit.Directory},
                      {"object_path", Unit.ObjectPath},
                      {"ir_path", Unit.IRPath},
                      {"remarks_path", Unit.RemarksPath},
                      {"language", Unit.Language},
                      {"target_triple", Unit.TargetTriple},
                      {"toolchain_version", Unit.ToolchainVersion},
                      {"source_content_hash", Unit.SourceContentHash},
                      {"command_fingerprint", Unit.CommandFingerprint},
                      {"arguments", stringsToJSON(Unit.Arguments)}};
}

json::Value llvm::advisor::toJSON(const JobRecord &Job) {
  return json::Object{{"id", Job.ID},
                      {"state", jobStateToString(Job.Current)},
                      {"message", Job.Message}};
}

json::Value llvm::advisor::toJSON(const EntityRecord &Entity) {
  // Entities are polymorphic: the kind is stored both as `entity_kind`
  // (canonical) and `kind` (for backward compatibility with older clients).
  json::Object Object = Entity.Data;
  Object["entity_kind"] = Entity.Kind;
  if (!Object.get("kind"))
    Object["kind"] = Entity.Kind;
  Object["id"] = Entity.ID;
  Object["snapshot_id"] = Entity.SnapshotID;
  Object["unit_id"] = Entity.UnitID;
  Object["owner_id"] = Entity.OwnerID;
  return Object;
}

json::Value llvm::advisor::toJSON(const HealthStatus &Health) {
  return json::Object{{"ok", Health.OK},
                      {"store", Health.Store},
                      {"snapshots", static_cast<int64_t>(Health.Snapshots)},
                      {"units", static_cast<int64_t>(Health.Units)}};
}

Expected<SnapshotRecord>
llvm::advisor::snapshotFromJSON(const json::Value &Value) {
  const json::Object *Object = Value.getAsObject();
  if (!Object)
    return createStringError(inconvertibleErrorCode(),
                             "snapshot is not an object");
  SnapshotRecord Snapshot;
  if (Expected<std::string> ID = readString(*Object, "id"))
    Snapshot.ID = *ID;
  else
    return ID.takeError();
  if (Expected<std::string> SourceRoot = readString(*Object, "source_root"))
    Snapshot.SourceRoot = *SourceRoot;
  else
    return SourceRoot.takeError();
  if (Expected<std::string> BuildRoot = readString(*Object, "build_root"))
    Snapshot.BuildRoot = *BuildRoot;
  else
    return BuildRoot.takeError();
  if (std::optional<StringRef> ParentID = Object->getString("parent_id"))
    Snapshot.ParentID = ParentID->str();
  if (std::optional<int64_t> Created = Object->getInteger("created_unix"))
    Snapshot.CreatedUnix = static_cast<uint64_t>(*Created);
  return Snapshot;
}

Expected<UnitRecord> llvm::advisor::unitFromJSON(const json::Value &Value) {
  const json::Object *Object = Value.getAsObject();
  if (!Object)
    return createStringError(inconvertibleErrorCode(), "unit is not an object");
  UnitRecord Unit;
  if (Expected<std::string> ID = readString(*Object, "id"))
    Unit.ID = *ID;
  else
    return ID.takeError();
  if (Expected<std::string> SnapshotID = readString(*Object, "snapshot_id"))
    Unit.SnapshotID = *SnapshotID;
  else
    return SnapshotID.takeError();
  if (Expected<std::string> SourcePath = readString(*Object, "source_path"))
    Unit.SourcePath = *SourcePath;
  else
    return SourcePath.takeError();
  if (std::optional<StringRef> Directory = Object->getString("directory"))
    Unit.Directory = Directory->str();
  if (std::optional<StringRef> ObjectPath = Object->getString("object_path"))
    Unit.ObjectPath = ObjectPath->str();
  if (std::optional<StringRef> IRPath = Object->getString("ir_path"))
    Unit.IRPath = IRPath->str();
  if (std::optional<StringRef> RemarksPath = Object->getString("remarks_path"))
    Unit.RemarksPath = RemarksPath->str();
  if (std::optional<StringRef> Language = Object->getString("language"))
    Unit.Language = Language->str();
  if (std::optional<StringRef> TargetTriple =
          Object->getString("target_triple"))
    Unit.TargetTriple = TargetTriple->str();
  if (std::optional<StringRef> ToolchainVersion =
          Object->getString("toolchain_version"))
    Unit.ToolchainVersion = ToolchainVersion->str();
  if (std::optional<StringRef> SourceContentHash =
          Object->getString("source_content_hash"))
    Unit.SourceContentHash = SourceContentHash->str();
  if (std::optional<StringRef> CommandFingerprint =
          Object->getString("command_fingerprint"))
    Unit.CommandFingerprint = CommandFingerprint->str();
  Unit.Arguments = stringsFromJSON(*Object, "arguments");
  return Unit;
}

Expected<JobRecord> llvm::advisor::jobFromJSON(const json::Value &Value) {
  const json::Object *Object = Value.getAsObject();
  if (!Object)
    return createStringError(inconvertibleErrorCode(), "job is not an object");
  JobRecord Job;
  if (Expected<std::string> ID = readString(*Object, "id"))
    Job.ID = *ID;
  else
    return ID.takeError();
  if (std::optional<StringRef> Message = Object->getString("message"))
    Job.Message = Message->str();
  if (std::optional<StringRef> State = Object->getString("state")) {
    if (Expected<JobRecord::State> S = jobStateFromString(*State))
      Job.Current = *S;
    else
      return S.takeError();
  }
  return Job;
}

Expected<EntityRecord> llvm::advisor::entityFromJSON(const json::Value &Value) {
  const json::Object *Object = Value.getAsObject();
  if (!Object)
    return createStringError(inconvertibleErrorCode(),
                             "entity is not an object");

  EntityRecord Entity;
  if (std::optional<StringRef> Kind = Object->getString("entity_kind"))
    Entity.Kind = Kind->str();
  else if (Expected<std::string> Kind = readString(*Object, "kind"))
    Entity.Kind = *Kind;
  else
    return Kind.takeError();
  if (Expected<std::string> ID = readString(*Object, "id"))
    Entity.ID = *ID;
  else
    return ID.takeError();
  if (std::optional<StringRef> SnapshotID = Object->getString("snapshot_id"))
    Entity.SnapshotID = SnapshotID->str();
  if (std::optional<StringRef> UnitID = Object->getString("unit_id"))
    Entity.UnitID = UnitID->str();
  if (std::optional<StringRef> OwnerID = Object->getString("owner_id"))
    Entity.OwnerID = OwnerID->str();
  Entity.Data = *Object;
  return Entity;
}
