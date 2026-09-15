//===------------------- AdvisorTypes.h - LLVM Advisor -------------------===//
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

#pragma once

#include "AdvisorCommon.h"

namespace llvm::advisor {

struct CompileCommand {
  std::string Directory;
  std::string File;
  SmallVector<std::string, 16> Arguments;
};

struct SnapshotRecord {
  std::string ID;
  std::string SourceRoot;
  std::string BuildRoot;
  std::string ParentID;
  uint64_t CreatedUnix = 0;
};

struct UnitRecord {
  std::string ID;
  std::string SnapshotID;
  std::string SourcePath;
  std::string Directory;
  std::string ObjectPath;
  std::string IRPath;
  std::string RemarksPath;
  std::string Language;
  std::string TargetTriple;
  std::string ToolchainVersion;
  std::string SourceContentHash;
  std::string CommandFingerprint;
  SmallVector<std::string, 16> Arguments;
};

struct CapabilityRequest {
  std::string SnapshotID;
  std::string UnitID;
  SmallVector<std::string, 8> CapabilityIDs;
};

struct JobRecord {
  enum State { Queued, Running, Succeeded, Failed, Cancelled };

  std::string ID;
  State Current = Queued;
  std::string Message;
};

struct EntityRecord {
  std::string Kind;
  std::string ID;
  std::string SnapshotID;
  std::string UnitID;
  std::string OwnerID;
  json::Object Data;
};

struct HealthStatus {
  bool OK = true;
  std::string Store;
  uint64_t Snapshots = 0;
  uint64_t Units = 0;
};

json::Value toJSON(const CompileCommand &Command);
json::Value toJSON(const SnapshotRecord &Snapshot);
json::Value toJSON(const UnitRecord &Unit);
json::Value toJSON(const JobRecord &Job);
json::Value toJSON(const EntityRecord &Entity);
json::Value toJSON(const HealthStatus &Health);

Expected<SnapshotRecord> snapshotFromJSON(const json::Value &Value);
Expected<UnitRecord> unitFromJSON(const json::Value &Value);
Expected<JobRecord> jobFromJSON(const json::Value &Value);
Expected<EntityRecord> entityFromJSON(const json::Value &Value);

} // namespace llvm::advisor
