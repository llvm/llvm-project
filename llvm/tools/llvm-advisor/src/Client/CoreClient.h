//===------------------- CoreClient.h - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared core access used by all clients (CLI, HTTP, LSP).
// Provides unified API for snapshots, units, queries, and results.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"
#include "Capability/CapabilityRegistry.h"
#include "Capability/CapabilitySpec.h"
#include "Capability/PluginRegistry.h"
#include "Core/AdvisorTypes.h"
#include "Storage/StorageManager.h"

namespace llvm::advisor {

struct InspectionFilter {
  std::string Function;
  std::string Pass;
  std::string Severity;
  std::string File;
  int64_t Line = -1;
  int64_t Index = -1;
};

class CoreClient {
public:
  static Expected<std::unique_ptr<CoreClient>> create(StringRef StoreRoot,
                                                      StringRef CapabilityDir);

  /// Load an external plugin (.so/.dylib/.dll) and register its capabilities.
  Error loadPlugin(StringRef Path);

  Expected<SnapshotRecord> createSnapshot(StringRef SourceRoot,
                                          StringRef BuildRoot,
                                          ArrayRef<std::string> Capabilities);
  SmallVector<SnapshotRecord, 16> listSnapshots() const;
  SmallVector<UnitRecord, 64> listUnits(StringRef SnapshotID) const;
  SmallVector<CapabilitySpec, 32> listCapabilities() const;
  Expected<json::Array> queryUnit(StringRef UnitID,
                                  ArrayRef<std::string> CapabilityIDs);
  Expected<json::Array> querySnapshot(StringRef SnapshotID,
                                      ArrayRef<std::string> CapabilityIDs);
  Expected<json::Value> ingestRuntime(StringRef SnapshotID, StringRef Kind,
                                      StringRef Path);
  Expected<json::Value> correlateRuntime(StringRef SnapshotID);
  Expected<json::Array> listInsights(StringRef SnapshotID,
                                     StringRef UnitID = StringRef());
  Expected<json::Object> runInsight(StringRef Name, StringRef SnapshotID,
                                    StringRef UnitID = StringRef(),
                                    StringRef BaselineSnapshotID = StringRef());
  Expected<std::string> resolveUnitID(StringRef SnapshotID,
                                      StringRef Selector) const;
  Expected<json::Object> inspect(StringRef SnapshotID, StringRef UnitSelector,
                                 StringRef CapabilityID,
                                 const InspectionFilter &Filter) const;
  Expected<json::Object> inspectSignals(StringRef SnapshotID,
                                        StringRef UnitSelector,
                                        const InspectionFilter &Filter) const;
  Expected<json::Object> inspectCompare(StringRef BaselineSnapshotID,
                                        StringRef SnapshotID,
                                        StringRef UnitSelector,
                                        StringRef CapabilityID,
                                        const InspectionFilter &Filter) const;
  SmallVector<JobRecord, 16> listJobs() const;
  Expected<JobRecord> getJob(StringRef JobID) const;
  json::Value compare(StringRef Before, StringRef After) const;
  json::Value compareCapability(StringRef Before, StringRef After,
                                StringRef CapID) const;
  HealthStatus health() const;
  json::Value inspectStorage() const;
  Error compactStorage();
  StorageManager &storage() { return *Storage; }

private:
  explicit CoreClient(std::unique_ptr<StorageManager> Storage);

  std::unique_ptr<StorageManager> Storage;
  CapabilityRegistry Registry;
  PluginRegistry Plugins;
};

} // namespace llvm::advisor
