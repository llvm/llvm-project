//===------------------- CaptureCore.h - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Creates snapshots, captures UnitIDs, and manages baseline capture.
// Coordinates with BuildIntegration and Storage to persist captured data.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"
#include "Capability/CapabilityRegistry.h"
#include "Core/BuildIntegration.h"
#include "Core/UnitIdentity.h"
#include "Storage/StorageManager.h"

namespace llvm::advisor {

class CaptureCore {
public:
  CaptureCore(StorageManager &Storage, CapabilityRegistry &Registry)
      : Storage(Storage), Registry(Registry) {}

  Expected<SnapshotRecord> createSnapshot(StringRef SourceRoot,
                                          StringRef BuildRoot,
                                          ArrayRef<std::string> Capabilities);

private:
  Expected<SnapshotRecord> initializeSnapshot(StringRef SourceRoot,
                                               StringRef BuildRoot);
  Expected<UnitRecord> prepareUnit(const CompileCommand &Command,
                                   const SnapshotRecord &Snapshot);

  StorageManager &Storage;
  CapabilityRegistry &Registry;
  BuildIntegration Builds;
  UnitIdentity Identity;
};

} // namespace llvm::advisor
