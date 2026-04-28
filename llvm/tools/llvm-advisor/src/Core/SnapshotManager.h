//===------------------- SnapshotManager.h - LLVM Advisor
//-------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Snapshot lifecycle: create, list, delete, and tier management.
// Handles snapshot retention and cleanup policies.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"
#include "Core/AdvisorTypes.h"
#include "Storage/StorageManager.h"

namespace llvm::advisor {

class SnapshotManager {
public:
  explicit SnapshotManager(StorageManager &Storage) : Storage(Storage) {}

  SmallVector<SnapshotRecord, 16> list() const {
    return Storage.metadata().listSnapshots();
  }

private:
  StorageManager &Storage;
};

} // namespace llvm::advisor
