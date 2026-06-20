//===------------------- StorageManager.h - LLVM Advisor -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of StorageManager in Storage
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"
#include "Storage/BlobStore.h"
#include "Storage/IndexManager.h"
#include "Storage/MetadataStore.h"
#include "Storage/ResultStore.h"
#include "Storage/RetentionManager.h"
#include "Storage/SchemaManager.h"

namespace llvm::advisor {

class StorageManager {
public:
  static Expected<std::unique_ptr<StorageManager>> create(StringRef Root);

  cas::ObjectStore &getCAS() { return *CAS; }
  BlobStore &blobs() { return Blobs; }
  MetadataStore &metadata() { return Metadata; }
  ResultStore &results() { return Results; }
  IndexManager &indexes() { return Indexes; }
  RetentionManager &retention() { return Retention; }
  SchemaManager &schema() { return Schema; }
  StringRef root() const { return Root; }

  Error storeRepresentation(const EntityRecord &Entity);
  Expected<EntityRecord> loadRepresentation(StringRef ID) const;
  Error healthCheck() const;

private:
  StorageManager(std::string Root, std::unique_ptr<cas::ObjectStore> CAS);

  std::string Root;
  std::unique_ptr<cas::ObjectStore> CAS;
  BlobStore Blobs;
  MetadataStore Metadata;
  ResultStore Results;
  IndexManager Indexes;
  RetentionManager Retention;
  SchemaManager Schema;
};

} // namespace llvm::advisor
