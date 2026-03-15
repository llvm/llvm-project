//===------------------- StorageManager.cpp - LLVM Advisor ----------------===//
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
#include "Storage/StorageManager.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::advisor;

StorageManager::StorageManager(std::string Root,
                               std::unique_ptr<cas::ObjectStore> CAS)
    : Root(std::move(Root)), CAS(std::move(CAS)), Blobs(*this->CAS),
      Metadata(Blobs, (Twine(this->Root) + "/metadata.root").str()),
      Results(Blobs, (Twine(this->Root) + "/results.root").str()),
      Indexes(Blobs, (Twine(this->Root) + "/index.root").str()),
      Retention(*this->CAS, this->Root),
      Schema(Blobs, (Twine(this->Root) + "/schema.root").str()) {}

Expected<std::unique_ptr<StorageManager>>
StorageManager::create(StringRef Root) {
  if (std::error_code EC = sys::fs::create_directories(Root))
    return createStringError(EC, "cannot create store root '%s'",
                             Root.str().c_str());

  SmallString<256> CASPath(Root);
  sys::path::append(CASPath, "cas");
  Expected<std::unique_ptr<cas::ObjectStore>> CAS =
      cas::createOnDiskCAS(CASPath);
  if (!CAS)
    return CAS.takeError();

  std::unique_ptr<StorageManager> Storage(
      new StorageManager(Root.str(), std::move(*CAS)));
  if (Error Err = Storage->Metadata.load())
    return std::move(Err);
  if (Error Err = Storage->Results.load())
    return std::move(Err);
  if (Error Err = Storage->Indexes.load())
    return std::move(Err);
  if (Error Err = Storage->Schema.validateOnStartup())
    return std::move(Err);
  return Storage;
}

Error StorageManager::storeRepresentation(const EntityRecord &Entity) {
  return Metadata.putEntity(Entity);
}

Expected<EntityRecord> StorageManager::loadRepresentation(StringRef ID) const {
  return Metadata.getEntity(ID);
}

Error StorageManager::healthCheck() const {
  if (std::error_code EC = sys::fs::access(Root, sys::fs::AccessMode::Write))
    return createStringError(EC, "storage root not writable: %s", Root.c_str());

  SmallString<256> CASPath(Root);
  sys::path::append(CASPath, "cas");
  if (!sys::fs::exists(CASPath))
    return createStringError(inconvertibleErrorCode(), "CAS directory missing");

  (void)Metadata.snapshotCount();
  (void)Metadata.unitCount();
  return Error::success();
}
