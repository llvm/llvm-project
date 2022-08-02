//===- DialectResourceBlobManager.cpp - Dialect Blob Management -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectResourceBlobManager.h"
#include "llvm/ADT/SmallString.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// DialectResourceBlobManager
//===---------------------------------------------------------------------===//

auto DialectResourceBlobManager::lookup(StringRef name) -> BlobEntry * {
  llvm::sys::SmartScopedReader<true> reader(blobMapLock);

  auto it = blobMap.find(name);
  return it != blobMap.end() ? &it->second : nullptr;
}

void DialectResourceBlobManager::update(StringRef name,
                                        AsmResourceBlob &&newBlob) {
  BlobEntry *entry = lookup(name);
  assert(entry && "`update` expects an existing entry for the provided name");
  entry->setBlob(std::move(newBlob));
}

auto DialectResourceBlobManager::insert(StringRef name,
                                        Optional<AsmResourceBlob> blob)
    -> BlobEntry & {
  llvm::sys::SmartScopedWriter<true> writer(blobMapLock);

  // Functor used to attempt insertion with a given name.
  auto tryInsertion = [&](StringRef name) -> BlobEntry * {
    auto it = blobMap.try_emplace(name, BlobEntry());
    if (it.second) {
      it.first->second.initialize(it.first->getKey(), std::move(blob));
      return &it.first->second;
    }
    return nullptr;
  };

  // Try inserting with the name provided by the user.
  if (BlobEntry *entry = tryInsertion(name))
    return *entry;

  // If an entry already exists for the user provided name, tweak the name and
  // re-attempt insertion until we find one that is unique.
  llvm::SmallString<32> nameStorage(name);
  nameStorage.push_back('_');
  size_t nameCounter = 1;
  do {
    Twine(nameCounter++).toVector(nameStorage);

    // Try inserting with the new name.
    if (BlobEntry *entry = tryInsertion(nameStorage))
      return *entry;
    nameStorage.resize(name.size() + 1);
  } while (true);
}
