//===- CASFSBuilder.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/CASFSBuilder.h"
#include "llvm/CAS/CachingOnDiskFileSystem.h"

using namespace llvm;
using namespace llvm::cas;

CASFSBuilder::CASFSBuilder(ObjectStore &DB, ArrayRef<MappedPrefix> PrefixMaps)
    : DB(DB), PrefixMaps(PrefixMaps) {}

CASFSBuilder::~CASFSBuilder() {}

static Error recursiveAccess(CachingOnDiskFileSystem &FS, const Twine &Path) {
  std::optional<llvm::cas::CASID> FileID;
  auto ST = FS.statusAndFileID(Path, FileID, /*FollowSymlinks=*/false);
  if (!ST)
    return createFileError(Path, ST.getError());

  if (ST->isDirectory()) {
    std::error_code EC;
    for (vfs::directory_iterator I = FS.dir_begin(Path, EC), IE; !EC && I != IE;
         I.increment(EC)) {
      auto Err = recursiveAccess(FS, I->path());
      if (Err)
        return Err;
    }
  }

  return Error::success();
}

Error CASFSBuilder::ingestFileSystemPath(const Twine &Path) {
  if (!FS) {
    auto FS = createCachingOnDiskFileSystem(DB);
    if (!FS)
      return FS.takeError();
    (*FS)->trackNewAccesses();
    this->FS = std::move(*FS);
  }

  if (Error E = recursiveAccess(*FS, Path))
    return E;
  return Error::success();
}

void CASFSBuilder::mergeCASFSRoot(ObjectRef Root, const Twine &Path) {
  Builder.pushTreeContent(Root, Path);
}

Expected<ObjectProxy> CASFSBuilder::finish() {
  if (FS) {
    auto createTree = [&]() -> Expected<ObjectProxy> {
      if (PrefixMaps.empty())
        return FS->createTreeFromNewAccesses();

      TreePathPrefixMapper Mapper(FS);
      Mapper.addRange(PrefixMaps);
      Mapper.sort();
      return FS->createTreeFromNewAccesses(
          [&](const llvm::vfs::CachedDirectoryEntry &Entry,
              SmallVectorImpl<char> &Storage) {
            return Mapper.mapDirEntry(Entry, Storage);
          });
    };

    auto TreeRef = createTree();
    if (!TreeRef)
      return TreeRef.takeError();
    Builder.pushTreeContent(TreeRef->getRef(), "");
  }

  return Builder.create(DB);
}
