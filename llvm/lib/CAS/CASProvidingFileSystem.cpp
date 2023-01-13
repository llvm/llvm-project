//===- CASProvidingFileSystem.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/CASProvidingFileSystem.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace llvm;
using namespace llvm::cas;

namespace {

class CASProvidingFile final : public vfs::File {
  std::shared_ptr<ObjectStore> DB;
  std::unique_ptr<File> UnderlyingFile;

public:
  CASProvidingFile(std::shared_ptr<ObjectStore> DB,
                   std::unique_ptr<File> UnderlyingFile)
      : DB(std::move(DB)), UnderlyingFile(std::move(UnderlyingFile)) {}

  ErrorOr<vfs::Status> status() override { return UnderlyingFile->status(); }

  ErrorOr<std::unique_ptr<MemoryBuffer>> getBuffer(const Twine &Name,
                                                   int64_t FileSize,
                                                   bool RequiresNullTerminator,
                                                   bool IsVolatile) override {
    return UnderlyingFile->getBuffer(Name, FileSize, RequiresNullTerminator,
                                     IsVolatile);
  }

  ErrorOr<Optional<cas::ObjectRef>> getObjectRefForContent() override {
    auto UnderlyingCASRef = UnderlyingFile->getObjectRefForContent();
    if (!UnderlyingCASRef || *UnderlyingCASRef)
      return UnderlyingCASRef;

    auto Buffer = UnderlyingFile->getBuffer("<contents>", /*FileSize*/ -1,
                                            /*RequiresNullTerminator*/ false);
    if (!Buffer)
      return Buffer.getError();
    auto Blob = DB->storeFromString(None, (*Buffer)->getBuffer());
    if (!Blob)
      return errorToErrorCode(Blob.takeError());
    return *Blob;
  }

  std::error_code close() override { return UnderlyingFile->close(); }
};

class CASProvidingFileSystem : public vfs::ProxyFileSystem {
  std::shared_ptr<ObjectStore> DB;

public:
  CASProvidingFileSystem(std::shared_ptr<ObjectStore> DB,
                         IntrusiveRefCntPtr<vfs::FileSystem> FS)
      : ProxyFileSystem(std::move(FS)), DB(std::move(DB)) {}

  ErrorOr<std::unique_ptr<vfs::File>>
  openFileForRead(const Twine &Path) override {
    auto UnderlyingFile = ProxyFileSystem::openFileForRead(Path);
    if (!UnderlyingFile)
      return UnderlyingFile.getError();
    return std::make_unique<CASProvidingFile>(DB, std::move(*UnderlyingFile));
  }
};

} // namespace

std::unique_ptr<vfs::FileSystem> cas::createCASProvidingFileSystem(
    std::shared_ptr<ObjectStore> DB,
    IntrusiveRefCntPtr<vfs::FileSystem> UnderlyingFS) {
  return std::make_unique<CASProvidingFileSystem>(std::move(DB),
                                                  std::move(UnderlyingFS));
}
