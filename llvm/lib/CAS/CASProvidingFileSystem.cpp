//===- CASProvidingFileSystem.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/CASFileSystem.h"
#include "llvm/CAS/CASProvidingFileSystem.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::vfs;

namespace {

class CASProvidingFile final : public CASBackedFile {
  std::unique_ptr<File> UnderlyingFile;
  ObjectRef Ref;

public:
  CASProvidingFile(ObjectRef Ref, std::unique_ptr<File> UnderlyingFile)
      : UnderlyingFile(std::move(UnderlyingFile)), Ref(Ref) {}

  ErrorOr<vfs::Status> status() final { return UnderlyingFile->status(); }

  ErrorOr<std::unique_ptr<MemoryBuffer>> getBuffer(const Twine &Name,
                                                   int64_t FileSize,
                                                   bool RequiresNullTerminator,
                                                   bool IsVolatile) final {
    return UnderlyingFile->getBuffer(Name, FileSize, RequiresNullTerminator,
                                     IsVolatile);
  }

  cas::ObjectRef getObjectRefForContent() final { return Ref; }

  std::error_code close() final { return UnderlyingFile->close(); }
};

class CASProvidingFileSystem final : public CASBackedFileSystem {
  std::shared_ptr<ObjectStore> DB;
  IntrusiveRefCntPtr<vfs::FileSystem> FS;

public:
  CASProvidingFileSystem(std::shared_ptr<ObjectStore> DB,
                         IntrusiveRefCntPtr<vfs::FileSystem> FS)
      : DB(std::move(DB)), FS(std::move(FS)) {}

  llvm::ErrorOr<Status> status(const Twine &Path) final {
    return FS->status(Path);
  }
  bool exists(const Twine &Path) final { return FS->exists(Path); }
  directory_iterator dir_begin(const Twine &Dir, std::error_code &EC) final {
    return FS->dir_begin(Dir, EC);
  }
  llvm::ErrorOr<std::string> getCurrentWorkingDirectory() const final {
    return FS->getCurrentWorkingDirectory();
  }
  std::error_code setCurrentWorkingDirectory(const Twine &Path) final {
    return FS->setCurrentWorkingDirectory(Path);
  }
  std::error_code getRealPath(const Twine &Path,
                              SmallVectorImpl<char> &Output) final {
    return FS->getRealPath(Path, Output);
  }
  std::error_code isLocal(const Twine &Path, bool &Result) final {
    return FS->isLocal(Path, Result);
  }

  llvm::Expected<std::unique_ptr<CASBackedFile>>
  openCASBackedFileForRead(const Twine &Path) final {
    auto F = FS->openFileForRead(Path);
    if (!F)
      return errorCodeToError(F.getError());

    auto Buffer = (*F)->getBuffer("<contents>", /*FileSize*/ -1,
                                  /*RequiresNullTerminator*/ false);
    if (!Buffer)
      return errorCodeToError(Buffer.getError());

    auto Blob = DB->storeFromString({}, (*Buffer)->getBuffer());
    if (!Blob)
      return Blob.takeError();

    return std::make_unique<CASProvidingFile>(*Blob, std::move(*F));
  }

  IntrusiveRefCntPtr<CASBackedFileSystem> createThreadSafeProxyFS() final {
    return makeIntrusiveRefCnt<CASProvidingFileSystem>(DB, FS);
  }
};
} // namespace


std::unique_ptr<vfs::FileSystem> cas::createCASProvidingFileSystem(
    std::shared_ptr<ObjectStore> DB,
    IntrusiveRefCntPtr<vfs::FileSystem> UnderlyingFS) {
  return std::make_unique<CASProvidingFileSystem>(std::move(DB),
                                                  std::move(UnderlyingFS));
}
