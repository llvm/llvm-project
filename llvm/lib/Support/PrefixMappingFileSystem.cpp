//===- PrefixMappingFileSystem.cpp - --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/PrefixMappingFileSystem.h"
#include "llvm/Support/PrefixMapper.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace llvm;
using namespace llvm::vfs;

namespace {

class PrefixMappingFileSystem final : public ProxyFileSystem {
  mutable PrefixMapper Mapper;

public:
  PrefixMappingFileSystem(PrefixMapper Mapper,
                          IntrusiveRefCntPtr<FileSystem> UnderlyingFS)
      : ProxyFileSystem(std::move(UnderlyingFS)), Mapper(std::move(Mapper)) {}

#define PREFIX_MAP_PATH(Old, New)                                              \
  SmallString<256> New;                                                        \
  Old.toVector(New);                                                           \
  Mapper.mapInPlace(New);

  llvm::ErrorOr<Status> status(const Twine &Path) override {
    PREFIX_MAP_PATH(Path, MappedPath)
    return ProxyFileSystem::status(MappedPath);
  }

  llvm::ErrorOr<std::unique_ptr<File>>
  openFileForRead(const Twine &Path) override {
    PREFIX_MAP_PATH(Path, MappedPath)
    return ProxyFileSystem::openFileForRead(MappedPath);
  }

  directory_iterator dir_begin(const Twine &Path,
                               std::error_code &EC) override {
    SmallString<256> MappedPath;
    Path.toVector(MappedPath);
    Mapper.mapInPlace(MappedPath);
    return ProxyFileSystem::dir_begin(MappedPath, EC);
  }

  std::error_code setCurrentWorkingDirectory(const Twine &Path) override {
    PREFIX_MAP_PATH(Path, MappedPath)
    return ProxyFileSystem::setCurrentWorkingDirectory(MappedPath);
  }

  std::error_code getRealPath(const Twine &Path,
                              SmallVectorImpl<char> &Output) const override {
    PREFIX_MAP_PATH(Path, MappedPath)
    return ProxyFileSystem::getRealPath(MappedPath, Output);
  }

  std::error_code isLocal(const Twine &Path, bool &Result) override {
    PREFIX_MAP_PATH(Path, MappedPath)
    return ProxyFileSystem::isLocal(MappedPath, Result);
  }
};

} // namespace

std::unique_ptr<FileSystem> vfs::createPrefixMappingFileSystem(
    PrefixMapper Mapper, IntrusiveRefCntPtr<FileSystem> UnderlyingFS) {
  return std::make_unique<PrefixMappingFileSystem>(std::move(Mapper),
                                                   std::move(UnderlyingFS));
}
