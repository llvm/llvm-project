//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/CASFileSystem.h"

using namespace llvm;
using namespace llvm::cas;

const char CASBackedFile::ID = 0;
const char CASBackedFileSystem::ID = 0;

namespace {
class CASFileWithFixedStatus : public cas::CASBackedFile {
  std::unique_ptr<cas::CASBackedFile> InnerFile;
  vfs::Status S;

public:
  CASFileWithFixedStatus(std::unique_ptr<CASBackedFile> InnerFile,
                         vfs::Status S)
      : InnerFile(std::move(InnerFile)), S(std::move(S)) {}

  ErrorOr<vfs::Status> status() override { return S; }
  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  getBuffer(const Twine &Name, int64_t FileSize, bool RequiresNullTerminator,
            bool IsVolatile) override {
    return InnerFile->getBuffer(Name, FileSize, RequiresNullTerminator,
                                IsVolatile);
  }

  std::error_code close() override { return InnerFile->close(); }

  void setPath(const Twine &Path) override { S = S.copyWithNewName(S, Path); }

  cas::ObjectRef getObjectRefForContent() override {
    return InnerFile->getObjectRefForContent();
  }
};
} // namespace

std::unique_ptr<vfs::File>
CASBackedFile::createWithFixedStatus(std::unique_ptr<vfs::File> Self,
                                     vfs::Status S) {
  return std::make_unique<CASFileWithFixedStatus>(
      cast<CASBackedFile>(std::move(Self)), std::move(S));
}

llvm::Expected<std::pair<std::unique_ptr<llvm::MemoryBuffer>, cas::ObjectRef>>
CASBackedFileSystem::getBufferAndObjectRefForFile(const Twine &Name,
                                                  int64_t FileSize,
                                                  bool RequiresNullTerminator,
                                                  bool IsVolatile) {
  auto F = openCASBackedFileForRead(Name);
  if (!F)
    return F.takeError();

  auto Buf =
      (*F)->getBuffer(Name, FileSize, RequiresNullTerminator, IsVolatile);
  if (!Buf)
    return errorCodeToError(Buf.getError());

  auto CASRef = (*F)->getObjectRefForContent();
  return std::pair{std::move(*Buf), CASRef};
}

llvm::Expected<cas::ObjectRef>
CASBackedFileSystem::getObjectRefForFileContent(const Twine &Name) {
  auto F = openCASBackedFileForRead(Name);
  if (!F)
    return F.takeError();
  return (*F)->getObjectRefForContent();
}
