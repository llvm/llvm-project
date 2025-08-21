//===- llvm/CAS/CASFileSystem.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_CASFILESYSTEM_H
#define LLVM_CAS_CASFILESYSTEM_H

#include "llvm/Support/Error.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "llvm/Support/VirtualFileSystem.h"

namespace llvm::cas {
class ObjectStore;
class CASID;

/// Abstract class represents an open file backed by a CAS.
class CASBackedFile : public RTTIExtends<CASBackedFile, vfs::File> {
public:
  static const char ID;
  /// Get the CAS reference for the contents of the file.
  virtual cas::ObjectRef getObjectRefForContent() = 0;
};

/// Abstract class represents a CAS backed file system.
class CASBackedFileSystem
    : public llvm::RTTIExtends<CASBackedFileSystem, vfs::FileSystem> {
public:
  static const char ID;

  /// This is a convenience method that opens a file, gets its content and then
  /// closes the file. It returns MemoryBuffer and ObjectRef in one call to avoid
  /// open the file twice.
  /// The IsText parameter is used to distinguish whether the file should be
  /// opened as a binary or text file.
  llvm::Expected<std::pair<std::unique_ptr<llvm::MemoryBuffer>, cas::ObjectRef>>
  getBufferAndObjectRefForFile(const Twine &Name, int64_t FileSize = -1,
                               bool RequiresNullTerminator = true,
                               bool IsVolatile = false, bool IsText = true);

  /// Get ObjectRef of a file from its path.
  llvm::Expected<cas::ObjectRef> getObjectRefForFileContent(const Twine &Name);

  /// Implementation for openFileForRead using CASBackedFile.
  ErrorOr<std::unique_ptr<vfs::File>>
  openFileForRead(const Twine &Path) override {
    auto F = openCASBackedFileForRead(Path);
    if (!F)
      return errorToErrorCode(F.takeError());
    return std::move(*F);
  }

  /// Get CASBackedFile for read.
  virtual llvm::Expected<std::unique_ptr<CASBackedFile>>
  openCASBackedFileForRead(const Twine &Path) = 0;

  /// Get a proxy FS that has an independent working directory.
  virtual IntrusiveRefCntPtr<CASBackedFileSystem>
  createThreadSafeProxyFS() = 0;
};

Expected<std::unique_ptr<vfs::FileSystem>>
createCASFileSystem(std::shared_ptr<ObjectStore> DB, const CASID &RootID,
                    sys::path::Style PathStyle = sys::path::Style::native);

Expected<std::unique_ptr<vfs::FileSystem>>
createCASFileSystem(ObjectStore &DB, const CASID &RootID,
                    sys::path::Style PathStyle = sys::path::Style::native);

} // namespace llvm::cas

#endif // LLVM_CAS_CASFILESYSTEM_H
