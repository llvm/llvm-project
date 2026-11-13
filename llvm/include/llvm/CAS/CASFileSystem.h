//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Interfaces for files and file systems whose contents are stored in a CAS.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_CASFILESYSTEM_H
#define LLVM_CAS_CASFILESYSTEM_H

#include "llvm/CAS/CASReference.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "llvm/Support/VirtualFileSystem.h"

namespace llvm::cas {

/// An open file whose contents are stored in a CAS.
class LLVM_ABI CASBackedFile : public RTTIExtends<CASBackedFile, vfs::File> {
public:
  static const char ID;
  /// Get the CAS reference for the contents of the file.
  virtual cas::ObjectRef getObjectRefForContent() = 0;

private:
  /// Keep the file CAS-backed when it is given a fixed status, e.g. when it
  /// is read through a \c vfs::RedirectingFileSystem.
  std::unique_ptr<vfs::File>
  createWithFixedStatus(std::unique_ptr<vfs::File> Self,
                        vfs::Status S) override;
};

/// A file system whose files are \c CASBackedFile.
class CASBackedFileSystem
    : public llvm::RTTIExtends<CASBackedFileSystem, vfs::FileSystem> {
public:
  LLVM_ABI static const char ID;

  /// Open a file and return both its contents and the CAS reference for its
  /// contents, so that callers that need both only open the file once.
  LLVM_ABI llvm::Expected<
      std::pair<std::unique_ptr<llvm::MemoryBuffer>, cas::ObjectRef>>
  getBufferAndObjectRefForFile(const Twine &Name, int64_t FileSize = -1,
                               bool RequiresNullTerminator = true,
                               bool IsVolatile = false);

  /// Get the CAS reference for the contents of the file at \p Name.
  LLVM_ABI llvm::Expected<cas::ObjectRef>
  getObjectRefForFileContent(const Twine &Name);

  /// Open a file for reading. The result is always a \c CASBackedFile.
  ErrorOr<std::unique_ptr<vfs::File>>
  openFileForRead(const Twine &Path) override {
    auto F = openCASBackedFileForRead(Path);
    if (!F)
      return errorToErrorCode(F.takeError());
    return std::move(*F);
  }

  /// Open a file for reading as a \c CASBackedFile.
  virtual llvm::Expected<std::unique_ptr<CASBackedFile>>
  openCASBackedFileForRead(const Twine &Path) = 0;
};

} // namespace llvm::cas

#endif // LLVM_CAS_CASFILESYSTEM_H
