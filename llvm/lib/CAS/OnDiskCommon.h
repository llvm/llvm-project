//===- OnDiskCommon.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CAS_ONDISKCOMMON_H
#define LLVM_LIB_CAS_ONDISKCOMMON_H

#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include <chrono>
#include <optional>

namespace llvm::cas::ondisk {

/// The version for all the ondisk database files. It needs to be bumped when
/// compatibility breaking changes are introduced.
constexpr StringLiteral CASFormatVersion = "v1";

/// Retrieves an overridden maximum mapping size for CAS files, if any,
/// speicified by LLVM_CAS_MAX_MAPPING_SIZE in the environment or set by
/// `setMaxMappingSize()`. If the value from environment is unreadable, returns
/// an error.
Expected<std::optional<uint64_t>> getOverriddenMaxMappingSize();

/// Set MaxMappingSize for ondisk CAS. This function is not thread-safe and
/// should be set before creaing any ondisk CAS and does not affect CAS already
/// created. Set value 0 to use default size.
LLVM_ABI_FOR_TEST void setMaxMappingSize(uint64_t Size);

/// Whether to use a small file mapping for ondisk databases created in \p Path.
///
/// For some file system that doesn't support sparse file, use a smaller file
/// mapping to avoid consuming too much disk space on creation.
bool useSmallMappingSize(const Twine &Path);

/// Thread-safe alternative to \c sys::fs::lockFile. This does not support all
/// the platforms that \c sys::fs::lockFile does, so keep it in the CAS library
/// for now.
std::error_code lockFileThreadSafe(int FD, llvm::sys::fs::LockKind Kind);

/// Thread-safe alternative to \c sys::fs::unlockFile. This does not support all
/// the platforms that \c sys::fs::lockFile does, so keep it in the CAS library
/// for now.
std::error_code unlockFileThreadSafe(int FD);

/// Thread-safe alternative to \c sys::fs::tryLockFile. This does not support
/// all the platforms that \c sys::fs::lockFile does, so keep it in the CAS
/// library for now.
std::error_code tryLockFileThreadSafe(
    int FD, std::chrono::milliseconds Timeout = std::chrono::milliseconds(0),
    llvm::sys::fs::LockKind Kind = llvm::sys::fs::LockKind::Exclusive);

/// Allocate space for the file \p FD on disk, if the filesystem supports it.
///
/// On filesystems that support this operation, this ensures errors such as
/// \c std::errc::no_space_on_device are detected before we write data.
///
/// \returns the new size of the file, or an \c Error.
Expected<size_t> preallocateFileTail(int FD, size_t CurrentSize,
                                     size_t NewSize);

/// Get boot time for the OS. This can be used to check if the CAS has been
/// validated since boot.
///
/// \returns the boot time in seconds (0 if operation not supported), or an \c
/// Error.
Expected<uint64_t> getBootTime();

/// Helper RAII class for copying a file to a unique file path. At destruction
/// time it will delete any new temporary files created.
class UniqueTempFile {
public:
  UniqueTempFile() = default;
  ~UniqueTempFile();

  /// Create a new unique file path under \p ParentPath and copy the contents
  /// of \p CopyFromPath into it. It will use file cloning when applicable.
  ///
  /// \returns the new unique file path.
  Expected<StringRef> createAndCopyFrom(StringRef ParentPath,
                                        StringRef CopyFromPath);

  /// Rename the new unique file to \p RenameToPath. This is useful to indicate
  /// that the unique file doesn't need to be cleared at destruction time.
  Error renameTo(StringRef RenameToPath);

private:
  SmallString<256> TmpPath;
  SmallString<256> UniqueTmpPath;
};

} // namespace llvm::cas::ondisk

#endif // LLVM_LIB_CAS_ONDISKCOMMON_H
