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
#include <chrono>
#include <optional>

namespace llvm::cas::ondisk {

/// Retrieves an overridden maximum mapping size for CAS files, if any,
/// speicified by LLVM_CAS_MAX_MAPPING_SIZE in the environment or set by
/// `setMaxMappingSize()`. If the value from environment is unreadable, returns
/// an error.
Expected<std::optional<uint64_t>> getOverriddenMaxMappingSize();

/// Set MaxMappingSize for ondisk CAS. This function is not thread-safe and
/// should be set before creaing any ondisk CAS and does not affect CAS already
/// created. Set value 0 to use default size.
LLVM_ABI_FOR_TEST void setMaxMappingSize(uint64_t Size);

/// Thread-safe alternative to \c sys::fs::lockFile. This does not support all
/// the platforms that \c sys::fs::lockFile does, so keep it in the CAS library
/// for now.
std::error_code lockFileThreadSafe(int FD, bool Exclusive = true);

/// Thread-safe alternative to \c sys::fs::unlockFile. This does not support all
/// the platforms that \c sys::fs::lockFile does, so keep it in the CAS library
/// for now.
std::error_code unlockFileThreadSafe(int FD);

/// Thread-safe alternative to \c sys::fs::tryLockFile. This does not support
/// all the platforms that \c sys::fs::lockFile does, so keep it in the CAS
/// library for now.
std::error_code tryLockFileThreadSafe(
    int FD, std::chrono::milliseconds Timeout = std::chrono::milliseconds(0),
    bool Exclusive = true);

/// Allocate space for the file \p FD on disk, if the filesystem supports it.
///
/// On filesystems that support this operation, this ensures errors such as
/// \c std::errc::no_space_on_device are detected before we write data.
///
/// \returns the new size of the file, or an \c Error.
Expected<size_t> preallocateFileTail(int FD, size_t CurrentSize, size_t NewSize);

} // namespace llvm::cas::ondisk

#endif // LLVM_LIB_CAS_ONDISKCOMMON_H
