//===- OnDiskCommon.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CAS_ONDISKCOMMON_H
#define LLVM_LIB_CAS_ONDISKCOMMON_H

#include <chrono>

namespace llvm::cas::ondisk {

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

} // namespace llvm::cas::ondisk

#endif // LLVM_LIB_CAS_ONDISKCOMMON_H
