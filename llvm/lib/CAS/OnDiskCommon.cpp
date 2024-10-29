//===- OnDiskCommon.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OnDiskCommon.h"
#include <thread>

#if __has_include(<sys/file.h>)
#include <sys/file.h>
#ifdef LOCK_SH
#define HAVE_FLOCK 1
#else
#define HAVE_FLOCK 0
#endif
#endif

using namespace llvm;

std::error_code cas::ondisk::lockFileThreadSafe(int FD, bool Exclusive) {
#if HAVE_FLOCK
  if (flock(FD, Exclusive ? LOCK_EX : LOCK_SH) == 0)
    return std::error_code();
  return std::error_code(errno, std::generic_category());
#elif defined(_WIN32)
  // On Windows this implementation is thread-safe.
  return sys::fs::lockFile(FD, Exclusive);
#else
  return make_error_code(std::errc::no_lock_available);
#endif
}

std::error_code cas::ondisk::unlockFileThreadSafe(int FD) {
#if HAVE_FLOCK
  if (flock(FD, LOCK_UN) == 0)
    return std::error_code();
  return std::error_code(errno, std::generic_category());
#elif defined(_WIN32)
  // On Windows this implementation is thread-safe.
  return sys::fs::unlockFile(FD);
#else
  return make_error_code(std::errc::no_lock_available);
#endif
}

std::error_code
cas::ondisk::tryLockFileThreadSafe(int FD, std::chrono::milliseconds Timeout,
                                   bool Exclusive) {
#if HAVE_FLOCK
  auto Start = std::chrono::steady_clock::now();
  auto End = Start + Timeout;
  do {
    if (flock(FD, (Exclusive ? LOCK_EX : LOCK_SH) | LOCK_NB) == 0)
      return std::error_code();
    int Error = errno;
    if (Error == EWOULDBLOCK) {
      // Match sys::fs::tryLockFile, which sleeps for 1 ms per attempt.
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    return std::error_code(Error, std::generic_category());
  } while (std::chrono::steady_clock::now() < End);
  return make_error_code(std::errc::no_lock_available);
#elif defined(_WIN32)
  // On Windows this implementation is thread-safe.
  return sys::fs::tryLockFile(FD, Timeout, Exclusive);
#else
  return make_error_code(std::errc::no_lock_available);
#endif
}
