//===--- LockFileManager.h - File-level locking utility ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_SUPPORT_LOCKFILEMANAGER_H
#define LLVM_SUPPORT_LOCKFILEMANAGER_H

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/AdvisoryLock.h"
#include <optional>
#include <string>
#include <variant>

namespace llvm {
/// Class that manages the creation of a lock file to aid implicit coordination
/// between different processes.
///
/// The implicit coordination works by creating a ".lock" file, using the
/// atomicity of the file system to ensure that only a single process can create
/// that ".lock" file. When the lock file is removed, the owning process has
/// finished the operation.
class LockFileManager : public AdvisoryLock {
  SmallString<128> FileName;
  SmallString<128> LockFileName;
  SmallString<128> UniqueLockFileName;

  struct OwnerUnknown {};
  struct OwnedByUs {};
  struct OwnedByAnother {
    std::string OwnerHostName;
    int OwnerPID;
  };
  std::variant<OwnerUnknown, OwnedByUs, OwnedByAnother> Owner;

  LockFileManager(const LockFileManager &) = delete;
  LockFileManager &operator=(const LockFileManager &) = delete;

  static std::optional<OwnedByAnother> readLockFile(StringRef LockFileName);

  static bool processStillExecuting(StringRef Hostname, int PID);

public:
  /// Does not try to acquire the lock.
  LockFileManager(StringRef FileName);

  /// Tries to acquire the lock without blocking.
  /// \returns true if the lock was successfully acquired, false if the lock is
  /// already held by someone else, or \c Error in case of unexpected failure.
  Expected<bool> tryLock() override;

  /// For a shared lock, wait until the owner releases the lock.
  ///
  /// \param MaxSeconds the maximum total wait time in seconds.
  WaitForUnlockResult
  waitForUnlockFor(std::chrono::seconds MaxSeconds) override;

  /// Remove the lock file.  This may delete a different lock file than
  /// the one previously read if there is a race.
  std::error_code unsafeMaybeUnlock() override;

  /// Unlocks the lock if previously acquired by \c tryLock().
  ~LockFileManager() override;
};
} // end namespace llvm

#endif // LLVM_SUPPORT_LOCKFILEMANAGER_H
