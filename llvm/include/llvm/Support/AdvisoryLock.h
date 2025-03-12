//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ADVISORYLOCK_H
#define LLVM_SUPPORT_ADVISORYLOCK_H

#include "llvm/Support/Error.h"

namespace llvm {
/// Describes the result of waiting for the owner to release the lock.
enum class WaitForUnlockResult {
  /// The lock was released successfully.
  Success,
  /// Owner died while holding the lock.
  OwnerDied,
  /// Reached timeout while waiting for the owner to release the lock.
  Timeout,
};

/// A synchronization primitive with weak mutual exclusion guarantees.
/// Implementations of this interface may allow multiple threads/processes to
/// acquire the lock simultaneously. Typically, threads/processes waiting for
/// the lock to be unlocked will validate the computation produced valid result.
class AdvisoryLock {
public:
  /// Tries to acquire the lock without blocking.
  ///
  /// \returns true if the lock was successfully acquired (owned lock), false if
  /// the lock is already held by someone else (shared lock), or \c Error in
  /// case of unexpected failure.
  virtual Expected<bool> tryLock() = 0;

  /// For a shared lock, wait until the owner releases the lock.
  ///
  /// \param MaxSeconds the maximum total wait time in seconds.
  virtual WaitForUnlockResult waitForUnlockFor(unsigned MaxSeconds) = 0;
  WaitForUnlockResult waitForUnlock() { return waitForUnlockFor(90); }

  /// Unlocks a shared lock. This may allow another thread/process to acquire
  /// the lock before the existing owner released it and notify waiting
  /// threads/processes. This is an unsafe operation.
  virtual std::error_code unsafeUnlockShared() = 0;

  /// Unlocks the lock if previously acquired by \c tryLock().
  virtual ~AdvisoryLock() = default;
};
} // end namespace llvm

#endif
