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

#include <chrono>

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
/// acquire the ownership of the lock simultaneously.
/// Typically, threads/processes waiting for the lock to be unlocked will
/// validate that the computation was performed by the expected thread/process
/// and re-run the computation if not.
class AdvisoryLock {
public:
  /// Tries to acquire ownership of the lock without blocking.
  ///
  /// \returns true if ownership of the lock was acquired successfully, false if
  /// the lock is already owned by someone else, or \c Error in case of an
  /// unexpected failure.
  virtual Expected<bool> tryLock() = 0;

  /// For a lock owned by someone else, wait until it is unlocked.
  ///
  /// \param MaxSeconds the maximum total wait time in seconds.
  virtual WaitForUnlockResult
  waitForUnlockFor(std::chrono::seconds MaxSeconds) = 0;

  /// For a lock owned by someone else, unlock it. A permitted side-effect is
  /// that another thread/process may acquire ownership of the lock before the
  /// existing owner unlocks it. This is an unsafe operation.
  virtual std::error_code unsafeMaybeUnlock() = 0;

  /// Unlocks the lock if its ownership was previously acquired by \c tryLock().
  virtual ~AdvisoryLock() = default;
};
} // end namespace llvm

#endif
