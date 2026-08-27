//===---------- LockedAccess.h - Locked access wrapper ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Convenience wrapper for simple locked access to a value.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_LOCKEDACCESS_H
#define ORC_RT_LOCKEDACCESS_H

#include <mutex>
#include <utility>

namespace orc_rt {

/// A convenience wrapper for simple locked access to a value.
///
/// LockedAccess acquires a lock on construction and releases it on
/// destruction, providing pointer-like access to the value in between.
/// All accessors are rvalue-ref-qualified, so LockedAccess can only be used
/// as a temporary — it cannot be stored in a variable or member.
///
/// This is intended for simple, short critical sections where a class wants
/// to return locked access to an internal value. For more complex locking
/// patterns (e.g. lock/unlock/relock, condition variables, multiple locks)
/// use std::unique_lock or std::scoped_lock directly.
template <typename T, typename LockT,
          typename MutexT = typename LockT::mutex_type>
class LockedAccess {
public:
  /// Construct a LockedAccess that references \p R and locks \p M.
  LockedAccess(T &R, MutexT &M) : Lock(M), R(R) {}

  // LockedAccess is not copyable or movable.
  LockedAccess(const LockedAccess &) = delete;
  LockedAccess &operator=(const LockedAccess &) = delete;
  LockedAccess(LockedAccess &&) = delete;
  LockedAccess &operator=(LockedAccess &&) = delete;

  /// Returns a reference to the locked value. The returned reference must not
  /// be used after this LockedAccess temporary is destroyed, as the lock will
  /// no longer be held.
  T &operator*() && noexcept { return R; }
  const T &operator*() const && noexcept { return R; }

  /// Returns a pointer to the locked value for member access. The pointer must
  /// not be used after this LockedAccess temporary is destroyed, as the lock
  /// will no longer be held.
  T *operator->() && noexcept { return &R; }
  const T *operator->() const && noexcept { return &R; }

  /// Calls \p Op with a mutable reference to the locked value, returning
  /// whatever \p Op returns. The lock is held for the duration of the call.
  /// Use this for multi-statement critical sections.
  template <typename OpT>
  decltype(auto)
  with_ref(OpT &&Op) && noexcept(noexcept(std::forward<OpT>(Op)(R))) {
    return std::forward<OpT>(Op)(R);
  }

  /// Calls \p Op with a const reference to the locked value, returning
  /// whatever \p Op returns. The lock is held for the duration of the call.
  template <typename OpT>
  decltype(auto) with_ref(OpT &&Op) const && noexcept(
      noexcept(std::forward<OpT>(Op)(std::as_const(R)))) {
    return std::forward<OpT>(Op)(std::as_const(R));
  }

private:
  LockT Lock;
  T &R;
};

/// Deduction guide: defaults LockT to std::scoped_lock<MutexT>.
template <typename T, typename MutexT>
LockedAccess(T &, MutexT &)
    -> LockedAccess<T, std::scoped_lock<MutexT>, MutexT>;

} // namespace orc_rt

#endif // ORC_RT_LOCKEDACCESS_H
