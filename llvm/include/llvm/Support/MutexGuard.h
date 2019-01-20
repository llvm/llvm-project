//===-- Support/MutexGuard.h - Acquire/Release Mutex In Scope ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a guard for a block of code that ensures a Mutex is locked
// upon construction and released upon destruction.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_MUTEXGUARD_H
#define LLVM_SUPPORT_MUTEXGUARD_H

#include "llvm/Support/Mutex.h"

namespace llvm {
  /// Instances of this class acquire a given Mutex Lock when constructed and
  /// hold that lock until destruction. The intention is to instantiate one of
  /// these on the stack at the top of some scope to be assured that C++
  /// destruction of the object will always release the Mutex and thus avoid
  /// a host of nasty multi-threading problems in the face of exceptions, etc.
  /// Guard a section of code with a Mutex.
  class MutexGuard {
    sys::Mutex &M;
    MutexGuard(const MutexGuard &) = delete;
    void operator=(const MutexGuard &) = delete;
  public:
    MutexGuard(sys::Mutex &m) : M(m) { M.lock(); }
    ~MutexGuard() { M.unlock(); }
    /// holds - Returns true if this locker instance holds the specified lock.
    /// This is mostly used in assertions to validate that the correct mutex
    /// is held.
    bool holds(const sys::Mutex& lock) const { return &M == &lock; }
  };
}

#endif // LLVM_SUPPORT_MUTEXGUARD_H
