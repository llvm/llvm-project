//===-- EJitBareMetal.h - Bare-metal stubs for EJIT -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// When EJIT_FREESTANDING is defined, provides no-op replacements for OS-dependent
// primitives (mutex, shared_mutex) so the EJIT runtime compiles without pthread.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITBAREMETAL_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITBAREMETAL_H

#ifdef EJIT_FREESTANDING

namespace llvm {
namespace ejit {

/// No-op mutex for bare-metal single-threaded operation.
/// Satisfies BasicLockable + Lockable + SharedLockable so it works with
/// std::lock_guard, std::unique_lock, and std::shared_lock.
struct BareMetalMutex {
  void lock() {}
  void unlock() {}
  bool try_lock() { return true; }
  void lock_shared() {}
  void unlock_shared() {}
  bool try_lock_shared() { return true; }
};

} // namespace ejit
} // namespace llvm

#endif // EJIT_FREESTANDING
#endif // LLVM_EXECUTIONENGINE_EJIT_EJITBAREMETAL_H
