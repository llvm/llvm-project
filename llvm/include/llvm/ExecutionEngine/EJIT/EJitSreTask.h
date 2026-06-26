//===-- EJitSreTask.h - Platform task abstraction for EJIT worker --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITSRETASK_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITSRETASK_H

#include "llvm/ExecutionEngine/EJIT/EJitAtomic.h"

namespace llvm {
namespace ejit {

class EJitSreTask {
public:
  using EntryFn = void (*)(void *);

  EJitSreTask() = default;
  EJitSreTask(const EJitSreTask &) = delete;
  EJitSreTask &operator=(const EJitSreTask &) = delete;

  static bool create(EJitSreTask &out, EntryFn entry, void *ctx,
                     const char *name = nullptr);
  static void destroy(EJitSreTask &task);

  /// Idle hint for the worker loop when the queue is empty. Host builds yield
  /// through std::this_thread::yield; freestanding builds delay for one
  /// scheduler tick through the platform task API.
  static void yield();

  bool stopRequested() const { return stopFlag_.loadAcquire() != 0; }

private:
  friend class EJitWorker;
  void *handle_ = nullptr;
  EntryFn entry_ = nullptr;
  void *ctx_ = nullptr;
  EJitAtomicU32 stopFlag_{0};
};

} // namespace ejit
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_EJIT_EJITSRETASK_H
