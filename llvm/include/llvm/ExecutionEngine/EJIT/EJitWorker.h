//===-- EJitWorker.h - Internal single worker for EJIT taskpool ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITWORKER_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITWORKER_H

#include "llvm/ExecutionEngine/EJIT/EJitAtomic.h"
#include "llvm/ExecutionEngine/EJIT/EJitSreTask.h"

namespace llvm {
namespace ejit {

class EJitTaskPool;

class EJitWorker {
public:
  explicit EJitWorker(EJitTaskPool &pool, const char *name = "ejit-worker")
      : pool_(pool), name_(name) {}
  ~EJitWorker() { stop(); }

  bool start();
  void stop();

  /// True while the worker loop is actually executing (set on loop entry,
  /// cleared on loop exit). Reflects real run state, not merely "start called".
  bool isRunning() const { return running_.loadAcquire() != 0; }
  uint64_t processedCount() const { return processed_.loadRelaxed(); }
  uint64_t spinCount() const { return spins_.loadRelaxed(); }

private:
  static void taskEntry(void *ctx);
  void run();

  EJitTaskPool &pool_;
  const char *name_;
  EJitSreTask task_;
  EJitAtomicU64 processed_{0};
  EJitAtomicU64 spins_{0};
  EJitAtomicU32 running_{0};
  EJitAtomicU32 started_{0};
};

} // namespace ejit
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_EJIT_EJITWORKER_H
