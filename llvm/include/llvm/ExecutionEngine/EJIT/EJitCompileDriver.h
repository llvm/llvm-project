//===-- EJitCompileDriver.h - Compilation Scheduler -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITCOMPILEDRIVER_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITCOMPILEDRIVER_H

#include "llvm/ExecutionEngine/EJIT/EJitCache.h"
#include "llvm/ExecutionEngine/EJIT/EJitModuleLoader.h"
#include "llvm/ExecutionEngine/EJIT/EJitOptions.h"
#include "llvm/ExecutionEngine/EJIT/EJitRuntimeState.h"
#ifdef EJIT_SRE_TASKPOOL
#include "llvm/ExecutionEngine/EJIT/EJitTaskPool.h"
#endif
#ifdef EJIT_SRE_SHARED_TASKPOOL
#include "llvm/ExecutionEngine/EJIT/EJitSharedTaskPool.h"
#endif
#include <memory>
#include <string>

namespace llvm {
namespace ejit {

class EJitOrcEngine;
class EJitLogger;

/// Unified entry point for sync and async compilation. Handles cache
/// lookup, time-window state verification, bitcode retrieval, and
/// compilation dispatch.
class EJitCompileDriver {
public:
  struct Result {
    void *funcPtr = nullptr;
    size_t compileTimeMs = 0;
    size_t codeSize = 0;
  };

  EJitCompileDriver(const Config &config, EJitCache &cache,
                    EJitRuntimeState &runtimeState, EJitModuleLoader &loader,
                    EJitLogger *logger = nullptr);

  ~EJitCompileDriver();

  /// Hot path: cache lookup on pre-computed uint64_t cacheKey.
  /// Cold path: decode cacheKey → load bitcode → JIT compile.
  /// Returns nullptr on miss that cannot be compiled (time window not
  /// active, no bitcode, or compile failure).
  void *getOrCompile(uint64_t cacheKey);

#ifdef EJIT_SRE_TASKPOOL
  /// Cold compile path WITHOUT storing into the LRU EJitCache. Used as the
  /// taskpool's compile callback (the taskpool owns its own fixed cache).
  /// Returns the JIT function pointer or nullptr.
  void *compileNow(const EJitCompileRequest &req);

  /// The SRE taskpool scheduler (non-null when EJIT_SRE_TASKPOOL is built).
  EJitTaskPool *taskPool() { return taskPool_.get(); }

  /// Start the taskpool's single async worker. Called by EJit ONLY after all
  /// registration is consumed/frozen and the ORC engine is installed. Returns
  /// false if the worker could not be started.
  bool startTaskPoolWorker() { return taskPool_ && taskPool_->startWorker(); }

  /// Whether the taskpool worker is currently running (test/diagnostic).
  bool isTaskPoolWorkerRunning() const {
    return taskPool_ && taskPool_->isWorkerRunning();
  }

  bool hasSyncEngine() const { return syncEngine_ != nullptr; }

  void stopTaskPoolWorker() {
    if (taskPool_)
      taskPool_->stopWorker();
  }
#endif

#ifdef EJIT_SRE_SHARED_TASKPOOL
  /// The cross-core shared taskpool driving the process-global shared state.
  /// When EJIT_SRE_SHARED_TASKPOOL is built, the taskpool C ABI binds here
  /// instead of the per-instance taskPool_.
  EJitSharedTaskPool *sharedTaskPool() { return &sharedPool_; }
  const EJitSharedTaskPool *sharedTaskPool() const { return &sharedPool_; }

  /// Run owner election over the shared state and, if this core becomes the
  /// owner, start the ONE shared worker. Returns false on a clean init failure
  /// (owner worker-start failed / ABI mismatch). Idempotent across instances.
  bool startSharedTaskPool();

  /// Owner-only orderly shutdown of the shared worker (soft-stop + join).
  void stopSharedTaskPool() { sharedPool_.ownerShutdown(); }
#endif

  EJitCache &getCache() { return cache_; }
  EJitRuntimeState &getRuntimeState() { return runtimeState_; }
  EJitModuleLoader &getLoader() { return loader_; }
  const Config &getConfig() { return config_; }
  EJitOrcEngine *getSyncEngine() { return syncEngine_.get(); }
#ifndef EJIT_FREESTANDING
  EJitLogger *getLogger() { return logger_; }
#else
  EJitLogger *getLogger() { return nullptr; }
#endif

  void setSyncEngine(std::unique_ptr<EJitOrcEngine> engine);
  void registerSymbol(const std::string &name, void *addr);

private:
  const Config &config_;
  EJitCache &cache_;
  EJitRuntimeState &runtimeState_;
  EJitModuleLoader &loader_;
#ifndef EJIT_FREESTANDING
  EJitLogger *logger_;
#endif

  std::unique_ptr<EJitOrcEngine> syncEngine_;
#ifdef EJIT_SRE_TASKPOOL
  std::unique_ptr<EJitTaskPool> taskPool_;
#endif
#ifdef EJIT_SRE_SHARED_TASKPOOL
  EJitSharedTaskPool sharedPool_;
  EJitSreTask sharedWorkerTask_;
  // Worker start/stop adapters bridging the shared pool to the platform task
  // abstraction (host std::thread / SRE platform task). Static so the shared
  // pool can call them through plain function pointers (never std::function).
  static bool sharedWorkerStart(void *ctx,
                                EJitSharedTaskPool::WorkerEntryFn entry,
                                void *entryCtx, uint64_t *outTaskId);
  static void sharedWorkerStop(void *ctx);
  /// Worker idle/yield hook: defers to the platform task abstraction
  /// (EJitSreTask::yield) so the shared worker never busy-spins.
  static void sharedWorkerIdle(void *ctx);
#endif
  // Async compiler will be added in EJitAsyncCompiler phase

  /// Cold compile path (decode → verify active → load bitcode → JIT compile).
  /// When \p storeLru is true the result is inserted into the LRU EJitCache.
  void *compileCold(uint64_t cacheKey, bool storeLru);
};

} // namespace ejit
} // namespace llvm

#endif
