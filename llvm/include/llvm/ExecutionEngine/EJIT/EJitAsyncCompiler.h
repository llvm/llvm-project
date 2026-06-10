//===-- EJitAsyncCompiler.h - Asynchronous JIT Compiler -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITASYNCCOMPILER_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITASYNCCOMPILER_H

#ifndef EJIT_FREESTANDING

#include "llvm/ExecutionEngine/EJIT/EJitCache.h"
#include "llvm/ExecutionEngine/EJIT/EJitOptions.h"
#include "llvm/ExecutionEngine/EJIT/EJitOrcEngine.h"
#include "llvm/ExecutionEngine/EJIT/EJitRuntimeState.h"
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <thread>

namespace llvm {
namespace ejit {

class EJitOrcEngine;
class EJitSyncCompiler;

/// Compile request queued for asynchronous compilation.
struct CompileRequest {
  std::string funcName;
  std::string bitcodeData;
  SpecializationContext ctx;
  uint64_t timestamp;
};

/// Asynchronous compiler with a dedicated worker thread and its own
/// LLJIT engine instance for thread isolation.
class EJitAsyncCompiler {
public:
  EJitAsyncCompiler(const Config &config,
                    EJitCache &cache,
                    EJitRuntimeState &runtimeState);
  ~EJitAsyncCompiler();

  void start();
  void stop();

  /// Submit a compile request. Non-blocking; duplicates are skipped.
  void submitRequest(CompileRequest req);

private:
  void workerLoop();
  void compileOne(const CompileRequest &req);

  const Config &config_;
  EJitCache &cache_;
  EJitRuntimeState &runtimeState_;

  std::thread workerThread_;
  std::queue<CompileRequest> requestQueue_;
  std::mutex queueMutex_;
  std::condition_variable queueCV_;
  std::atomic<bool> running_{false};
  std::atomic<bool> stopping_{false};

  std::set<uint64_t> requestsInFlight_;
  std::mutex inFlightMutex_;

  std::unique_ptr<EJitOrcEngine> workerEngine_;
  std::unique_ptr<EJitSyncCompiler> syncCompiler_;
};

} // namespace ejit
} // namespace llvm

#endif // EJIT_FREESTANDING
#endif
