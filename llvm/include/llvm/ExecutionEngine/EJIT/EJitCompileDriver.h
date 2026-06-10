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

  EJitCompileDriver(const Config &config,
                    EJitCache &cache,
                    EJitRuntimeState &runtimeState,
                    EJitModuleLoader &loader,
                    EJitLogger *logger = nullptr);

  ~EJitCompileDriver();

  /// Hot path: cache lookup on pre-computed uint64_t cacheKey.
  /// Cold path: decode cacheKey → load bitcode → JIT compile.
  /// Returns nullptr on miss that cannot be compiled (time window not
  /// active, no bitcode, or compile failure).
  void *getOrCompile(uint64_t cacheKey);

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
  // Async compiler will be added in EJitAsyncCompiler phase
};

} // namespace ejit
} // namespace llvm

#endif
