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
                    PeriodArrayRegistry &periodReg,
                    EJitRuntimeState &runtimeState,
                    EJitModuleLoader &loader,
                    EJitLogger *logger = nullptr);

  ~EJitCompileDriver();

  /// Get or compile a function for the given specialization.
  /// Returns nullptr if compilation cannot proceed (e.g., time window
  /// not active, no bitcode found, or compile failure).
  void *getOrCompile(const std::string &funcName,
                     const std::pair<std::string, uint8_t> *dims,
                     unsigned count);

  EJitCache &getCache() { return cache_; }
  EJitRuntimeState &getRuntimeState() { return runtimeState_; }

  void setSyncEngine(std::unique_ptr<EJitOrcEngine> engine);

  /// Register a user-defined symbol for JIT resolution (bare-metal).
  void registerSymbol(const std::string &name, void *addr);

private:
  const Config &config_;
  EJitCache &cache_;
  PeriodArrayRegistry &periodReg_;
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
