//===-- EJit.h - EmbeddedJIT Main C++ API ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJIT_H
#define LLVM_EXECUTIONENGINE_EJIT_EJIT_H

#include "llvm/ExecutionEngine/EJIT/EJitCache.h"
#include "llvm/ExecutionEngine/EJIT/EJitError.h"
#include "llvm/ExecutionEngine/EJIT/EJitModuleLoader.h"
#include "llvm/ExecutionEngine/EJIT/EJitOptions.h"
#include "llvm/ExecutionEngine/EJIT/EJitRuntimeState.h"
#include <memory>
#include <string>

namespace llvm {
namespace ejit {

class EJitCompileDriver;
class EJitLogger;

/// Main user-facing class for EmbeddedJIT. Owns all runtime components.
class EJit {
public:
  EJit(const Config &config = {});
  ~EJit();

  // Lifecycle
  void activate(const std::string &periodName, uint8_t cellIdx);
  void deactivate(const std::string &periodName, uint8_t cellIdx);
  void activateAll(const std::string &periodName);
  void deactivateAll(const std::string &periodName);
  bool isActive(const std::string &periodName, uint8_t cellIdx) const;

  // Compilation
  void *getOrCompile(const std::string &funcName,
                     const std::pair<std::string, uint8_t> *dims,
                     unsigned count);

  /// v2: funcIdx-based entry point. Zero string ops on cache hit.
  void *getOrCompile(uint32_t funcIdx,
                     const std::pair<std::string, uint8_t> *dims,
                     unsigned count);

  // Cache management
  void clearCache();
  void invalidateByPeriod(const std::string &periodName, uint8_t cellIdx);
  void invalidateAllByPeriod(const std::string &periodName);

  // Configuration
  void setCompileMode(CompileMode mode);
  CompileMode getCompileMode() const;
  void setOptimizationLevel(OptimizationLevel level);
  OptimizationLevel getOptimizationLevel() const;

  /// Register a user-defined external symbol for JIT resolution.
  /// Required for bare-metal where dlsym is unavailable.
  void registerSymbol(const std::string &name, void *addr);

  /// Manual registration of bitcode / period arrays / static vars.
  /// These can be called after ejit_init() to register data at runtime
  /// (bare-metal friendly, avoiding global constructors).
  void registerBitcode(const std::string &funcName,
                       const uint8_t *data, size_t size);
  void registerPeriodArray(const std::string &periodName,
                           const std::string &varName,
                           void *baseAddr, uint64_t arraySize);
  void registerStaticVar(const std::string &varName, void *varAddr);

  // Registry access (for C API validation)
  const PeriodArrayRegistry &getRegistry() const { return runtimeState_->getRegistry(); }

  // Statistics
  EJitCache::Stats getStats() const;

  // Error
  const EJitError *getLastError() const;

private:
  Config config_;
  std::unique_ptr<EJitRuntimeState> runtimeState_;
  std::unique_ptr<EJitModuleLoader> moduleLoader_;
  std::unique_ptr<EJitCache> cache_;
#ifndef EJIT_FREESTANDING
  std::unique_ptr<EJitLogger> logger_;
#endif
  std::unique_ptr<EJitCompileDriver> compileDriver_;
};

} // namespace ejit
} // namespace llvm

#endif
