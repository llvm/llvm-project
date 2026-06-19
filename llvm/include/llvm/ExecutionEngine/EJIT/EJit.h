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
class EJitTaskPool;

/// Main user-facing class for EmbeddedJIT. Owns all runtime components.
class EJit {
public:
  EJit(const Config &config = {});
  ~EJit();

  // Lifecycle — period-level (fans out to all arrays under periodName).
  // Returns false only in a taskpool build when periodName is not a registered
  // lifecycle (no state is changed); always true in the legacy build.
  bool activate(const std::string &periodName, uint8_t cellIdx);
  bool deactivate(const std::string &periodName, uint8_t cellIdx);

  // Lifecycle — array-level (single array only). Validates that arrayPtr is a
  // registered period array whose period matches periodName; in a taskpool
  // build it also requires a registered lifecycle and syncs the taskpool
  // SwitchController. Returns false (changing no state) on any mismatch.
  bool activateArray(const std::string &periodName, void *arrayPtr,
                     uint8_t cellIdx);
  bool deactivateArray(const std::string &periodName, void *arrayPtr,
                       uint8_t cellIdx);

  bool activateAll(const std::string &periodName);
  bool deactivateAll(const std::string &periodName);
  bool isActive(const std::string &periodName, uint8_t cellIdx) const;

  // Compilation
  /// Pre-computed cacheKey = funcIdx(32b) | dims(4x8b). The AOT wrapper
  /// computes this in registers; no dims array construction.
  void *getOrCompile(uint64_t cacheKey);

  // Cache management
  void clearCache();
  void invalidateByPeriod(const std::string &periodName, uint8_t cellIdx);
  void invalidateAllByPeriod(const std::string &periodName);

  // Configuration
  /// Change compile mode. In a taskpool build, switching to Async requires a
  /// ready ORC engine and a successfully started worker; on failure the current
  /// mode is preserved and false is returned.
  bool setCompileMode(CompileMode mode);
  CompileMode getCompileMode() const;
  void setOptimizationLevel(OptimizationLevel level);
  OptimizationLevel getOptimizationLevel() const;

  /// Register a user-defined external symbol for JIT resolution.
  /// Required for bare-metal where dlsym is unavailable.
  void registerSymbol(const std::string &name, void *addr);

  /// Manual registration of bitcode / period arrays / static vars.
  /// These can be called after ejit_init() to register data at runtime
  /// (bare-metal friendly, avoiding global constructors).
  /// registerBitcode returns false on a null/zero payload, funcIndex capacity
  /// exhaustion, or a same-name re-registration with a different payload. In a
  /// taskpool build, all three reject once registration is frozen (after
  /// ejit_init) so the running worker never races a registry write.
  bool registerBitcode(const std::string &funcName, const uint8_t *data,
                       size_t size);
  bool registerPeriodArray(const std::string &periodName,
                           const std::string &varName, void *baseAddr,
                           uint64_t arraySize);
  bool registerStaticVar(const std::string &varName, void *varAddr);

  // Registry access (for C API validation)
  const PeriodArrayRegistry &getRegistry() const {
    return runtimeState_->getRegistry();
  }

  // Statistics
  EJitCache::Stats getStats() const;

  // Error
  const EJitError *getLastError() const;

  /// True if registration during construction failed (funcIndex/lifecycle
  /// capacity exhausted, a malformed or conflicting bitcode payload, or a null
  /// fixup pointer). ejit_init() returns failure and tears down the instance
  /// rather than exposing a half-registered taskpool.
  bool initFailed() const { return initFailed_; }
  const EJitError &initError() const { return initError_; }

  /// True once registration is frozen (taskpool build: after ejit_init has
  /// consumed all registration data, completed funcIndex/lifecycle fixup and
  /// started the worker). While frozen, every runtime registration entry point
  /// rejects so the single worker never races a lock-free registry write.
  bool registrationFrozen() const {
    return regPhase_ != RegistrationPhase::Open;
  }

#ifdef EJIT_SRE_TASKPOOL
  /// Access the SRE taskpool scheduler (used by the taskpool C ABI). May be
  /// null if the compile driver was not constructed.
  EJitTaskPool *taskPool();
#endif

private:
  Config config_;
  std::unique_ptr<EJitRuntimeState> runtimeState_;
  std::unique_ptr<EJitModuleLoader> moduleLoader_;
  std::unique_ptr<EJitCache> cache_;
#ifndef EJIT_FREESTANDING
  std::unique_ptr<EJitLogger> logger_;
#endif
  std::unique_ptr<EJitCompileDriver> compileDriver_;

  /// Record the first construction-time registration failure (later ones are
  /// ignored so the earliest root cause is reported).
  void recordInitError(int code, const std::string &message,
                       const std::string &funcName);
  bool initFailed_ = false;
  EJitError initError_;

  /// Registration lifecycle: Open during construction (and forever in a legacy
  /// build), Frozen once a taskpool init completes, Failed on init error.
  enum class RegistrationPhase { Open, Frozen, Failed };
  RegistrationPhase regPhase_ = RegistrationPhase::Open;
};

} // namespace ejit
} // namespace llvm

#endif
