//===-- EJitOrcEngine.h - OrcJIT Engine Wrapper ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITORCENGINE_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITORCENGINE_H

#include "llvm/ExecutionEngine/EJIT/EJitOptions.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/Support/Error.h"
#include <memory>
#include <string>

namespace llvm {
namespace ejit {

class PeriodArrayRegistry;
class EJitRuntimeState;

struct SpecializationContext {
  std::string fnName;
  uint64_t cacheKey = 0;
  struct DimInfo {
    std::string periodName;
    uint8_t cellIdx;
  };
  SmallVector<DimInfo, 4> dimensions;
  OptimizationLevel optLevel = OptimizationLevel::L2;
};

/// Wraps an LLJIT instance with EmbeddedJIT-specific configuration:
/// custom memory manager and IR transform layer for the JIT pipeline.
class EJitOrcEngine {
public:
  EJitOrcEngine();
  ~EJitOrcEngine();

  /// Create a configured engine. Thread-safe to call once.
  static Expected<std::unique_ptr<EJitOrcEngine>>
  Create(const Config &config,
         PeriodArrayRegistry &periodReg,
         EJitRuntimeState &runtimeState);

  /// Load a bitcode module into a per-specialization JITDylib identified
  /// by cacheKey. Each specialization gets its own JITDylib so symbols
  /// from the same TU bitcode can be defined multiple times without conflict.
  Error loadBitcodeModule(StringRef bitcodeData,
                          uint64_t cacheKey,
                          const std::string &origFnName);

  /// Look up a compiled function symbol in the specialization JITDylib
  /// identified by cacheKey.
  Expected<void *> lookup(uint64_t cacheKey, const std::string &name);

  /// Set the active specialization context (used during compilation).
  void setActiveContext(const SpecializationContext *ctx);
  const SpecializationContext *getActiveContext() const;

  /// Register a user-defined external symbol (function or global) that the
  /// JIT can resolve when compiling bitcode modules. Required for bare-metal
  /// environments where dynamic symbol lookup is unavailable.
  void addUserSymbol(const std::string &name, void *addr);

private:
  struct Impl;
  std::unique_ptr<Impl> P;
};

} // namespace ejit
} // namespace llvm

#endif
