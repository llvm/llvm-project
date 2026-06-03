//===-- EJitSyncCompiler.h - Synchronous JIT Compiler ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITSYNCCOMPILER_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITSYNCCOMPILER_H

#ifndef EJIT_FREESTANDING

#include "llvm/ExecutionEngine/EJIT/EJitOptions.h"
#include "llvm/ExecutionEngine/EJIT/EJitOrcEngine.h"
#include <cstddef>
#include <string>

namespace llvm {
namespace ejit {

class EJitOrcEngine;

/// Blocking JIT compiler that executes the full compilation pipeline
/// on the calling thread.
class EJitSyncCompiler {
public:
  struct Result {
    void *funcPtr = nullptr;
    size_t compileTimeMs = 0;
    size_t codeSize = 0;
  };

  /// Compile a function synchronously. Uses ctx.cacheKey.
  Result compile(EJitOrcEngine &engine,
                 const std::string &bitcodeData,
                 const SpecializationContext &ctx);
};

} // namespace ejit
} // namespace llvm

#endif // EJIT_FREESTANDING
#endif
