//===-- EJitOptions.h - EmbeddedJIT Configuration -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITOPTIONS_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITOPTIONS_H

#include <cstddef>
#include <cstdint>

namespace llvm {
namespace ejit {

enum class CompileMode { Sync, Async };
enum class OptimizationLevel { L1 = 1, L2 = 2, L3 = 3 };

struct Config {
  CompileMode compileMode = CompileMode::Sync;
  OptimizationLevel optLevel = OptimizationLevel::L2;
  size_t maxCodeMemory = 384 * 1024;
  size_t maxDataMemory = 128 * 1024;
  size_t maxCacheEntries = 256;
  size_t maxCacheSize = 32 * 1024 * 1024;
  size_t maxSingleFuncSize = 512 * 1024;
  bool enableLogger = true;
};

} // namespace ejit
} // namespace llvm

#endif
