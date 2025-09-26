//===- AllocToken.h - Allocation token instrumentation --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the AllocTokenPass, an instrumentation pass that
// replaces allocation calls with ones including an allocation token.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_ALLOCTOKEN_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_ALLOCTOKEN_H

#include "llvm/IR/Analysis.h"
#include "llvm/IR/PassManager.h"
#include <optional>

namespace llvm {

class Module;

struct AllocTokenOptions {
  std::optional<uint64_t> MaxTokens;
  bool FastABI = false;
  bool Extended = false;
  AllocTokenOptions() = default;
};

/// A module pass that rewrites heap allocations to use token-enabled
/// allocation functions based on various source-level properties.
class AllocTokenPass : public PassInfoMixin<AllocTokenPass> {
public:
  LLVM_ABI explicit AllocTokenPass(AllocTokenOptions Opts = {});
  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
  static bool isRequired() { return true; }

private:
  const AllocTokenOptions Options;
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_ALLOCTOKEN_H
