//===- LowFatSanitizer.h - LowFat Pointer Bounds Checking -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_LOWFATSANITIZER_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_LOWFATSANITIZER_H

#include "llvm/IR/PassManager.h"

namespace llvm {
class Module;

struct LowFatSanitizerOptions {
  bool Recover = false;

  enum class LowFatMode {
    Fast,       /// instrument at OptimizerLastEP
    Safe,       /// Barrier at PipelineStartEP + instrument at OptimizerLastEP
    RightAlign, /// Fast instrumentation + right-align allocations within class
                /// slots to improve detection of right-side (overflow) OOB at
                /// the cost of a blind spot on the left (underflow) side.
  };
  LowFatMode Mode = LowFatMode::Fast;

  bool InternalBarrierOnly_ = false;
};

class LowFatSanitizerPass : public PassInfoMixin<LowFatSanitizerPass> {
public:
  LLVM_ABI
  LowFatSanitizerPass(const LowFatSanitizerOptions &Options);
  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }

private:
  LowFatSanitizerOptions Options;
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_LOWFATSANITIZER_H
