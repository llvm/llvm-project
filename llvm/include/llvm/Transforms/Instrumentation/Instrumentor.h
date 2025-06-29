//===- Transforms/Instrumentation/Instrumentor.h --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A highly configurable instrumentation pass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_INSTRUMENTOR_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_INSTRUMENTOR_H

#include "llvm/IR/Instruction.h"
#include "llvm/IR/PassManager.h"

#include <functional>

namespace llvm {

/// Configuration for the Instrumentor. First generic configuration, followed by
/// the selection of what instruction classes and instructions should be
/// instrumented and how.
struct InstrumentorConfig {

  /// An optional callback that takes the instruction that is about to be
  /// instrumented and can return false if it should be skipped.
  using CallbackTy = std::function<bool(Instruction &)>;

#define SECTION_START(SECTION, CLASS) struct {

#define CONFIG_INTERNAL(SECTION, TYPE, NAME, DEFAULT_VALUE)                    \
  TYPE NAME = DEFAULT_VALUE;

#define CONFIG(SECTION, TYPE, NAME, DEFAULT_VALUE) TYPE NAME = DEFAULT_VALUE;

#define SECTION_END(SECTION)                                                   \
  }                                                                            \
  SECTION;

#include "llvm/Transforms/Instrumentation/InstrumentorConfig.def"
};

class InstrumentorPass : public PassInfoMixin<InstrumentorPass> {
  InstrumentorConfig IC;

public:
  InstrumentorPass(InstrumentorConfig IC = InstrumentorConfig{}) : IC(IC) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_INSTRUMENTOR_H
