//===-- Sink.h - Code Sinking -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass moves instructions into successor blocks, when possible, so that
// they aren't executed on paths where their results aren't needed.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_SINK_H
#define LLVM_TRANSFORMS_SCALAR_SINK_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Function;

/// Move instructions into successor blocks when possible.
class SinkingPass : public OptionalPassInfoMixin<SinkingPass> {
  /// Do not sink an instruction into a block that postdominates it if that
  /// would keep more operand bits live than it frees. Such a move saves no
  /// execution and can only increase register pressure. On targets with
  /// divergent branches, only divergent values are counted.
  bool AvoidPostDominators;

public:
  explicit SinkingPass(bool AvoidPostDominators = false)
      : AvoidPostDominators(AvoidPostDominators) {}

  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  LLVM_ABI void
  printPipeline(raw_ostream &OS,
                function_ref<StringRef(StringRef)> MapClassName2PassName);
};
}

#endif // LLVM_TRANSFORMS_SCALAR_SINK_H
