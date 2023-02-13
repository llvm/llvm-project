//===- Transforms/Instrumentation/TraceRecorder.h - TSan Pass -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the thread sanitizer pass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_TRACERECORDER_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_TRACERECORDER_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include <map>

namespace llvm {
// Insert TraceRecorder (race detection) instrumentation

/// A function pass for trec instrumentation.
///
/// Instruments functions to detect race conditions reads. This function pass
/// inserts calls to runtime library functions. If the functions aren't declared
/// yet, the pass inserts the declarations. Otherwise the existing globals are



struct TraceRecorderPass : public PassInfoMixin<TraceRecorderPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
  static bool isRequired() { return true; }
};

struct ModuleTraceRecorderPass
  : public PassInfoMixin<ModuleTraceRecorderPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

} // namespace llvm
#endif /* LLVM_TRANSFORMS_INSTRUMENTATION_TRACERECORDER_H */
