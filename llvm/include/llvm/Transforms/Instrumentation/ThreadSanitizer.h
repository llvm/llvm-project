//===- ThreadSanitizer.h - ThreadSanitizer instrumentation ------*- C++ -*-===//
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

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_THREADSANITIZER_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_THREADSANITIZER_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class Function;
class Module;

struct ThreadSanitizerOptions {
  bool InstrumentMemoryAccesses = true;
  bool InstrumentAtomics = true;
  bool InstrumentMemIntrinsics = true;
  bool AlwaysInstrumentFuncEntryExit = false;
};

/// A function pass for tsan instrumentation.
///
/// Instruments functions to detect race conditions reads. This function pass
/// inserts calls to runtime library functions. If the functions aren't declared
/// yet, the pass inserts the declarations. Otherwise the existing globals are
class ThreadSanitizerPass : public RequiredPassInfoMixin<ThreadSanitizerPass> {
public:
  LLVM_ABI
  ThreadSanitizerPass(const ThreadSanitizerOptions &Options = {})
      : Options(Options) {}
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);

private:
  ThreadSanitizerOptions Options;
};

/// A module pass for tsan instrumentation.
///
/// Create ctor and init functions.
struct ModuleThreadSanitizerPass
    : public RequiredPassInfoMixin<ModuleThreadSanitizerPass> {
  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm
#endif /* LLVM_TRANSFORMS_INSTRUMENTATION_THREADSANITIZER_H */
