//===--------- Definition of the MemProfiler class --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MemProfiler class.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_MEMPROFILER_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_MEMPROFILER_H

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// Public interface to the memory profiler pass for instrumenting code to
/// profile memory accesses.
///
/// The profiler itself is a function pass that works by inserting various
/// calls to the MemProfiler runtime library functions. The runtime library
/// essentially replaces malloc() and free() with custom implementations that
/// record data about the allocations.
class MemProfilerPass : public PassInfoMixin<MemProfilerPass> {
public:
  explicit MemProfilerPass();
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

/// Public interface to the memory profiler module pass for instrumenting code
/// to profile memory allocations and accesses.
class ModuleMemProfilerPass : public PassInfoMixin<ModuleMemProfilerPass> {
public:
  explicit ModuleMemProfilerPass();
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

// Insert MemProfiler instrumentation
FunctionPass *createMemProfilerFunctionPass();
ModulePass *createModuleMemProfilerLegacyPassPass();

} // namespace llvm

#endif
