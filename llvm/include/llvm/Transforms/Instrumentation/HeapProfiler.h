//===--------- Definition of the HeapProfiler class ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the HeapProfiler class.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_HEAPPROFILER_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_HEAPPROFILER_H

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// Public interface to the heap profiler pass for instrumenting code to
/// profile heap memory accesses.
///
/// The profiler itself is a function pass that works by inserting various
/// calls to the HeapProfiler runtime library functions. The runtime library
/// essentially replaces malloc() and free() with custom implementations that
/// record data about the allocations.
class HeapProfilerPass : public PassInfoMixin<HeapProfilerPass> {
public:
  explicit HeapProfilerPass();
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

/// Public interface to the heap profiler module pass for instrumenting code
/// to profile heap memory allocations and accesses.
class ModuleHeapProfilerPass : public PassInfoMixin<ModuleHeapProfilerPass> {
public:
  explicit ModuleHeapProfilerPass();
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:
};

// Insert HeapProfiler instrumentation
FunctionPass *createHeapProfilerFunctionPass();
ModulePass *createModuleHeapProfilerLegacyPassPass();

} // namespace llvm

#endif
