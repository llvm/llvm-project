//===--------- Definition of the MemProfiler class --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the MemProfiler class.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_MEMPROFILER_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_MEMPROFILER_H

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/IR/PassManager.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/Support/VirtualFileSystem.h"

namespace llvm {
class Function;
class Module;
class TargetLibraryInfo;

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
  static bool isRequired() { return true; }
};

/// Public interface to the memory profiler module pass for instrumenting code
/// to profile memory allocations and accesses.
class ModuleMemProfilerPass : public PassInfoMixin<ModuleMemProfilerPass> {
public:
  explicit ModuleMemProfilerPass();
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

class MemProfUsePass : public PassInfoMixin<MemProfUsePass> {
public:
  explicit MemProfUsePass(std::string MemoryProfileFile,
                          IntrusiveRefCntPtr<vfs::FileSystem> FS = nullptr);
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

  struct AllocMatchInfo {
    uint64_t TotalSize = 0;
    AllocationType AllocType = AllocationType::None;
    bool Matched = false;
  };

  void
  readMemprof(Function &F, const IndexedMemProfReader &MemProfReader,
              const TargetLibraryInfo &TLI,
              std::map<uint64_t, AllocMatchInfo> &FullStackIdToAllocMatchInfo);

private:
  std::string MemoryProfileFileName;
  IntrusiveRefCntPtr<vfs::FileSystem> FS;
};

} // namespace llvm

#endif
