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
#include "llvm/IR/PassManager.h"
#include "llvm/ProfileData/MemProf.h"

#include <unordered_map>

namespace llvm {
class Function;
class IndexedInstrProfReader;
class Module;
class TargetLibraryInfo;

namespace vfs {
class FileSystem;
} // namespace vfs

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

private:
  std::string MemoryProfileFileName;
  IntrusiveRefCntPtr<vfs::FileSystem> FS;
};

namespace memprof {

// Extract all calls from the IR.  Arrange them in a map from caller GUIDs to a
// list of call sites, each of the form {LineLocation, CalleeGUID}.
DenseMap<uint64_t, SmallVector<CallEdgeTy, 0>> extractCallsFromIR(
    Module &M, const TargetLibraryInfo &TLI,
    function_ref<bool(uint64_t)> IsPresentInProfile = [](uint64_t) {
      return true;
    });

struct LineLocationHash {
  uint64_t operator()(const LineLocation &Loc) const {
    return Loc.getHashCode();
  }
};

using LocToLocMap =
    std::unordered_map<LineLocation, LineLocation, LineLocationHash>;

// Compute an undrifting map.  The result is a map from caller GUIDs to an inner
// map that maps source locations in the profile to those in the current IR.
DenseMap<uint64_t, LocToLocMap>
computeUndriftMap(Module &M, IndexedInstrProfReader *MemProfReader,
                  const TargetLibraryInfo &TLI);

} // namespace memprof
} // namespace llvm

#endif
