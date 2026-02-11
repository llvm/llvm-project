//===--------- MemProfUse.h - Memory profiler use pass ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the MemProfUsePass class and related utilities.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_MEMPROFUSE_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_MEMPROFUSE_H

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/IR/PassManager.h"
#include "llvm/ProfileData/DataAccessProf.h"
#include "llvm/ProfileData/MemProf.h"
#include "llvm/Support/Compiler.h"

#include <unordered_map>

namespace llvm {
class IndexedInstrProfReader;
class Module;
class TargetLibraryInfo;

namespace vfs {
class FileSystem;
} // namespace vfs

class MemProfUsePass : public PassInfoMixin<MemProfUsePass> {
public:
  LLVM_ABI explicit MemProfUsePass(
      std::string MemoryProfileFile,
      IntrusiveRefCntPtr<vfs::FileSystem> FS = nullptr);
  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:
  // Annotate global variables' section prefix based on data access profile,
  // return true if any global variable is annotated and false otherwise.
  bool
  annotateGlobalVariables(Module &M,
                          const memprof::DataAccessProfData *DataAccessProf);
  std::string MemoryProfileFileName;
  IntrusiveRefCntPtr<vfs::FileSystem> FS;
};

namespace memprof {

// Extract all calls from the IR.  Arrange them in a map from caller GUIDs to a
// list of call sites, each of the form {LineLocation, CalleeGUID}.
LLVM_ABI DenseMap<uint64_t, SmallVector<CallEdgeTy, 0>> extractCallsFromIR(
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
LLVM_ABI DenseMap<uint64_t, LocToLocMap>
computeUndriftMap(Module &M, IndexedInstrProfReader *MemProfReader,
                  const TargetLibraryInfo &TLI);

} // namespace memprof
} // namespace llvm

#endif
