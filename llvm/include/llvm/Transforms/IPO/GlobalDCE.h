//===-- GlobalDCE.h - DCE unreachable internal functions ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transform is designed to eliminate unreachable internal globals from the
// program.  It uses an aggressive algorithm, searching out globals that are
// known to be alive.  After it finds all of the globals which are needed, it
// deletes whatever is left over.  This allows it to delete recursive chunks of
// the program which are unreachable.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_GLOBALDCE_H
#define LLVM_TRANSFORMS_IPO_GLOBALDCE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include <unordered_map>

namespace llvm {

/// Pass to remove unused function declarations.
class GlobalDCEPass : public PassInfoMixin<GlobalDCEPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);

private:
  SmallPtrSet<GlobalValue*, 32> AliveGlobals;

  /// Global -> Global that uses this global.
  DenseMap<GlobalValue *, SmallPtrSet<GlobalValue *, 4>> GVDependencies;

  /// Constant -> Globals that use this global cache.
  std::unordered_map<Constant *, SmallPtrSet<GlobalValue *, 8>>
      ConstantDependenciesCache;

  /// Comdat -> Globals in that Comdat section.
  std::unordered_multimap<Comdat *, GlobalValue *> ComdatMembers;

  /// !type metadata -> set of (vtable, offset) pairs
  DenseMap<Metadata *, SmallSet<std::pair<GlobalVariable *, uint64_t>, 4>>
      TypeIdMap;

  /// VTable -> set of vfuncs. This only contains vtables for which we have
  /// enough information to safely do dead virtual function elimination, and
  /// only contains vfuncs that are within the range specified in
  /// !vcall_visibility).
  DenseMap<GlobalValue *, SmallPtrSet<GlobalValue *, 8>> VFESafeVTablesAndFns;

  void UpdateGVDependencies(GlobalValue &GV);
  void MarkLive(GlobalValue &GV,
                SmallVectorImpl<GlobalValue *> *Updates = nullptr);
  bool RemoveUnusedGlobalValue(GlobalValue &GV);

  // Dead virtual function elimination.
  void AddVirtualFunctionDependencies(Module &M);
  void ScanVTables(Module &M);
  void ScanTypeCheckedLoadIntrinsics(Module &M);
  void ScanVTableLoad(Function *Caller, Metadata *TypeId, uint64_t CallOffset);

  void ComputeDependencies(Value *V, SmallPtrSetImpl<GlobalValue *> &U);
};

}

#endif // LLVM_TRANSFORMS_IPO_GLOBALDCE_H
