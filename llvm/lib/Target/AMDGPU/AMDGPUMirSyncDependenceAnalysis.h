//===- MirSyncDependenceAnalysis.h - MirDivergent Branch Dependence -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// This file defines the SyncDependenceAnalysis class, which computes for
// every divergent branch the set of phi nodes that the branch will make
// divergent.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include <memory>
#include <map>

namespace llvm {
class MachineBasicBlock;
class MachineDominatorTree;
class MachineLoop;
class MachinePostDominatorTree;
class MachineLoopInfo;
class MachineFunction;
class MachineInstr;

using DivergentJoinMapTy =
    llvm::DenseMap<const llvm::MachineBasicBlock *,
                   llvm::SmallPtrSet<const llvm::MachineBasicBlock *, 4>>;

using ConstBlockSet = llvm::SmallPtrSet<const MachineBasicBlock *, 4>;

/// \brief Relates points of divergent control to join points in
/// reducible CFGs.
///
/// This analysis relates points of divergent control to points of converging
/// divergent control. The analysis requires all loops to be reducible.
class SyncDependenceAnalysis {
  void visitSuccessor(const MachineBasicBlock &succBlock, const MachineLoop *termLoop,
                      const MachineBasicBlock *defBlock);

public:
  bool inRegion(const MachineBasicBlock &BB) const;

  ~SyncDependenceAnalysis();
  SyncDependenceAnalysis(const MachineDominatorTree &DT, const MachinePostDominatorTree &PDT,
                         const MachineLoopInfo &LI,
                         // AMDGPU change begin
                         DivergentJoinMapTy &JoinMap
                         // AMDGPU change end
  );

  /// \brief Computes divergent join points and loop exits caused by branch
  /// divergence in \p Term.
  ///
  /// The set of blocks which are reachable by disjoint paths from \p Term.
  /// The set also contains loop exits if there two disjoint paths:
  /// one from \p Term to the loop exit and another from \p Term to the loop
  /// header. Those exit blocks are added to the returned set.
  /// If L is the parent loop of \p Term and an exit of L is in the returned
  /// set then L is a divergent loop.
  const ConstBlockSet &join_blocks(const MachineInstr &Term);

  /// \brief Computes divergent join points and loop exits (in the surrounding
  /// loop) caused by the divergent loop exits of\p MachineLoop.
  ///
  /// The set of blocks which are reachable by disjoint paths from the
  /// loop exits of \p MachineLoop.
  /// This treats the loop as a single node in \p MachineLoop's parent loop.
  /// The returned set has the same properties as for join_blocks(TermInst&).
  const ConstBlockSet &join_blocks(const MachineLoop &MachineLoop);

private:
  static ConstBlockSet EmptyBlockSet;

  llvm::ReversePostOrderTraversal<const llvm::MachineFunction *> FuncRPOT;
  const MachineDominatorTree &DT;
  const MachinePostDominatorTree &PDT;
  const MachineLoopInfo &LI;
  // AMDGPU change begin.
  DivergentJoinMapTy &DivergentJoinMap;
  // AMDGPU change end.
  std::map<const MachineLoop *, std::unique_ptr<ConstBlockSet>> CachedLoopExitJoins;
  std::map<const MachineInstr *, std::unique_ptr<ConstBlockSet>>
      CachedBranchJoins;
};

} // namespace llvm


