//===- SyncDependenceAnalysis.h - Divergent Branch Dependence -*- C++ -*-===//
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

#ifndef LLVM_ANALYSIS_SYNCDEPENDENCEANALYSIS_H
#define LLVM_ANALYSIS_SYNCDEPENDENCEANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include <memory>
#include <unordered_map>

namespace llvm {

class BasicBlock;
class DominatorTree;
class Loop;
class PostDominatorTree;

using ConstBlockSet = SmallPtrSet<const BasicBlock *, 4>;
struct ControlDivergenceDesc {
  // Join points of divergent disjoint paths.
  ConstBlockSet JoinDivBlocks;
  // Divergent loop exits
  ConstBlockSet LoopDivBlocks;
};

struct ModifiedPO {
  std::vector<const BasicBlock *> LoopPO;
  std::unordered_map<const BasicBlock *, unsigned> POIndex;
  void appendBlock(const BasicBlock &BB) {
    POIndex[&BB] = LoopPO.size();
    LoopPO.push_back(&BB);
  }
  unsigned getIndexOf(const BasicBlock &BB) const {
    return POIndex.find(&BB)->second;
  }
  unsigned size() const { return LoopPO.size(); }
  const BasicBlock *getBlockAt(unsigned Idx) const { return LoopPO[Idx]; }
};

/// \brief Relates points of divergent control to join points in
/// reducible CFGs.
///
/// This analysis relates points of divergent control to points of converging
/// divergent control. The analysis requires all loops to be reducible.
class SyncDependenceAnalysis {
public:
  ~SyncDependenceAnalysis();
  SyncDependenceAnalysis(const DominatorTree &DT, const PostDominatorTree &PDT,
                         const LoopInfo &LI);

  /// \brief Computes divergent join points and loop exits caused by branch
  /// divergence in \p Term.
  ///
  /// The set of blocks which are reachable by disjoint paths from \p Term.
  /// The set also contains loop exits if there two disjoint paths:
  /// one from \p Term to the loop exit and another from \p Term to the loop
  /// header. Those exit blocks are added to the returned set.
  /// If L is the parent loop of \p Term and an exit of L is in the returned
  /// set then L is a divergent loop.
  const ControlDivergenceDesc &getJoinBlocks(const Instruction &Term);

private:
  static ControlDivergenceDesc EmptyDivergenceDesc;

  ModifiedPO LoopPO;

  const DominatorTree &DT;
  const PostDominatorTree &PDT;
  const LoopInfo &LI;

  std::map<const Instruction *, std::unique_ptr<ControlDivergenceDesc>>
      CachedControlDivDescs;
};

} // namespace llvm

#endif // LLVM_ANALYSIS_SYNCDEPENDENCEANALYSIS_H
