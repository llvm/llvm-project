//===- PostIndexCandidateCache.h - Candidate queries -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_SELECTIONDAG_POSTINDEXCANDIDATECACHE_H
#define LLVM_LIB_CODEGEN_SELECTIONDAG_POSTINDEXCANDIDATECACHE_H

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/CodeGen/SelectionDAG.h"

namespace llvm {

/// A DAGCombiner-owned cache of bases with no ADD/SUB users of any result.
/// A failed target-specific match is not cached: another memory operation may
/// still be able to use the same ADD/SUB. No candidate search is truncated.
class PostIndexCandidateCache final : public SelectionDAG::DAGUpdateListener {
  SmallPtrSet<SDNode *, 16> BasesWithoutAddSubUsers;

  void invalidate(SDNode *N) {
    // Insertion also clears stale entries when an address is reused after a
    // direct DeleteNode call, which does not issue NodeDeleted.
    BasesWithoutAddSubUsers.erase(N);
    if (N->getOpcode() == ISD::ADD || N->getOpcode() == ISD::SUB)
      for (SDValue Op : N->op_values())
        BasesWithoutAddSubUsers.erase(Op.getNode());
  }

  void NodeInserted(SDNode *N) override { invalidate(N); }
  void NodeUpdated(SDNode *N) override { invalidate(N); }
  void NodeOperandsChanged(SDNode *N) override { invalidate(N); }
  void NodeDeleted(SDNode *N, SDNode *) override {
    // A node merged by CSE can already have new operands here, without having
    // emitted NodeUpdated. Invalidate them before the node is deallocated.
    invalidate(N);
  }

public:
  explicit PostIndexCandidateCache(SelectionDAG &DAG)
      : DAGUpdateListener(DAG) {}
  PostIndexCandidateCache(const PostIndexCandidateCache &) = delete;
  PostIndexCandidateCache &operator=(const PostIndexCandidateCache &) = delete;

  bool hasAnyAddOrSubUser(SDNode *Base) {
    if (BasesWithoutAddSubUsers.contains(Base))
      return false;
    for (SDNode *User : Base->users())
      if (User->getOpcode() == ISD::ADD || User->getOpcode() == ISD::SUB)
        return true;
    BasesWithoutAddSubUsers.insert(Base);
    return false;
  }
};

} // namespace llvm

#endif
