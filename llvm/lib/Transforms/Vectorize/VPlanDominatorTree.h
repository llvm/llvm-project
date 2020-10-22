//===-- VPlanDominatorTree.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements dominator tree analysis for a single level of a VPlan's
/// H-CFG.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_VPLANDOMINATORTREE_H
#define LLVM_TRANSFORMS_VECTORIZE_VPLANDOMINATORTREE_H

#include "VPlan.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/CfgTraits.h"

namespace llvm {

/// Partial CFG traits for VPlan's CFG, without a value type.
class VPCfgTraitsBase : public CfgTraitsBase {
public:
  using ParentType = VPRegionBlock;
  using BlockRef = VPBlockBase *;
  using ValueRef = void;

  static CfgBlockRef wrapRef(BlockRef block) {
    return makeOpaque<CfgBlockRefTag>(block);
  }
  static BlockRef unwrapRef(CfgBlockRef block) {
    return static_cast<BlockRef>(getOpaque(block));
  }
};

class VPCfgTraits : public CfgTraits<VPCfgTraitsBase, VPCfgTraits> {
public:
  static VPRegionBlock *getBlockParent(VPBlockBase *block) {
    return block->getParent();
  }

  static auto predecessors(VPBlockBase *block) {
    return llvm::inverse_children<VPBlockBase *>(block);
  }

  static auto successors(VPBlockBase *block) {
    return llvm::children<VPBlockBase *>(block);
  }
};

template <> struct CfgTraitsFor<VPBlockBase> { using CfgTraits = VPCfgTraits; };

/// Template specialization of the standard LLVM dominator tree utility for
/// VPBlockBases.
using VPDominatorTree = DomTreeBase<VPBlockBase>;

using VPDomTreeNode = DomTreeNodeBase<VPBlockBase>;

/// Template specializations of GraphTraits for VPDomTreeNode.
template <>
struct GraphTraits<VPDomTreeNode *>
    : public DomTreeGraphTraitsBase<VPDomTreeNode,
                                    VPDomTreeNode::const_iterator> {};

template <>
struct GraphTraits<const VPDomTreeNode *>
    : public DomTreeGraphTraitsBase<const VPDomTreeNode,
                                    VPDomTreeNode::const_iterator> {};
} // namespace llvm
#endif // LLVM_TRANSFORMS_VECTORIZE_VPLANDOMINATORTREE_H
