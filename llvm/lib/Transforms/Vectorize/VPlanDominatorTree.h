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

namespace llvm {

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
