//===-- VPlanPredicator.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the VPlanPredicator class which contains the public
/// interfaces to predicate and linearize the VPlan region.
///
//===----------------------------------------------------------------------===//

#include "VPlanPredicator.h"
#include "VPlan.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "VPlanPredicator"

using namespace llvm;

// Generate VPInstructions at the beginning of CurrBB that calculate the
// predicate being propagated from PredBB to CurrBB depending on the edge type
// between them. For example if:
//  i.  PredBB is controlled by predicate %BP, and
//  ii. The edge PredBB->CurrBB is the false edge, controlled by the condition
//  bit value %CBV then this function will generate the following two
//  VPInstructions at the start of CurrBB:
//   %IntermediateVal = not %CBV
//   %FinalVal        = and %BP %IntermediateVal
// It returns %FinalVal.
VPValue *VPlanPredicator::getOrCreateNotPredicate(VPBasicBlock *PredBB,
                                                  VPBasicBlock *CurrBB) {
  VPValue *CBV = PredBB->getCondBit();

  // Set the intermediate value - this is either 'CBV', or 'not CBV'
  // depending on the edge type.
  EdgeType ET = getEdgeTypeBetween(PredBB, CurrBB);
  VPValue *IntermediateVal = nullptr;
  switch (ET) {
  case EdgeType::TRUE_EDGE:
    // CurrBB is the true successor of PredBB - nothing to do here.
    IntermediateVal = CBV;
    break;

  case EdgeType::FALSE_EDGE:
    // CurrBB is the False successor of PredBB - compute not of CBV.
    IntermediateVal = Builder.createNot(CBV);
    break;
  }

  // Now AND intermediate value with PredBB's block predicate if it has one.
  VPValue *BP = PredBB->getPredicate();
  if (BP)
    return Builder.createAnd(BP, IntermediateVal);
  else
    return IntermediateVal;
}

// Generate a tree of ORs for all IncomingPredicates in  WorkList.
// Note: This function destroys the original Worklist.
//
// P1 P2 P3 P4 P5
//  \ /   \ /  /
//  OR1   OR2 /
//    \    | /
//     \   +/-+
//      \  /  |
//       OR3  |
//         \  |
//          OR4 <- Returns this
//           |
//
// The algorithm uses a worklist of predicates as its main data structure.
// We pop a pair of values from the front (e.g. P1 and P2), generate an OR
// (in this example OR1), and push it back. In this example the worklist
// contains {P3, P4, P5, OR1}.
// The process iterates until we have only one element in the Worklist (OR4).
// The last element is the root predicate which is returned.
VPValue *VPlanPredicator::genPredicateTree(std::list<VPValue *> &Worklist) {
  if (Worklist.empty())
    return nullptr;

  // The worklist initially contains all the leaf nodes. Initialize the tree
  // using them.
  while (Worklist.size() >= 2) {
    // Pop a pair of values from the front.
    VPValue *LHS = Worklist.front();
    Worklist.pop_front();
    VPValue *RHS = Worklist.front();
    Worklist.pop_front();

    // Create an OR of these values.
    VPValue *Or = Builder.createOr(LHS, RHS);

    // Push OR to the back of the worklist.
    Worklist.push_back(Or);
  }

  assert(Worklist.size() == 1 && "Expected 1 item in worklist");

  // The root is the last node in the worklist.
  VPValue *Root = Worklist.front();

  // This root needs to replace the existing block predicate. This is done in
  // the caller function.
  return Root;
}

// Return whether the edge FromBlock -> ToBlock is a TRUE_EDGE or FALSE_EDGE
VPlanPredicator::EdgeType
VPlanPredicator::getEdgeTypeBetween(VPBlockBase *FromBlock,
                                    VPBlockBase *ToBlock) {
  unsigned Count = 0;
  for (VPBlockBase *SuccBlock : FromBlock->getSuccessors()) {
    if (SuccBlock == ToBlock) {
      assert(Count < 2 && "Switch not supported currently");
      return (Count == 0) ? EdgeType::TRUE_EDGE : EdgeType::FALSE_EDGE;
    }
    Count++;
  }

  llvm_unreachable("Broken getEdgeTypeBetween");
}

// Generate all predicates needed for CurrBlock by going through its immediate
// predecessor blocks.
void VPlanPredicator::createOrPropagatePredicates(VPBlockBase *CurrBlock,
                                                  VPRegionBlock *Region) {
  // Blocks that dominate region exit inherit the predicate from the region.
  // Return after setting the predicate.
  if (VPDomTree.dominates(CurrBlock, Region->getExit())) {
    VPValue *RegionBP = Region->getPredicate();
    CurrBlock->setPredicate(RegionBP);
    return;
  }

  // Collect all incoming predicates in a worklist.
  std::list<VPValue *> IncomingPredicates;

  // Set the builder's insertion point to the top of the current BB
  VPBasicBlock *CurrBB = cast<VPBasicBlock>(CurrBlock->getEntryBasicBlock());
  Builder.setInsertPoint(CurrBB, CurrBB->begin());

  // For each predecessor, generate the VPInstructions required for
  // computing 'BP AND (not) CBV" at the top of CurrBB.
  // Collect the outcome of this calculation for all predecessors
  // into IncomingPredicates.
  for (VPBlockBase *PredBlock : CurrBlock->getPredecessors()) {
    // Skip back-edges
    if (VPBlockUtils::isBackEdge(PredBlock, CurrBlock, VPLI))
      continue;

    VPValue *IncomingPredicate = nullptr;
    unsigned NumPredSuccsNoBE =
        VPBlockUtils::countSuccessorsNoBE(PredBlock, VPLI);

    // If there is an unconditional branch to the currBB, then we don't create
    // edge predicates. We use the predecessor's block predicate instead.
    if (NumPredSuccsNoBE == 1)
      IncomingPredicate = PredBlock->getPredicate();
    else if (NumPredSuccsNoBE == 2) {
      // Emit recipes into CurrBlock if required
      assert(isa<VPBasicBlock>(PredBlock) && "Only BBs have multiple exits");
      IncomingPredicate =
          getOrCreateNotPredicate(cast<VPBasicBlock>(PredBlock), CurrBB);
    } else
      llvm_unreachable("FIXME: switch statement ?");

    if (IncomingPredicate)
      IncomingPredicates.push_back(IncomingPredicate);
  }

  // Logically OR all incoming predicates by building the Predicate Tree.
  VPValue *Predicate = genPredicateTree(IncomingPredicates);

  // Now update the block's predicate with the new one.
  CurrBlock->setPredicate(Predicate);
}

// Generate all predicates needed for Region.
void VPlanPredicator::predicateRegionRec(VPRegionBlock *Region) {
  VPBasicBlock *EntryBlock = cast<VPBasicBlock>(Region->getEntry());
  ReversePostOrderTraversal<VPBlockBase *> RPOT(EntryBlock);

  // Generate edge predicates and append them to the block predicate. RPO is
  // necessary since the predecessor blocks' block predicate needs to be set
  // before the current block's block predicate can be computed.
  for (VPBlockBase *Block : RPOT) {
    // TODO: Handle nested regions once we start generating the same.
    assert(!isa<VPRegionBlock>(Block) && "Nested region not expected");
    createOrPropagatePredicates(Block, Region);
  }
}

// Linearize the CFG within Region.
// TODO: Predication and linearization need RPOT for every region.
// This traversal is expensive. Since predication is not adding new
// blocks, we should be able to compute RPOT once in predication and
// reuse it here. This becomes even more important once we have nested
// regions.
void VPlanPredicator::linearizeRegionRec(VPRegionBlock *Region) {
  ReversePostOrderTraversal<VPBlockBase *> RPOT(Region->getEntry());
  VPBlockBase *PrevBlock = nullptr;

  for (VPBlockBase *CurrBlock : RPOT) {
    // TODO: Handle nested regions once we start generating the same.
    assert(!isa<VPRegionBlock>(CurrBlock) && "Nested region not expected");

    // Linearize control flow by adding an unconditional edge between PrevBlock
    // and CurrBlock skipping loop headers and latches to keep intact loop
    // header predecessors and loop latch successors.
    if (PrevBlock && !VPLI->isLoopHeader(CurrBlock) &&
        !VPBlockUtils::blockIsLoopLatch(PrevBlock, VPLI)) {

      LLVM_DEBUG(dbgs() << "Linearizing: " << PrevBlock->getName() << "->"
                        << CurrBlock->getName() << "\n");

      PrevBlock->clearSuccessors();
      CurrBlock->clearPredecessors();
      VPBlockUtils::connectBlocks(PrevBlock, CurrBlock);
    }

    PrevBlock = CurrBlock;
  }
}

// Entry point. The driver function for the predicator.
void VPlanPredicator::predicate(void) {
  // Predicate the blocks within Region.
  predicateRegionRec(cast<VPRegionBlock>(Plan.getEntry()));

  // Linearlize the blocks with Region.
  linearizeRegionRec(cast<VPRegionBlock>(Plan.getEntry()));
}

VPlanPredicator::VPlanPredicator(VPlan &Plan)
    : Plan(Plan), VPLI(&(Plan.getVPLoopInfo())) {
  // FIXME: Predicator is currently computing the dominator information for the
  // top region. Once we start storing dominator information in a VPRegionBlock,
  // we can avoid this recalculation.
  VPDomTree.recalculate(*(cast<VPRegionBlock>(Plan.getEntry())));
}
