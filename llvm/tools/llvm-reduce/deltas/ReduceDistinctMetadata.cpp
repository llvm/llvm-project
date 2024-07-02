//===- ReduceDistinctMetadata.cpp - Specialized Delta Pass ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------------------===//
//
// This file implements two functions used by the Generic Delta Debugging
// Algorithm, which are used to reduce unnamed distinct metadata nodes.
//
//===------------------------------------------------------------------------------===//

#include "ReduceDistinctMetadata.h"
#include "Delta.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/InstIterator.h"
#include <algorithm>
#include <queue>
#include <set>

using namespace llvm;

// Traverse the graph breadth-first and try to remove unnamed metadata nodes
void reduceNodes(MDNode *Root,
                 std::set<std::pair<unsigned int, MDNode *>> &NodesToDelete,
                 MDNode *TemporaryNode, Oracle &O, Module &Program) {
  std::queue<MDNode *> NodesToTraverse{};
  // Keep track of visited nodes not to get into loops
  std::set<MDNode *> VisitedNodes{};
  NodesToTraverse.push(Root);

  while (!NodesToTraverse.empty()) {
    MDNode *CurrentNode = NodesToTraverse.front();
    NodesToTraverse.pop();

    // Mark the nodes for removal
    for (unsigned int I = 0; I < CurrentNode->getNumOperands(); ++I) {
      Metadata *Operand = CurrentNode->getOperand(I).get();
      if (isa_and_nonnull<MDNode>(Operand)) {
        // Check whether node has been visited
        if (VisitedNodes.find(static_cast<MDNode *>(Operand)) ==
            VisitedNodes.end()) {
          NodesToTraverse.push(static_cast<MDNode *>(Operand));
          VisitedNodes.insert(static_cast<MDNode *>(Operand));
        }
        // Delete the node only if it is distinct
        if (static_cast<MDNode *>(Operand)->isDistinct())
          // Add to removal list
          NodesToDelete.insert(std::make_pair(I, CurrentNode));
      }
    }

    // Remove the nodes
    for (std::pair<unsigned int, MDNode *> Node : NodesToDelete) {
      if (!O.shouldKeep())
        Node.second->replaceOperandWith(Node.first, TemporaryNode);
    }
    NodesToDelete.clear();
  }
}

// After reducing metadata, we need to remove references to the temporary node,
// this is also done with BFS
void cleanUpTemporaries(NamedMDNode &NamedNode, MDTuple *TemporaryTuple,
                        Module &Program) {
  std::queue<MDTuple *> NodesToTraverse{};
  std::set<MDTuple *> VisitedNodes{};

  // Push all first level operands of the named node to the queue
  for (auto I = NamedNode.op_begin(); I != NamedNode.op_end(); ++I) {
    // If the node hasn't been traversed yet, add it to the queue of nodes to
    // traverse.
    if (VisitedNodes.find(static_cast<MDTuple *>(*I)) == VisitedNodes.end()) {
      NodesToTraverse.push(static_cast<MDTuple *>(*I));
      VisitedNodes.insert(static_cast<MDTuple *>(*I));
    }
  }

  while (!NodesToTraverse.empty()) {
    MDTuple *CurrentTuple = NodesToTraverse.front();
    NodesToTraverse.pop();

    // Shift all of the interesting elements to the left, pop remaining
    // afterwards
    if (CurrentTuple
            ->isDistinct()) { // Do resizing and cleaning operations only if
                              // the node is distinct, as resizing is not
                              // supported for unique nodes and is redundant.
      unsigned int NotToRemove = 0;
      for (unsigned int I = 0; I < CurrentTuple->getNumOperands(); ++I) {
        Metadata *Operand = CurrentTuple->getOperand(I).get();
        // If current operand is not the temporary node, move it to the front
        // and increase notToRemove so that it will be saved
        if (Operand != TemporaryTuple) {
          Metadata *TemporaryMetadata =
              CurrentTuple->getOperand(NotToRemove).get();
          CurrentTuple->replaceOperandWith(NotToRemove, Operand);
          CurrentTuple->replaceOperandWith(I, TemporaryMetadata);
          ++NotToRemove;
        }
      }

      // Remove all the uninteresting elements
      unsigned int OriginalOperands = CurrentTuple->getNumOperands();
      for (unsigned int I = 0; I < OriginalOperands - NotToRemove; ++I)
        CurrentTuple->pop_back();
    }

    // Push the remaining nodes into the queue
    for (unsigned int I = 0; I < CurrentTuple->getNumOperands(); ++I) {
      Metadata *Operand = CurrentTuple->getOperand(I).get();
      if (isa_and_nonnull<MDTuple>(Operand) &&
          VisitedNodes.find(static_cast<MDTuple *>(Operand)) ==
              VisitedNodes.end()) {
        NodesToTraverse.push(static_cast<MDTuple *>(Operand));
        // If the node hasn't been traversed yet, add it to the queue of nodes
        // to traverse.
        VisitedNodes.insert(static_cast<MDTuple *>(Operand));
      }
    }
  }
}

static void extractDistinctMetadataFromModule(Oracle &O,
                                              ReducerWorkItem &WorkItem) {
  Module &Program = WorkItem.getModule();
  MDTuple *TemporaryTuple = MDTuple::getDistinct(
      Program.getContext(), SmallVector<Metadata *, 1>{llvm::MDString::get(
                                Program.getContext(), "temporary_tuple")});
  std::set<std::pair<unsigned int, MDNode *>> NodesToDelete{};
  for (NamedMDNode &NamedNode :
       Program.named_metadata()) { // Iterate over the named nodes
    for (unsigned int I = 0; I < NamedNode.getNumOperands();
         ++I) { // Iterate over first level unnamed nodes..
      Metadata *Operand = NamedNode.getOperand(I);
      if (isa_and_nonnull<MDTuple>(Operand))
        reduceNodes(static_cast<MDTuple *>(Operand), NodesToDelete,
                    TemporaryTuple, O, Program);
    }
  }
  for (NamedMDNode &NamedNode : Program.named_metadata())
    cleanUpTemporaries(NamedNode, TemporaryTuple, Program);
}

void llvm::reduceDistinctMetadataDeltaPass(TestRunner &Test) {
  runDeltaPass(Test, extractDistinctMetadataFromModule,
               "Reducing Distinct Metadata");
}
