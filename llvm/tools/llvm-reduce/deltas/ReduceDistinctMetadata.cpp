//===- ReduceDistinctMetadata.cpp - Specialized Delta Pass
//------------------------===//
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
void BFS_removal(MDNode *root,
                 std::set<std::pair<unsigned int, MDNode *>> &nodesToDelete,
                 MDNode *tmp, Oracle &O, Module &Program) {
  std::queue<MDNode *> q{};
  std::set<MDNode *>
      visited{}; // Keep track of visited nodes not to get into loops
  q.push(root);

  while (!q.empty()) {
    MDNode *current = q.front();
    q.pop();

    // Mark the nodes for removal
    for (unsigned int i = 0; i < current->getNumOperands(); ++i) {
      Metadata *operand = current->getOperand(i).get();
      if (isa_and_nonnull<MDNode>(operand)) {
        if (std::find(visited.begin(), visited.end(), operand) ==
            visited.end()) { // Check whether node has been visited
          q.push(static_cast<MDNode *>(operand));
          visited.insert(static_cast<MDNode *>(operand));
        }
        // Delete the node only if it is distinct
        if (static_cast<MDNode *>(operand)->isDistinct())
          nodesToDelete.insert(
              std::make_pair(i, current)); // Add to removal list
      }
    }

    // Remove the nodes
    for (std::pair<unsigned int, MDNode *> node : nodesToDelete) {
      if (!O.shouldKeep())
        node.second->replaceOperandWith(node.first, tmp);
    }
    nodesToDelete.clear();
  }
}

// After reducing metadata, we need to remove references to the temporary node,
// this is also done with BFS
void cleanup(NamedMDNode &namedNode, MDTuple *tmp, Module &Program) {
  std::queue<MDTuple *> q{};
  std::set<MDTuple *> visited{};

  // Push all first level operands of the named node to the queue
  for (auto i = namedNode.op_begin(); i != namedNode.op_end(); ++i) {
    // If the node hasn't been traversed yet, add it to the queue of nodes to
    // traverse.
    if (std::find(visited.begin(), visited.end(), *i) == visited.end()) {
      q.push(static_cast<MDTuple *>(*i));
      visited.insert(static_cast<MDTuple *>(*i));
    }
  }

  while (!q.empty()) {
    MDTuple *current = q.front();
    q.pop();

    // Shift all of the interesting elements to the left, pop remaining
    // afterwards
    if (current->isDistinct()) { // Do resizing and cleaning operations only if
                                 // the node is distinct, as resizing is not
                                 // supported for unique nodes and is redundant.
      unsigned int notToRemove = 0;
      for (unsigned int i = 0; i < current->getNumOperands(); ++i) {
        Metadata *operand = current->getOperand(i).get();
        // If current operand is not the temporary node, move it to the front
        // and increase notToRemove so that it will be saved
        if (operand != tmp) {
          Metadata *temporary = current->getOperand(notToRemove).get();
          current->replaceOperandWith(notToRemove, operand);
          current->replaceOperandWith(i, temporary);
          ++notToRemove;
        }
      }

      // Remove all the uninteresting elements
      unsigned int originalOperands = current->getNumOperands();
      for (unsigned int i = 0; i < originalOperands - notToRemove; ++i)
        current->pop_back();
    }

    // Push the remaining nodes into the queue
    for (unsigned int i = 0; i < current->getNumOperands(); ++i) {
      Metadata *operand = current->getOperand(i).get();
      if (isa_and_nonnull<MDTuple>(operand) &&
          std::find(visited.begin(), visited.end(), operand) == visited.end()) {
        q.push(static_cast<MDTuple *>(operand));
        visited.insert(static_cast<MDTuple *>(
            operand)); // If the node hasn't been traversed yet, add it to the
                       // queue of nodes to traverse.
      }
    }
  }
}

static void extractDistinctMetadataFromModule(Oracle &O,
                                              ReducerWorkItem &WorkItem) {
  Module &Program = WorkItem.getModule();
  MDTuple *tmp = MDTuple::getDistinct(
      Program.getContext(), SmallVector<Metadata *, 1>{llvm::MDString::get(
                                Program.getContext(), "temporary_node")});
  std::set<std::pair<unsigned int, MDNode *>> nodesToDelete{};
  for (NamedMDNode &namedNode :
       Program.named_metadata()) { // Iterate over the named nodes
    for (unsigned int i = 0; i < namedNode.getNumOperands();
         ++i) { // Iterate over first level unnamed nodes..
      Metadata *operand = namedNode.getOperand(i);
      if (isa_and_nonnull<MDTuple>(operand))
        BFS_removal(static_cast<MDTuple *>(operand), nodesToDelete, tmp, O,
                    Program);
    }
  }
  for (NamedMDNode &namedNode : Program.named_metadata())
    cleanup(namedNode, tmp, Program);
}

void llvm::reduceDistinctMetadataDeltaPass(TestRunner &Test) {
  runDeltaPass(Test, extractDistinctMetadataFromModule,
               "Reducing Distinct Metadata");
}