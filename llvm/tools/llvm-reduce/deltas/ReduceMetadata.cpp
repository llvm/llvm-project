//===- ReduceMetadata.cpp - Specialized Delta Pass ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements two functions used by the Generic Delta Debugging
// Algorithm, which are used to reduce Metadata nodes.
//
//===----------------------------------------------------------------------===//

#include "ReduceMetadata.h"
#include "Delta.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/InstIterator.h"
#include <vector>

using namespace llvm;

/// Removes all the Named and Unnamed Metadata Nodes, as well as any debug
/// functions that aren't inside the desired Chunks.
static void extractMetadataFromModule(Oracle &O, Module &Program) {
  SmallSetVector<MDNode *, 8> NodesToVisit;

  // Get out-of-chunk Named metadata nodes
  SmallVector<NamedMDNode *> NamedNodesToDelete;
  for (NamedMDNode &MD : Program.named_metadata()) {
    if (O.shouldKeep()) {
      for (auto *Op : MD.operands())
        NodesToVisit.insert(Op);
    } else {
      NamedNodesToDelete.push_back(&MD);
    }
  }

  for (NamedMDNode *NN : NamedNodesToDelete) {
    for (auto I : seq<unsigned>(0, NN->getNumOperands()))
      NN->setOperand(I, nullptr);
    NN->eraseFromParent();
  }

  // Delete elements from named metadata lists
  for (auto &NamedList : Program.named_metadata()) {
    SmallVector<MDNode *> NewOperands;
    for (auto *Op : NamedList.operands())
      if (O.shouldKeep())
        NewOperands.push_back(Op);
    if (NewOperands.size() == NamedList.getNumOperands())
      continue;
    NamedList.clearOperands();
    for (auto *Op : NewOperands)
      NamedList.addOperand(Op);
  }

  // Delete out-of-chunk metadata attached to globals.
  for (GlobalVariable &GV : Program.globals()) {
    SmallVector<std::pair<unsigned, MDNode *>> MDs;
    GV.getAllMetadata(MDs);
    for (std::pair<unsigned, MDNode *> &MD : MDs) {
      if (O.shouldKeep()) {
        NodesToVisit.insert(MD.second);
      } else {
        GV.setMetadata(MD.first, nullptr);
      }
    }
  }

  for (Function &F : Program) {
    {
      SmallVector<std::pair<unsigned, MDNode *>> MDs;
      // Delete out-of-chunk metadata attached to functions.
      F.getAllMetadata(MDs);
      for (std::pair<unsigned, MDNode *> &MD : MDs) {
        if (O.shouldKeep()) {
          NodesToVisit.insert(MD.second);
        } else {
          F.setMetadata(MD.first, nullptr);
        }
      }
    }

    // Delete out-of-chunk metadata attached to instructions.
    for (Instruction &I : instructions(F)) {
      SmallVector<std::pair<unsigned, MDNode *>> MDs;
      I.getAllMetadata(MDs);
      for (std::pair<unsigned, MDNode *> &MD : MDs) {
        if (O.shouldKeep()) {
          NodesToVisit.insert(MD.second);
        } else {
          I.setMetadata(MD.first, nullptr);
        }
      }
    }
  }

  // Gather all metadata tuples and their parents
  SmallVector<std::pair<MDNode *, unsigned>> OperandsOfTuples;
  SmallSet<Metadata *, 8> VisitedNodes;
  while (!NodesToVisit.empty()) {
    auto *Node = NodesToVisit.pop_back_val();
    if (!VisitedNodes.insert(Node).second)
      continue;
    for (auto I : seq<unsigned>(0, Node->getNumOperands())) {
      auto *Op = Node->getOperand(I).get();
      if (auto *MD = dyn_cast_or_null<MDNode>(Op))
        NodesToVisit.insert(MD);
      if (isa_and_nonnull<MDTuple>(Op))
        OperandsOfTuples.push_back(std::make_pair(Node, I));
    }
  }

  // Delete elements from metadata tuples
  for (auto [Node, NodeOpID] : OperandsOfTuples) {
    auto *Tuple = dyn_cast<MDTuple>(Node->getOperand(NodeOpID));
    SmallVector<Metadata *> NewOperands;
    for (auto &Op : Tuple->operands())
      if (O.shouldKeep())
        NewOperands.push_back(Op.get());
    if (NewOperands.size() == Tuple->getNumOperands())
      continue;
    Node->replaceOperandWith(
        NodeOpID, MDTuple::get(Tuple->getContext(), makeArrayRef(NewOperands)));
  }
}

void llvm::reduceMetadataDeltaPass(TestRunner &Test) {
  outs() << "*** Reducing Metadata...\n";
  runDeltaPass(Test, extractMetadataFromModule);
  outs() << "----------------------------\n";
}
