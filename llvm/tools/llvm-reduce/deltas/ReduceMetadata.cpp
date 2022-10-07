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
#include "llvm/IR/IntrinsicInst.h"
#include <vector>

using namespace llvm;

static bool shouldKeepDebugIntrinsicMetadata(Instruction &I, MDNode &MD) {
  return isa<DILocation>(MD) && isa<DbgInfoIntrinsic>(I);
}

static bool shouldKeepDebugNamedMetadata(NamedMDNode &MD) {
  return MD.getName() == "llvm.dbg.cu" && MD.getNumOperands() != 0;
}

/// Removes all the Named and Unnamed Metadata Nodes, as well as any debug
/// functions that aren't inside the desired Chunks.
static void extractMetadataFromModule(Oracle &O, Module &Program) {
  // Get out-of-chunk Named metadata nodes
  SmallVector<NamedMDNode *> NamedNodesToDelete;
  for (NamedMDNode &MD : Program.named_metadata())
    if (!shouldKeepDebugNamedMetadata(MD) && !O.shouldKeep())
      NamedNodesToDelete.push_back(&MD);

  for (NamedMDNode *NN : NamedNodesToDelete) {
    for (auto I : seq<unsigned>(0, NN->getNumOperands()))
      NN->setOperand(I, nullptr);
    NN->eraseFromParent();
  }

  // Delete out-of-chunk metadata attached to globals.
  for (GlobalVariable &GV : Program.globals()) {
    SmallVector<std::pair<unsigned, MDNode *>> MDs;
    GV.getAllMetadata(MDs);
    for (std::pair<unsigned, MDNode *> &MD : MDs)
      if (!O.shouldKeep())
        GV.setMetadata(MD.first, nullptr);
  }

  for (Function &F : Program) {
    {
      SmallVector<std::pair<unsigned, MDNode *>> MDs;
      // Delete out-of-chunk metadata attached to functions.
      F.getAllMetadata(MDs);
      for (std::pair<unsigned, MDNode *> &MD : MDs)
        if (!O.shouldKeep())
          F.setMetadata(MD.first, nullptr);
    }

    // Delete out-of-chunk metadata attached to instructions.
    for (Instruction &I : instructions(F)) {
      SmallVector<std::pair<unsigned, MDNode *>> MDs;
      I.getAllMetadata(MDs);
      for (std::pair<unsigned, MDNode *> &MD : MDs) {
        if (!shouldKeepDebugIntrinsicMetadata(I, *MD.second) && !O.shouldKeep())
          I.setMetadata(MD.first, nullptr);
      }
    }
  }
}

void llvm::reduceMetadataDeltaPass(TestRunner &Test) {
  outs() << "*** Reducing Metadata...\n";
  runDeltaPass(Test, extractMetadataFromModule);
  outs() << "----------------------------\n";
}
