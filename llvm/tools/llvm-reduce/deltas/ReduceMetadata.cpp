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
  // Get out-of-chunk Named metadata nodes
  SmallVector<NamedMDNode *> NamedNodesToDelete;
  for (NamedMDNode &MD : Program.named_metadata())
    if (!O.shouldKeep())
      NamedNodesToDelete.push_back(&MD);

  for (NamedMDNode *NN : NamedNodesToDelete) {
    for (auto I : seq<unsigned>(0, NN->getNumOperands()))
      NN->setOperand(I, NULL);
    NN->eraseFromParent();
  }

  // Delete out-of-chunk metadata attached to globals.
  SmallVector<std::pair<unsigned, MDNode *>> MDs;
  for (GlobalVariable &GV : Program.globals()) {
    GV.getAllMetadata(MDs);
    for (std::pair<unsigned, MDNode *> &MD : MDs)
      if (!O.shouldKeep())
        GV.setMetadata(MD.first, NULL);
  }

  for (Function &F : Program) {
    // Delete out-of-chunk metadata attached to functions.
    F.getAllMetadata(MDs);
    for (std::pair<unsigned, MDNode *> &MD : MDs)
      if (!O.shouldKeep())
        F.setMetadata(MD.first, NULL);

    // Delete out-of-chunk metadata attached to instructions.
    for (Instruction &I : instructions(F)) {
      I.getAllMetadata(MDs);
      for (std::pair<unsigned, MDNode *> &MD : MDs)
        if (!O.shouldKeep())
          I.setMetadata(MD.first, NULL);
    }
  }
}

static int countMetadataTargets(Module &Program) {
  int NamedMetadataNodes = Program.named_metadata_size();

  // Get metadata attached to globals.
  int GlobalMetadataArgs = 0;
  SmallVector<std::pair<unsigned, MDNode *>> MDs;
  for (GlobalVariable &GV : Program.globals()) {
    GV.getAllMetadata(MDs);
    GlobalMetadataArgs += MDs.size();
  }

  // Get metadata attached to functions & instructions.
  int FunctionMetadataArgs = 0;
  int InstructionMetadataArgs = 0;
  for (Function &F : Program) {
    F.getAllMetadata(MDs);
    FunctionMetadataArgs += MDs.size();

    for (Instruction &I : instructions(F)) {
      I.getAllMetadata(MDs);
      InstructionMetadataArgs += MDs.size();
    }
  }

  return NamedMetadataNodes + GlobalMetadataArgs + FunctionMetadataArgs +
         InstructionMetadataArgs;
}

void llvm::reduceMetadataDeltaPass(TestRunner &Test) {
  outs() << "*** Reducing Metadata...\n";
  int MDCount = countMetadataTargets(Test.getProgram());
  runDeltaPass(Test, MDCount, extractMetadataFromModule);
  outs() << "----------------------------\n";
}
