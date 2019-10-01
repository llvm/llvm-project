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
#include "llvm/ADT/SmallVector.h"
#include <set>
#include <vector>

using namespace llvm;

/// Adds all Unnamed Metadata Nodes that are inside desired Chunks to set
template <class T>
static void getChunkMetadataNodes(T &MDUser, int &I,
                                  const std::vector<Chunk> &ChunksToKeep,
                                  std::set<MDNode *> &SeenNodes,
                                  std::set<MDNode *> &NodesToKeep) {
  SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
  MDUser.getAllMetadata(MDs);
  for (auto &MD : MDs) {
    SeenNodes.insert(MD.second);
    if (I < (int)ChunksToKeep.size()) {
      if (ChunksToKeep[I].contains(SeenNodes.size()))
        NodesToKeep.insert(MD.second);
      if (ChunksToKeep[I].end == (int)SeenNodes.size())
        ++I;
    }
  }
}

/// Erases out-of-chunk unnamed metadata nodes from its user
template <class T>
static void eraseMetadataIfOutsideChunk(T &MDUser,
                                        const std::set<MDNode *> &NodesToKeep) {
  SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
  MDUser.getAllMetadata(MDs);
  for (int I = 0, E = MDs.size(); I != E; ++I)
    if (!NodesToKeep.count(MDs[I].second))
      MDUser.setMetadata(I, NULL);
}

/// Removes all the Named and Unnamed Metadata Nodes, as well as any debug
/// functions that aren't inside the desired Chunks.
static void extractMetadataFromModule(const std::vector<Chunk> &ChunksToKeep,
                                      Module *Program) {
  std::set<MDNode *> SeenNodes;
  std::set<MDNode *> NodesToKeep;
  int I = 0;

  // Add chunk MDNodes used by GVs, Functions, and Instructions to set
  for (auto &GV : Program->globals())
    getChunkMetadataNodes(GV, I, ChunksToKeep, SeenNodes, NodesToKeep);

  for (auto &F : *Program) {
    getChunkMetadataNodes(F, I, ChunksToKeep, SeenNodes, NodesToKeep);
    for (auto &BB : F)
      for (auto &Inst : BB)
        getChunkMetadataNodes(Inst, I, ChunksToKeep, SeenNodes, NodesToKeep);
  }

  // Once more, go over metadata nodes, but deleting the ones outside chunks
  for (auto &GV : Program->globals())
    eraseMetadataIfOutsideChunk(GV, NodesToKeep);

  for (auto &F : *Program) {
    eraseMetadataIfOutsideChunk(F, NodesToKeep);
    for (auto &BB : F)
      for (auto &Inst : BB)
        eraseMetadataIfOutsideChunk(Inst, NodesToKeep);
  }


  // Get out-of-chunk Named metadata nodes
  unsigned MetadataCount = SeenNodes.size();
  std::vector<NamedMDNode *> NamedNodesToDelete;
  for (auto &MD : Program->named_metadata()) {
    if (I < (int)ChunksToKeep.size()) {
      if (!ChunksToKeep[I].contains(++MetadataCount))
        NamedNodesToDelete.push_back(&MD);
      if (ChunksToKeep[I].end == (int)SeenNodes.size())
        ++I;
    } else
      NamedNodesToDelete.push_back(&MD);
  }

  for (auto *NN : NamedNodesToDelete) {
    for (int I = 0, E = NN->getNumOperands(); I != E; ++I)
      NN->setOperand(I, NULL);
    NN->eraseFromParent();
  }
}

// Gets unnamed metadata nodes used by a given instruction/GV/function and adds
// them to the set of seen nodes
template <class T>
static void addMetadataToSet(T &MDUser, std::set<MDNode *> &UnnamedNodes) {
  SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
  MDUser.getAllMetadata(MDs);
  for (auto &MD : MDs)
    UnnamedNodes.insert(MD.second);
}

/// Returns the amount of Named and Unnamed Metadata Nodes
static int countMetadataTargets(Module *Program) {
  std::set<MDNode *> UnnamedNodes;
  int NamedMetadataNodes = Program->named_metadata_size();

  // Get metadata nodes used by globals
  for (auto &GV : Program->globals())
    addMetadataToSet(GV, UnnamedNodes);

  // Do the same for nodes used by functions & instructions
  for (auto &F : *Program) {
    addMetadataToSet(F, UnnamedNodes);
    for (auto &BB : F)
      for (auto &I : BB)
        addMetadataToSet(I, UnnamedNodes);
  }

  return UnnamedNodes.size() + NamedMetadataNodes;
}

void llvm::reduceMetadataDeltaPass(TestRunner &Test) {
  outs() << "*** Reducing Metadata...\n";
  int MDCount = countMetadataTargets(Test.getProgram());
  runDeltaPass(Test, MDCount, extractMetadataFromModule);
  outs() << "----------------------------\n";
}
