//===- llvm/CodeGen/BasicBlockMatchingAndInference.cpp ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// In Propeller's profile, we have already read the hash values of basic blocks,
// as well as the weights of basic blocks and edges in the CFG. In this file,
// we first match the basic blocks in the profile with those in the current
// MachineFunction using the basic block hash, thereby obtaining the weights of
// some basic blocks and edges. Subsequently, we infer the weights of all basic
// blocks using an inference algorithm.
//
// TODO: Integrate part of the code in this file with BOLT's implementation into
// the LLVM infrastructure, enabling both BOLT and Propeller to reuse it.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/BasicBlockMatchingAndInference.h"
#include "llvm/CodeGen/BasicBlockSectionsProfileReader.h"
#include "llvm/CodeGen/MachineBlockHashInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"
#include <llvm/Support/CommandLine.h>
#include <unordered_map>

using namespace llvm;

static cl::opt<float>
    PropellerInferThreshold("propeller-infer-threshold",
                            cl::desc("Threshold for infer stale profile"),
                            cl::init(0.6), cl::Optional);

/// The object is used to identify and match basic blocks given their hashes.
class StaleMatcher {
public:
  /// Initialize stale matcher.
  void init(const std::vector<MachineBasicBlock *> &Blocks,
            const std::vector<BlendedBlockHash> &Hashes) {
    assert(Blocks.size() == Hashes.size() &&
           "incorrect matcher initialization");
    for (size_t I = 0; I < Blocks.size(); I++) {
      MachineBasicBlock *Block = Blocks[I];
      uint16_t OpHash = Hashes[I].getOpcodeHash();
      OpHashToBlocks[OpHash].push_back(std::make_pair(Hashes[I], Block));
    }
  }

  /// Find the most similar block for a given hash.
  MachineBasicBlock *matchBlock(BlendedBlockHash BlendedHash) const {
    auto BlockIt = OpHashToBlocks.find(BlendedHash.getOpcodeHash());
    if (BlockIt == OpHashToBlocks.end()) {
      return nullptr;
    }
    MachineBasicBlock *BestBlock = nullptr;
    uint64_t BestDist = std::numeric_limits<uint64_t>::max();
    for (auto It : BlockIt->second) {
      MachineBasicBlock *Block = It.second;
      BlendedBlockHash Hash = It.first;
      uint64_t Dist = Hash.distance(BlendedHash);
      if (BestBlock == nullptr || Dist < BestDist) {
        BestDist = Dist;
        BestBlock = Block;
      }
    }
    return BestBlock;
  }

private:
  using HashBlockPairType = std::pair<BlendedBlockHash, MachineBasicBlock *>;
  std::unordered_map<uint16_t, std::vector<HashBlockPairType>> OpHashToBlocks;
};

INITIALIZE_PASS_BEGIN(BasicBlockMatchingAndInference,
                      "machine-block-match-infer",
                      "Machine Block Matching and Inference Analysis", true,
                      true)
INITIALIZE_PASS_DEPENDENCY(MachineBlockHashInfo)
INITIALIZE_PASS_DEPENDENCY(BasicBlockSectionsProfileReaderWrapperPass)
INITIALIZE_PASS_END(BasicBlockMatchingAndInference, "machine-block-match-infer",
                    "Machine Block Matching and Inference Analysis", true, true)

char BasicBlockMatchingAndInference::ID = 0;

BasicBlockMatchingAndInference::BasicBlockMatchingAndInference()
    : MachineFunctionPass(ID) {}

void BasicBlockMatchingAndInference::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineBlockHashInfo>();
  AU.addRequired<BasicBlockSectionsProfileReaderWrapperPass>();
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

std::optional<BasicBlockMatchingAndInference::WeightInfo>
BasicBlockMatchingAndInference::getWeightInfo(StringRef FuncName) const {
  auto It = ProgramWeightInfo.find(FuncName);
  if (It == ProgramWeightInfo.end()) {
    return std::nullopt;
  }
  return It->second;
}

BasicBlockMatchingAndInference::WeightInfo
BasicBlockMatchingAndInference::initWeightInfoByMatching(MachineFunction &MF) {
  std::vector<MachineBasicBlock *> Blocks;
  std::vector<BlendedBlockHash> Hashes;
  auto BSPR = &getAnalysis<BasicBlockSectionsProfileReaderWrapperPass>();
  auto MBHI = &getAnalysis<MachineBlockHashInfo>();
  for (auto &Block : MF) {
    Blocks.push_back(&Block);
    Hashes.push_back(BlendedBlockHash(MBHI->getMBBHash(Block)));
  }
  StaleMatcher Matcher;
  Matcher.init(Blocks, Hashes);
  BasicBlockMatchingAndInference::WeightInfo MatchWeight;
  const CFGProfile *CFG = BSPR->getFunctionCFGProfile(MF.getName());
  if (CFG == nullptr)
    return MatchWeight;
  for (auto &BlockCount : CFG->NodeCounts) {
    if (CFG->BBHashes.count(BlockCount.first.BaseID)) {
      auto Hash = CFG->BBHashes.lookup(BlockCount.first.BaseID);
      MachineBasicBlock *Block = Matcher.matchBlock(BlendedBlockHash(Hash));
      // When a basic block has clone copies, sum their counts.
      if (Block != nullptr)
        MatchWeight.BlockWeights[Block] += BlockCount.second;
    }
  }
  for (auto &PredItem : CFG->EdgeCounts) {
    auto PredID = PredItem.first.BaseID;
    if (!CFG->BBHashes.count(PredID))
      continue;
    auto PredHash = CFG->BBHashes.lookup(PredID);
    MachineBasicBlock *PredBlock =
        Matcher.matchBlock(BlendedBlockHash(PredHash));
    if (PredBlock == nullptr)
      continue;
    for (auto &SuccItem : PredItem.second) {
      auto SuccID = SuccItem.first.BaseID;
      auto EdgeWeight = SuccItem.second;
      if (CFG->BBHashes.count(SuccID)) {
        auto SuccHash = CFG->BBHashes.lookup(SuccID);
        MachineBasicBlock *SuccBlock =
            Matcher.matchBlock(BlendedBlockHash(SuccHash));
        // When an edge has clone copies, sum their counts.
        if (SuccBlock != nullptr)
          MatchWeight.EdgeWeights[std::make_pair(PredBlock, SuccBlock)] +=
              EdgeWeight;
      }
    }
  }
  return MatchWeight;
}

void BasicBlockMatchingAndInference::generateWeightInfoByInference(
    MachineFunction &MF,
    BasicBlockMatchingAndInference::WeightInfo &MatchWeight) {
  BlockEdgeMap Successors;
  for (auto &Block : MF) {
    for (auto *Succ : Block.successors())
      Successors[&Block].push_back(Succ);
  }
  SampleProfileInference<MachineFunction> SPI(
      MF, Successors, MatchWeight.BlockWeights, MatchWeight.EdgeWeights);
  BlockWeightMap BlockWeights;
  EdgeWeightMap EdgeWeights;
  SPI.apply(BlockWeights, EdgeWeights);
  ProgramWeightInfo.try_emplace(
      MF.getName(), BasicBlockMatchingAndInference::WeightInfo{
                        std::move(BlockWeights), std::move(EdgeWeights)});
}

bool BasicBlockMatchingAndInference::runOnMachineFunction(MachineFunction &MF) {
  if (MF.empty())
    return false;
  auto MatchWeight = initWeightInfoByMatching(MF);
  // If the ratio of the number of MBBs in matching to the total number of MBBs
  // in the function is less than the threshold value, the processing should be
  // abandoned.
  if (static_cast<float>(MatchWeight.BlockWeights.size()) / MF.size() <
      PropellerInferThreshold) {
    return false;
  }
  generateWeightInfoByInference(MF, MatchWeight);
  return false;
}

MachineFunctionPass *llvm::createBasicBlockMatchingAndInferencePass() {
  return new BasicBlockMatchingAndInference();
}
