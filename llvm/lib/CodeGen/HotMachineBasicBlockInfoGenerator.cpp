#include "llvm/CodeGen/HotMachineBasicBlockInfoGenerator.h"
#include "llvm/InitializePasses.h"
#include "llvm/CodeGen/MachineBlockHashInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Transforms/Utils/CodeLayout.h"
#include "llvm/Target/TargetMachine.h"
#include <unordered_set>
#include <llvm/Support/CommandLine.h>

using namespace llvm;

static cl::opt<bool> PropellerMatchInfer("propeller-match-infer", 
    cl::desc("Use match&infer to evaluate stale profile"), cl::init(false), cl::Optional);

static cl::opt<float> PropellerInferThreshold("propeller-infer-threshold", 
    cl::desc("Threshold for infer stale profile"), cl::init(0.6), cl::Optional);    

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
      uint16_t OpHash = Hashes[I].OpcodeHash;
      OpHashToBlocks[OpHash].push_back(std::make_pair(Hashes[I], Block));
    }
  }

  /// Find the most similar block for a given hash.
  MachineBasicBlock *matchBlock(BlendedBlockHash BlendedHash) const {
    auto BlockIt = OpHashToBlocks.find(BlendedHash.OpcodeHash);
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

INITIALIZE_PASS_BEGIN(HotMachineBasicBlockInfoGenerator, "machine-block-match-infer",
                      "Machine Block Matching and Inference Analysis", true, true)
INITIALIZE_PASS_DEPENDENCY(MachineBlockHashInfo)
INITIALIZE_PASS_DEPENDENCY(FuncHotBBHashesProfileReader)               
INITIALIZE_PASS_END(HotMachineBasicBlockInfoGenerator, "machine-block-match-infer",
                    "Machine Block Matching and Inference Analysis", true, true)

char HotMachineBasicBlockInfoGenerator::ID = 0;

HotMachineBasicBlockInfoGenerator::HotMachineBasicBlockInfoGenerator() : MachineFunctionPass(ID) {
    initializeHotMachineBasicBlockInfoGeneratorPass(*PassRegistry::getPassRegistry());
}

void HotMachineBasicBlockInfoGenerator::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineBlockHashInfo>();
  AU.addRequired<FuncHotBBHashesProfileReader>();
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

std::optional<SmallVector<MachineBasicBlock *, 4>> 
HotMachineBasicBlockInfoGenerator::getHotMBBs(StringRef FuncName) const {
  auto It = FuncToHotMBBs.find(FuncName);
  if (It == FuncToHotMBBs.end()) {
    return std::nullopt;
  }
  return It->second;
}
    
void HotMachineBasicBlockInfoGenerator::matchHotBBsByHashes(
    MachineFunction &MF, 
    SmallVector<HotBBInfo, 4> &HotMBBInfos,
    BlockWeightMap &MBBToFreq, 
    BlockEdgeMap &Successors,
    SmallVector<MachineBasicBlock *, 4> &HotBBs) {
  std::vector<MachineBasicBlock *> Blocks;
  std::vector<BlendedBlockHash> Hashes;
  auto MBHI = &getAnalysis<MachineBlockHashInfo>();
  for (auto &Block : MF) {
    Blocks.push_back(&Block);
    Hashes.push_back(BlendedBlockHash(MBHI->getMBBHash(Block)));
    for (auto *Succ : Block.successors()) {
      Successors[&Block].push_back(Succ);
    }
  }
  StaleMatcher Matcher;
  Matcher.init(Blocks, Hashes);
  for (auto &item : HotMBBInfos) {
    MachineBasicBlock *Block 
        = Matcher.matchBlock(BlendedBlockHash(item.BBHash));
    if (Block != nullptr) {
      HotBBs.push_back(Block);
      MBBToFreq[Block] = item.Freq;
    }
  }
}

void HotMachineBasicBlockInfoGenerator::generateHotBBsforFunction(
    MachineFunction &MF,
    BlockWeightMap &OriBlockWeights,
    BlockWeightMap &BlockWeights, 
    EdgeWeightMap &EdgeWeights,
    SmallVector<MachineBasicBlock *, 4> &HotBBs) {
  if (!PropellerMatchInfer) {
    for (auto MBB : HotBBs) {
      if (MBB->isEntryBlock() || OriBlockWeights[MBB] > 0) {
        FuncToHotMBBs[MF.getName()].push_back(MBB);
      }
    }
    return;
  }
 
  if (MF.size() <= 2) {
    for (auto &MBB : MF) {
      if (MBB.isEntryBlock() || BlockWeights[&MBB] > 0) {
        FuncToHotMBBs[MF.getName()].push_back(&MBB);
      }
    }
    return;
  }

  MF.RenumberBlocks();

  SmallVector<uint64_t, 0> BlockSizes(MF.size());
  SmallVector<uint64_t, 0> BlockCounts(MF.size());
  std::vector<MachineBasicBlock *> OrigOrder;
  OrigOrder.reserve(MF.size());
  SmallVector<codelayout::EdgeCount, 0> JumpCounts;

  // Init the MBB size and count.
  for (auto &MBB : MF) {
    auto NonDbgInsts =
        instructionsWithoutDebug(MBB.instr_begin(), MBB.instr_end());
    int NumInsts = std::distance(NonDbgInsts.begin(), NonDbgInsts.end());
    BlockSizes[MBB.getNumber()] = 4 * NumInsts;
    BlockCounts[MBB.getNumber()] = BlockWeights[&MBB];
    OrigOrder.push_back(&MBB);
  }
  
  // Init the edge count.
  for (auto &MBB : MF) {
    for (auto *Succ : MBB.successors()) {
      auto EdgeWeight = EdgeWeights[std::make_pair(&MBB, Succ)];
      JumpCounts.push_back({static_cast<uint64_t>(MBB.getNumber()), static_cast<uint64_t>(Succ->getNumber()), EdgeWeight});
    }
  }
  
  // Run the layout algorithm
  auto Result = computeExtTspLayout(BlockSizes, BlockCounts, JumpCounts);
  for (uint64_t R : Result) {
    auto Block = OrigOrder[R];
    if (Block->isEntryBlock() || BlockWeights[Block] > 0)
      FuncToHotMBBs[MF.getName()].push_back(Block);
  }
}

bool HotMachineBasicBlockInfoGenerator::runOnMachineFunction(MachineFunction &MF) {
  auto [FindFlag, HotMBBInfos] = 
      getAnalysis<FuncHotBBHashesProfileReader>()
      .getHotBBInfosForFunction(MF.getName());
  if (!FindFlag) {
    return false;
  }
  BlockWeightMap MBBToFreq;
  BlockEdgeMap Successors;
  SmallVector<MachineBasicBlock *, 4> HotBBs;
  matchHotBBsByHashes(MF, HotMBBInfos, MBBToFreq, Successors, HotBBs);

  // If the ratio of the number of MBBs in matching to the total number of MBBs in the 
  // function is less than the threshold value, the processing should be abandoned.
  if (static_cast<float>(HotBBs.size()) / MF.size() < PropellerInferThreshold) {
    return false;
  }

  SampleProfileInference<MachineFunction> SPI(MF, Successors, MBBToFreq);
  BlockWeightMap BlockWeights;
  EdgeWeightMap EdgeWeights;
  SPI.apply(BlockWeights, EdgeWeights);
  generateHotBBsforFunction(MF, MBBToFreq, BlockWeights, EdgeWeights, HotBBs);
  return false;
}

MachineFunctionPass *llvm::createHotMachineBasicBlockInfoGeneratorPass() {
  return new HotMachineBasicBlockInfoGenerator();
}
