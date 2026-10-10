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
#include "CodeGenOptions.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/CodeGen/BasicBlockSectionsProfileReader.h"
#include "llvm/CodeGen/MachineBlockHashInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/ProfileData/SampleProf.h"
#include "llvm/ProfileData/SampleProfReader.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace llvm;

#define DEBUG_TYPE "machine-block-match-infer"

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
  DenseMap<uint16_t, std::vector<HashBlockPairType>> OpHashToBlocks;
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

namespace {
/// Looks up the sampled weights of the instructions of one machine function in
/// a line-based sample profile. This is the lookup part of
/// SampleProfileLoaderBaseImpl without its weight propagation, remarks and
/// coverage tracking, which are not needed to supplement the Propeller profile.
class FunctionSamplesQuery {
public:
  FunctionSamplesQuery(const FunctionSamples &Samples,
                       SampleProfileReaderItaniumRemapper *Remapper)
      : Samples(Samples), Remapper(Remapper) {}

  /// Returns the weight of \p MI, or an error if it has no samples.
  ErrorOr<uint64_t> getInstWeight(const MachineInstr &MI) {
    const DILocation *DIL = MI.getDebugLoc();
    if (!DIL)
      return std::error_code();
    // Find the samples of the inlined instance MI comes from.
    auto [It, Inserted] = InlinedSamplesCache.try_emplace(DIL, nullptr);
    if (Inserted)
      It->second = Samples.findFunctionSamples(DIL, Remapper);
    if (!It->second)
      return std::error_code();
    uint32_t Discriminator = EnableFSDiscriminator
                                 ? DIL->getDiscriminator()
                                 : DIL->getBaseDiscriminator();
    return It->second->findSamplesAt(FunctionSamples::getOffset(DIL),
                                     Discriminator);
  }

  /// Returns the maximum weight among the non-meta instructions of \p MBB, or
  /// an error if none of them has samples.
  ErrorOr<uint64_t> getBlockWeight(const MachineBasicBlock &MBB) {
    uint64_t Max = 0;
    bool HasWeight = false;
    for (const MachineInstr &MI : MBB) {
      if (MI.isMetaInstruction())
        continue;
      if (ErrorOr<uint64_t> W = getInstWeight(MI)) {
        Max = std::max(Max, *W);
        HasWeight = true;
      }
    }
    if (!HasWeight)
      return std::error_code();
    return Max;
  }

private:
  const FunctionSamples &Samples;
  SampleProfileReaderItaniumRemapper *Remapper;
  // Samples of the inlined instance each location comes from.
  DenseMap<const DILocation *, const FunctionSamples *> InlinedSamplesCache;
};
} // namespace

BasicBlockMatchingAndInference::BasicBlockMatchingAndInference()
    : MachineFunctionPass(ID) {}

BasicBlockMatchingAndInference::~BasicBlockMatchingAndInference() = default;

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

BasicBlockMatchingAndInference::WeightInfo
BasicBlockMatchingAndInference::initWeightInfoByBBID(MachineFunction &MF,
                                                     const CFGProfile &CFG) {
  // Match blocks by their full UniqueBBID (including CloneID). If a cloned
  // block in the profile does not exist in MF (e.g., because its cloning path
  // was rejected), fall back to the original block so its counts are
  // attributed to the original block.
  DenseMap<UniqueBBID, MachineBasicBlock *> BBIDToMBB;
  for (auto &MBB : MF)
    BBIDToMBB[*MBB.getBBID()] = &MBB;
  auto FindMBB = [&](const UniqueBBID &BBID) {
    if (MachineBasicBlock *MBB = BBIDToMBB.lookup(BBID))
      return MBB;
    return BBIDToMBB.lookup(UniqueBBID{BBID.BaseID, 0});
  };

  // A basic block is matched only if it exists in CFG.NodeCounts and all of its
  // edges in the CFG profile are also edges in the MIR. The profile may omit
  // edges without samples, so MIR edges missing from the profile are allowed.
  // Both ends of a profile edge that is not in the MIR are kept as unknown.
  DenseSet<const MachineBasicBlock *> MismatchedMBBs;
  for (const auto &[PredID, Succs] : CFG.EdgeCounts) {
    MachineBasicBlock *PredMBB = FindMBB(PredID);
    for (const auto &[SuccID, EdgeCount] : Succs) {
      MachineBasicBlock *SuccMBB = FindMBB(SuccID);
      if (PredMBB && SuccMBB && PredMBB->isSuccessor(SuccMBB))
        continue;
      if (PredMBB)
        MismatchedMBBs.insert(PredMBB);
      if (SuccMBB)
        MismatchedMBBs.insert(SuccMBB);
    }
  }
  DenseSet<const MachineBasicBlock *> ProfiledMBBs;
  for (const auto &[BBID, BlockCount] : CFG.NodeCounts)
    if (MachineBasicBlock *MBB = FindMBB(BBID))
      ProfiledMBBs.insert(MBB);
  DenseSet<const MachineBasicBlock *> MatchedMBBs;
  for (auto &MBB : MF) {
    if (!ProfiledMBBs.contains(&MBB))
      continue;
    if (MismatchedMBBs.contains(&MBB))
      LLVM_DEBUG(dbgs() << MF.getName() << ": bb." << MBB.getNumber()
                        << " has profile edges that are not in the CFG\n");
    else
      MatchedMBBs.insert(&MBB);
  }

  BasicBlockMatchingAndInference::WeightInfo MatchWeight;
  for (const auto &[BBID, BlockCount] : CFG.NodeCounts) {
    MachineBasicBlock *MBB = FindMBB(BBID);
    if (MBB && MatchedMBBs.count(MBB))
      MatchWeight.BlockWeights[MBB] += BlockCount;
  }
  for (const auto &[PredID, Succs] : CFG.EdgeCounts) {
    MachineBasicBlock *PredMBB = FindMBB(PredID);
    if (!PredMBB || !MatchedMBBs.count(PredMBB))
      continue;
    for (const auto &[SuccID, EdgeCount] : Succs) {
      MachineBasicBlock *SuccMBB = FindMBB(SuccID);
      if (!SuccMBB || !MatchedMBBs.count(SuccMBB))
        continue;
      MatchWeight.EdgeWeights[std::make_pair(PredMBB, SuccMBB)] += EdgeCount;
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

bool BasicBlockMatchingAndInference::doInitialization(Module &M) {
  const CodeGenOptions &Opts = getCodeGenOptions(M);
  StringRef ProfileFile = Opts.sample_profile_supplement;
  // Only the weights matched by BBIDs are supplemented.
  if (ProfileFile.empty() || !Opts.propeller_match_bbid)
    return false;
  LLVMContext &Ctx = M.getContext();
  if (ProfileSummaryInfo(M).hasSampleProfile() ||
      !Opts.fs_profile_file.empty()) {
    Ctx.diagnose(DiagnosticInfoSampleProfile(
        ProfileFile,
        "Supplemental sample profiles are not supported with SampleFDO",
        DS_Warning));
    return false;
  }
  auto ReaderOrErr = SampleProfileReader::create(
      ProfileFile, Ctx, *vfs::getRealFileSystem(),
      EnableFSDiscriminator ? FSDiscriminatorPass::PassLast
                            : FSDiscriminatorPass::Base,
      Opts.sample_profile_supplement_remapping_file);
  if (std::error_code EC = ReaderOrErr.getError()) {
    Ctx.diagnose(DiagnosticInfoSampleProfile(
        ProfileFile,
        "Could not open supplemental sample profile: " + EC.message()));
    return false;
  }
  std::unique_ptr<SampleProfileReader> Reader = std::move(*ReaderOrErr);
  Reader->setModule(&M);
  if (Reader->read() != sampleprof_error::success) {
    Ctx.diagnose(DiagnosticInfoSampleProfile(
        ProfileFile, "Could not read supplemental sample profile"));
    return false;
  }
  if (Reader->profileIsProbeBased()) {
    Ctx.diagnose(DiagnosticInfoSampleProfile(
        ProfileFile,
        "Pseudo probe based supplemental sample profiles are not supported",
        DS_Warning));
    return false;
  }
  if (Reader->profileIsCS()) {
    Ctx.diagnose(DiagnosticInfoSampleProfile(
        ProfileFile,
        "Context-sensitive supplemental sample profiles are not supported",
        DS_Warning));
    return false;
  }
  if (Reader->profileIsFS() && !EnableFSDiscriminator) {
    Ctx.diagnose(DiagnosticInfoSampleProfile(
        ProfileFile,
        "Flow-sensitive supplemental sample profiles require "
        "-enable-fs-discriminator",
        DS_Warning));
    return false;
  }
  SampleReader = std::move(Reader);
  return false;
}

void BasicBlockMatchingAndInference::supplementWithSampleProfile(
    MachineFunction &MF, WeightInfo &MatchWeight) {
  if (!SampleReader)
    return;
  const FunctionSamples *Samples =
      SampleReader->getSamplesFor(MF.getFunction());
  if (!Samples || Samples->empty())
    return;
  FunctionSamplesQuery Query(*Samples, SampleReader->getRemapper());

  // Scale the weights of the supplemental profile so that the entry block
  // weights of both profiles agree.
  const MachineBasicBlock &EntryBB = MF.front();
  uint64_t PropellerEntryWeight =
      std::max<uint64_t>(MatchWeight.BlockWeights.lookup(&EntryBB), 1);
  uint64_t SampleEntryWeight = 0;
  if (ErrorOr<uint64_t> W = Query.getBlockWeight(EntryBB))
    SampleEntryWeight = *W;
  if (SampleEntryWeight == 0)
    SampleEntryWeight = Samples->getHeadSamples();
  SampleEntryWeight = std::max<uint64_t>(SampleEntryWeight, 1);

  for (const MachineBasicBlock &MBB : MF) {
    // Only supplement blocks that have no samples in the Propeller profile, so
    // that code executed in production is not assumed cold prior to inference.
    // Blocks that already have Propeller samples keep their Propeller weight.
    if (MatchWeight.BlockWeights.lookup(&MBB) != 0)
      continue;
    ErrorOr<uint64_t> SampleWeight = Query.getBlockWeight(MBB);
    if (!SampleWeight)
      continue;
    // Skip weights that round to zero, so that blocks missing from
    // BlockWeights stay unknown to inference rather than known to be cold.
    uint64_t Weight =
        mulDivNearest(*SampleWeight, PropellerEntryWeight, SampleEntryWeight);
    if (Weight != 0)
      MatchWeight.BlockWeights[&MBB] = Weight;
  }
}

bool BasicBlockMatchingAndInference::doFinalization(Module &M) {
  // The profile is read for each module.
  SampleReader.reset();
  return false;
}

bool BasicBlockMatchingAndInference::runOnMachineFunction(MachineFunction &MF) {
  if (MF.empty())
    return false;
  auto *BSPR = &getAnalysis<BasicBlockSectionsProfileReaderWrapperPass>();
  const CFGProfile *CFG = BSPR->getFunctionCFGProfile(MF.getName());
  if (CFG == nullptr || (CFG->NodeCounts.empty() && CFG->EdgeCounts.empty()))
    return false;

  const CodeGenOptions &Opts = getCodeGenOptions(MF);
  BasicBlockMatchingAndInference::WeightInfo MatchWeight;
  if (Opts.propeller_match_bbid) {
    MatchWeight = initWeightInfoByBBID(MF, *CFG);
    supplementWithSampleProfile(MF, MatchWeight);
  } else {
    MatchWeight = initWeightInfoByMatching(MF);
    // If the ratio of the number of MBBs in matching to the total number of
    // MBBs in the function is less than the threshold value, the processing
    // should be abandoned.
    if (static_cast<float>(MatchWeight.BlockWeights.size()) / MF.size() <
        Opts.propeller_infer_threshold) {
      return false;
    }
  }
  generateWeightInfoByInference(MF, MatchWeight);
  return false;
}

MachineFunctionPass *llvm::createBasicBlockMatchingAndInferencePass() {
  return new BasicBlockMatchingAndInference();
}
