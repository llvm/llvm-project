//===- IndirectBrExpandPass.cpp - Expand indirectbr to switch -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// Implements an expansion pass to turn `indirectbr` instructions in the IR
/// into `switch` instructions. This works by enumerating the basic blocks in
/// a dense range of integers, replacing each `blockaddr` constant with the
/// corresponding integer constant, and then building a switch that maps from
/// the integers to the actual blocks. All of the indirectbr instructions in the
/// function are redirected to this common switch.
///
/// While this is generically useful if a target is unable to codegen
/// `indirectbr` natively, it is primarily useful when there is some desire to
/// get the builtin non-jump-table lowering of a switch even when the input
/// source contained an explicit indirect branch construct.
///
/// Note that it doesn't make any sense to enable this pass unless a target also
/// disables jump-table lowering of switches. Doing that is likely to pessimize
/// the code.
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LazyBlockFrequencyInfo.h"
#include "llvm/CodeGen/IndirectBrExpand.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ScaledNumber.h"
#include "llvm/Target/TargetMachine.h"
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "indirectbr-expand"

namespace llvm {
extern cl::opt<bool> ProfcheckDisableMetadataFixes;
} // namespace llvm

namespace {

class IndirectBrExpandLegacyPass : public FunctionPass {
  CodeGenOptLevel OptLevel;

public:
  static char ID; // Pass identification, replacement for typeid

  IndirectBrExpandLegacyPass(CodeGenOptLevel OptLevel)
      : FunctionPass(ID), OptLevel(OptLevel) {}

  IndirectBrExpandLegacyPass()
      : IndirectBrExpandLegacyPass(CodeGenOptLevel::None) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    if (OptLevel != CodeGenOptLevel::None)
      LazyBlockFrequencyInfoPass::getLazyBFIAnalysisUsage(AU);
    AU.addPreserved<DominatorTreeWrapperPass>();
  }

  bool runOnFunction(Function &F) override;
};

} // end anonymous namespace

static bool runImpl(Function &F, const TargetLowering *TLI, DomTreeUpdater *DTU,
                    function_ref<BlockFrequencyInfo *()> GetBFI,
                    bool PreserveProfile);

PreservedAnalyses IndirectBrExpandPass::run(Function &F,
                                            FunctionAnalysisManager &FAM) {
  auto *STI = TM->getSubtargetImpl(F);
  if (!STI->enableIndirectBrExpand())
    return PreservedAnalyses::all();

  auto *TLI = STI->getTargetLowering();
  auto *DT = FAM.getCachedResult<DominatorTreeAnalysis>(F);
  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Lazy);

  bool Changed = runImpl(
      F, TLI, DT ? &DTU : nullptr,
      [&]() { return &FAM.getResult<BlockFrequencyAnalysis>(F); },
      /*PreserveProfile=*/true);
  if (!Changed)
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  return PA;
}

char IndirectBrExpandLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(IndirectBrExpandLegacyPass, DEBUG_TYPE,
                      "Expand indirectbr instructions", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(IndirectBrExpandLegacyPass, DEBUG_TYPE,
                    "Expand indirectbr instructions", false, false)

FunctionPass *llvm::createIndirectBrExpandPass(CodeGenOptLevel OptLevel) {
  return new IndirectBrExpandLegacyPass(OptLevel);
}

bool runImpl(Function &F, const TargetLowering *TLI, DomTreeUpdater *DTU,
             function_ref<BlockFrequencyInfo *()> GetBFI,
             bool PreserveProfile) {
  auto &DL = F.getDataLayout();

  SmallVector<IndirectBrInst *, 1> IndirectBrs;
  SmallVector<uint64_t, 1> IndirectBrsBlockFrequencies;
  SmallVector<uint64_t, 1> IndirectBrsBranchWeightSums;
  bool SkipProfileUpdates = !PreserveProfile;
  BlockFrequencyInfo *BFI = nullptr;

  struct IndirectBrSuccessor {
    // The index into the IndirectBrs, IndirectBrsBlockFrequencies, and
    // IndirectBrsBranchWeightSums vectors.
    size_t IndirectBrIndex = 0;
    uint64_t SuccessorBranchWeight = 0;
  };

  // Set of all potential successors for indirectbr instructions.
  DenseMap<const BasicBlock *, SmallVector<IndirectBrSuccessor>>
      IndirectBrSuccToIndirectBr;

  // Build a list of indirectbrs that we want to rewrite.
  for (BasicBlock &BB : F)
    if (auto *IBr = dyn_cast<IndirectBrInst>(BB.getTerminator())) {
      // Handle the degenerate case of no successors by replacing the indirectbr
      // with unreachable as there is no successor available.
      if (IBr->getNumSuccessors() == 0) {
        (void)new UnreachableInst(F.getContext(), IBr->getIterator());
        IBr->eraseFromParent();
        continue;
      }

      IndirectBrs.push_back(IBr);
      const size_t CurrentIndirectBrIndex = IndirectBrs.size() - 1;
      for (const BasicBlock *SuccessorBB : IBr->successors())
        IndirectBrSuccToIndirectBr.insert({SuccessorBB, {}});

      if (SkipProfileUpdates)
        continue;
      if (!BFI)
        BFI = GetBFI();
      std::optional<uint64_t> BlockFrequency = BFI->getBlockProfileCount(&BB);
      if (!BlockFrequency.has_value()) {
        SkipProfileUpdates = true;
        continue;
      }
      IndirectBrsBlockFrequencies.push_back(*BlockFrequency);
      SmallVector<uint32_t> IndirectBrBranchWeights;
      bool HasBranchWeights =
          extractBranchWeights(*IBr, IndirectBrBranchWeights);
      if (!HasBranchWeights) {
        SkipProfileUpdates = true;
        continue;
      }
      for (const auto [SuccessorBB, SuccessorBranchWeight] :
           zip_equal(IBr->successors(), IndirectBrBranchWeights))
        IndirectBrSuccToIndirectBr[SuccessorBB].push_back(
            {CurrentIndirectBrIndex, SuccessorBranchWeight});
      IndirectBrsBranchWeightSums.push_back(sum_of(IndirectBrBranchWeights));
      assert(IndirectBrsBranchWeightSums.size() == IndirectBrs.size() &&
             "expected an identical number of blocks in both vectors");
    }

  if (IndirectBrs.empty())
    return false;

  // If we need to replace any indirectbrs we need to establish integer
  // constants that will correspond to each of the basic blocks in the function
  // whose address escapes. We do that here and rewrite all the blockaddress
  // constants to just be those integer constants cast to a pointer type.
  SmallVector<BasicBlock *, 4> BBs;
  SmallVector<ScaledNumber<uint64_t>, 4> BBWeights;

  for (BasicBlock &BB : F) {
    // Skip blocks that aren't successors to an indirectbr we're going to
    // rewrite.
    auto IndirectBrSuccToIndirectBrIt = IndirectBrSuccToIndirectBr.find(&BB);
    if (IndirectBrSuccToIndirectBrIt == IndirectBrSuccToIndirectBr.end())
      continue;

    auto *BA = BlockAddress::lookup(&BB);

    // Skip if the constant was formed but ended up not being used (due to DCE
    // or whatever).
    if (!BA || !BA->isConstantUsed())
      continue;

    // Compute the index we want to use for this basic block. We can't use zero
    // because null can be compared with block addresses.
    int BBIndex = BBs.size() + 1;
    BBs.push_back(&BB);

    auto *ITy = cast<IntegerType>(DL.getIntPtrType(BA->getType()));
    ConstantInt *BBIndexC = ConstantInt::get(ITy, BBIndex);

    // Now rewrite the blockaddress to an integer constant based on the index.
    // FIXME: This part doesn't properly recognize other uses of blockaddress
    // expressions, for instance, where they are used to pass labels to
    // asm-goto. This part of the pass needs a rework.
    BA->replaceAllUsesWith(ConstantExpr::getIntToPtr(BBIndexC, BA->getType()));

    if (SkipProfileUpdates)
      continue;
    ScaledNumber<uint64_t> BranchWeightSumsProduct(1, 0);
    for (uint64_t BranchWeightSum : IndirectBrsBranchWeightSums)
      BranchWeightSumsProduct *= ScaledNumber<uint64_t>(BranchWeightSum, 0);
    ScaledNumber<uint64_t> BlockWeight(0, 0);
    for (const auto &[IndirectBrIndex, BlockBranchProbability] :
         IndirectBrSuccToIndirectBrIt->second) {
      // If the branch weight sum is zero, skip adding the block weight or
      // otherwise we end up dividing by zero.
      const uint64_t CurrentBranchWeightSum =
          IndirectBrsBranchWeightSums[IndirectBrIndex];
      if (CurrentBranchWeightSum == 0)
        continue;
      BlockWeight += ScaledNumber<uint64_t>(
                         IndirectBrsBlockFrequencies[IndirectBrIndex], 0) *
                     ScaledNumber<uint64_t>(BlockBranchProbability, 0) *
                     (BranchWeightSumsProduct /
                      ScaledNumber<uint64_t>(CurrentBranchWeightSum, 0));
    }
    BBWeights.push_back(BlockWeight);
  }

  if (BBs.empty()) {
    // There are no blocks whose address is taken, so any indirectbr instruction
    // cannot get a valid input and we can replace all of them with unreachable.
    SmallVector<DominatorTree::UpdateType, 8> Updates;
    if (DTU)
      Updates.reserve(IndirectBrSuccToIndirectBr.size());
    for (auto *IBr : IndirectBrs) {
      if (DTU) {
        for (BasicBlock *SuccBB : IBr->successors())
          Updates.push_back({DominatorTree::Delete, IBr->getParent(), SuccBB});
      }
      (void)new UnreachableInst(F.getContext(), IBr->getIterator());
      IBr->eraseFromParent();
    }
    if (DTU) {
      assert(Updates.size() == IndirectBrSuccToIndirectBr.size() &&
             "Got unexpected update count.");
      DTU->applyUpdates(Updates);
    }
    return true;
  }

  BasicBlock *SwitchBB;
  Value *SwitchValue;

  // Compute a common integer type across all the indirectbr instructions.
  IntegerType *CommonITy = nullptr;
  for (auto *IBr : IndirectBrs) {
    auto *ITy =
        cast<IntegerType>(DL.getIntPtrType(IBr->getAddress()->getType()));
    if (!CommonITy || ITy->getBitWidth() > CommonITy->getBitWidth())
      CommonITy = ITy;
  }

  auto GetSwitchValue = [CommonITy](IndirectBrInst *IBr) {
    return CastInst::CreatePointerCast(IBr->getAddress(), CommonITy,
                                       Twine(IBr->getAddress()->getName()) +
                                           ".switch_cast",
                                       IBr->getIterator());
  };

  SmallVector<DominatorTree::UpdateType, 8> Updates;

  if (IndirectBrs.size() == 1) {
    // If we only have one indirectbr, we can just directly replace it within
    // its block.
    IndirectBrInst *IBr = IndirectBrs[0];
    SwitchBB = IBr->getParent();
    SwitchValue = GetSwitchValue(IBr);
    if (DTU) {
      Updates.reserve(IndirectBrSuccToIndirectBr.size());
      for (BasicBlock *SuccBB : IBr->successors())
        Updates.push_back({DominatorTree::Delete, IBr->getParent(), SuccBB});
      assert(Updates.size() == IndirectBrSuccToIndirectBr.size() &&
             "Got unexpected update count.");
    }
    IBr->eraseFromParent();
  } else {
    // Otherwise we need to create a new block to hold the switch across BBs,
    // jump to that block instead of each indirectbr, and phi together the
    // values for the switch.
    SwitchBB = BasicBlock::Create(F.getContext(), "switch_bb", &F);
    auto *SwitchPN = PHINode::Create(CommonITy, IndirectBrs.size(),
                                     "switch_value_phi", SwitchBB);
    SwitchValue = SwitchPN;

    // Now replace the indirectbr instructions with direct branches to the
    // switch block and fill out the PHI operands.
    if (DTU)
      Updates.reserve(IndirectBrs.size() +
                      2 * IndirectBrSuccToIndirectBr.size());
    for (auto *IBr : IndirectBrs) {
      SwitchPN->addIncoming(GetSwitchValue(IBr), IBr->getParent());
      UncondBrInst::Create(SwitchBB, IBr->getIterator());
      if (DTU) {
        Updates.push_back({DominatorTree::Insert, IBr->getParent(), SwitchBB});
        for (BasicBlock *SuccBB : IBr->successors())
          Updates.push_back({DominatorTree::Delete, IBr->getParent(), SuccBB});
      }
      IBr->eraseFromParent();
    }
  }

  // Now build the switch in the block. The block will have no terminator
  // already.
  auto *SI = SwitchInst::Create(SwitchValue, BBs[0], BBs.size(), SwitchBB);

  // Add a case for each block.
  for (int i : llvm::seq<int>(1, BBs.size()))
    SI->addCase(ConstantInt::get(CommonITy, i + 1), BBs[i]);

  if (DTU) {
    // If there were multiple indirectbr's, they may have common successors,
    // but in the dominator tree, we only track unique edges.
    SmallPtrSet<BasicBlock *, 8> UniqueSuccessors;
    Updates.reserve(Updates.size() + BBs.size());
    for (BasicBlock *BB : BBs) {
      if (UniqueSuccessors.insert(BB).second)
        Updates.push_back({DominatorTree::Insert, SwitchBB, BB});
    }
    DTU->applyUpdates(Updates);
  }

  if (SkipProfileUpdates || ProfcheckDisableMetadataFixes) {
    setExplicitlyUnknownBranchWeightsIfProfiled(*SI, DEBUG_TYPE);
    return true;
  }

  // We need to convert the ScaledNumber weights (which might not be
  // representable in 64 bits) back to normal 64 bit integers so we can apply
  // them as metadata. They might not have the same scale though, so we find the
  // max scale and then scale down any weights that have a scale less than the
  // max scale. This ensures that all the weights have the same scale.
  int16_t MaxScale = 0;
  for (const ScaledNumber<uint64_t> &BBWeight : BBWeights)
    MaxScale = std::max(MaxScale, BBWeight.getScale());
  SmallVector<uint64_t, 4> ExtractedBBWeights;
  ExtractedBBWeights.reserve(BBWeights.size());
  for (ScaledNumber<uint64_t> &BBWeight : BBWeights) {
    int16_t Shift = MaxScale - BBWeight.getScale();
    assert(Shift >= 0 && "expected non-negative shift");
    ExtractedBBWeights.push_back(BBWeight.getDigits() >> Shift);
  }
  setFittedBranchWeights(*SI, ExtractedBBWeights, false);

  return true;
}

bool IndirectBrExpandLegacyPass::runOnFunction(Function &F) {
  auto *TPC = getAnalysisIfAvailable<TargetPassConfig>();
  if (!TPC)
    return false;

  auto &TM = TPC->getTM<TargetMachine>();
  auto &STI = *TM.getSubtargetImpl(F);
  if (!STI.enableIndirectBrExpand())
    return false;
  auto *TLI = STI.getTargetLowering();

  std::optional<DomTreeUpdater> DTU;
  if (auto *DTWP = getAnalysisIfAvailable<DominatorTreeWrapperPass>())
    DTU.emplace(DTWP->getDomTree(), DomTreeUpdater::UpdateStrategy::Lazy);

  return runImpl(
      F, TLI, DTU ? &*DTU : nullptr,
      [&]() { return &getAnalysis<LazyBlockFrequencyInfoPass>().getBFI(); },
      OptLevel != CodeGenOptLevel::None);
}
