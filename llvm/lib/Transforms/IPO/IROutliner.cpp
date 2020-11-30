//===- IROutliner.cpp -- Outline Similar Regions ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
// Implementation for the IROutliner which is used by the IROutliner Pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/IROutliner.h"
#include "llvm/Analysis/IRSimilarityIdentifier.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/IPO.h"
#include <map>
#include <set>
#include <vector>

#define DEBUG_TYPE "iroutliner"

using namespace llvm;
using namespace IRSimilarity;

/// The OutlinableGroup holds all the overarching information for outlining
/// a set of regions that are structurally similar to one another, such as the
/// types of the overall function, the output blocks, the sets of stores needed
/// and a list of the different regions. This information is used in the
/// deduplication of extracted regions with the same structure.
struct OutlinableGroup {
  /// The sections that could be outlined
  std::vector<OutlinableRegion *> Regions;

  /// Flag for whether we should not consider this group of OutlinableRegions
  /// for extraction.
  bool IgnoreGroup = false;
};

/// Move the contents of \p SourceBB to before the last instruction of \p
/// TargetBB.
/// \param SourceBB - the BasicBlock to pull Instructions from.
/// \param TargetBB - the BasicBlock to put Instruction into.
static void moveBBContents(BasicBlock &SourceBB, BasicBlock &TargetBB) {
  BasicBlock::iterator BBCurr, BBEnd, BBNext;
  for (BBCurr = SourceBB.begin(), BBEnd = SourceBB.end(); BBCurr != BBEnd;
       BBCurr = BBNext) {
    BBNext = std::next(BBCurr);
    BBCurr->moveBefore(TargetBB, TargetBB.end());
  }
}

void OutlinableRegion::splitCandidate() {
  assert(!CandidateSplit && "Candidate already split!");

  Instruction *StartInst = (*Candidate->begin()).Inst;
  Instruction *EndInst = (*Candidate->end()).Inst;
  assert(StartInst && EndInst && "Expected a start and end instruction?");
  StartBB = StartInst->getParent();
  PrevBB = StartBB;

  // The basic block gets split like so:
  // block:                 block:
  //   inst1                  inst1
  //   inst2                  inst2
  //   region1               br block_to_outline
  //   region2              block_to_outline:
  //   region3          ->    region1
  //   region4                region2
  //   inst3                  region3
  //   inst4                  region4
  //                          br block_after_outline
  //                        block_after_outline:
  //                          inst3
  //                          inst4

  std::string OriginalName = PrevBB->getName().str();

  StartBB = PrevBB->splitBasicBlock(StartInst, OriginalName + "_to_outline");

  // This is the case for the inner block since we do not have to include
  // multiple blocks.
  EndBB = StartBB;
  FollowBB = EndBB->splitBasicBlock(EndInst, OriginalName + "_after_outline");

  CandidateSplit = true;
}

void OutlinableRegion::reattachCandidate() {
  assert(CandidateSplit && "Candidate is not split!");

  // The basic block gets reattached like so:
  // block:                        block:
  //   inst1                         inst1
  //   inst2                         inst2
  //   br block_to_outline           region1
  // block_to_outline:        ->     region2
  //   region1                       region3
  //   region2                       region4
  //   region3                       inst3
  //   region4                       inst4
  //   br block_after_outline
  // block_after_outline:
  //   inst3
  //   inst4
  assert(StartBB != nullptr && "StartBB for Candidate is not defined!");
  assert(FollowBB != nullptr && "StartBB for Candidate is not defined!");

  // StartBB should only have one predecessor since we put an unconditional
  // branch at the end of PrevBB when we split the BasicBlock.
  PrevBB = StartBB->getSinglePredecessor();
  assert(PrevBB != nullptr &&
         "No Predecessor for the region start basic block!");

  assert(PrevBB->getTerminator() && "Terminator removed from PrevBB!");
  assert(EndBB->getTerminator() && "Terminator removed from EndBB!");
  PrevBB->getTerminator()->eraseFromParent();
  EndBB->getTerminator()->eraseFromParent();

  moveBBContents(*StartBB, *PrevBB);

  BasicBlock *PlacementBB = PrevBB;
  if (StartBB != EndBB)
    PlacementBB = EndBB;
  moveBBContents(*FollowBB, *PlacementBB);

  PrevBB->replaceSuccessorsPhiUsesWith(StartBB, PrevBB);
  PrevBB->replaceSuccessorsPhiUsesWith(FollowBB, PlacementBB);
  StartBB->eraseFromParent();
  FollowBB->eraseFromParent();

  // Make sure to save changes back to the StartBB.
  StartBB = PrevBB;
  EndBB = nullptr;
  PrevBB = nullptr;
  FollowBB = nullptr;

  CandidateSplit = false;
}

void IROutliner::pruneIncompatibleRegions(
    std::vector<IRSimilarityCandidate> &CandidateVec,
    OutlinableGroup &CurrentGroup) {
  bool PreviouslyOutlined;

  // Sort from beginning to end, so the IRSimilarityCandidates are in order.
  stable_sort(CandidateVec, [](const IRSimilarityCandidate &LHS,
                               const IRSimilarityCandidate &RHS) {
    return LHS.getStartIdx() < RHS.getStartIdx();
  });

  unsigned CurrentEndIdx = 0;
  for (IRSimilarityCandidate &IRSC : CandidateVec) {
    PreviouslyOutlined = false;
    unsigned StartIdx = IRSC.getStartIdx();
    unsigned EndIdx = IRSC.getEndIdx();

    for (unsigned Idx = StartIdx; Idx <= EndIdx; Idx++)
      if (Outlined.find(Idx) != Outlined.end()) {
        PreviouslyOutlined = true;
        break;
      }

    if (PreviouslyOutlined)
      continue;

    // TODO: If in the future we can outline across BasicBlocks, we will need to
    // check all BasicBlocks contained in the region.
    if (IRSC.getStartBB()->hasAddressTaken())
      continue;

    // Greedily prune out any regions that will overlap with already chosen
    // regions.
    if (CurrentEndIdx != 0 && StartIdx <= CurrentEndIdx)
      continue;

    OutlinableRegion *OS = new (OutlinerAllocator.Allocate<OutlinableRegion>())
        OutlinableRegion(IRSC, CurrentGroup);
    CurrentGroup.Regions.push_back(OS);

    CurrentEndIdx = EndIdx;
  }
}

bool IROutliner::extractSection(OutlinableRegion &Region) {
  assert(Region.StartBB != nullptr &&
         "StartBB for the OutlinableRegion is nullptr!");
  assert(Region.FollowBB != nullptr &&
         "StartBB for the OutlinableRegion is nullptr!");
  Function *OrigF = Region.StartBB->getParent();
  CodeExtractorAnalysisCache CEAC(*OrigF);
  Region.ExtractedFunction = Region.CE->extractCodeRegion(CEAC);

  // If the extraction was successful, find the BasicBlock, and reassign the
  // OutlinableRegion blocks
  if (!Region.ExtractedFunction) {
    LLVM_DEBUG(dbgs() << "CodeExtractor failed to outline " << Region.StartBB
                      << "\n");
    Region.reattachCandidate();
    return false;
  }

  BasicBlock *RewrittenBB = Region.FollowBB->getSinglePredecessor();
  Region.StartBB = RewrittenBB;
  Region.EndBB = RewrittenBB;

  // The sequences of outlinable regions has now changed.  We must fix the
  // IRInstructionDataList for consistency.  Although they may not be illegal
  // instructions, they should not be compared with anything else as they
  // should not be outlined in this round.  So marking these as illegal is
  // allowed.
  IRInstructionDataList *IDL = Region.Candidate->front()->IDL;
  Region.NewFront = new (OutlinerAllocator.Allocate<IRInstructionData>())
      IRInstructionData(*RewrittenBB->begin(), false, *IDL);
  Region.NewBack = new (OutlinerAllocator.Allocate<IRInstructionData>())
      IRInstructionData(*std::prev(std::prev(RewrittenBB->end())), false, *IDL);

  // Insert the first IRInstructionData of the new region in front of the
  // first IRInstructionData of the IRSimilarityCandidate.
  IDL->insert(Region.Candidate->begin(), *Region.NewFront);
  // Insert the first IRInstructionData of the new region after the
  // last IRInstructionData of the IRSimilarityCandidate.
  IDL->insert(Region.Candidate->end(), *Region.NewBack);
  // Remove the IRInstructionData from the IRSimilarityCandidate.
  IDL->erase(Region.Candidate->begin(), std::prev(Region.Candidate->end()));

  assert(RewrittenBB != nullptr &&
         "Could not find a predecessor after extraction!");

  // Iterate over the new set of instructions to find the new call
  // instruction.
  for (Instruction &I : *RewrittenBB)
    if (CallInst *CI = dyn_cast<CallInst>(&I))
      if (Region.ExtractedFunction == CI->getCalledFunction()) {
        Region.Call = CI;
        break;
      }
  Region.reattachCandidate();
  return true;
}

unsigned IROutliner::doOutline(Module &M) {
  // Find the possibile similarity sections.
  IRSimilarityIdentifier &Identifier = getIRSI(M);
  SimilarityGroupList &SimilarityCandidates = *Identifier.getSimilarity();

  // Sort them by size of extracted sections
  unsigned OutlinedFunctionNum = 0;
  // If we only have one SimilarityGroup in SimilarityCandidates, we do not have
  // to sort them by the potential number of instructions to be outlined
  if (SimilarityCandidates.size() > 1)
    llvm::stable_sort(SimilarityCandidates,
                      [](const std::vector<IRSimilarityCandidate> &LHS,
                         const std::vector<IRSimilarityCandidate> &RHS) {
                        return LHS[0].getLength() * LHS.size() >
                               RHS[0].getLength() * RHS.size();
                      });

  // Iterate over the possible sets of similarity.
  for (SimilarityGroup &CandidateVec : SimilarityCandidates) {
    OutlinableGroup CurrentGroup;

    // Remove entries that were previously outlined
    pruneIncompatibleRegions(CandidateVec, CurrentGroup);

    // We pruned the number of regions to 0 to 1, meaning that it's not worth
    // trying to outlined since there is no compatible similar instance of this
    // code.
    if (CurrentGroup.Regions.size() < 2)
      continue;

    // Create a CodeExtractor for each outlinable region.
    for (OutlinableRegion *OS : CurrentGroup.Regions) {
      // Break the outlinable region out of its parent BasicBlock into its own
      // BasicBlocks (see function implementation).
      OS->splitCandidate();
      std::vector<BasicBlock *> BE = {OS->StartBB};
      OS->CE = new (OutlinerAllocator.Allocate<CodeExtractor>())
          CodeExtractor(BE, nullptr, false, nullptr, nullptr, nullptr, false,
                        false, "outlined");
    }

    // Create functions out of all the sections, and mark them as outlined
    std::vector<OutlinableRegion *> OutlinedRegions;
    for (OutlinableRegion *OS : CurrentGroup.Regions) {
      OutlinedFunctionNum++;
      bool FunctionOutlined = extractSection(*OS);
      if (FunctionOutlined) {
        unsigned StartIdx = OS->Candidate->getStartIdx();
        unsigned EndIdx = OS->Candidate->getEndIdx();
        for (unsigned Idx = StartIdx; Idx <= EndIdx; Idx++)
          Outlined.insert(Idx);

        OutlinedRegions.push_back(OS);
      }
    }

    CurrentGroup.Regions = std::move(OutlinedRegions);
  }

  return OutlinedFunctionNum;
}

bool IROutliner::run(Module &M) { return doOutline(M) > 0; }

// Pass Manager Boilerplate
class IROutlinerLegacyPass : public ModulePass {
public:
  static char ID;
  IROutlinerLegacyPass() : ModulePass(ID) {
    initializeIROutlinerLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<IRSimilarityIdentifierWrapperPass>();
  }

  bool runOnModule(Module &M) override;
};

bool IROutlinerLegacyPass::runOnModule(Module &M) {
  if (skipModule(M))
    return false;

  auto GTTI = [this](Function &F) -> TargetTransformInfo & {
    return this->getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  };

  auto GIRSI = [this](Module &) -> IRSimilarityIdentifier & {
    return this->getAnalysis<IRSimilarityIdentifierWrapperPass>().getIRSI();
  };

  return IROutliner(GTTI, GIRSI).run(M);
}

PreservedAnalyses IROutlinerPass::run(Module &M, ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  std::function<TargetTransformInfo &(Function &)> GTTI =
      [&FAM](Function &F) -> TargetTransformInfo & {
    return FAM.getResult<TargetIRAnalysis>(F);
  };

  std::function<IRSimilarityIdentifier &(Module &)> GIRSI =
      [&AM](Module &M) -> IRSimilarityIdentifier & {
    return AM.getResult<IRSimilarityAnalysis>(M);
  };

  if (IROutliner(GTTI, GIRSI).run(M))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

char IROutlinerLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(IROutlinerLegacyPass, "iroutliner", "IR Outliner", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(IRSimilarityIdentifierWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(IROutlinerLegacyPass, "iroutliner", "IR Outliner", false,
                    false)

ModulePass *llvm::createIROutlinerPass() { return new IROutlinerLegacyPass(); }
