//===- CodeExtractor.cpp - Pull code region into a new function -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the interface to tear out a code region, such as an
// individual loop or a parallel section, into a new function, replacing it with
// a call to the new function.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BlockFrequencyInfoImpl.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Support/BlockFrequency.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include <cassert>
#include <cstdint>
#include <iterator>
#include <map>
#include <set>
#include <utility>
#include <vector>

using namespace llvm;
using namespace llvm::PatternMatch;
using ProfileCount = Function::ProfileCount;

#define DEBUG_TYPE "code-extractor"

// Provide a command-line option to aggregate function arguments into a struct
// for functions produced by the code extractor. This is useful when converting
// extracted functions to pthread-based code, as only one argument (void*) can
// be passed in to pthread_create().
static cl::opt<bool>
AggregateArgsOpt("aggregate-extracted-args", cl::Hidden,
                 cl::desc("Aggregate arguments to code-extracted functions"));

/// Test whether a block is valid for extraction.
static bool isBlockValidForExtraction(const BasicBlock &BB,
                                      const SetVector<BasicBlock *> &Result,
                                      bool AllowVarArgs, bool AllowAlloca) {
  // taking the address of a basic block moved to another function is illegal
  if (BB.hasAddressTaken())
    return false;

  // don't hoist code that uses another basicblock address, as it's likely to
  // lead to unexpected behavior, like cross-function jumps
  SmallPtrSet<User const *, 16> Visited;
  SmallVector<User const *, 16> ToVisit;

  for (Instruction const &Inst : BB)
    ToVisit.push_back(&Inst);

  while (!ToVisit.empty()) {
    User const *Curr = ToVisit.pop_back_val();
    if (!Visited.insert(Curr).second)
      continue;
    if (isa<BlockAddress const>(Curr))
      return false; // even a reference to self is likely to be not compatible

    if (isa<Instruction>(Curr) && cast<Instruction>(Curr)->getParent() != &BB)
      continue;

    for (auto const &U : Curr->operands()) {
      if (auto *UU = dyn_cast<User>(U))
        ToVisit.push_back(UU);
    }
  }

  // If explicitly requested, allow vastart and alloca. For invoke instructions
  // verify that extraction is valid.
  for (BasicBlock::const_iterator I = BB.begin(), E = BB.end(); I != E; ++I) {
    if (isa<AllocaInst>(I)) {
       if (!AllowAlloca)
         return false;
       continue;
    }

    if (const auto *II = dyn_cast<InvokeInst>(I)) {
      // Unwind destination (either a landingpad, catchswitch, or cleanuppad)
      // must be a part of the subgraph which is being extracted.
      if (auto *UBB = II->getUnwindDest())
        if (!Result.count(UBB))
          return false;
      continue;
    }

    // All catch handlers of a catchswitch instruction as well as the unwind
    // destination must be in the subgraph.
    if (const auto *CSI = dyn_cast<CatchSwitchInst>(I)) {
      if (auto *UBB = CSI->getUnwindDest())
        if (!Result.count(UBB))
          return false;
      for (auto *HBB : CSI->handlers())
        if (!Result.count(const_cast<BasicBlock*>(HBB)))
          return false;
      continue;
    }

    // Make sure that entire catch handler is within subgraph. It is sufficient
    // to check that catch return's block is in the list.
    if (const auto *CPI = dyn_cast<CatchPadInst>(I)) {
      for (const auto *U : CPI->users())
        if (const auto *CRI = dyn_cast<CatchReturnInst>(U))
          if (!Result.count(const_cast<BasicBlock*>(CRI->getParent())))
            return false;
      continue;
    }

    // And do similar checks for cleanup handler - the entire handler must be
    // in subgraph which is going to be extracted. For cleanup return should
    // additionally check that the unwind destination is also in the subgraph.
    if (const auto *CPI = dyn_cast<CleanupPadInst>(I)) {
      for (const auto *U : CPI->users())
        if (const auto *CRI = dyn_cast<CleanupReturnInst>(U))
          if (!Result.count(const_cast<BasicBlock*>(CRI->getParent())))
            return false;
      continue;
    }
    if (const auto *CRI = dyn_cast<CleanupReturnInst>(I)) {
      if (auto *UBB = CRI->getUnwindDest())
        if (!Result.count(UBB))
          return false;
      continue;
    }

    if (const CallInst *CI = dyn_cast<CallInst>(I)) {
      if (const Function *F = CI->getCalledFunction()) {
        auto IID = F->getIntrinsicID();
        if (IID == Intrinsic::vastart) {
          if (AllowVarArgs)
            continue;
          else
            return false;
        }

        // Currently, we miscompile outlined copies of eh_typid_for. There are
        // proposals for fixing this in llvm.org/PR39545.
        if (IID == Intrinsic::eh_typeid_for)
          return false;
      }
    }
  }

  return true;
}

/// Build a set of blocks to extract if the input blocks are viable.
static SetVector<BasicBlock *>
buildExtractionBlockSet(ArrayRef<BasicBlock *> BBs, DominatorTree *DT,
                        bool AllowVarArgs, bool AllowAlloca) {
  assert(!BBs.empty() && "The set of blocks to extract must be non-empty");
  SetVector<BasicBlock *> Result;

  // Loop over the blocks, adding them to our set-vector, and aborting with an
  // empty set if we encounter invalid blocks.
  for (BasicBlock *BB : BBs) {
    // If this block is dead, don't process it.
    if (DT && !DT->isReachableFromEntry(BB))
      continue;

    if (!Result.insert(BB))
      llvm_unreachable("Repeated basic blocks in extraction input");
  }

  LLVM_DEBUG(dbgs() << "Region front block: " << Result.front()->getName()
                    << '\n');

  for (auto *BB : Result) {
    if (!isBlockValidForExtraction(*BB, Result, AllowVarArgs, AllowAlloca))
      return {};

    // Make sure that the first block is not a landing pad.
    if (BB == Result.front()) {
      if (BB->isEHPad()) {
        LLVM_DEBUG(dbgs() << "The first block cannot be an unwind block\n");
        return {};
      }
      continue;
    }

    // All blocks other than the first must not have predecessors outside of
    // the subgraph which is being extracted.
    for (auto *PBB : predecessors(BB))
      if (!Result.count(PBB)) {
        LLVM_DEBUG(dbgs() << "No blocks in this region may have entries from "
                             "outside the region except for the first block!\n"
                          << "Problematic source BB: " << BB->getName() << "\n"
                          << "Problematic destination BB: " << PBB->getName()
                          << "\n");
        return {};
      }
  }

  return Result;
}

CodeExtractor::CodeExtractor(ArrayRef<BasicBlock *> BBs, DominatorTree *DT,
                             bool AggregateArgs, BlockFrequencyInfo *BFI,
                             BranchProbabilityInfo *BPI, AssumptionCache *AC,
                             bool AllowVarArgs, bool AllowAlloca,
                             std::string Suffix)
    : DT(DT), AggregateArgs(AggregateArgs || AggregateArgsOpt), BFI(BFI),
      BPI(BPI), AC(AC), AllowVarArgs(AllowVarArgs),
      Blocks(buildExtractionBlockSet(BBs, DT, AllowVarArgs, AllowAlloca)),
      Suffix(Suffix) {}

CodeExtractor::CodeExtractor(DominatorTree &DT, Loop &L, bool AggregateArgs,
                             BlockFrequencyInfo *BFI,
                             BranchProbabilityInfo *BPI, AssumptionCache *AC,
                             std::string Suffix)
    : DT(&DT), AggregateArgs(AggregateArgs || AggregateArgsOpt), BFI(BFI),
      BPI(BPI), AC(AC), AllowVarArgs(false),
      Blocks(buildExtractionBlockSet(L.getBlocks(), &DT,
                                     /* AllowVarArgs */ false,
                                     /* AllowAlloca */ false)),
      Suffix(Suffix) {}

/// definedInRegion - Return true if the specified value is defined in the
/// extracted region.
static bool definedInRegion(const SetVector<BasicBlock *> &Blocks, Value *V) {
  if (Instruction *I = dyn_cast<Instruction>(V))
    if (Blocks.count(I->getParent()))
      return true;
  return false;
}

/// definedInCaller - Return true if the specified value is defined in the
/// function being code extracted, but not in the region being extracted.
/// These values must be passed in as live-ins to the function.
static bool definedInCaller(const SetVector<BasicBlock *> &Blocks, Value *V) {
  if (isa<Argument>(V)) return true;
  if (Instruction *I = dyn_cast<Instruction>(V))
    if (!Blocks.count(I->getParent()))
      return true;
  return false;
}

static BasicBlock *getCommonExitBlock(const SetVector<BasicBlock *> &Blocks) {
  BasicBlock *CommonExitBlock = nullptr;
  auto hasNonCommonExitSucc = [&](BasicBlock *Block) {
    for (auto *Succ : successors(Block)) {
      // Internal edges, ok.
      if (Blocks.count(Succ))
        continue;
      if (!CommonExitBlock) {
        CommonExitBlock = Succ;
        continue;
      }
      if (CommonExitBlock != Succ)
        return true;
    }
    return false;
  };

  if (any_of(Blocks, hasNonCommonExitSucc))
    return nullptr;

  return CommonExitBlock;
}

CodeExtractorAnalysisCache::CodeExtractorAnalysisCache(Function &F) {
  for (BasicBlock &BB : F) {
    for (Instruction &II : BB.instructionsWithoutDebug())
      if (auto *AI = dyn_cast<AllocaInst>(&II))
        Allocas.push_back(AI);

    findSideEffectInfoForBlock(BB);
  }
}

void CodeExtractorAnalysisCache::findSideEffectInfoForBlock(BasicBlock &BB) {
  for (Instruction &II : BB.instructionsWithoutDebug()) {
    unsigned Opcode = II.getOpcode();
    Value *MemAddr = nullptr;
    switch (Opcode) {
    case Instruction::Store:
    case Instruction::Load: {
      if (Opcode == Instruction::Store) {
        StoreInst *SI = cast<StoreInst>(&II);
        MemAddr = SI->getPointerOperand();
      } else {
        LoadInst *LI = cast<LoadInst>(&II);
        MemAddr = LI->getPointerOperand();
      }
      // Global variable can not be aliased with locals.
      if (dyn_cast<Constant>(MemAddr))
        break;
      Value *Base = MemAddr->stripInBoundsConstantOffsets();
      if (!isa<AllocaInst>(Base)) {
        SideEffectingBlocks.insert(&BB);
        return;
      }
      BaseMemAddrs[&BB].insert(Base);
      break;
    }
    default: {
      IntrinsicInst *IntrInst = dyn_cast<IntrinsicInst>(&II);
      if (IntrInst) {
        if (IntrInst->isLifetimeStartOrEnd())
          break;
        SideEffectingBlocks.insert(&BB);
        return;
      }
      // Treat all the other cases conservatively if it has side effects.
      if (II.mayHaveSideEffects()) {
        SideEffectingBlocks.insert(&BB);
        return;
      }
    }
    }
  }
}

bool CodeExtractorAnalysisCache::doesBlockContainClobberOfAddr(
    BasicBlock &BB, AllocaInst *Addr) const {
  if (SideEffectingBlocks.count(&BB))
    return true;
  auto It = BaseMemAddrs.find(&BB);
  if (It != BaseMemAddrs.end())
    return It->second.count(Addr);
  return false;
}

bool CodeExtractor::isLegalToShrinkwrapLifetimeMarkers(
    const CodeExtractorAnalysisCache &CEAC, Instruction *Addr) const {
  AllocaInst *AI = cast<AllocaInst>(Addr->stripInBoundsConstantOffsets());
  Function *Func = (*Blocks.begin())->getParent();
  for (BasicBlock &BB : *Func) {
    if (Blocks.count(&BB))
      continue;
    if (CEAC.doesBlockContainClobberOfAddr(BB, AI))
      return false;
  }
  return true;
}

BasicBlock *
CodeExtractor::findOrCreateBlockForHoisting(BasicBlock *CommonExitBlock) {
  BasicBlock *SinglePredFromOutlineRegion = nullptr;
  assert(!Blocks.count(CommonExitBlock) &&
         "Expect a block outside the region!");
  for (auto *Pred : predecessors(CommonExitBlock)) {
    if (!Blocks.count(Pred))
      continue;
    if (!SinglePredFromOutlineRegion) {
      SinglePredFromOutlineRegion = Pred;
    } else if (SinglePredFromOutlineRegion != Pred) {
      SinglePredFromOutlineRegion = nullptr;
      break;
    }
  }

  if (SinglePredFromOutlineRegion)
    return SinglePredFromOutlineRegion;

#ifndef NDEBUG
  auto getFirstPHI = [](BasicBlock *BB) {
    BasicBlock::iterator I = BB->begin();
    PHINode *FirstPhi = nullptr;
    while (I != BB->end()) {
      PHINode *Phi = dyn_cast<PHINode>(I);
      if (!Phi)
        break;
      if (!FirstPhi) {
        FirstPhi = Phi;
        break;
      }
    }
    return FirstPhi;
  };
  // If there are any phi nodes, the single pred either exists or has already
  // be created before code extraction.
  assert(!getFirstPHI(CommonExitBlock) && "Phi not expected");
#endif

  BasicBlock *NewExitBlock = CommonExitBlock->splitBasicBlock(
      CommonExitBlock->getFirstNonPHI()->getIterator());

  for (auto PI = pred_begin(CommonExitBlock), PE = pred_end(CommonExitBlock);
       PI != PE;) {
    BasicBlock *Pred = *PI++;
    if (Blocks.count(Pred))
      continue;
    Pred->getTerminator()->replaceUsesOfWith(CommonExitBlock, NewExitBlock);
  }
  // Now add the old exit block to the outline region.
  Blocks.insert(CommonExitBlock);
  return CommonExitBlock;
}

// Find the pair of life time markers for address 'Addr' that are either
// defined inside the outline region or can legally be shrinkwrapped into the
// outline region. If there are not other untracked uses of the address, return
// the pair of markers if found; otherwise return a pair of nullptr.
CodeExtractor::LifetimeMarkerInfo
CodeExtractor::getLifetimeMarkers(const CodeExtractorAnalysisCache &CEAC,
                                  Instruction *Addr,
                                  BasicBlock *ExitBlock) const {
  LifetimeMarkerInfo Info;

  for (User *U : Addr->users()) {
    IntrinsicInst *IntrInst = dyn_cast<IntrinsicInst>(U);
    if (IntrInst) {
      if (IntrInst->getIntrinsicID() == Intrinsic::lifetime_start) {
        // Do not handle the case where Addr has multiple start markers.
        if (Info.LifeStart)
          return {};
        Info.LifeStart = IntrInst;
      }
      if (IntrInst->getIntrinsicID() == Intrinsic::lifetime_end) {
        if (Info.LifeEnd)
          return {};
        Info.LifeEnd = IntrInst;
      }
      continue;
    }
    // Find untracked uses of the address, bail.
    if (!definedInRegion(Blocks, U))
      return {};
  }

  if (!Info.LifeStart || !Info.LifeEnd)
    return {};

  Info.SinkLifeStart = !definedInRegion(Blocks, Info.LifeStart);
  Info.HoistLifeEnd = !definedInRegion(Blocks, Info.LifeEnd);
  // Do legality check.
  if ((Info.SinkLifeStart || Info.HoistLifeEnd) &&
      !isLegalToShrinkwrapLifetimeMarkers(CEAC, Addr))
    return {};

  // Check to see if we have a place to do hoisting, if not, bail.
  if (Info.HoistLifeEnd && !ExitBlock)
    return {};

  return Info;
}

void CodeExtractor::findAllocas(const CodeExtractorAnalysisCache &CEAC,
                                ValueSet &SinkCands, ValueSet &HoistCands,
                                BasicBlock *&ExitBlock) const {
  Function *Func = (*Blocks.begin())->getParent();
  ExitBlock = getCommonExitBlock(Blocks);

  auto moveOrIgnoreLifetimeMarkers =
      [&](const LifetimeMarkerInfo &LMI) -> bool {
    if (!LMI.LifeStart)
      return false;
    if (LMI.SinkLifeStart) {
      LLVM_DEBUG(dbgs() << "Sinking lifetime.start: " << *LMI.LifeStart
                        << "\n");
      SinkCands.insert(LMI.LifeStart);
    }
    if (LMI.HoistLifeEnd) {
      LLVM_DEBUG(dbgs() << "Hoisting lifetime.end: " << *LMI.LifeEnd << "\n");
      HoistCands.insert(LMI.LifeEnd);
    }
    return true;
  };

  // Look up allocas in the original function in CodeExtractorAnalysisCache, as
  // this is much faster than walking all the instructions.
  for (AllocaInst *AI : CEAC.getAllocas()) {
    BasicBlock *BB = AI->getParent();
    if (Blocks.count(BB))
      continue;

    // As a prior call to extractCodeRegion() may have shrinkwrapped the alloca,
    // check whether it is actually still in the original function.
    Function *AIFunc = BB->getParent();
    if (AIFunc != Func)
      continue;

    LifetimeMarkerInfo MarkerInfo = getLifetimeMarkers(CEAC, AI, ExitBlock);
    bool Moved = moveOrIgnoreLifetimeMarkers(MarkerInfo);
    if (Moved) {
      LLVM_DEBUG(dbgs() << "Sinking alloca: " << *AI << "\n");
      SinkCands.insert(AI);
      continue;
    }

    // Follow any bitcasts.
    SmallVector<Instruction *, 2> Bitcasts;
    SmallVector<LifetimeMarkerInfo, 2> BitcastLifetimeInfo;
    for (User *U : AI->users()) {
      if (U->stripInBoundsConstantOffsets() == AI) {
        Instruction *Bitcast = cast<Instruction>(U);
        LifetimeMarkerInfo LMI = getLifetimeMarkers(CEAC, Bitcast, ExitBlock);
        if (LMI.LifeStart) {
          Bitcasts.push_back(Bitcast);
          BitcastLifetimeInfo.push_back(LMI);
          continue;
        }
      }

      // Found unknown use of AI.
      if (!definedInRegion(Blocks, U)) {
        Bitcasts.clear();
        break;
      }
    }

    // Either no bitcasts reference the alloca or there are unknown uses.
    if (Bitcasts.empty())
      continue;

    LLVM_DEBUG(dbgs() << "Sinking alloca (via bitcast): " << *AI << "\n");
    SinkCands.insert(AI);
    for (unsigned I = 0, E = Bitcasts.size(); I != E; ++I) {
      Instruction *BitcastAddr = Bitcasts[I];
      const LifetimeMarkerInfo &LMI = BitcastLifetimeInfo[I];
      assert(LMI.LifeStart &&
             "Unsafe to sink bitcast without lifetime markers");
      moveOrIgnoreLifetimeMarkers(LMI);
      if (!definedInRegion(Blocks, BitcastAddr)) {
        LLVM_DEBUG(dbgs() << "Sinking bitcast-of-alloca: " << *BitcastAddr
                          << "\n");
        SinkCands.insert(BitcastAddr);
      }
    }
  }
}

bool CodeExtractor::isEligible() const {
  if (Blocks.empty())
    return false;
  BasicBlock *Header = *Blocks.begin();
  Function *F = Header->getParent();

  // For functions with varargs, check that varargs handling is only done in the
  // outlined function, i.e vastart and vaend are only used in outlined blocks.
  if (AllowVarArgs && F->getFunctionType()->isVarArg()) {
    auto containsVarArgIntrinsic = [](const Instruction &I) {
      if (const CallInst *CI = dyn_cast<CallInst>(&I))
        if (const Function *Callee = CI->getCalledFunction())
          return Callee->getIntrinsicID() == Intrinsic::vastart ||
                 Callee->getIntrinsicID() == Intrinsic::vaend;
      return false;
    };

    for (auto &BB : *F) {
      if (Blocks.count(&BB))
        continue;
      if (llvm::any_of(BB, containsVarArgIntrinsic))
        return false;
    }
  }
  return true;
}

void CodeExtractor::findInputsOutputs(ValueSet &Inputs, ValueSet &Outputs,
                                      const ValueSet &SinkCands) const {
  for (BasicBlock *BB : Blocks) {
    // If a used value is defined outside the region, it's an input.  If an
    // instruction is used outside the region, it's an output.
    for (Instruction &II : *BB) {
      for (auto &OI : II.operands()) {
        Value *V = OI;
        if (!SinkCands.count(V) && definedInCaller(Blocks, V))
          Inputs.insert(V);
      }

      for (User *U : II.users())
        if (!definedInRegion(Blocks, U)) {
          Outputs.insert(&II);
          break;
        }
    }
  }
}

/// severSplitPHINodesOfEntry - If a PHI node has multiple inputs from outside
/// of the region, we need to split the entry block of the region so that the
/// PHI node is easier to deal with.
void CodeExtractor::severSplitPHINodesOfEntry(BasicBlock *&Header) {
  unsigned NumPredsFromRegion = 0;
  unsigned NumPredsOutsideRegion = 0;

  if (Header != &Header->getParent()->getEntryBlock()) {
    PHINode *PN = dyn_cast<PHINode>(Header->begin());
    if (!PN) return;  // No PHI nodes.

    // If the header node contains any PHI nodes, check to see if there is more
    // than one entry from outside the region.  If so, we need to sever the
    // header block into two.
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
      if (Blocks.count(PN->getIncomingBlock(i)))
        ++NumPredsFromRegion;
      else
        ++NumPredsOutsideRegion;

    // If there is one (or fewer) predecessor from outside the region, we don't
    // need to do anything special.
    if (NumPredsOutsideRegion <= 1) return;
  }

  // Otherwise, we need to split the header block into two pieces: one
  // containing PHI nodes merging values from outside of the region, and a
  // second that contains all of the code for the block and merges back any
  // incoming values from inside of the region.
  BasicBlock *NewBB = SplitBlock(Header, Header->getFirstNonPHI(), DT);

  // We only want to code extract the second block now, and it becomes the new
  // header of the region.
  BasicBlock *OldPred = Header;
  Blocks.remove(OldPred);
  Blocks.insert(NewBB);
  Header = NewBB;

  // Okay, now we need to adjust the PHI nodes and any branches from within the
  // region to go to the new header block instead of the old header block.
  if (NumPredsFromRegion) {
    PHINode *PN = cast<PHINode>(OldPred->begin());
    // Loop over all of the predecessors of OldPred that are in the region,
    // changing them to branch to NewBB instead.
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
      if (Blocks.count(PN->getIncomingBlock(i))) {
        Instruction *TI = PN->getIncomingBlock(i)->getTerminator();
        TI->replaceUsesOfWith(OldPred, NewBB);
      }

    // Okay, everything within the region is now branching to the right block, we
    // just have to update the PHI nodes now, inserting PHI nodes into NewBB.
    BasicBlock::iterator AfterPHIs;
    for (AfterPHIs = OldPred->begin(); isa<PHINode>(AfterPHIs); ++AfterPHIs) {
      PHINode *PN = cast<PHINode>(AfterPHIs);
      // Create a new PHI node in the new region, which has an incoming value
      // from OldPred of PN.
      PHINode *NewPN = PHINode::Create(PN->getType(), 1 + NumPredsFromRegion,
                                       PN->getName() + ".ce", &NewBB->front());
      PN->replaceAllUsesWith(NewPN);
      NewPN->addIncoming(PN, OldPred);

      // Loop over all of the incoming value in PN, moving them to NewPN if they
      // are from the extracted region.
      for (unsigned i = 0; i != PN->getNumIncomingValues(); ++i) {
        if (Blocks.count(PN->getIncomingBlock(i))) {
          NewPN->addIncoming(PN->getIncomingValue(i), PN->getIncomingBlock(i));
          PN->removeIncomingValue(i);
          --i;
        }
      }
    }
  }
}

/// severSplitPHINodesOfExits - if PHI nodes in exit blocks have inputs from
/// outlined region, we split these PHIs on two: one with inputs from region
/// and other with remaining incoming blocks; then first PHIs are placed in
/// outlined region.
void CodeExtractor::severSplitPHINodesOfExits(
    const SmallPtrSetImpl<BasicBlock *> &Exits) {
  for (BasicBlock *ExitBB : Exits) {
    BasicBlock *NewBB = nullptr;

    for (PHINode &PN : ExitBB->phis()) {
      // Find all incoming values from the outlining region.
      SmallVector<unsigned, 2> IncomingVals;
      for (unsigned i = 0; i < PN.getNumIncomingValues(); ++i)
        if (Blocks.count(PN.getIncomingBlock(i)))
          IncomingVals.push_back(i);

      // Do not process PHI if there is one (or fewer) predecessor from region.
      // If PHI has exactly one predecessor from region, only this one incoming
      // will be replaced on codeRepl block, so it should be safe to skip PHI.
      if (IncomingVals.size() <= 1)
        continue;

      // Create block for new PHIs and add it to the list of outlined if it
      // wasn't done before.
      if (!NewBB) {
        NewBB = BasicBlock::Create(ExitBB->getContext(),
                                   ExitBB->getName() + ".split",
                                   ExitBB->getParent(), ExitBB);
        SmallVector<BasicBlock *, 4> Preds(pred_begin(ExitBB),
                                           pred_end(ExitBB));
        for (BasicBlock *PredBB : Preds)
          if (Blocks.count(PredBB))
            PredBB->getTerminator()->replaceUsesOfWith(ExitBB, NewBB);
        BranchInst::Create(ExitBB, NewBB);
        Blocks.insert(NewBB);
      }

      // Split this PHI.
      PHINode *NewPN =
          PHINode::Create(PN.getType(), IncomingVals.size(),
                          PN.getName() + ".ce", NewBB->getFirstNonPHI());
      for (unsigned i : IncomingVals)
        NewPN->addIncoming(PN.getIncomingValue(i), PN.getIncomingBlock(i));
      for (unsigned i : reverse(IncomingVals))
        PN.removeIncomingValue(i, false);
      PN.addIncoming(NewPN, NewBB);
    }
  }
}

void CodeExtractor::splitReturnBlocks() {
  for (BasicBlock *Block : Blocks)
    if (ReturnInst *RI = dyn_cast<ReturnInst>(Block->getTerminator())) {
      BasicBlock *New =
          Block->splitBasicBlock(RI->getIterator(), Block->getName() + ".ret");
      if (DT) {
        // Old dominates New. New node dominates all other nodes dominated
        // by Old.
        DomTreeNode *OldNode = DT->getNode(Block);
        SmallVector<DomTreeNode *, 8> Children(OldNode->begin(),
                                               OldNode->end());

        DomTreeNode *NewNode = DT->addNewBlock(New, Block);

        for (DomTreeNode *I : Children)
          DT->changeImmediateDominator(I, NewNode);
      }
    }
}

/// constructFunction - make a function based on inputs and outputs, as follows:
/// f(in0, ..., inN, out0, ..., outN)
Function *CodeExtractor::constructFunction(const ValueSet &inputs,
                                           const ValueSet &outputs,
                                           BasicBlock *header,
                                           BasicBlock *newRootNode,
                                           BasicBlock *newHeader,
                                           Function *oldFunction,
                                           Module *M) {
  LLVM_DEBUG(dbgs() << "inputs: " << inputs.size() << "\n");
  LLVM_DEBUG(dbgs() << "outputs: " << outputs.size() << "\n");

  // This function returns unsigned, outputs will go back by reference.
  switch (NumExitBlocks) {
  case 0:
  case 1: RetTy = Type::getVoidTy(header->getContext()); break;
  case 2: RetTy = Type::getInt1Ty(header->getContext()); break;
  default: RetTy = Type::getInt16Ty(header->getContext()); break;
  }

  std::vector<Type *> paramTy;

  // Add the types of the input values to the function's argument list
  for (Value *value : inputs) {
    LLVM_DEBUG(dbgs() << "value used in func: " << *value << "\n");
    paramTy.push_back(value->getType());
  }

  // Add the types of the output values to the function's argument list.
  for (Value *output : outputs) {
    LLVM_DEBUG(dbgs() << "instr used in func: " << *output << "\n");
    if (AggregateArgs)
      paramTy.push_back(output->getType());
    else
      paramTy.push_back(PointerType::getUnqual(output->getType()));
  }

  LLVM_DEBUG({
    dbgs() << "Function type: " << *RetTy << " f(";
    for (Type *i : paramTy)
      dbgs() << *i << ", ";
    dbgs() << ")\n";
  });

  StructType *StructTy = nullptr;
  if (AggregateArgs && (inputs.size() + outputs.size() > 0)) {
    StructTy = StructType::get(M->getContext(), paramTy);
    paramTy.clear();
    paramTy.push_back(PointerType::getUnqual(StructTy));
  }
  FunctionType *funcType =
                  FunctionType::get(RetTy, paramTy,
                                    AllowVarArgs && oldFunction->isVarArg());

  std::string SuffixToUse =
      Suffix.empty()
          ? (header->getName().empty() ? "extracted" : header->getName().str())
          : Suffix;
  // Create the new function
  Function *newFunction = Function::Create(
      funcType, GlobalValue::InternalLinkage, oldFunction->getAddressSpace(),
      oldFunction->getName() + "." + SuffixToUse, M);
  // If the old function is no-throw, so is the new one.
  if (oldFunction->doesNotThrow())
    newFunction->setDoesNotThrow();

  // Inherit the uwtable attribute if we need to.
  if (oldFunction->hasUWTable())
    newFunction->setHasUWTable();

  // Inherit all of the target dependent attributes and white-listed
  // target independent attributes.
  //  (e.g. If the extracted region contains a call to an x86.sse
  //  instruction we need to make sure that the extracted region has the
  //  "target-features" attribute allowing it to be lowered.
  // FIXME: This should be changed to check to see if a specific
  //           attribute can not be inherited.
  for (const auto &Attr : oldFunction->getAttributes().getFnAttributes()) {
    if (Attr.isStringAttribute()) {
      if (Attr.getKindAsString() == "thunk")
        continue;
    } else
      switch (Attr.getKindAsEnum()) {
      // Those attributes cannot be propagated safely. Explicitly list them
      // here so we get a warning if new attributes are added. This list also
      // includes non-function attributes.
      case Attribute::Alignment:
      case Attribute::AllocSize:
      case Attribute::ArgMemOnly:
      case Attribute::Builtin:
      case Attribute::ByVal:
      case Attribute::Convergent:
      case Attribute::Dereferenceable:
      case Attribute::DereferenceableOrNull:
      case Attribute::InAlloca:
      case Attribute::InReg:
      case Attribute::InaccessibleMemOnly:
      case Attribute::InaccessibleMemOrArgMemOnly:
      case Attribute::JumpTable:
      case Attribute::Naked:
      case Attribute::Nest:
      case Attribute::NoAlias:
      case Attribute::NoBuiltin:
      case Attribute::NoCapture:
      case Attribute::NoReturn:
      case Attribute::NoSync:
      case Attribute::None:
      case Attribute::NonNull:
      case Attribute::ReadNone:
      case Attribute::ReadOnly:
      case Attribute::Returned:
      case Attribute::ReturnsTwice:
      case Attribute::SExt:
      case Attribute::Speculatable:
      case Attribute::StackAlignment:
      case Attribute::StructRet:
      case Attribute::SwiftError:
      case Attribute::SwiftSelf:
      case Attribute::WillReturn:
      case Attribute::WriteOnly:
      case Attribute::ZExt:
      case Attribute::ImmArg:
      case Attribute::EndAttrKinds:
      case Attribute::EmptyKey:
      case Attribute::TombstoneKey:
        continue;
      // Those attributes should be safe to propagate to the extracted function.
      case Attribute::AlwaysInline:
      case Attribute::Cold:
      case Attribute::NoRecurse:
      case Attribute::InlineHint:
      case Attribute::MinSize:
      case Attribute::NoDuplicate:
      case Attribute::NoFree:
      case Attribute::NoImplicitFloat:
      case Attribute::NoInline:
      case Attribute::NonLazyBind:
      case Attribute::NoRedZone:
      case Attribute::NoUnwind:
      case Attribute::OptForFuzzing:
      case Attribute::OptimizeNone:
      case Attribute::OptimizeForSize:
      case Attribute::SafeStack:
      case Attribute::ShadowCallStack:
      case Attribute::SanitizeAddress:
      case Attribute::SanitizeMemory:
      case Attribute::SanitizeThread:
      case Attribute::SanitizeHWAddress:
      case Attribute::SanitizeMemTag:
      case Attribute::SpeculativeLoadHardening:
      case Attribute::StackProtect:
      case Attribute::StackProtectReq:
      case Attribute::StackProtectStrong:
      case Attribute::StrictFP:
      case Attribute::UWTable:
      case Attribute::NoCfCheck:
        break;
      }

    newFunction->addFnAttr(Attr);
  }
  newFunction->getBasicBlockList().push_back(newRootNode);

  // Create an iterator to name all of the arguments we inserted.
  Function::arg_iterator AI = newFunction->arg_begin();

  // Rewrite all users of the inputs in the extracted region to use the
  // arguments (or appropriate addressing into struct) instead.
  for (unsigned i = 0, e = inputs.size(); i != e; ++i) {
    Value *RewriteVal;
    if (AggregateArgs) {
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Type::getInt32Ty(header->getContext()));
      Idx[1] = ConstantInt::get(Type::getInt32Ty(header->getContext()), i);
      Instruction *TI = newFunction->begin()->getTerminator();
      GetElementPtrInst *GEP = GetElementPtrInst::Create(
          StructTy, &*AI, Idx, "gep_" + inputs[i]->getName(), TI);
      RewriteVal = new LoadInst(StructTy->getElementType(i), GEP,
                                "loadgep_" + inputs[i]->getName(), TI);
    } else
      RewriteVal = &*AI++;

    std::vector<User *> Users(inputs[i]->user_begin(), inputs[i]->user_end());
    for (User *use : Users)
      if (Instruction *inst = dyn_cast<Instruction>(use))
        if (Blocks.count(inst->getParent()))
          inst->replaceUsesOfWith(inputs[i], RewriteVal);
  }

  // Set names for input and output arguments.
  if (!AggregateArgs) {
    AI = newFunction->arg_begin();
    for (unsigned i = 0, e = inputs.size(); i != e; ++i, ++AI)
      AI->setName(inputs[i]->getName());
    for (unsigned i = 0, e = outputs.size(); i != e; ++i, ++AI)
      AI->setName(outputs[i]->getName()+".out");
  }

  // Rewrite branches to basic blocks outside of the loop to new dummy blocks
  // within the new function. This must be done before we lose track of which
  // blocks were originally in the code region.
  std::vector<User *> Users(header->user_begin(), header->user_end());
  for (auto &U : Users)
    // The BasicBlock which contains the branch is not in the region
    // modify the branch target to a new block
    if (Instruction *I = dyn_cast<Instruction>(U))
      if (I->isTerminator() && I->getFunction() == oldFunction &&
          !Blocks.count(I->getParent()))
        I->replaceUsesOfWith(header, newHeader);

  return newFunction;
}

/// Erase lifetime.start markers which reference inputs to the extraction
/// region, and insert the referenced memory into \p LifetimesStart.
///
/// The extraction region is defined by a set of blocks (\p Blocks), and a set
/// of allocas which will be moved from the caller function into the extracted
/// function (\p SunkAllocas).
static void eraseLifetimeMarkersOnInputs(const SetVector<BasicBlock *> &Blocks,
                                         const SetVector<Value *> &SunkAllocas,
                                         SetVector<Value *> &LifetimesStart) {
  for (BasicBlock *BB : Blocks) {
    for (auto It = BB->begin(), End = BB->end(); It != End;) {
      auto *II = dyn_cast<IntrinsicInst>(&*It);
      ++It;
      if (!II || !II->isLifetimeStartOrEnd())
        continue;

      // Get the memory operand of the lifetime marker. If the underlying
      // object is a sunk alloca, or is otherwise defined in the extraction
      // region, the lifetime marker must not be erased.
      Value *Mem = II->getOperand(1)->stripInBoundsOffsets();
      if (SunkAllocas.count(Mem) || definedInRegion(Blocks, Mem))
        continue;

      if (II->getIntrinsicID() == Intrinsic::lifetime_start)
        LifetimesStart.insert(Mem);
      II->eraseFromParent();
    }
  }
}

/// Insert lifetime start/end markers surrounding the call to the new function
/// for objects defined in the caller.
static void insertLifetimeMarkersSurroundingCall(
    Module *M, ArrayRef<Value *> LifetimesStart, ArrayRef<Value *> LifetimesEnd,
    CallInst *TheCall) {
  LLVMContext &Ctx = M->getContext();
  auto Int8PtrTy = Type::getInt8PtrTy(Ctx);
  auto NegativeOne = ConstantInt::getSigned(Type::getInt64Ty(Ctx), -1);
  Instruction *Term = TheCall->getParent()->getTerminator();

  // The memory argument to a lifetime marker must be a i8*. Cache any bitcasts
  // needed to satisfy this requirement so they may be reused.
  DenseMap<Value *, Value *> Bitcasts;

  // Emit lifetime markers for the pointers given in \p Objects. Insert the
  // markers before the call if \p InsertBefore, and after the call otherwise.
  auto insertMarkers = [&](Function *MarkerFunc, ArrayRef<Value *> Objects,
                           bool InsertBefore) {
    for (Value *Mem : Objects) {
      assert((!isa<Instruction>(Mem) || cast<Instruction>(Mem)->getFunction() ==
                                            TheCall->getFunction()) &&
             "Input memory not defined in original function");
      Value *&MemAsI8Ptr = Bitcasts[Mem];
      if (!MemAsI8Ptr) {
        if (Mem->getType() == Int8PtrTy)
          MemAsI8Ptr = Mem;
        else
          MemAsI8Ptr =
              CastInst::CreatePointerCast(Mem, Int8PtrTy, "lt.cast", TheCall);
      }

      auto Marker = CallInst::Create(MarkerFunc, {NegativeOne, MemAsI8Ptr});
      if (InsertBefore)
        Marker->insertBefore(TheCall);
      else
        Marker->insertBefore(Term);
    }
  };

  if (!LifetimesStart.empty()) {
    auto StartFn = llvm::Intrinsic::getDeclaration(
        M, llvm::Intrinsic::lifetime_start, Int8PtrTy);
    insertMarkers(StartFn, LifetimesStart, /*InsertBefore=*/true);
  }

  if (!LifetimesEnd.empty()) {
    auto EndFn = llvm::Intrinsic::getDeclaration(
        M, llvm::Intrinsic::lifetime_end, Int8PtrTy);
    insertMarkers(EndFn, LifetimesEnd, /*InsertBefore=*/false);
  }
}

/// emitCallAndSwitchStatement - This method sets up the caller side by adding
/// the call instruction, splitting any PHI nodes in the header block as
/// necessary.
CallInst *CodeExtractor::emitCallAndSwitchStatement(Function *newFunction,
                                                    BasicBlock *codeReplacer,
                                                    ValueSet &inputs,
                                                    ValueSet &outputs) {
  // Emit a call to the new function, passing in: *pointer to struct (if
  // aggregating parameters), or plan inputs and allocated memory for outputs
  std::vector<Value *> params, StructValues, ReloadOutputs, Reloads;

  Module *M = newFunction->getParent();
  LLVMContext &Context = M->getContext();
  const DataLayout &DL = M->getDataLayout();
  CallInst *call = nullptr;

  // Add inputs as params, or to be filled into the struct
  unsigned ArgNo = 0;
  SmallVector<unsigned, 1> SwiftErrorArgs;
  for (Value *input : inputs) {
    if (AggregateArgs)
      StructValues.push_back(input);
    else {
      params.push_back(input);
      if (input->isSwiftError())
        SwiftErrorArgs.push_back(ArgNo);
    }
    ++ArgNo;
  }

  // Create allocas for the outputs
  for (Value *output : outputs) {
    if (AggregateArgs) {
      StructValues.push_back(output);
    } else {
      AllocaInst *alloca =
        new AllocaInst(output->getType(), DL.getAllocaAddrSpace(),
                       nullptr, output->getName() + ".loc",
                       &codeReplacer->getParent()->front().front());
      ReloadOutputs.push_back(alloca);
      params.push_back(alloca);
    }
  }

  StructType *StructArgTy = nullptr;
  AllocaInst *Struct = nullptr;
  if (AggregateArgs && (inputs.size() + outputs.size() > 0)) {
    std::vector<Type *> ArgTypes;
    for (ValueSet::iterator v = StructValues.begin(),
           ve = StructValues.end(); v != ve; ++v)
      ArgTypes.push_back((*v)->getType());

    // Allocate a struct at the beginning of this function
    StructArgTy = StructType::get(newFunction->getContext(), ArgTypes);
    Struct = new AllocaInst(StructArgTy, DL.getAllocaAddrSpace(), nullptr,
                            "structArg",
                            &codeReplacer->getParent()->front().front());
    params.push_back(Struct);

    for (unsigned i = 0, e = inputs.size(); i != e; ++i) {
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Type::getInt32Ty(Context));
      Idx[1] = ConstantInt::get(Type::getInt32Ty(Context), i);
      GetElementPtrInst *GEP = GetElementPtrInst::Create(
          StructArgTy, Struct, Idx, "gep_" + StructValues[i]->getName());
      codeReplacer->getInstList().push_back(GEP);
      StoreInst *SI = new StoreInst(StructValues[i], GEP);
      codeReplacer->getInstList().push_back(SI);
    }
  }

  // Emit the call to the function
  call = CallInst::Create(newFunction, params,
                          NumExitBlocks > 1 ? "targetBlock" : "");
  // Add debug location to the new call, if the original function has debug
  // info. In that case, the terminator of the entry block of the extracted
  // function contains the first debug location of the extracted function,
  // set in extractCodeRegion.
  if (codeReplacer->getParent()->getSubprogram()) {
    if (auto DL = newFunction->getEntryBlock().getTerminator()->getDebugLoc())
      call->setDebugLoc(DL);
  }
  codeReplacer->getInstList().push_back(call);

  // Set swifterror parameter attributes.
  for (unsigned SwiftErrArgNo : SwiftErrorArgs) {
    call->addParamAttr(SwiftErrArgNo, Attribute::SwiftError);
    newFunction->addParamAttr(SwiftErrArgNo, Attribute::SwiftError);
  }

  Function::arg_iterator OutputArgBegin = newFunction->arg_begin();
  unsigned FirstOut = inputs.size();
  if (!AggregateArgs)
    std::advance(OutputArgBegin, inputs.size());

  // Reload the outputs passed in by reference.
  for (unsigned i = 0, e = outputs.size(); i != e; ++i) {
    Value *Output = nullptr;
    if (AggregateArgs) {
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Type::getInt32Ty(Context));
      Idx[1] = ConstantInt::get(Type::getInt32Ty(Context), FirstOut + i);
      GetElementPtrInst *GEP = GetElementPtrInst::Create(
          StructArgTy, Struct, Idx, "gep_reload_" + outputs[i]->getName());
      codeReplacer->getInstList().push_back(GEP);
      Output = GEP;
    } else {
      Output = ReloadOutputs[i];
    }
    LoadInst *load = new LoadInst(outputs[i]->getType(), Output,
                                  outputs[i]->getName() + ".reload");
    Reloads.push_back(load);
    codeReplacer->getInstList().push_back(load);
    std::vector<User *> Users(outputs[i]->user_begin(), outputs[i]->user_end());
    for (unsigned u = 0, e = Users.size(); u != e; ++u) {
      Instruction *inst = cast<Instruction>(Users[u]);
      if (!Blocks.count(inst->getParent()))
        inst->replaceUsesOfWith(outputs[i], load);
    }
  }

  // Now we can emit a switch statement using the call as a value.
  SwitchInst *TheSwitch =
      SwitchInst::Create(Constant::getNullValue(Type::getInt16Ty(Context)),
                         codeReplacer, 0, codeReplacer);

  // Since there may be multiple exits from the original region, make the new
  // function return an unsigned, switch on that number.  This loop iterates
  // over all of the blocks in the extracted region, updating any terminator
  // instructions in the to-be-extracted region that branch to blocks that are
  // not in the region to be extracted.
  std::map<BasicBlock *, BasicBlock *> ExitBlockMap;

  unsigned switchVal = 0;
  for (BasicBlock *Block : Blocks) {
    Instruction *TI = Block->getTerminator();
    for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
      if (!Blocks.count(TI->getSuccessor(i))) {
        BasicBlock *OldTarget = TI->getSuccessor(i);
        // add a new basic block which returns the appropriate value
        BasicBlock *&NewTarget = ExitBlockMap[OldTarget];
        if (!NewTarget) {
          // If we don't already have an exit stub for this non-extracted
          // destination, create one now!
          NewTarget = BasicBlock::Create(Context,
                                         OldTarget->getName() + ".exitStub",
                                         newFunction);
          unsigned SuccNum = switchVal++;

          Value *brVal = nullptr;
          switch (NumExitBlocks) {
          case 0:
          case 1: break;  // No value needed.
          case 2:         // Conditional branch, return a bool
            brVal = ConstantInt::get(Type::getInt1Ty(Context), !SuccNum);
            break;
          default:
            brVal = ConstantInt::get(Type::getInt16Ty(Context), SuccNum);
            break;
          }

          ReturnInst::Create(Context, brVal, NewTarget);

          // Update the switch instruction.
          TheSwitch->addCase(ConstantInt::get(Type::getInt16Ty(Context),
                                              SuccNum),
                             OldTarget);
        }

        // rewrite the original branch instruction with this new target
        TI->setSuccessor(i, NewTarget);
      }
  }

  // Store the arguments right after the definition of output value.
  // This should be proceeded after creating exit stubs to be ensure that invoke
  // result restore will be placed in the outlined function.
  Function::arg_iterator OAI = OutputArgBegin;
  for (unsigned i = 0, e = outputs.size(); i != e; ++i) {
    auto *OutI = dyn_cast<Instruction>(outputs[i]);
    if (!OutI)
      continue;

    // Find proper insertion point.
    BasicBlock::iterator InsertPt;
    // In case OutI is an invoke, we insert the store at the beginning in the
    // 'normal destination' BB. Otherwise we insert the store right after OutI.
    if (auto *InvokeI = dyn_cast<InvokeInst>(OutI))
      InsertPt = InvokeI->getNormalDest()->getFirstInsertionPt();
    else if (auto *Phi = dyn_cast<PHINode>(OutI))
      InsertPt = Phi->getParent()->getFirstInsertionPt();
    else
      InsertPt = std::next(OutI->getIterator());

    Instruction *InsertBefore = &*InsertPt;
    assert((InsertBefore->getFunction() == newFunction ||
            Blocks.count(InsertBefore->getParent())) &&
           "InsertPt should be in new function");
    assert(OAI != newFunction->arg_end() &&
           "Number of output arguments should match "
           "the amount of defined values");
    if (AggregateArgs) {
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Type::getInt32Ty(Context));
      Idx[1] = ConstantInt::get(Type::getInt32Ty(Context), FirstOut + i);
      GetElementPtrInst *GEP = GetElementPtrInst::Create(
          StructArgTy, &*OAI, Idx, "gep_" + outputs[i]->getName(),
          InsertBefore);
      new StoreInst(outputs[i], GEP, InsertBefore);
      // Since there should be only one struct argument aggregating
      // all the output values, we shouldn't increment OAI, which always
      // points to the struct argument, in this case.
    } else {
      new StoreInst(outputs[i], &*OAI, InsertBefore);
      ++OAI;
    }
  }

  // Now that we've done the deed, simplify the switch instruction.
  Type *OldFnRetTy = TheSwitch->getParent()->getParent()->getReturnType();
  switch (NumExitBlocks) {
  case 0:
    // There are no successors (the block containing the switch itself), which
    // means that previously this was the last part of the function, and hence
    // this should be rewritten as a `ret'

    // Check if the function should return a value
    if (OldFnRetTy->isVoidTy()) {
      ReturnInst::Create(Context, nullptr, TheSwitch);  // Return void
    } else if (OldFnRetTy == TheSwitch->getCondition()->getType()) {
      // return what we have
      ReturnInst::Create(Context, TheSwitch->getCondition(), TheSwitch);
    } else {
      // Otherwise we must have code extracted an unwind or something, just
      // return whatever we want.
      ReturnInst::Create(Context,
                         Constant::getNullValue(OldFnRetTy), TheSwitch);
    }

    TheSwitch->eraseFromParent();
    break;
  case 1:
    // Only a single destination, change the switch into an unconditional
    // branch.
    BranchInst::Create(TheSwitch->getSuccessor(1), TheSwitch);
    TheSwitch->eraseFromParent();
    break;
  case 2:
    BranchInst::Create(TheSwitch->getSuccessor(1), TheSwitch->getSuccessor(2),
                       call, TheSwitch);
    TheSwitch->eraseFromParent();
    break;
  default:
    // Otherwise, make the default destination of the switch instruction be one
    // of the other successors.
    TheSwitch->setCondition(call);
    TheSwitch->setDefaultDest(TheSwitch->getSuccessor(NumExitBlocks));
    // Remove redundant case
    TheSwitch->removeCase(SwitchInst::CaseIt(TheSwitch, NumExitBlocks-1));
    break;
  }

  // Insert lifetime markers around the reloads of any output values. The
  // allocas output values are stored in are only in-use in the codeRepl block.
  insertLifetimeMarkersSurroundingCall(M, ReloadOutputs, ReloadOutputs, call);

  return call;
}

void CodeExtractor::moveCodeToFunction(Function *newFunction) {
  Function *oldFunc = (*Blocks.begin())->getParent();
  Function::BasicBlockListType &oldBlocks = oldFunc->getBasicBlockList();
  Function::BasicBlockListType &newBlocks = newFunction->getBasicBlockList();

  for (BasicBlock *Block : Blocks) {
    // Delete the basic block from the old function, and the list of blocks
    oldBlocks.remove(Block);

    // Insert this basic block into the new function
    newBlocks.push_back(Block);
  }
}

void CodeExtractor::calculateNewCallTerminatorWeights(
    BasicBlock *CodeReplacer,
    DenseMap<BasicBlock *, BlockFrequency> &ExitWeights,
    BranchProbabilityInfo *BPI) {
  using Distribution = BlockFrequencyInfoImplBase::Distribution;
  using BlockNode = BlockFrequencyInfoImplBase::BlockNode;

  // Update the branch weights for the exit block.
  Instruction *TI = CodeReplacer->getTerminator();
  SmallVector<unsigned, 8> BranchWeights(TI->getNumSuccessors(), 0);

  // Block Frequency distribution with dummy node.
  Distribution BranchDist;

  // Add each of the frequencies of the successors.
  for (unsigned i = 0, e = TI->getNumSuccessors(); i < e; ++i) {
    BlockNode ExitNode(i);
    uint64_t ExitFreq = ExitWeights[TI->getSuccessor(i)].getFrequency();
    if (ExitFreq != 0)
      BranchDist.addExit(ExitNode, ExitFreq);
    else
      BPI->setEdgeProbability(CodeReplacer, i, BranchProbability::getZero());
  }

  // Check for no total weight.
  if (BranchDist.Total == 0)
    return;

  // Normalize the distribution so that they can fit in unsigned.
  BranchDist.normalize();

  // Create normalized branch weights and set the metadata.
  for (unsigned I = 0, E = BranchDist.Weights.size(); I < E; ++I) {
    const auto &Weight = BranchDist.Weights[I];

    // Get the weight and update the current BFI.
    BranchWeights[Weight.TargetNode.Index] = Weight.Amount;
    BranchProbability BP(Weight.Amount, BranchDist.Total);
    BPI->setEdgeProbability(CodeReplacer, Weight.TargetNode.Index, BP);
  }
  TI->setMetadata(
      LLVMContext::MD_prof,
      MDBuilder(TI->getContext()).createBranchWeights(BranchWeights));
}

/// Erase debug info intrinsics which refer to values in \p F but aren't in
/// \p F.
static void eraseDebugIntrinsicsWithNonLocalRefs(Function &F) {
  for (Instruction &I : instructions(F)) {
    SmallVector<DbgVariableIntrinsic *, 4> DbgUsers;
    findDbgUsers(DbgUsers, &I);
    for (DbgVariableIntrinsic *DVI : DbgUsers)
      if (DVI->getFunction() != &F)
        DVI->eraseFromParent();
  }
}

/// Fix up the debug info in the old and new functions by pointing line
/// locations and debug intrinsics to the new subprogram scope, and by deleting
/// intrinsics which point to values outside of the new function.
static void fixupDebugInfoPostExtraction(Function &OldFunc, Function &NewFunc,
                                         CallInst &TheCall) {
  DISubprogram *OldSP = OldFunc.getSubprogram();
  LLVMContext &Ctx = OldFunc.getContext();

  if (!OldSP) {
    // Erase any debug info the new function contains.
    stripDebugInfo(NewFunc);
    // Make sure the old function doesn't contain any non-local metadata refs.
    eraseDebugIntrinsicsWithNonLocalRefs(NewFunc);
    return;
  }

  // Create a subprogram for the new function. Leave out a description of the
  // function arguments, as the parameters don't correspond to anything at the
  // source level.
  assert(OldSP->getUnit() && "Missing compile unit for subprogram");
  DIBuilder DIB(*OldFunc.getParent(), /*AllowUnresolvedNodes=*/false,
                OldSP->getUnit());
  auto SPType = DIB.createSubroutineType(DIB.getOrCreateTypeArray(None));
  DISubprogram::DISPFlags SPFlags = DISubprogram::SPFlagDefinition |
                                    DISubprogram::SPFlagOptimized |
                                    DISubprogram::SPFlagLocalToUnit;
  auto NewSP = DIB.createFunction(
      OldSP->getUnit(), NewFunc.getName(), NewFunc.getName(), OldSP->getFile(),
      /*LineNo=*/0, SPType, /*ScopeLine=*/0, DINode::FlagZero, SPFlags);
  NewFunc.setSubprogram(NewSP);

  // Debug intrinsics in the new function need to be updated in one of two
  // ways:
  //  1) They need to be deleted, because they describe a value in the old
  //     function.
  //  2) They need to point to fresh metadata, e.g. because they currently
  //     point to a variable in the wrong scope.
  SmallDenseMap<DINode *, DINode *> RemappedMetadata;
  SmallVector<Instruction *, 4> DebugIntrinsicsToDelete;
  for (Instruction &I : instructions(NewFunc)) {
    auto *DII = dyn_cast<DbgInfoIntrinsic>(&I);
    if (!DII)
      continue;

    // Point the intrinsic to a fresh label within the new function.
    if (auto *DLI = dyn_cast<DbgLabelInst>(&I)) {
      DILabel *OldLabel = DLI->getLabel();
      DINode *&NewLabel = RemappedMetadata[OldLabel];
      if (!NewLabel)
        NewLabel = DILabel::get(Ctx, NewSP, OldLabel->getName(),
                                OldLabel->getFile(), OldLabel->getLine());
      DLI->setArgOperand(0, MetadataAsValue::get(Ctx, NewLabel));
      continue;
    }

    // If the location isn't a constant or an instruction, delete the
    // intrinsic.
    auto *DVI = cast<DbgVariableIntrinsic>(DII);
    Value *Location = DVI->getVariableLocation();
    if (!Location ||
        (!isa<Constant>(Location) && !isa<Instruction>(Location))) {
      DebugIntrinsicsToDelete.push_back(DVI);
      continue;
    }

    // If the variable location is an instruction but isn't in the new
    // function, delete the intrinsic.
    Instruction *LocationInst = dyn_cast<Instruction>(Location);
    if (LocationInst && LocationInst->getFunction() != &NewFunc) {
      DebugIntrinsicsToDelete.push_back(DVI);
      continue;
    }

    // Point the intrinsic to a fresh variable within the new function.
    DILocalVariable *OldVar = DVI->getVariable();
    DINode *&NewVar = RemappedMetadata[OldVar];
    if (!NewVar)
      NewVar = DIB.createAutoVariable(
          NewSP, OldVar->getName(), OldVar->getFile(), OldVar->getLine(),
          OldVar->getType(), /*AlwaysPreserve=*/false, DINode::FlagZero,
          OldVar->getAlignInBits());
    DVI->setArgOperand(1, MetadataAsValue::get(Ctx, NewVar));
  }
  for (auto *DII : DebugIntrinsicsToDelete)
    DII->eraseFromParent();
  DIB.finalizeSubprogram(NewSP);

  // Fix up the scope information attached to the line locations in the new
  // function.
  for (Instruction &I : instructions(NewFunc)) {
    if (const DebugLoc &DL = I.getDebugLoc())
      I.setDebugLoc(DebugLoc::get(DL.getLine(), DL.getCol(), NewSP));

    // Loop info metadata may contain line locations. Fix them up.
    auto updateLoopInfoLoc = [&Ctx,
                              NewSP](const DILocation &Loc) -> DILocation * {
      return DILocation::get(Ctx, Loc.getLine(), Loc.getColumn(), NewSP,
                             nullptr);
    };
    updateLoopMetadataDebugLocations(I, updateLoopInfoLoc);
  }
  if (!TheCall.getDebugLoc())
    TheCall.setDebugLoc(DebugLoc::get(0, 0, OldSP));

  eraseDebugIntrinsicsWithNonLocalRefs(NewFunc);
}

Function *
CodeExtractor::extractCodeRegion(const CodeExtractorAnalysisCache &CEAC) {
  if (!isEligible())
    return nullptr;

  // Assumption: this is a single-entry code region, and the header is the first
  // block in the region.
  BasicBlock *header = *Blocks.begin();
  Function *oldFunction = header->getParent();

  // Calculate the entry frequency of the new function before we change the root
  //   block.
  BlockFrequency EntryFreq;
  if (BFI) {
    assert(BPI && "Both BPI and BFI are required to preserve profile info");
    for (BasicBlock *Pred : predecessors(header)) {
      if (Blocks.count(Pred))
        continue;
      EntryFreq +=
          BFI->getBlockFreq(Pred) * BPI->getEdgeProbability(Pred, header);
    }
  }

  // Remove @llvm.assume calls that will be moved to the new function from the
  // old function's assumption cache.
  for (BasicBlock *Block : Blocks) {
    for (auto It = Block->begin(), End = Block->end(); It != End;) {
      Instruction *I = &*It;
      ++It;

      if (match(I, m_Intrinsic<Intrinsic::assume>())) {
        if (AC)
          AC->unregisterAssumption(cast<CallInst>(I));
        I->eraseFromParent();
      }
    }
  }

  // If we have any return instructions in the region, split those blocks so
  // that the return is not in the region.
  splitReturnBlocks();

  // Calculate the exit blocks for the extracted region and the total exit
  // weights for each of those blocks.
  DenseMap<BasicBlock *, BlockFrequency> ExitWeights;
  SmallPtrSet<BasicBlock *, 1> ExitBlocks;
  for (BasicBlock *Block : Blocks) {
    for (succ_iterator SI = succ_begin(Block), SE = succ_end(Block); SI != SE;
         ++SI) {
      if (!Blocks.count(*SI)) {
        // Update the branch weight for this successor.
        if (BFI) {
          BlockFrequency &BF = ExitWeights[*SI];
          BF += BFI->getBlockFreq(Block) * BPI->getEdgeProbability(Block, *SI);
        }
        ExitBlocks.insert(*SI);
      }
    }
  }
  NumExitBlocks = ExitBlocks.size();

  // If we have to split PHI nodes of the entry or exit blocks, do so now.
  severSplitPHINodesOfEntry(header);
  severSplitPHINodesOfExits(ExitBlocks);

  // This takes place of the original loop
  BasicBlock *codeReplacer = BasicBlock::Create(header->getContext(),
                                                "codeRepl", oldFunction,
                                                header);

  // The new function needs a root node because other nodes can branch to the
  // head of the region, but the entry node of a function cannot have preds.
  BasicBlock *newFuncRoot = BasicBlock::Create(header->getContext(),
                                               "newFuncRoot");
  auto *BranchI = BranchInst::Create(header);
  // If the original function has debug info, we have to add a debug location
  // to the new branch instruction from the artificial entry block.
  // We use the debug location of the first instruction in the extracted
  // blocks, as there is no other equivalent line in the source code.
  if (oldFunction->getSubprogram()) {
    any_of(Blocks, [&BranchI](const BasicBlock *BB) {
      return any_of(*BB, [&BranchI](const Instruction &I) {
        if (!I.getDebugLoc())
          return false;
        BranchI->setDebugLoc(I.getDebugLoc());
        return true;
      });
    });
  }
  newFuncRoot->getInstList().push_back(BranchI);

  ValueSet inputs, outputs, SinkingCands, HoistingCands;
  BasicBlock *CommonExit = nullptr;
  findAllocas(CEAC, SinkingCands, HoistingCands, CommonExit);
  assert(HoistingCands.empty() || CommonExit);

  // Find inputs to, outputs from the code region.
  findInputsOutputs(inputs, outputs, SinkingCands);

  // Now sink all instructions which only have non-phi uses inside the region.
  // Group the allocas at the start of the block, so that any bitcast uses of
  // the allocas are well-defined.
  AllocaInst *FirstSunkAlloca = nullptr;
  for (auto *II : SinkingCands) {
    if (auto *AI = dyn_cast<AllocaInst>(II)) {
      AI->moveBefore(*newFuncRoot, newFuncRoot->getFirstInsertionPt());
      if (!FirstSunkAlloca)
        FirstSunkAlloca = AI;
    }
  }
  assert((SinkingCands.empty() || FirstSunkAlloca) &&
         "Did not expect a sink candidate without any allocas");
  for (auto *II : SinkingCands) {
    if (!isa<AllocaInst>(II)) {
      cast<Instruction>(II)->moveAfter(FirstSunkAlloca);
    }
  }

  if (!HoistingCands.empty()) {
    auto *HoistToBlock = findOrCreateBlockForHoisting(CommonExit);
    Instruction *TI = HoistToBlock->getTerminator();
    for (auto *II : HoistingCands)
      cast<Instruction>(II)->moveBefore(TI);
  }

  // Collect objects which are inputs to the extraction region and also
  // referenced by lifetime start markers within it. The effects of these
  // markers must be replicated in the calling function to prevent the stack
  // coloring pass from merging slots which store input objects.
  ValueSet LifetimesStart;
  eraseLifetimeMarkersOnInputs(Blocks, SinkingCands, LifetimesStart);

  // Construct new function based on inputs/outputs & add allocas for all defs.
  Function *newFunction =
      constructFunction(inputs, outputs, header, newFuncRoot, codeReplacer,
                        oldFunction, oldFunction->getParent());

  // Update the entry count of the function.
  if (BFI) {
    auto Count = BFI->getProfileCountFromFreq(EntryFreq.getFrequency());
    if (Count.hasValue())
      newFunction->setEntryCount(
          ProfileCount(Count.getValue(), Function::PCT_Real)); // FIXME
    BFI->setBlockFreq(codeReplacer, EntryFreq.getFrequency());
  }

  CallInst *TheCall =
      emitCallAndSwitchStatement(newFunction, codeReplacer, inputs, outputs);

  moveCodeToFunction(newFunction);

  // Replicate the effects of any lifetime start/end markers which referenced
  // input objects in the extraction region by placing markers around the call.
  insertLifetimeMarkersSurroundingCall(
      oldFunction->getParent(), LifetimesStart.getArrayRef(), {}, TheCall);

  // Propagate personality info to the new function if there is one.
  if (oldFunction->hasPersonalityFn())
    newFunction->setPersonalityFn(oldFunction->getPersonalityFn());

  // Update the branch weights for the exit block.
  if (BFI && NumExitBlocks > 1)
    calculateNewCallTerminatorWeights(codeReplacer, ExitWeights, BPI);

  // Loop over all of the PHI nodes in the header and exit blocks, and change
  // any references to the old incoming edge to be the new incoming edge.
  for (BasicBlock::iterator I = header->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
      if (!Blocks.count(PN->getIncomingBlock(i)))
        PN->setIncomingBlock(i, newFuncRoot);
  }

  for (BasicBlock *ExitBB : ExitBlocks)
    for (PHINode &PN : ExitBB->phis()) {
      Value *IncomingCodeReplacerVal = nullptr;
      for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i) {
        // Ignore incoming values from outside of the extracted region.
        if (!Blocks.count(PN.getIncomingBlock(i)))
          continue;

        // Ensure that there is only one incoming value from codeReplacer.
        if (!IncomingCodeReplacerVal) {
          PN.setIncomingBlock(i, codeReplacer);
          IncomingCodeReplacerVal = PN.getIncomingValue(i);
        } else
          assert(IncomingCodeReplacerVal == PN.getIncomingValue(i) &&
                 "PHI has two incompatbile incoming values from codeRepl");
      }
    }

  fixupDebugInfoPostExtraction(*oldFunction, *newFunction, *TheCall);

  // Mark the new function `noreturn` if applicable. Terminators which resume
  // exception propagation are treated as returning instructions. This is to
  // avoid inserting traps after calls to outlined functions which unwind.
  bool doesNotReturn = none_of(*newFunction, [](const BasicBlock &BB) {
    const Instruction *Term = BB.getTerminator();
    return isa<ReturnInst>(Term) || isa<ResumeInst>(Term);
  });
  if (doesNotReturn)
    newFunction->setDoesNotReturn();

  LLVM_DEBUG(if (verifyFunction(*newFunction, &errs())) {
    newFunction->dump();
    report_fatal_error("verification of newFunction failed!");
  });
  LLVM_DEBUG(if (verifyFunction(*oldFunction))
             report_fatal_error("verification of oldFunction failed!"));
  LLVM_DEBUG(if (AC && verifyAssumptionCache(*oldFunction, *newFunction, AC))
                 report_fatal_error("Stale Asumption cache for old Function!"));
  return newFunction;
}

bool CodeExtractor::verifyAssumptionCache(const Function &OldFunc,
                                          const Function &NewFunc,
                                          AssumptionCache *AC) {
  for (auto AssumeVH : AC->assumptions()) {
    CallInst *I = dyn_cast_or_null<CallInst>(AssumeVH);
    if (!I)
      continue;

    // There shouldn't be any llvm.assume intrinsics in the new function.
    if (I->getFunction() != &OldFunc)
      return true;

    // There shouldn't be any stale affected values in the assumption cache
    // that were previously in the old function, but that have now been moved
    // to the new function.
    for (auto AffectedValVH : AC->assumptionsFor(I->getOperand(0))) {
      CallInst *AffectedCI = dyn_cast_or_null<CallInst>(AffectedValVH);
      if (!AffectedCI)
        continue;
      if (AffectedCI->getFunction() != &OldFunc)
        return true;
      auto *AssumedInst = dyn_cast<Instruction>(AffectedCI->getOperand(0));
      if (AssumedInst->getFunction() != &OldFunc)
        return true;
    }
  }
  return false;
}
