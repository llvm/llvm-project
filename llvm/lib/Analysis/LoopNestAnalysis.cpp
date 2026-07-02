//===- LoopNestAnalysis.cpp - Loop Nest Analysis --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// The implementation for the loop nest analysis.
///
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopNestAnalysis.h"
#include "llvm/ADT/BreadthFirstIterator.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/Analysis/ValueTracking.h"

using namespace llvm;

#define DEBUG_TYPE "loopnest"
#ifndef NDEBUG
static const char *VerboseDebug = DEBUG_TYPE "-verbose";
#endif

/// Determine whether the loops structure violates basic requirements for
/// perfect nesting:
///  - the inner loop should be the outer loop's only child
///  - the outer loop header should 'flow' into the inner loop preheader
///    through at most empty blocks (no conditional branches allowed)
///  - if the inner loop latch exits the inner loop, it should 'flow' into
///    the outer loop latch.
/// Returns true if the loop structure satisfies the basic requirements and
/// false otherwise.
static bool checkLoopsStructure(const Loop &OuterLoop, const Loop &InnerLoop,
                                ScalarEvolution &SE);

//===----------------------------------------------------------------------===//
// LoopNest implementation
//

LoopNest::LoopNest(Loop &Root, ScalarEvolution &SE)
    : MaxPerfectDepth(getMaxPerfectDepth(Root, SE)) {
  append_range(Loops, breadth_first(&Root));
}

std::unique_ptr<LoopNest> LoopNest::getLoopNest(Loop &Root,
                                                ScalarEvolution &SE) {
  return std::make_unique<LoopNest>(Root, SE);
}

static CmpInst *getOuterLoopLatchCmp(const Loop &OuterLoop) {

  const BasicBlock *Latch = OuterLoop.getLoopLatch();
  assert(Latch && "Expecting a valid loop latch");

  const CondBrInst *BI = dyn_cast<CondBrInst>(Latch->getTerminator());
  assert(BI && "Expecting loop latch terminator to be a branch instruction");

  CmpInst *OuterLoopLatchCmp = dyn_cast<CmpInst>(BI->getCondition());
  DEBUG_WITH_TYPE(
      VerboseDebug, if (OuterLoopLatchCmp) {
        dbgs() << "Outer loop latch compare instruction: " << *OuterLoopLatchCmp
               << "\n";
      });
  return OuterLoopLatchCmp;
}

static bool checkSafeInstruction(const Instruction &I,
                                 const CmpInst *OuterLoopLatchCmp,
                                 std::optional<Loop::LoopBounds> OuterLoopLB) {

  bool IsAllowed = isSafeToSpeculativelyExecute(&I) || isa<PHINode>(I) ||
                   isa<UncondBrInst>(I) || isa<CondBrInst>(I);
  if (!IsAllowed)
    return false;
  // The only binary instruction allowed is the outer loop step instruction,
  // and the only comparison instruction allowed is the outer loop latch
  // compare instruction.
  if ((isa<BinaryOperator>(I) && &I != &OuterLoopLB->getStepInst()) ||
      (isa<CmpInst>(I) && &I != OuterLoopLatchCmp)) {
    return false;
  }
  return true;
}

bool LoopNest::arePerfectlyNested(const Loop &OuterLoop, const Loop &InnerLoop,
                                  ScalarEvolution &SE) {
  return (analyzeLoopNestForPerfectNest(OuterLoop, InnerLoop, SE) ==
          PerfectLoopNest);
}

LoopNest::LoopNestEnum LoopNest::analyzeLoopNestForPerfectNest(
    const Loop &OuterLoop, const Loop &InnerLoop, ScalarEvolution &SE) {

  assert(!OuterLoop.isInnermost() && "Outer loop should have subloops");
  assert(!InnerLoop.isOutermost() && "Inner loop should have a parent");
  LLVM_DEBUG(dbgs() << "Checking whether loop '" << OuterLoop.getName()
                    << "' and '" << InnerLoop.getName()
                    << "' are perfectly nested.\n");

  // Determine whether the loops structure satisfies the following requirements:
  //  - the inner loop should be the outer loop's only child
  //  - the outer loop header should 'flow' into the inner loop preheader
  //    through at most empty blocks
  //  - if the inner loop latch exits the inner loop, it should 'flow' into
  //    the outer loop latch.
  if (!checkLoopsStructure(OuterLoop, InnerLoop, SE)) {
    LLVM_DEBUG(dbgs() << "Not perfectly nested: invalid loop structure.\n");
    return InvalidLoopStructure;
  }

  // Bail out if we cannot retrieve the outer loop bounds.
  auto OuterLoopLB = OuterLoop.getBounds(SE);
  if (OuterLoopLB == std::nullopt) {
    LLVM_DEBUG(dbgs() << "Cannot compute loop bounds of OuterLoop: "
                      << OuterLoop << "\n";);
    return OuterLoopLowerBoundUnknown;
  }

  CmpInst *OuterLoopLatchCmp = getOuterLoopLatchCmp(OuterLoop);

  // Determine whether instructions in a basic block are one of:
  //  - the outer loop latch comparison
  //  - the outer loop induction variable increment
  //  - a phi node, a cast or a branch
  auto containsOnlySafeInstructions = [&](const BasicBlock &BB) {
    return llvm::all_of(BB, [&](const Instruction &I) {
      bool IsSafeInstr =
          checkSafeInstruction(I, OuterLoopLatchCmp, OuterLoopLB);
      if (!IsSafeInstr) {
        DEBUG_WITH_TYPE(VerboseDebug, {
          dbgs() << "Instruction: " << I << "\nin basic block:" << BB
                 << "is unsafe.\n";
        });
      }
      return IsSafeInstr;
    });
  };

  // Check the code surrounding the inner loop for instructions that are deemed
  // unsafe.
  const BasicBlock *OuterLoopHeader = OuterLoop.getHeader();
  const BasicBlock *OuterLoopLatch = OuterLoop.getLoopLatch();
  const BasicBlock *InnerLoopPreHeader = InnerLoop.getLoopPreheader();

  if (!containsOnlySafeInstructions(*OuterLoopHeader) ||
      !containsOnlySafeInstructions(*OuterLoopLatch) ||
      (InnerLoopPreHeader != OuterLoopHeader &&
       !containsOnlySafeInstructions(*InnerLoopPreHeader)) ||
      !containsOnlySafeInstructions(*InnerLoop.getExitBlock())) {
    LLVM_DEBUG(dbgs() << "Not perfectly nested: code surrounding inner loop is "
                         "unsafe\n";);
    return ImperfectLoopNest;
  }

  LLVM_DEBUG(dbgs() << "Loop '" << OuterLoop.getName() << "' and '"
                    << InnerLoop.getName() << "' are perfectly nested.\n");

  return PerfectLoopNest;
}

LoopNest::InstrVectorTy LoopNest::getInterveningInstructions(
    const Loop &OuterLoop, const Loop &InnerLoop, ScalarEvolution &SE) {
  InstrVectorTy Instr;
  switch (analyzeLoopNestForPerfectNest(OuterLoop, InnerLoop, SE)) {
  case PerfectLoopNest:
    LLVM_DEBUG(dbgs() << "The loop Nest is Perfect, returning empty "
                         "instruction vector. \n";);
    return Instr;

  case InvalidLoopStructure:
    LLVM_DEBUG(dbgs() << "Not perfectly nested: invalid loop structure. "
                         "Instruction vector is empty.\n";);
    return Instr;

  case OuterLoopLowerBoundUnknown:
    LLVM_DEBUG(dbgs() << "Cannot compute loop bounds of OuterLoop: "
                      << OuterLoop << "\nInstruction vector is empty.\n";);
    return Instr;

  case ImperfectLoopNest:
    break;
  }

  // Identify the outer loop latch comparison instruction.
  auto OuterLoopLB = OuterLoop.getBounds(SE);

  CmpInst *OuterLoopLatchCmp = getOuterLoopLatchCmp(OuterLoop);

  auto GetUnsafeInstructions = [&](const BasicBlock &BB) {
    for (const Instruction &I : BB) {
      if (!checkSafeInstruction(I, OuterLoopLatchCmp, OuterLoopLB)) {
        Instr.push_back(&I);
        DEBUG_WITH_TYPE(VerboseDebug, {
          dbgs() << "Instruction: " << I << "\nin basic block:" << BB
                 << "is unsafe.\n";
        });
      }
    }
  };

  // Check the code surrounding the inner loop for instructions that are deemed
  // unsafe.
  const BasicBlock *OuterLoopHeader = OuterLoop.getHeader();
  const BasicBlock *OuterLoopLatch = OuterLoop.getLoopLatch();
  const BasicBlock *InnerLoopPreHeader = InnerLoop.getLoopPreheader();
  const BasicBlock *InnerLoopExitBlock = InnerLoop.getExitBlock();

  GetUnsafeInstructions(*OuterLoopHeader);
  GetUnsafeInstructions(*OuterLoopLatch);
  GetUnsafeInstructions(*InnerLoopExitBlock);

  if (InnerLoopPreHeader != OuterLoopHeader) {
    GetUnsafeInstructions(*InnerLoopPreHeader);
  }
  return Instr;
}

SmallVector<LoopVectorTy, 4>
LoopNest::getPerfectLoops(ScalarEvolution &SE) const {
  SmallVector<LoopVectorTy, 4> LV;
  LoopVectorTy PerfectNest;

  for (Loop *L : depth_first(const_cast<Loop *>(Loops.front()))) {
    if (PerfectNest.empty())
      PerfectNest.push_back(L);

    auto &SubLoops = L->getSubLoops();
    if (SubLoops.size() == 1 && arePerfectlyNested(*L, *SubLoops.front(), SE)) {
      PerfectNest.push_back(SubLoops.front());
    } else {
      LV.push_back(PerfectNest);
      PerfectNest.clear();
    }
  }

  return LV;
}

unsigned LoopNest::getMaxPerfectDepth(const Loop &Root, ScalarEvolution &SE) {
  LLVM_DEBUG(dbgs() << "Get maximum perfect depth of loop nest rooted by loop '"
                    << Root.getName() << "'\n");

  const Loop *CurrentLoop = &Root;
  const auto *SubLoops = &CurrentLoop->getSubLoops();
  unsigned CurrentDepth = 1;

  while (SubLoops->size() == 1) {
    const Loop *InnerLoop = SubLoops->front();
    if (!arePerfectlyNested(*CurrentLoop, *InnerLoop, SE)) {
      LLVM_DEBUG({
        dbgs() << "Not a perfect nest: loop '" << CurrentLoop->getName()
               << "' is not perfectly nested with loop '"
               << InnerLoop->getName() << "'\n";
      });
      break;
    }

    CurrentLoop = InnerLoop;
    SubLoops = &CurrentLoop->getSubLoops();
    ++CurrentDepth;
  }

  return CurrentDepth;
}

const BasicBlock &LoopNest::skipEmptyBlockUntil(const BasicBlock *From,
                                                const BasicBlock *End,
                                                bool CheckUniquePred) {
  assert(From && "Expecting valid From");
  assert(End && "Expecting valid End");

  if (From == End || !From->getUniqueSuccessor())
    return *From;

  auto IsEmpty = [](const BasicBlock *BB) {
    return (BB->size() == 1);
  };

  // Visited is used to avoid running into an infinite loop.
  SmallPtrSet<const BasicBlock *, 4> Visited;
  const BasicBlock *BB = From->getUniqueSuccessor();
  const BasicBlock *PredBB = From;
  while (BB && BB != End && IsEmpty(BB) && !Visited.count(BB) &&
         (!CheckUniquePred || BB->getUniquePredecessor())) {
    Visited.insert(BB);
    PredBB = BB;
    BB = BB->getUniqueSuccessor();
  }

  return (BB == End) ? *End : *PredBB;
}

static bool checkLoopsStructure(const Loop &OuterLoop, const Loop &InnerLoop,
                                ScalarEvolution &SE) {
  // The inner loop must be the only outer loop's child.
  if ((OuterLoop.getSubLoops().size() != 1) ||
      (InnerLoop.getParentLoop() != &OuterLoop))
    return false;

  // We expect loops in normal form which have a preheader, header, latch...
  if (!OuterLoop.isLoopSimplifyForm() || !InnerLoop.isLoopSimplifyForm())
    return false;

  const BasicBlock *OuterLoopHeader = OuterLoop.getHeader();
  const BasicBlock *OuterLoopLatch = OuterLoop.getLoopLatch();
  const BasicBlock *InnerLoopPreHeader = InnerLoop.getLoopPreheader();
  const BasicBlock *InnerLoopLatch = InnerLoop.getLoopLatch();
  const BasicBlock *InnerLoopExit = InnerLoop.getExitBlock();

  // We expect rotated loops. The inner loop should have a single exit block.
  if (OuterLoop.getExitingBlock() != OuterLoopLatch ||
      InnerLoop.getExitingBlock() != InnerLoopLatch || !InnerLoopExit)
    return false;

  // Ensure the outer loop header flows directly into the inner loop preheader
  // through at most empty (unconditional) blocks. Loop-invariant guards that
  // previously branched from the outer header to the latch are removed by
  // SimpleLoopUnswitch run, so no conditional branch is expected here.
  if (OuterLoopHeader != InnerLoopPreHeader) {
    const BasicBlock &SingleSucc =
        LoopNest::skipEmptyBlockUntil(OuterLoopHeader, InnerLoopPreHeader);
    if (&SingleSucc != InnerLoopPreHeader)
      return false;
  }

  // Ensure the inner loop exit block leads to the outer loop latch possibly
  // through empty blocks.
  if (&LoopNest::skipEmptyBlockUntil(InnerLoop.getExitBlock(),
                                     OuterLoopLatch) != OuterLoopLatch) {
    DEBUG_WITH_TYPE(
        VerboseDebug,
        dbgs() << "Inner loop exit block " << *InnerLoopExit
               << " does not directly lead to the outer loop latch.\n";);
    return false;
  }

  return true;
}

AnalysisKey LoopNestAnalysis::Key;

raw_ostream &llvm::operator<<(raw_ostream &OS, const LoopNest &LN) {
  OS << "IsPerfect=";
  if (LN.getMaxPerfectDepth() == LN.getNestDepth())
    OS << "true";
  else
    OS << "false";
  OS << ", Depth=" << LN.getNestDepth();
  OS << ", OutermostLoop: " << LN.getOutermostLoop().getName();
  OS << ", Loops: ( ";
  for (const Loop *L : LN.getLoops())
    OS << L->getName() << " ";
  OS << ")";

  return OS;
}

//===----------------------------------------------------------------------===//
// LoopNestPrinterPass implementation
//

PreservedAnalyses LoopNestPrinterPass::run(Loop &L, LoopAnalysisManager &AM,
                                           LoopStandardAnalysisResults &AR,
                                           LPMUpdater &U) {
  if (auto LN = LoopNest::getLoopNest(L, AR.SE))
    OS << *LN << "\n";

  return PreservedAnalyses::all();
}
