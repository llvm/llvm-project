//===-- SISinkAsyncDMA.cpp - Sink async DMA out of divergent branches ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Sink async DMA intrinsics out of divergent then-blocks after
/// SIAnnotateControlFlow. Fully-masked waves otherwise skip the DMA, so the
/// ASYNCcnt observed at the join depends on whether the wave took the branch.
/// Software-pipelined kernels then have to use a conservative async waitcnt.
///
/// Move the DMA before llvm.amdgcn.else or llvm.amdgcn.end.cf to keep the
/// then-block's EXEC mask while making every wave update ASYNCcnt.
///
///        MBB                     MBB        llvm.amdgcn.if sets EXEC to then
///       /   \                   /   \       block mask before the branch, so
///   ThenBB   |               ThenBB  |      both edges carry it and per-lane
///    [DMA]   |      ==>          \  /       behavior is unchanged. But every
///       \   /                   JoinBB      wave now issues the DMA, so
///      JoinBB                    [DMA]      ASYNCcnt at the join no longer
///    [amdgcn.end.cf]                   |        depends on the branch.
///                             [amdgcn.end.cf]
///
/// The join is split so that amdgcn.end.cf starts a block of its own, because
/// SILowerControlFlow emits the EXEC restore at the top of the block holding
/// it, which would otherwise place it above the sunk DMAs.

//
//===----------------------------------------------------------------------===//

#include "SISinkAsyncDMA.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "si-sink-async-dma"

namespace {

class SISinkAsyncDMA {
  DomTreeUpdater &DTU;
  LoopInfo &LI;

  bool sinkFromBoundary(IntrinsicInst &Boundary);

public:
  SISinkAsyncDMA(DomTreeUpdater &DTU, LoopInfo &LI) : DTU(DTU), LI(LI) {}

  bool run(Function &F);
};

class SISinkAsyncDMALegacy : public FunctionPass {
public:
  static char ID;

  SISinkAsyncDMALegacy() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

  StringRef getPassName() const override {
    return "SI sink async DMA out of divergent then-blocks";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<LoopInfoWrapperPass>();
    FunctionPass::getAnalysisUsage(AU);
  }
};

} // namespace

char SISinkAsyncDMALegacy::ID = 0;

INITIALIZE_PASS_BEGIN(SISinkAsyncDMALegacy, DEBUG_TYPE,
                      "SI sink async DMA out of divergent then-blocks", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(SISinkAsyncDMALegacy, DEBUG_TYPE,
                    "SI sink async DMA out of divergent then-blocks", false,
                    false)

char &llvm::SISinkAsyncDMALegacyID = SISinkAsyncDMALegacy::ID;

#define GENERATE_ASYNC_DMA_CASES(NAME)                                         \
  case Intrinsic::NAME##_b8:                                                   \
  case Intrinsic::NAME##_b32:                                                  \
  case Intrinsic::NAME##_b64:                                                  \
  case Intrinsic::NAME##_b128:

static bool usesAsynccnt(const IntrinsicInst &II) {
  switch (II.getIntrinsicID()) {
    GENERATE_ASYNC_DMA_CASES(amdgcn_cluster_load_async_to_lds)
    GENERATE_ASYNC_DMA_CASES(amdgcn_global_load_async_to_lds)
    GENERATE_ASYNC_DMA_CASES(amdgcn_global_store_async_from_lds)
    return true;
  default:
    return false;
  }
}

#undef GENERATE_ASYNC_DMA_CASES

bool SISinkAsyncDMA::sinkFromBoundary(IntrinsicInst &Boundary) {
  using namespace PatternMatch;

  BasicBlock *JoinBB = Boundary.getParent();
  if (JoinBB->getFirstInsertionPt() != Boundary.getIterator() ||
      pred_size(JoinBB) != 2)
    return false;

  // The saved-EXEC operand of the boundary identifies the if/else that opened
  // the region.
  Value *ControlValue;
  if (!match(
          Boundary.getArgOperand(0),
          m_ExtractValue<1>(m_Value(
              ControlValue,
              m_AnyIntrinsic<Intrinsic::amdgcn_if, Intrinsic::amdgcn_else>()))))
    return false;

  auto *Control = cast<IntrinsicInst>(ControlValue);
  BasicBlock *HeadBB = Control->getParent();
  auto *HeadBr = dyn_cast<CondBrInst>(HeadBB->getTerminator());
  if (!HeadBr ||
      !match(HeadBr->getCondition(), m_ExtractValue<0>(m_Specific(Control))) ||
      HeadBr->getSuccessor(1) != JoinBB)
    return false;

  BasicBlock *ThenBB = HeadBr->getSuccessor(0);
  auto *ThenBr = dyn_cast<UncondBrInst>(ThenBB->getTerminator());
  if (ThenBB->getSinglePredecessor() != HeadBB || !ThenBr ||
      ThenBr->getSuccessor(0) != JoinBB)
    return false;

  SmallVector<IntrinsicInst *, 4> ToSink;
  bool CrossedBarrier = false;
  for (Instruction &I :
       reverse(make_range(ThenBB->begin(), ThenBr->getIterator()))) {
    if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
      // Keep async marker groups intact.
      Intrinsic::ID ID = II->getIntrinsicID();
      if (ID == Intrinsic::amdgcn_asyncmark ||
          ID == Intrinsic::amdgcn_wait_asyncmark)
        return false;

      if (usesAsynccnt(*II)) {
        if (CrossedBarrier)
          break;
        ToSink.push_back(II);
        continue;
      }
    }

    CrossedBarrier |= mayHaveNonDefUseDependency(I);
  }
  if (ToSink.empty())
    return false;

  SmallDenseMap<Instruction *, PHINode *, 4> MergedValues;
  for (IntrinsicInst *DMA : reverse(ToSink)) {
    for (Use &U : DMA->args()) {
      auto *Def = dyn_cast<Instruction>(U.get());
      if (!Def || Def->getParent() != ThenBB)
        continue;

      PHINode *&Merged = MergedValues[Def];
      if (!Merged) {
        Merged = PHINode::Create(Def->getType(), 2, Def->getName() + ".sink",
                                 JoinBB->getFirstNonPHIIt());
        // The head edge reaches the boundary with an empty EXEC mask, so this
        // value is not observed by the DMA.
        Merged->addIncoming(PoisonValue::get(Def->getType()), HeadBB);
        Merged->addIncoming(Def, ThenBB);
      }
      U.set(Merged);
    }

    LLVM_DEBUG(dbgs() << "Sinking async DMA out of divergent then-block: "
                      << *DMA);
    DMA->moveBefore(Boundary.getIterator());
  }

  // SILowerControlFlow inserts the EXEC restore at the beginning of the block
  // that contains the boundary.
  SplitBlock(JoinBB, Boundary.getIterator(), &DTU, &LI);
  return true;
}

bool SISinkAsyncDMA::run(Function &F) {
  // Collect the boundaries up front because sinking splits their blocks.
  SmallVector<IntrinsicInst *, 4> Boundaries;
  for (Instruction &I : instructions(F)) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (II && (II->getIntrinsicID() == Intrinsic::amdgcn_else ||
               II->getIntrinsicID() == Intrinsic::amdgcn_end_cf))
      Boundaries.push_back(II);
  }

  bool Changed = false;
  for (IntrinsicInst *Boundary : Boundaries)
    Changed |= sinkFromBoundary(*Boundary);
  return Changed;
}

bool SISinkAsyncDMALegacy::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  const TargetMachine &TM =
      getAnalysis<TargetPassConfig>().getTM<TargetMachine>();
  if (!TM.getSubtarget<GCNSubtarget>(F).hasAsynccnt())
    return false;

  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Lazy);
  return SISinkAsyncDMA(DTU, LI).run(F);
}

PreservedAnalyses SISinkAsyncDMAPass::run(Function &F,
                                          FunctionAnalysisManager &FAM) {
  if (!TM.getSubtarget<GCNSubtarget>(F).hasAsynccnt())
    return PreservedAnalyses::all();

  DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);
  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Lazy);
  if (!SISinkAsyncDMA(DTU, LI).run(F))
    return PreservedAnalyses::all();

  PreservedAnalyses PA = PreservedAnalyses::none();
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<LoopAnalysis>();
  return PA;
}
