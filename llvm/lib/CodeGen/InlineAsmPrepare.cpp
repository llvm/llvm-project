//===-- InlineAsmPrepare - Prepare inline asm for code generation ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers inline asm calls in LLVM IR in order to to assist
// SelectionDAG's codegen.
//
// CallBrInst:
//
//   Assists in inserting register copies for the output values of a callbr
//   along the edges leading to the indirect target blocks. Though the output
//   SSA value is defined by the callbr instruction itself in the IR
//   representation, the value cannot be copied to the appropriate virtual
//   registers prior to jumping to an indirect label, since the jump occurs
//   within the user-provided assembly blob.
//
//   Instead, those copies must occur separately at the beginning of each
//   indirect target. That requires that we create a separate SSA definition in
//   each of them (via llvm.callbr.landingpad), and may require splitting
//   critical edges so we have a location to place the intrinsic. Finally, we
//   remap users of the original callbr output SSA value to instead point to
//   the appropriate llvm.callbr.landingpad value.
//
//   Ideally, this could be done inside SelectionDAG, or in the
//   MachineInstruction representation, without the use of an IR-level
//   intrinsic.  But, within the current framework, it’s simpler to implement
//   as an IR pass.  (If support for callbr in GlobalISel is implemented, it’s
//   worth considering whether this is still required.)
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/InlineAsmPrepare.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"

using namespace llvm;

#define DEBUG_TYPE "inline-asm-prepare"

namespace {

class InlineAsmPrepare : public FunctionPass {
public:
  InlineAsmPrepare() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addPreserved<DominatorTreeWrapperPass>();
  }
  bool runOnFunction(Function &F) override;

  static char ID;
};

char InlineAsmPrepare::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(InlineAsmPrepare, "inline-asm-prepare",
                      "Prepare inline asm insts", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(InlineAsmPrepare, "inline-asm-prepare",
                    "Prepare inline asm insts", false, false)

FunctionPass *llvm::createInlineAsmPreparePass() {
  return new InlineAsmPrepare();
}

#ifndef NDEBUG
static void printDebugDomInfo(const DominatorTree &DT, const Use &U,
                              const BasicBlock *BB, bool IsDefaultDest) {
  if (isa<Instruction>(U.getUser()))
    LLVM_DEBUG(dbgs() << "Use: " << *U.getUser() << ", in block "
                      << cast<Instruction>(U.getUser())->getParent()->getName()
                      << ", is " << (DT.dominates(BB, U) ? "" : "NOT ")
                      << "dominated by " << BB->getName() << " ("
                      << (IsDefaultDest ? "in" : "") << "direct)\n");
}
#endif

/// The Use is in the same BasicBlock as the intrinsic call.
static bool isInSameBasicBlock(const Use &U, const BasicBlock *BB) {
  const auto *I = dyn_cast<Instruction>(U.getUser());
  return I && I->getParent() == BB;
}

static void updateSSA(DominatorTree &DT, CallBrInst *CBR, CallInst *Intrinsic,
                      SSAUpdater &SSAUpdate) {
  SmallPtrSet<Use *, 4> Visited;
  BasicBlock *DefaultDest = CBR->getDefaultDest();
  BasicBlock *LandingPad = Intrinsic->getParent();

  SmallVector<Use *, 4> Uses(make_pointer_range(CBR->uses()));
  for (Use *U : Uses) {
    if (!Visited.insert(U).second)
      continue;

#ifndef NDEBUG
    printDebugDomInfo(DT, *U, LandingPad, /*IsDefaultDest*/ false);
    printDebugDomInfo(DT, *U, DefaultDest, /*IsDefaultDest*/ true);
#endif

    // Don't rewrite the use in the newly inserted intrinsic.
    if (const auto *II = dyn_cast<IntrinsicInst>(U->getUser()))
      if (II->getIntrinsicID() == Intrinsic::callbr_landingpad)
        continue;

    // If the Use is in the same BasicBlock as the Intrinsic call, replace
    // the Use with the value of the Intrinsic call.
    if (isInSameBasicBlock(*U, LandingPad)) {
      U->set(Intrinsic);
      continue;
    }

    // If the Use is dominated by the default dest, do not touch it.
    if (DT.dominates(DefaultDest, *U))
      continue;

    SSAUpdate.RewriteUse(*U);
  }
}

static bool splitCriticalEdges(CallBrInst *CBR, DominatorTree *DT) {
  bool Changed = false;

  CriticalEdgeSplittingOptions Options(DT);
  Options.setMergeIdenticalEdges();

  // The indirect destination might be duplicated between another parameter...
  //
  //   %0 = callbr ... [label %x, label %x]
  //
  // ...hence MergeIdenticalEdges and AllowIndentical edges, but we don't need
  // to split the default destination if it's duplicated between an indirect
  // destination...
  //
  //   %1 = callbr ... to label %x [label %x]
  //
  // ...hence starting at 1 and checking against successor 0 (aka the default
  // destination).
  for (unsigned I = 1, E = CBR->getNumSuccessors(); I != E; ++I)
    if (CBR->getSuccessor(I) == CBR->getSuccessor(0) ||
        isCriticalEdge(CBR, I, /*AllowIdenticalEdges*/ true))
      if (SplitKnownCriticalEdge(CBR, I, Options))
        Changed = true;

  return Changed;
}

/// Create a separate SSA definition in each indirect target (via
/// llvm.callbr.landingpad). This may require splitting critical edges so we
/// have a location to place the intrinsic. Then remap users of the original
/// callbr output SSA value to instead point to the appropriate
/// llvm.callbr.landingpad value.
static bool insertIntrinsicCalls(CallBrInst *CBR, DominatorTree &DT) {
  bool Changed = false;
  SmallPtrSet<const BasicBlock *, 4> Visited;
  IRBuilder<> Builder(CBR->getContext());

  if (!CBR->getNumIndirectDests())
    return false;

  SSAUpdater SSAUpdate;
  SSAUpdate.Initialize(CBR->getType(), CBR->getName());
  SSAUpdate.AddAvailableValue(CBR->getParent(), CBR);
  SSAUpdate.AddAvailableValue(CBR->getDefaultDest(), CBR);

  for (BasicBlock *IndDest : CBR->getIndirectDests()) {
    if (!Visited.insert(IndDest).second)
      continue;

    Builder.SetInsertPoint(&*IndDest->begin());
    CallInst *Intrinsic = Builder.CreateIntrinsic(
        CBR->getType(), Intrinsic::callbr_landingpad, {CBR});
    SSAUpdate.AddAvailableValue(IndDest, Intrinsic);
    updateSSA(DT, CBR, Intrinsic, SSAUpdate);
    Changed = true;
  }

  return Changed;
}

static bool processCallBrInst(Function &F, CallBrInst *CBR, DominatorTree *DT) {
  bool Changed = false;

  Changed |= splitCriticalEdges(CBR, DT);
  Changed |= insertIntrinsicCalls(CBR, *DT);

  return Changed;
}

static SmallVector<CallBrInst *, 2> findCallBrs(Function &F) {
  SmallVector<CallBrInst *, 2> CBRs;
  for (BasicBlock &BB : F)
    if (auto *CBR = dyn_cast<CallBrInst>(BB.getTerminator()))
      if (!CBR->getType()->isVoidTy() && !CBR->use_empty())
        CBRs.push_back(CBR);
  return CBRs;
}

static bool runImpl(Function &F, ArrayRef<CallBrInst *> CBRs,
                    DominatorTree *DT) {
  bool Changed = false;

  for (CallBrInst *CBR : CBRs)
    Changed |= processCallBrInst(F, CBR, DT);

  return Changed;
}

bool InlineAsmPrepare::runOnFunction(Function &F) {
  SmallVector<CallBrInst *, 2> CBRs = findCallBrs(F);
  if (CBRs.empty())
    return false;

  // It's highly likely that most programs do not contain CallBrInsts. Follow a
  // similar pattern from SafeStackLegacyPass::runOnFunction to reuse previous
  // domtree analysis if available, otherwise compute it lazily. This avoids
  // forcing Dominator Tree Construction at -O0 for programs that likely do not
  // contain CallBrInsts. It does pessimize programs with callbr at higher
  // optimization levels, as the DominatorTree created here is not reused by
  // subsequent passes.
  DominatorTree *DT;
  std::optional<DominatorTree> LazilyComputedDomTree;
  if (auto *DTWP = getAnalysisIfAvailable<DominatorTreeWrapperPass>())
    DT = &DTWP->getDomTree();
  else {
    LazilyComputedDomTree.emplace(F);
    DT = &*LazilyComputedDomTree;
  }

  return runImpl(F, CBRs, DT);
}

PreservedAnalyses InlineAsmPreparePass::run(Function &F,
                                            FunctionAnalysisManager &FAM) {
  SmallVector<CallBrInst *, 2> CBRs = findCallBrs(F);
  if (CBRs.empty())
    return PreservedAnalyses::all();

  auto *DT = &FAM.getResult<DominatorTreeAnalysis>(F);

  if (runImpl(F, CBRs, DT)) {
    PreservedAnalyses PA;
    PA.preserve<DominatorTreeAnalysis>();
    return PA;
  }

  return PreservedAnalyses::all();
}
