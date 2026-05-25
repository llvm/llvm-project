//===-- InlineAsmPrepare - Prepare inline asm for code generation ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers inline asm calls in LLVM IR in order to assist
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
// llvm.asm.constraint.br:
//
//   Remove the "llvm.asm.constraint.br" call and opt to prefer either
//   "registers" (on the callbr's default path) or "memory" (on the callbr's
//   indirect path). We choose the latter only when compiling at '-O0', because
//   the fast register allocator isn't equipped to fold registers if register
//   pressure is too great.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/InlineAsmPrepare.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"

using namespace llvm;

#define DEBUG_TYPE "inline-asm-prepare"

namespace {

class InlineAsmPrepare : public FunctionPass {
public:
  InlineAsmPrepare() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
  }
  bool runOnFunction(Function &F) override;

  static char ID;
};

char InlineAsmPrepare::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(InlineAsmPrepare, DEBUG_TYPE, "Prepare inline asm insts",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(InlineAsmPrepare, DEBUG_TYPE, "Prepare inline asm insts",
                    false, false)

FunctionPass *llvm::createInlineAsmPreparePass() {
  return new InlineAsmPrepare();
}

//===----------------------------------------------------------------------===//
//                      Process CallBr instructions
//===----------------------------------------------------------------------===//

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

static bool splitCriticalEdges(CallBrInst *CBR, DominatorTree &DT) {
  bool Changed = false;

  CriticalEdgeSplittingOptions Options(&DT);
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

static bool processCallBrInst(CallBrInst *CBR, DominatorTree &DT) {
  bool Changed = false;

  Changed |= splitCriticalEdges(CBR, DT);
  Changed |= insertIntrinsicCalls(CBR, DT);

  return Changed;
}

//===----------------------------------------------------------------------===//
//               Process 'llvm.asm.constraint.br' instructions
//===----------------------------------------------------------------------===//

static bool processAsmConstraintBrInst(CallBrInst &CBR, bool IsOptLevelNone,
                                       DomTreeUpdater &DTU) {
  BasicBlock *BB = CBR.getParent();
  BasicBlock *PrefReg = CBR.getDefaultDest();
  BasicBlock *PrefMem = CBR.getIndirectDest(0);
  BasicBlock *Merge = isa<CallBrInst>(PrefReg->getTerminator())
                          ? nullptr
                          : PrefReg->getSingleSuccessor();

  CBR.eraseFromParent();

  if (IsOptLevelNone) {
    DeleteDeadBlock(PrefReg, &DTU);
    IRBuilder(BB).CreateBr(PrefMem);
    MergeBlockIntoPredecessor(PrefMem, &DTU);
    if (Merge)
      MergeBlockIntoPredecessor(Merge, &DTU);
  } else {
    DeleteDeadBlock(PrefMem, &DTU);
    IRBuilder(BB).CreateBr(PrefReg);
    MergeBlockIntoPredecessor(PrefReg, &DTU);
    if (Merge)
      MergeBlockIntoPredecessor(Merge, &DTU);
  }

  DTU.flush();

  return true;
}

static bool runImpl(Function &F, bool IsOptLevelNone, DomTreeUpdater &DTU) {
  bool Changed = false;
  SmallVector<CallBrInst *, 4> AsmConstraintBrs;

  // Collect asm_constraint_br instructions first.
  for (auto &BB : F)
    if (auto *CBR = dyn_cast<CallBrInst>(BB.getTerminator()))
      if (CBR->getIntrinsicID() == Intrinsic::asm_constraint_br)
        AsmConstraintBrs.push_back(CBR);

  // Process 'llvm.asm.constraint.br' instructions first. At -O0 this deletes
  // the PrefReg block (and its callbr) via DeleteDeadBlock, which immediately
  // removes it from the function's block list. Collect OtherCallBrs only
  // after this loop to avoid holding dangling pointers into deleted blocks.
  for (auto *CBR : AsmConstraintBrs)
    Changed |= processAsmConstraintBrInst(*CBR, IsOptLevelNone, DTU);

  // Collect and process the remaining 'callbr' instructions.
  SmallVector<CallBrInst *, 4> OtherCallBrs;
  for (auto &BB : F)
    if (auto *CBR = dyn_cast<CallBrInst>(BB.getTerminator()))
      if (!CBR->getType()->isVoidTy() && !CBR->use_empty())
        OtherCallBrs.push_back(CBR);

  for (auto *CBR : OtherCallBrs)
    Changed |= processCallBrInst(CBR, DTU.getDomTree());

  return Changed;
}

bool InlineAsmPrepare::runOnFunction(Function &F) {
  const auto *TM = &getAnalysis<TargetPassConfig>().getTM<TargetMachine>();
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Lazy);

  bool IsOptLevelNone =
      skipFunction(F) ? true : TM->getOptLevel() == CodeGenOptLevel::None;

  return runImpl(F, IsOptLevelNone, DTU);
}

PreservedAnalyses InlineAsmPreparePass::run(Function &F,
                                            FunctionAnalysisManager &FAM) {
  auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Lazy);
  bool IsOptLevelNone =
      F.hasOptNone() ? true : TM->getOptLevel() == CodeGenOptLevel::None;

  if (runImpl(F, IsOptLevelNone, DTU)) {
    PreservedAnalyses PA;
    PA.preserve<DominatorTreeAnalysis>();
    return PA;
  }

  return PreservedAnalyses::all();
}
