//===-- AGXAnnotateControlFlow.cpp - Annotate structured control flow -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Annotates the structurized control flow graph with the execution mask
// intrinsics llvm.agx.if, llvm.agx.else, llvm.agx.end.cf, llvm.agx.loop.begin
// and llvm.agx.loop, following SIAnnotateControlFlow.
//
// AGX keeps the execution mask in the per-thread nesting counter r0l: a
// thread is active when its counter is zero. Unlike AMDGPU no mask needs to
// be saved: "if" deactivates the threads where the condition is false
// (counter 1) and pushes the inactive ones one level deeper, "else" swaps
// the threads at the innermost level and "end.cf" pops a level. A loop
// pushes a level on entry (llvm.agx.loop.begin, in a dedicated preheader);
// at the latch, llvm.agx.loop deactivates the threads leaving the loop and
// branches back while any thread remains; the exit pops the level.
//
// All branches are annotated, uniform or not: there are no scalar branches.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAGX.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

using namespace llvm;

#define DEBUG_TYPE "agx-annotate-control-flow"

namespace {

class AGXAnnotateControlFlowImpl {
  Function &F;
  DominatorTree &DT;
  LoopInfo &LI;

  /// Blocks where the innermost open construct reconverges.
  SmallVector<BasicBlock *, 16> Stack;

  ConstantInt *BoolTrue;
  ConstantInt *BoolFalse;

  bool isTopOfStack(BasicBlock *BB) const {
    return !Stack.empty() && Stack.back() == BB;
  }
  bool isElse(PHINode *Phi) const;
  void prepareLoops();

  bool openIf(CondBrInst *Term);
  bool insertElse(CondBrInst *Term);
  bool handleLoop(CondBrInst *Term);
  bool closeControlFlow(BasicBlock *BB);
  bool layoutBlocks();

  CallInst *createCall(IRBuilder<> &IRB, Intrinsic::ID ID,
                       ArrayRef<Value *> Args = {}) {
    return IRB.CreateCall(
        Intrinsic::getOrInsertDeclaration(F.getParent(), ID), Args);
  }

public:
  AGXAnnotateControlFlowImpl(Function &F, DominatorTree &DT, LoopInfo &LI)
      : F(F), DT(DT), LI(LI) {
    BoolTrue = ConstantInt::getTrue(F.getContext());
    BoolFalse = ConstantInt::getFalse(F.getContext());
  }

  bool run();
};

class AGXAnnotateControlFlow : public FunctionPass {
public:
  static char ID;
  AGXAnnotateControlFlow() : FunctionPass(ID) {}

  StringRef getPassName() const override {
    return "AGX annotate control flow";
  }

  bool runOnFunction(Function &F) override {
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    return AGXAnnotateControlFlowImpl(F, DT, LI).run();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<LoopInfoWrapperPass>();
  }
};

} // namespace

char AGXAnnotateControlFlow::ID = 0;

INITIALIZE_PASS_BEGIN(AGXAnnotateControlFlow, DEBUG_TYPE,
                      "Annotate AGX control flow", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(AGXAnnotateControlFlow, DEBUG_TYPE,
                    "Annotate AGX control flow", false, false)

FunctionPass *llvm::createAGXAnnotateControlFlowPass() {
  return new AGXAnnotateControlFlow();
}

/// Is \p Phi the condition of the else part of an if-then-else produced by
/// the structurizer: true coming from the if block, false from the then
/// block?
bool AGXAnnotateControlFlowImpl::isElse(PHINode *Phi) const {
  BasicBlock *IDom = DT.getNode(Phi->getParent())->getIDom()->getBlock();
  for (unsigned I = 0, E = Phi->getNumIncomingValues(); I != E; ++I) {
    Value *Expected = Phi->getIncomingBlock(I) == IDom ? BoolTrue : BoolFalse;
    if (Phi->getIncomingValue(I) != Expected)
      return false;
  }
  return true;
}

/// Give every loop a dedicated preheader that only branches to the header,
/// so that the loop entry push is executed once, with the execution mask of
/// the loop.
void AGXAnnotateControlFlowImpl::prepareLoops() {
  SmallVector<Loop *, 8> Loops(LI.getLoopsInPreorder());
  for (Loop *L : Loops) {
    BasicBlock *Pre = L->getLoopPreheader();
    if (!Pre || !isa<UncondBrInst>(Pre->getTerminator()) ||
        Pre->getSingleSuccessor() != L->getHeader())
      Pre = InsertPreheaderForLoop(L, &DT, &LI, nullptr,
                                   /*PreserveLCSSA=*/false);
    if (!Pre)
      report_fatal_error("AGX: cannot create a loop preheader");
  }
}

bool AGXAnnotateControlFlowImpl::openIf(CondBrInst *Term) {
  IRBuilder<> IRB(Term);
  Term->setCondition(createCall(IRB, Intrinsic::agx_if, Term->getCondition()));
  Stack.push_back(Term->getSuccessor(1));
  return true;
}

bool AGXAnnotateControlFlowImpl::insertElse(CondBrInst *Term) {
  Stack.pop_back();
  IRBuilder<> IRB(Term);
  Term->setCondition(createCall(IRB, Intrinsic::agx_else));
  Stack.push_back(Term->getSuccessor(1));
  return true;
}

bool AGXAnnotateControlFlowImpl::handleLoop(CondBrInst *Term) {
  BasicBlock *BB = Term->getParent();
  Loop *L = LI.getLoopFor(BB);
  if (!L)
    report_fatal_error("AGX: backward branch outside of a loop");

  // The structurizer leaves loops with a single latch that branches to the
  // exit when its condition is true.
  BasicBlock *Header = Term->getSuccessor(1);
  BasicBlock *Pre = L->getLoopPreheader();
  assert(Pre && "loop preheaders are created by prepareLoops");
  IRBuilder<> PreIRB(Pre->getTerminator());
  createCall(PreIRB, Intrinsic::agx_loop_begin);

  IRBuilder<> IRB(Term);
  Term->setCondition(
      createCall(IRB, Intrinsic::agx_loop, Term->getCondition()));
  (void)Header;
  Stack.push_back(Term->getSuccessor(0));
  return true;
}

bool AGXAnnotateControlFlowImpl::closeControlFlow(BasicBlock *BB) {
  assert(isTopOfStack(BB));
  Loop *L = LI.getLoopFor(BB);
  if (L && L->getHeader() == BB) {
    // end.cf must not run on every iteration: move it to a block before the
    // header, on the entry edges.
    SmallVector<BasicBlock *, 2> Preds;
    for (BasicBlock *Pred : predecessors(BB))
      if (!L->contains(Pred))
        Preds.push_back(Pred);
    BB = SplitBlockPredecessors(BB, Preds, "endcf.split", &DT, &LI, nullptr,
                                false);
  }

  Stack.pop_back();
  BasicBlock::iterator InsertPt = BB->getFirstInsertionPt();
  IRBuilder<> IRB(BB, InsertPt);
  IRB.SetCurrentDebugLocation(DebugLoc());
  createCall(IRB, Intrinsic::agx_end_cf);
  return true;
}

bool AGXAnnotateControlFlowImpl::run() {
  prepareLoops();

  bool Changed = false;
  for (df_iterator<BasicBlock *> I = df_begin(&F.getEntryBlock()),
                                 E = df_end(&F.getEntryBlock());
       I != E; ++I) {
    BasicBlock *BB = *I;
    auto *Term = dyn_cast<CondBrInst>(BB->getTerminator());

    if (!Term) {
      if (isTopOfStack(BB))
        Changed |= closeControlFlow(BB);
      continue;
    }

    if (I.nodeVisited(Term->getSuccessor(1))) {
      if (isTopOfStack(BB))
        Changed |= closeControlFlow(BB);
      if (DT.dominates(Term->getSuccessor(1), BB))
        Changed |= handleLoop(Term);
      continue;
    }

    if (isTopOfStack(BB)) {
      auto *Phi = dyn_cast<PHINode>(Term->getCondition());
      if (Phi && Phi->getParent() == BB && isElse(Phi)) {
        Changed |= insertElse(Term);
        RecursivelyDeleteDeadPHINode(Phi);
        continue;
      }
      Changed |= closeControlFlow(BB);
    }

    Changed |= openIf(Term);
  }

  if (!Stack.empty())
    report_fatal_error("AGX: failed to annotate the control flow graph (it "
                       "is not structured)");
  Changed |= layoutBlocks();
  return Changed;
}

/// Execution mask code runs each construct in order, falling through from
/// the then part into the flow block and from the loop body into the latch.
/// The reverse post-order of the structured graph is that order; it becomes
/// the block layout, which machine block placement leaves alone.
bool AGXAnnotateControlFlowImpl::layoutBlocks() {
  ReversePostOrderTraversal<Function *> RPOT(&F);
  bool Changed = false;
  BasicBlock *Prev = nullptr;
  for (BasicBlock *BB : RPOT) {
    if (Prev && Prev->getNextNode() != BB) {
      BB->moveAfter(Prev);
      Changed = true;
    }
    Prev = BB;
  }
  return Changed;
}
