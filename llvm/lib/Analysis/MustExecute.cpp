//===- MustExecute.cpp - Printer for isGuaranteedToExecute ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/MustExecute.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

bool LoopSafetyInfo::headerMayThrow() const {
  return HeaderMayThrow;
}

bool LoopSafetyInfo::anyBlockMayThrow() const {
  return MayThrow;
}

void LoopSafetyInfo::computeLoopSafetyInfo(Loop *CurLoop) {
  assert(CurLoop != nullptr && "CurLoop can't be null");
  BasicBlock *Header = CurLoop->getHeader();
  // Iterate over header and compute safety info.
  HeaderMayThrow = !isGuaranteedToTransferExecutionToSuccessor(Header);
  MayThrow = HeaderMayThrow;
  // Iterate over loop instructions and compute safety info.
  // Skip header as it has been computed and stored in HeaderMayThrow.
  // The first block in loopinfo.Blocks is guaranteed to be the header.
  assert(Header == *CurLoop->getBlocks().begin() &&
         "First block must be header");
  for (Loop::block_iterator BB = std::next(CurLoop->block_begin()),
                            BBE = CurLoop->block_end();
       (BB != BBE) && !MayThrow; ++BB)
    MayThrow |= !isGuaranteedToTransferExecutionToSuccessor(*BB);

  // Compute funclet colors if we might sink/hoist in a function with a funclet
  // personality routine.
  Function *Fn = CurLoop->getHeader()->getParent();
  if (Fn->hasPersonalityFn())
    if (Constant *PersonalityFn = Fn->getPersonalityFn())
      if (isScopedEHPersonality(classifyEHPersonality(PersonalityFn)))
        BlockColors = colorEHFunclets(*Fn);
}

/// Return true if we can prove that the given ExitBlock is not reached on the
/// first iteration of the given loop.  That is, the backedge of the loop must
/// be executed before the ExitBlock is executed in any dynamic execution trace.
static bool CanProveNotTakenFirstIteration(const BasicBlock *ExitBlock,
                                           const DominatorTree *DT,
                                           const Loop *CurLoop) {
  auto *CondExitBlock = ExitBlock->getSinglePredecessor();
  if (!CondExitBlock)
    // expect unique exits
    return false;
  assert(CurLoop->contains(CondExitBlock) && "meaning of exit block");
  auto *BI = dyn_cast<BranchInst>(CondExitBlock->getTerminator());
  if (!BI || !BI->isConditional())
    return false;
  // If condition is constant and false leads to ExitBlock then we always
  // execute the true branch.
  if (auto *Cond = dyn_cast<ConstantInt>(BI->getCondition()))
    return BI->getSuccessor(Cond->getZExtValue() ? 1 : 0) == ExitBlock;
  auto *Cond = dyn_cast<CmpInst>(BI->getCondition());
  if (!Cond)
    return false;
  // todo: this would be a lot more powerful if we used scev, but all the
  // plumbing is currently missing to pass a pointer in from the pass
  // Check for cmp (phi [x, preheader] ...), y where (pred x, y is known
  auto *LHS = dyn_cast<PHINode>(Cond->getOperand(0));
  auto *RHS = Cond->getOperand(1);
  if (!LHS || LHS->getParent() != CurLoop->getHeader())
    return false;
  auto DL = ExitBlock->getModule()->getDataLayout();
  auto *IVStart = LHS->getIncomingValueForBlock(CurLoop->getLoopPreheader());
  auto *SimpleValOrNull = SimplifyCmpInst(Cond->getPredicate(),
                                          IVStart, RHS,
                                          {DL, /*TLI*/ nullptr,
                                              DT, /*AC*/ nullptr, BI});
  auto *SimpleCst = dyn_cast_or_null<Constant>(SimpleValOrNull);
  if (!SimpleCst)
    return false;
  if (ExitBlock == BI->getSuccessor(0))
    return SimpleCst->isZeroValue();
  assert(ExitBlock == BI->getSuccessor(1) && "implied by above");
  return SimpleCst->isAllOnesValue();
}


bool LoopSafetyInfo::allLoopPathsLeadToBlock(const Loop *CurLoop,
                                             const BasicBlock *BB,
                                             const DominatorTree *DT) const {
  assert(CurLoop->contains(BB) && "Should only be called for loop blocks!");

  // Fast path: header is always reached once the loop is entered.
  if (BB == CurLoop->getHeader())
    return true;

  // Collect all transitive predecessors of BB in the same loop. This set will
  // be a subset of the blocks within the loop.
  SmallPtrSet<const BasicBlock *, 4> Predecessors;
  SmallVector<const BasicBlock *, 4> WorkList;
  for (auto *Pred : predecessors(BB)) {
    Predecessors.insert(Pred);
    WorkList.push_back(Pred);
  }
  while (!WorkList.empty()) {
    auto *Pred = WorkList.pop_back_val();
    assert(CurLoop->contains(Pred) && "Should only reach loop blocks!");
    // We are not interested in backedges and we don't want to leave loop.
    if (Pred == CurLoop->getHeader())
      continue;
    // TODO: If BB lies in an inner loop of CurLoop, this will traverse over all
    // blocks of this inner loop, even those that are always executed AFTER the
    // BB. It may make our analysis more conservative than it could be, see test
    // @nested and @nested_no_throw in test/Analysis/MustExecute/loop-header.ll.
    // We can ignore backedge of all loops containing BB to get a sligtly more
    // optimistic result.
    for (auto *PredPred : predecessors(Pred))
      if (Predecessors.insert(PredPred).second)
        WorkList.push_back(PredPred);
  }

  // Make sure that all successors of all predecessors of BB are either:
  // 1) BB,
  // 2) Also predecessors of BB,
  // 3) Exit blocks which are not taken on 1st iteration.
  // Memoize blocks we've already checked.
  SmallPtrSet<const BasicBlock *, 4> CheckedSuccessors;
  for (auto *Pred : Predecessors)
    for (auto *Succ : successors(Pred))
      if (CheckedSuccessors.insert(Succ).second &&
          Succ != BB && !Predecessors.count(Succ))
        // By discharging conditions that are not executed on the 1st iteration,
        // we guarantee that *at least* on the first iteration all paths from
        // header that *may* execute will lead us to the block of interest. So
        // that if we had virtually peeled one iteration away, in this peeled
        // iteration the set of predecessors would contain only paths from
        // header to BB without any exiting edges that may execute.
        //
        // TODO: We only do it for exiting edges currently. We could use the
        // same function to skip some of the edges within the loop if we know
        // that they will not be taken on the 1st iteration.
        //
        // TODO: If we somehow know the number of iterations in loop, the same
        // check may be done for any arbitrary N-th iteration as long as N is
        // not greater than minimum number of iterations in this loop.
        if (CurLoop->contains(Succ) ||
            !CanProveNotTakenFirstIteration(Succ, DT, CurLoop))
          return false;

  // All predecessors can only lead us to BB.
  return true;
}

/// Returns true if the instruction in a loop is guaranteed to execute at least
/// once.
bool llvm::isGuaranteedToExecute(const Instruction &Inst,
                                 const DominatorTree *DT, const Loop *CurLoop,
                                 const LoopSafetyInfo *SafetyInfo) {
  // We have to check to make sure that the instruction dominates all
  // of the exit blocks.  If it doesn't, then there is a path out of the loop
  // which does not execute this instruction, so we can't hoist it.

  // If the instruction is in the header block for the loop (which is very
  // common), it is always guaranteed to dominate the exit blocks.  Since this
  // is a common case, and can save some work, check it now.
  if (Inst.getParent() == CurLoop->getHeader())
    // If there's a throw in the header block, we can't guarantee we'll reach
    // Inst unless we can prove that Inst comes before the potential implicit
    // exit.  At the moment, we use a (cheap) hack for the common case where
    // the instruction of interest is the first one in the block.
    return !SafetyInfo->headerMayThrow() ||
      Inst.getParent()->getFirstNonPHIOrDbg() == &Inst;

  // Somewhere in this loop there is an instruction which may throw and make us
  // exit the loop.
  if (SafetyInfo->anyBlockMayThrow())
    return false;

  // If there is a path from header to exit or latch that doesn't lead to our
  // instruction's block, return false.
  if (!SafetyInfo->allLoopPathsLeadToBlock(CurLoop, Inst.getParent(), DT))
    return false;

  return true;
}


namespace {
  struct MustExecutePrinter : public FunctionPass {

    static char ID; // Pass identification, replacement for typeid
    MustExecutePrinter() : FunctionPass(ID) {
      initializeMustExecutePrinterPass(*PassRegistry::getPassRegistry());
    }
    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesAll();
      AU.addRequired<DominatorTreeWrapperPass>();
      AU.addRequired<LoopInfoWrapperPass>();
    }
    bool runOnFunction(Function &F) override;
  };
}

char MustExecutePrinter::ID = 0;
INITIALIZE_PASS_BEGIN(MustExecutePrinter, "print-mustexecute",
                      "Instructions which execute on loop entry", false, true)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(MustExecutePrinter, "print-mustexecute",
                    "Instructions which execute on loop entry", false, true)

FunctionPass *llvm::createMustExecutePrinter() {
  return new MustExecutePrinter();
}

static bool isMustExecuteIn(const Instruction &I, Loop *L, DominatorTree *DT) {
  // TODO: merge these two routines.  For the moment, we display the best
  // result obtained by *either* implementation.  This is a bit unfair since no
  // caller actually gets the full power at the moment.
  LoopSafetyInfo LSI;
  LSI.computeLoopSafetyInfo(L);
  return isGuaranteedToExecute(I, DT, L, &LSI) ||
    isGuaranteedToExecuteForEveryIteration(&I, L);
}

namespace {
/// An assembly annotator class to print must execute information in
/// comments.
class MustExecuteAnnotatedWriter : public AssemblyAnnotationWriter {
  DenseMap<const Value*, SmallVector<Loop*, 4> > MustExec;

public:
  MustExecuteAnnotatedWriter(const Function &F,
                             DominatorTree &DT, LoopInfo &LI) {
    for (auto &I: instructions(F)) {
      Loop *L = LI.getLoopFor(I.getParent());
      while (L) {
        if (isMustExecuteIn(I, L, &DT)) {
          MustExec[&I].push_back(L);
        }
        L = L->getParentLoop();
      };
    }
  }
  MustExecuteAnnotatedWriter(const Module &M,
                             DominatorTree &DT, LoopInfo &LI) {
    for (auto &F : M)
    for (auto &I: instructions(F)) {
      Loop *L = LI.getLoopFor(I.getParent());
      while (L) {
        if (isMustExecuteIn(I, L, &DT)) {
          MustExec[&I].push_back(L);
        }
        L = L->getParentLoop();
      };
    }
  }


  void printInfoComment(const Value &V, formatted_raw_ostream &OS) override {
    if (!MustExec.count(&V))
      return;

    const auto &Loops = MustExec.lookup(&V);
    const auto NumLoops = Loops.size();
    if (NumLoops > 1)
      OS << " ; (mustexec in " << NumLoops << " loops: ";
    else
      OS << " ; (mustexec in: ";

    bool first = true;
    for (const Loop *L : Loops) {
      if (!first)
        OS << ", ";
      first = false;
      OS << L->getHeader()->getName();
    }
    OS << ")";
  }
};
} // namespace

bool MustExecutePrinter::runOnFunction(Function &F) {
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();

  MustExecuteAnnotatedWriter Writer(F, DT, LI);
  F.print(dbgs(), &Writer);

  return false;
}
