//===- MustExecute.cpp - Printer for isGuaranteedToExecute ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
using namespace llvm;

namespace {
  struct MustExecutePrinter : public FunctionPass {
    DenseMap<Value*, SmallVector<Loop*, 4> > MustExec;
    SmallVector<Value *, 4> Ordering;

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
    void print(raw_ostream &OS, const Module * = nullptr) const override;
    void releaseMemory() override {
      MustExec.clear();
      Ordering.clear();
    }
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

bool isMustExecuteIn(Instruction &I, Loop *L, DominatorTree *DT) {
  // TODO: move loop specific code to analysis
  //LoopSafetyInfo LSI;
  //computeLoopSafetyInfo(&LSI, L);
  //return isGuaranteedToExecute(I, DT, L, &LSI);
  return isGuaranteedToExecuteForEveryIteration(&I, L);
}

bool MustExecutePrinter::runOnFunction(Function &F) {
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  for (auto &I: instructions(F)) {
    Loop *L = LI.getLoopFor(I.getParent());
    while (L) {
      if (isMustExecuteIn(I, L, &DT)) {
        if (!MustExec.count(&I))
          Ordering.push_back(&I);
        MustExec[&I].push_back(L);
      }
      L = L->getParentLoop();
    };
  }
  return false;
}

void MustExecutePrinter::print(raw_ostream &OS, const Module *M) const {
  OS << "The following are guaranteed to execute (for the respective loops):\n";
  for (Value *V: Ordering) {
    V->printAsOperand(OS);
    auto NumLoops = MustExec.lookup(V).size();
    if (NumLoops > 1)
      OS << "\t(mustexec in " << NumLoops << " loops: ";
    else
      OS << "\t(mustexec in: ";
    
    bool first = true;
    for (const Loop *L : MustExec.lookup(V)) {
      if (!first)
        OS << ", ";
      first = false;
      OS << L->getHeader()->getName();
    }
    OS << ")\n";
  }
  OS << "\n";
}
