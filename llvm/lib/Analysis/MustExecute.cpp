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
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
using namespace llvm;

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

bool isMustExecuteIn(const Instruction &I, Loop *L, DominatorTree *DT) {
  // TODO: move loop specific code to analysis
  //LoopSafetyInfo LSI;
  //computeLoopSafetyInfo(&LSI, L);
  //return isGuaranteedToExecute(I, DT, L, &LSI);
  return isGuaranteedToExecuteForEveryIteration(&I, L);
}

/// \brief An assembly annotator class to print must execute information in
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

bool MustExecutePrinter::runOnFunction(Function &F) {
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();

  MustExecuteAnnotatedWriter Writer(F, DT, LI);
  F.print(dbgs(), &Writer);
  
  return false;
}
