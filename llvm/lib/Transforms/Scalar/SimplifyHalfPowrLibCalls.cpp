//===- SimplifyHalfPowrLibCalls.cpp - Optimize specific half_powr calls ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple pass that applies an experimental
// transformation on calls to specific functions.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "simplify-libcalls-halfpowr"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Target/TargetData.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Config/config.h"
using namespace llvm;

namespace {
  /// This pass optimizes well half_powr function calls.
  ///
  class VISIBILITY_HIDDEN SimplifyHalfPowrLibCalls : public FunctionPass {
    const TargetData *TD;
  public:
    static char ID; // Pass identification
    SimplifyHalfPowrLibCalls() : FunctionPass(&ID) {}

    bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<TargetData>();
    }

    Instruction *
    InlineHalfPowrs(const std::vector<Instruction *> &HalfPowrs,
                    Instruction *InsertPt);
  };
  char SimplifyHalfPowrLibCalls::ID = 0;
} // end anonymous namespace.

static RegisterPass<SimplifyHalfPowrLibCalls>
X("simplify-libcalls-halfpowr", "Simplify half_powr library calls");

// Public interface to the Simplify HalfPowr LibCalls pass.
FunctionPass *llvm::createSimplifyHalfPowrLibCallsPass() {
  return new SimplifyHalfPowrLibCalls(); 
}

/// InlineHalfPowrs - Inline a sequence of adjacent half_powr calls, rearranging
/// their control flow to better facilitate subsequent optimization.
Instruction *
SimplifyHalfPowrLibCalls::InlineHalfPowrs(const std::vector<Instruction *> &HalfPowrs,
                                        Instruction *InsertPt) {
  std::vector<BasicBlock *> Bodies;
  BasicBlock *NewBlock = 0;

  for (unsigned i = 0, e = HalfPowrs.size(); i != e; ++i) {
    CallInst *Call = cast<CallInst>(HalfPowrs[i]);
    Function *Callee = Call->getCalledFunction();

    // Minimally sanity-check the CFG of half_powr to ensure that it contains
    // the the kind of code we expect.  If we're running this pass, we have
    // reason to believe it will be what we expect.
    Function::iterator I = Callee->begin();
    BasicBlock *Prologue = I++;
    if (I == Callee->end()) break;
    BasicBlock *SubnormalHandling = I++;
    if (I == Callee->end()) break;
    BasicBlock *Body = I++;
    if (I != Callee->end()) break;
    if (SubnormalHandling->getSinglePredecessor() != Prologue)
      break;
    BranchInst *PBI = dyn_cast<BranchInst>(Prologue->getTerminator());
    if (!PBI || !PBI->isConditional())
      break;
    BranchInst *SNBI = dyn_cast<BranchInst>(SubnormalHandling->getTerminator());
    if (!SNBI || SNBI->isConditional())
      break;
    if (!isa<ReturnInst>(Body->getTerminator()))
      break;

    Instruction *NextInst = next(BasicBlock::iterator(Call));

    // Inline the call, taking care of what code ends up where.
    NewBlock = SplitBlock(NextInst->getParent(), NextInst, this);

    bool B = InlineFunction(Call, 0, TD);
    assert(B && "half_powr didn't inline?"); B=B;

    BasicBlock *NewBody = NewBlock->getSinglePredecessor();
    assert(NewBody);
    Bodies.push_back(NewBody);
  }

  if (!NewBlock)
    return InsertPt;

  // Put the code for all the bodies into one block, to facilitate
  // subsequent optimization.
  (void)SplitEdge(NewBlock->getSinglePredecessor(), NewBlock, this);
  for (unsigned i = 0, e = Bodies.size(); i != e; ++i) {
    BasicBlock *Body = Bodies[i];
    Instruction *FNP = Body->getFirstNonPHI();
    // Splice the insts from body into NewBlock.
    NewBlock->getInstList().splice(NewBlock->begin(), Body->getInstList(),
                                   FNP, Body->getTerminator());
  }

  return NewBlock->begin();
}

/// runOnFunction - Top level algorithm.
///
bool SimplifyHalfPowrLibCalls::runOnFunction(Function &F) {
  TD = &getAnalysis<TargetData>();
  
  bool Changed = false;
  std::vector<Instruction *> HalfPowrs;
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
      // Look for calls.
      bool IsHalfPowr = false;
      if (CallInst *CI = dyn_cast<CallInst>(I)) {
        // Look for direct calls and calls to non-external functions.
        Function *Callee = CI->getCalledFunction();
        if (Callee && Callee->hasExternalLinkage()) {
          // Look for calls with well-known names.
          const char *CalleeName = Callee->getNameStart();
          if (strcmp(CalleeName, "__half_powrf4") == 0)
            IsHalfPowr = true;
        }
      }
      if (IsHalfPowr)
        HalfPowrs.push_back(I);
      // We're looking for sequences of up to three such calls, which we'll
      // simplify as a group.
      if ((!IsHalfPowr && !HalfPowrs.empty()) || HalfPowrs.size() == 3) {
        I = InlineHalfPowrs(HalfPowrs, I);
        E = I->getParent()->end();
        HalfPowrs.clear();
        Changed = true;
      }
    }
    assert(HalfPowrs.empty() && "Block had no terminator!");
  }

  return Changed;
}
