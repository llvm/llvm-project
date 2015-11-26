//===-- WebAssemblyOptimizeReturned.cpp - Optimize "returned" attributes --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Optimize calls with "returned" attributes for WebAssembly.
///
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-optimize-returned"

namespace {
class OptimizeReturned final : public FunctionPass,
                               public InstVisitor<OptimizeReturned> {
  const char *getPassName() const override {
    return "WebAssembly Optimize Returned";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    FunctionPass::getAnalysisUsage(AU);
  }

  bool runOnFunction(Function &F) override;

  DominatorTree *DT;

public:
  static char ID;
  OptimizeReturned() : FunctionPass(ID), DT(nullptr) {}

  void visitCallSite(CallSite CS);
};
} // End anonymous namespace

char OptimizeReturned::ID = 0;
FunctionPass *llvm::createWebAssemblyOptimizeReturned() {
  return new OptimizeReturned();
}

void OptimizeReturned::visitCallSite(CallSite CS) {
  for (unsigned i = 0, e = CS.getNumArgOperands(); i < e; ++i)
    if (CS.paramHasAttr(1 + i, Attribute::Returned)) {
      Instruction *Inst = CS.getInstruction();
      Value *Arg = CS.getArgOperand(i);
      // Like replaceDominatedUsesWith but using Instruction/Use dominance.
      for (auto UI = Arg->use_begin(), UE = Arg->use_end(); UI != UE;) {
        Use &U = *UI++;
        if (DT->dominates(Inst, U))
          U.set(Inst);
      }
    }
}

bool OptimizeReturned::runOnFunction(Function &F) {
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  visit(F);
  return true;
}
