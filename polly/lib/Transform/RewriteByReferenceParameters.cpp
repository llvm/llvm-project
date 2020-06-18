//===------ RewriteByReferenceParameters.cpp --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass introduces separate 'alloca' instructions for read-only
// by-reference function parameters to indicate that these paramters are
// read-only. After this transformation -mem2reg has more freedom to promote
// variables to registers, which allows SCEV to work in more cases.
//
//===----------------------------------------------------------------------===//

#include "polly/LinkAllPasses.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"

#define DEBUG_TYPE "polly-rewrite-byref-params"

using namespace llvm;

namespace {

class RewriteByrefParams : public FunctionPass {
private:
  RewriteByrefParams(const RewriteByrefParams &) = delete;
  const RewriteByrefParams &operator=(const RewriteByrefParams &) = delete;

public:
  static char ID;
  explicit RewriteByrefParams() : FunctionPass(ID) {}

  virtual void getAnalysisUsage(AnalysisUsage &AU) const override {}

  void tryRewriteInstruction(Instruction &Inst) {
    BasicBlock *Entry = &Inst.getParent()->getParent()->getEntryBlock();

    auto *Call = dyn_cast<CallInst>(&Inst);

    if (!Call)
      return;

    llvm::Function *F = Call->getCalledFunction();

    if (!F)
      return;

    // We currently match for a very specific function. In case this proves
    // useful, we can make this code dependent on readonly metadata.
    if (!F->hasName() || F->getName() != "_gfortran_transfer_integer_write")
      return;

    auto *BitCast = dyn_cast<BitCastInst>(Call->getOperand(1));

    if (!BitCast)
      return;

    auto *Alloca = dyn_cast<AllocaInst>(BitCast->getOperand(0));

    if (!Alloca)
      return;

    std::string InstName = Alloca->getName().str();

    auto NewAlloca =
        new AllocaInst(Alloca->getAllocatedType(), 0,
                       "polly_byref_alloca_" + InstName, &*Entry->begin());

    auto *LoadedVal = new LoadInst(Alloca->getAllocatedType(), Alloca,
                                   "polly_byref_load_" + InstName, &Inst);

    new StoreInst(LoadedVal, NewAlloca, &Inst);
    auto *NewBitCast = new BitCastInst(NewAlloca, BitCast->getType(),
                                       "polly_byref_cast_" + InstName, &Inst);
    Call->setOperand(1, NewBitCast);
  }

  virtual bool runOnFunction(Function &F) override {
    for (BasicBlock &BB : F)
      for (Instruction &Inst : BB)
        tryRewriteInstruction(Inst);

    return true;
  }
};

char RewriteByrefParams::ID;
} // anonymous namespace

Pass *polly::createRewriteByrefParamsPass() { return new RewriteByrefParams(); }

INITIALIZE_PASS_BEGIN(RewriteByrefParams, "polly-rewrite-byref-params",
                      "Polly - Rewrite by reference parameters", false, false)
INITIALIZE_PASS_END(RewriteByrefParams, "polly-rewrite-byref-params",
                    "Polly - Rewrite by reference parameters", false, false)
