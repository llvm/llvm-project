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

#include "polly/RewriteByReferenceParameters.h"
#include "polly/LinkAllPasses.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"

#define DEBUG_TYPE "polly-rewrite-byref-params"

using namespace llvm;

namespace {
static void tryRewriteInstruction(Instruction &Inst) {
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

static void runRewriteByrefParams(Function &F) {
  for (BasicBlock &BB : F)
    for (Instruction &Inst : BB)
      tryRewriteInstruction(Inst);
}

class RewriteByrefParamsWrapperPass : public FunctionPass {
private:
  RewriteByrefParamsWrapperPass(const RewriteByrefParamsWrapperPass &) = delete;
  const RewriteByrefParamsWrapperPass &
  operator=(const RewriteByrefParamsWrapperPass &) = delete;

public:
  static char ID;
  explicit RewriteByrefParamsWrapperPass() : FunctionPass(ID) {}

  virtual void getAnalysisUsage(AnalysisUsage &AU) const override {}

  virtual bool runOnFunction(Function &F) override {
    runRewriteByrefParams(F);
    return true;
  }
};

char RewriteByrefParamsWrapperPass::ID;
} // anonymous namespace

Pass *polly::createRewriteByrefParamsWrapperPass() {
  return new RewriteByrefParamsWrapperPass();
}

llvm::PreservedAnalyses
polly ::RewriteByrefParamsPass::run(llvm::Function &F,
                                    llvm::FunctionAnalysisManager &FAM) {
  runRewriteByrefParams(F);
  return PreservedAnalyses::none();
}

INITIALIZE_PASS_BEGIN(RewriteByrefParamsWrapperPass,
                      "polly-rewrite-byref-params",
                      "Polly - Rewrite by reference parameters", false, false)
INITIALIZE_PASS_END(RewriteByrefParamsWrapperPass, "polly-rewrite-byref-params",
                    "Polly - Rewrite by reference parameters", false, false)
