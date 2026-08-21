//===----- RISCVZacasABIFix.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements a fence insertion for an atomic cmpxchg in a case that
// isn't easy to do with the current AtomicExpandPass hooks API.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVTargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-zacas-abi-fix"
#define PASS_NAME "RISC-V Zacas ABI fix"

namespace {
class RISCVZacasABIFixImpl : public InstVisitor<RISCVZacasABIFixImpl, bool> {
  const RISCVSubtarget *ST;

public:
  RISCVZacasABIFixImpl(const RISCVSubtarget *ST) : ST(ST) {}
  bool run(Function &F);
  bool visitInstruction(Instruction &I) { return false; }
  bool visitAtomicCmpXchgInst(AtomicCmpXchgInst &I);
};
} // namespace

namespace {
class RISCVZacasABIFixLegacy : public FunctionPass {
public:
  static char ID;

  RISCVZacasABIFixLegacy() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

  StringRef getPassName() const override { return PASS_NAME; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<TargetPassConfig>();
  }
};
} // namespace

// Insert a leading fence (needed for broadest atomics ABI compatibility)
// only if the Zacas extension is enabled and the AtomicCmpXchgInst has a
// SequentiallyConsistent failure ordering.
bool RISCVZacasABIFixImpl::visitAtomicCmpXchgInst(AtomicCmpXchgInst &I) {
  assert(ST->hasStdExtZacas() && "only necessary to run in presence of zacas");
  IRBuilder<> Builder(&I);
  if (I.getFailureOrdering() != AtomicOrdering::SequentiallyConsistent)
    return false;

  Builder.CreateFence(AtomicOrdering::SequentiallyConsistent);
  return true;
}

bool RISCVZacasABIFixImpl::run(Function &F) {
  if (!ST->hasStdExtZacas())
    return false;

  bool MadeChange = false;
  for (auto &BB : F)
    for (Instruction &I : llvm::make_early_inc_range(BB))
      MadeChange |= visit(I);

  return MadeChange;
}

bool RISCVZacasABIFixLegacy::runOnFunction(Function &F) {
  auto &TPC = getAnalysis<TargetPassConfig>();
  auto &TM = TPC.getTM<RISCVTargetMachine>();
  auto *ST = &TM.getSubtarget<RISCVSubtarget>(F);

  if (skipFunction(F))
    return false;

  return RISCVZacasABIFixImpl(ST).run(F);
}

INITIALIZE_PASS_BEGIN(RISCVZacasABIFixLegacy, DEBUG_TYPE, PASS_NAME, false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(RISCVZacasABIFixLegacy, DEBUG_TYPE, PASS_NAME, false, false)

char RISCVZacasABIFixLegacy::ID = 0;

FunctionPass *llvm::createRISCVZacasABIFixLegacyPass() {
  return new RISCVZacasABIFixLegacy();
}

PreservedAnalyses RISCVZacasABIFixPass::run(Function &F,
                                            FunctionAnalysisManager &FAM) {
  auto *ST = &TM->getSubtarget<RISCVSubtarget>(F);

  bool Changed = RISCVZacasABIFixImpl(ST).run(F);
  if (!Changed)
    return PreservedAnalyses::all();

  return PreservedAnalyses::allInSet<CFGAnalyses>();
}
