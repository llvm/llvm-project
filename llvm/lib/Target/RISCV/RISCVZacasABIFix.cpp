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
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-zacas-abi-fix"
#define PASS_NAME "RISC-V Zacas ABI fix"

namespace {

class RISCVZacasABIFix : public FunctionPass,
                         public InstVisitor<RISCVZacasABIFix, bool> {
  const RISCVSubtarget *ST;

public:
  static char ID;

  RISCVZacasABIFix() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

  StringRef getPassName() const override { return PASS_NAME; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<TargetPassConfig>();
  }

  bool visitInstruction(Instruction &I) { return false; }
  bool visitAtomicCmpXchgInst(AtomicCmpXchgInst &I);
};

} // end anonymous namespace

// Insert a leading fence (needed for broadest atomics ABI compatibility)
// only if the Zacas extension is enabled and the AtomicCmpXchgInst has a
// SequentiallyConsistent failure ordering.
bool RISCVZacasABIFix::visitAtomicCmpXchgInst(AtomicCmpXchgInst &I) {
  assert(ST->hasStdExtZacas() && "only necessary to run in presence of zacas");
  IRBuilder<> Builder(&I);
  if (I.getFailureOrdering() != AtomicOrdering::SequentiallyConsistent)
    return false;

  Builder.CreateFence(AtomicOrdering::SequentiallyConsistent);
  return true;
}

bool RISCVZacasABIFix::runOnFunction(Function &F) {
  auto &TPC = getAnalysis<TargetPassConfig>();
  auto &TM = TPC.getTM<RISCVTargetMachine>();
  ST = &TM.getSubtarget<RISCVSubtarget>(F);

  if (skipFunction(F) || !ST->hasStdExtZacas())
    return false;

  bool MadeChange = false;
  for (auto &BB : F)
    for (Instruction &I : llvm::make_early_inc_range(BB))
      MadeChange |= visit(I);

  return MadeChange;
}

INITIALIZE_PASS_BEGIN(RISCVZacasABIFix, DEBUG_TYPE, PASS_NAME, false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(RISCVZacasABIFix, DEBUG_TYPE, PASS_NAME, false, false)

char RISCVZacasABIFix::ID = 0;

FunctionPass *llvm::createRISCVZacasABIFixPass() {
  return new RISCVZacasABIFix();
}
