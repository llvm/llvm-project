//===-- AGXNarrowInductions.cpp - 64-bit induction variables in 32 bits ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// PTX computes addresses in 64 bits, so its loops count byte offsets in 64-bit
// induction variables even where the offsets are small (an n-body tile of 256
// float4: 0 to 4096). AGX's ALU is 32 bits wide: a 64-bit induction variable
// takes a 64-bit add per iteration, a register pair, and an exit test of both
// halves. A 64-bit header phi incremented by a loop-invariant amount, whose
// value and increment scalar evolution proves within [0, 2^31), becomes a
// 32-bit phi; its 64-bit uses take the zero extension, which instruction
// selection folds into 64-bit address arithmetic and into compares with
// constants.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "agx-narrow-inductions"

static cl::opt<bool> AGXNarrowInductionsOpt(
    "agx-narrow-ivs", cl::init(true), cl::Hidden,
    cl::desc("Count 64-bit induction variables that stay below 2^31 in 32 "
             "bits"));

namespace {

class AGXNarrowInductions : public FunctionPass {
public:
  static char ID;
  AGXNarrowInductions() : FunctionPass(ID) {}

  StringRef getPassName() const override { return "AGX Narrow Inductions"; }
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.setPreservesCFG();
  }
};

} // namespace

char AGXNarrowInductions::ID = 0;

INITIALIZE_PASS_BEGIN(AGXNarrowInductions, DEBUG_TYPE, "AGX Narrow Inductions",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_END(AGXNarrowInductions, DEBUG_TYPE, "AGX Narrow Inductions",
                    false, false)

FunctionPass *llvm::createAGXNarrowInductionsPass() {
  return new AGXNarrowInductions();
}

static bool fitsSigned32(ScalarEvolution &SE, Value *V) {
  return SE.getUnsignedRange(SE.getSCEV(V)).getUnsignedMax().ult(1u << 31);
}

static bool narrow(PHINode &Phi, Loop &L, ScalarEvolution &SE) {
  BasicBlock *Preheader = L.getLoopPreheader(), *Latch = L.getLoopLatch();
  if (!Preheader || !Latch || Phi.getNumIncomingValues() != 2)
    return false;
  Value *Start = Phi.getIncomingValueForBlock(Preheader);
  auto *Inc = dyn_cast<BinaryOperator>(Phi.getIncomingValueForBlock(Latch));
  Value *Step;
  if (!Inc || !match(Inc, m_c_Add(m_Specific(&Phi), m_Value(Step))) ||
      !L.isLoopInvariant(Step))
    return false;
  if (!fitsSigned32(SE, &Phi) || !fitsSigned32(SE, Inc) ||
      !fitsSigned32(SE, Step))
    return false;

  SE.forgetLoop(&L);
  Type *I32 = Type::getInt32Ty(Phi.getContext());
  IRBuilder<> B(Preheader->getTerminator());
  Value *NarrowStart = B.CreateTrunc(Start, I32, Start->getName() + ".lo");
  Value *NarrowStep = B.CreateTrunc(Step, I32, Step->getName() + ".lo");
  PHINode *NarrowPhi =
      PHINode::Create(I32, 2, Phi.getName() + ".lo", Phi.getIterator());
  B.SetInsertPoint(Inc);
  Value *NarrowInc =
      B.CreateAdd(NarrowPhi, NarrowStep, Inc->getName() + ".lo",
                  /*HasNUW=*/true, /*HasNSW=*/true);
  NarrowPhi->addIncoming(NarrowStart, Preheader);
  NarrowPhi->addIncoming(NarrowInc, Latch);

  B.SetInsertPoint(Phi.getParent(), Phi.getParent()->getFirstInsertionPt());
  Value *WidePhi = B.CreateZExt(NarrowPhi, Phi.getType(), Phi.getName());
  B.SetInsertPoint(Inc);
  Value *WideInc = B.CreateZExt(NarrowInc, Inc->getType(), Inc->getName());
  // The zero extension of the increment replaces it after it is computed.
  cast<Instruction>(WideInc)->moveAfter(cast<Instruction>(NarrowInc));
  Inc->replaceAllUsesWith(WideInc);
  Phi.replaceAllUsesWith(WidePhi);
  Inc->eraseFromParent();
  Phi.eraseFromParent();
  return true;
}

bool AGXNarrowInductions::runOnFunction(Function &F) {
  if (!AGXNarrowInductionsOpt || skipFunction(F))
    return false;
  LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  ScalarEvolution &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  bool Changed = false;
  for (Loop *L : LI.getLoopsInPreorder()) {
    SmallVector<PHINode *, 4> Phis;
    for (PHINode &Phi : L->getHeader()->phis())
      if (Phi.getType()->isIntegerTy(64))
        Phis.push_back(&Phi);
    for (PHINode *Phi : Phis)
      Changed |= narrow(*Phi, *L, SE);
  }
  return Changed;
}
