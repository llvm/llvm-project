//===-- PISAExpandIntrinsics.cpp - modify function signatures -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PISA.h"
#include "PISASubtarget.h"
#include "PISATargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/InitializePasses.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"

#define DEBUG_TYPE "pisa-expand-intrinsics"
#define DEBUG_NAME "PISA expand intrinsics"

using namespace llvm;

namespace {

class PISAExpandIntrinsics : public FunctionPass {
public:
  static char ID;

  PISAExpandIntrinsics() : FunctionPass(ID) {}
  StringRef getPassName() const override { return DEBUG_NAME; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
  }
  bool runOnFunction(Function &F) override;
};

} // namespace

char PISAExpandIntrinsics::ID = 0;

INITIALIZE_PASS_BEGIN(PISAExpandIntrinsics, DEBUG_TYPE, DEBUG_NAME, false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(PISAExpandIntrinsics, DEBUG_TYPE, DEBUG_NAME, false, false)

bool PISAExpandIntrinsics::runOnFunction(Function &F) {
  auto &TPC = getAnalysis<TargetPassConfig>();
  auto &TM = TPC.getTM<TargetMachine>();
  const auto *ST = TM.getSubtargetImpl(F);
  const auto *TLI = ST->getTargetLowering();

  SmallVector<MemIntrinsic *> MemIntrs;
  for (auto &I : instructions(F)) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (!II)
      continue;
    if (II->getIntrinsicID() == Intrinsic::memset ||
        II->getIntrinsicID() == Intrinsic::memcpy ||
        II->getIntrinsicID() == Intrinsic::memmove) {
      MemIntrinsic *MI = cast<MemIntrinsic>(II);
      uint64_t Len = ~0, Limit = 0;
      if (ConstantInt *LenCI = dyn_cast<ConstantInt>(MI->getLength()))
        Len = LenCI->getZExtValue();
      switch (II->getIntrinsicID()) {
      case Intrinsic::memset:
        Limit = TLI->getMaxStoresPerMemset(F.hasOptSize());
        break;
      case Intrinsic::memcpy:
        Limit = TLI->getMaxStoresPerMemcpy(F.hasOptSize());
        break;
      case Intrinsic::memmove:
        Limit = TLI->getMaxStoresPerMemmove(F.hasOptSize());
        break;
      }
      if (Len > Limit)
        MemIntrs.push_back(MI);
    }
  }

  bool Changed = false;

  // Expand llvm.mem* intrinsics to a loop
  const TargetTransformInfo &TTI =
      getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  for (MemIntrinsic *MemCall : MemIntrs) {
    if (auto *Memcpy = dyn_cast<MemCpyInst>(MemCall))
      expandMemCpyAsLoop(Memcpy, TTI);
    else if (auto *Memmove = dyn_cast<MemMoveInst>(MemCall))
      expandMemMoveAsLoop(Memmove, TTI);
    else if (auto *Memset = dyn_cast<MemSetInst>(MemCall))
      expandMemSetAsLoop(Memset);
    Changed = true;
    MemCall->eraseFromParent();
  }

  return Changed;
}

FunctionPass *llvm::createPISAExpandIntrinsicsPass() {
  return new PISAExpandIntrinsics();
}
