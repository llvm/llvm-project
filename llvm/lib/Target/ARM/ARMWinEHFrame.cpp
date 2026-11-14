//===-- ARMWinEHFrame.cpp - No realignment in Windows C++ EH frames -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The C++ EH tables find UnwindHelp and the catch objects by their offset from
// SP at function entry, and a funclet reaches the parent frame only through r6,
// because the runtime doesn't restore SP or r11. In a realigned frame no object
// is at a fixed offset from both, so, as with MSVC, functions with C++ EH
// funclets must not realign the stack. This pass marks them "no-realign-stack"
// and gives over-aligned allocas an aligned address inside a bigger block.
// Over-aligned catch objects and locals passed to llvm.localescape can't be
// moved like that and are diagnosed.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/EHPersonalities.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "arm-wineh-frame"

namespace {
class ARMWinEHFrame : public FunctionPass {
public:
  static char ID;
  ARMWinEHFrame() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.setPreservesCFG();
  }

  StringRef getPassName() const override {
    return "ARM Windows C++ EH frame preparation";
  }
};
} // end anonymous namespace

char ARMWinEHFrame::ID = 0;

INITIALIZE_PASS_BEGIN(ARMWinEHFrame, DEBUG_TYPE,
                      "ARM Windows C++ EH frame preparation", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(ARMWinEHFrame, DEBUG_TYPE,
                    "ARM Windows C++ EH frame preparation", false, false)

FunctionPass *llvm::createARMWinEHFramePass() { return new ARMWinEHFrame(); }

bool ARMWinEHFrame::runOnFunction(Function &F) {
  if (!F.hasPersonalityFn() ||
      classifyEHPersonality(F.getPersonalityFn()) != EHPersonality::MSVC_CXX ||
      none_of(F, [](const BasicBlock &BB) { return BB.isEHPad(); }))
    return false;
  const TargetMachine &TM =
      getAnalysis<TargetPassConfig>().getTM<TargetMachine>();

  const Align StackAlign =
      TM.getSubtargetImpl(F)->getFrameLowering()->getStackAlign();
  const DataLayout &DL = F.getDataLayout();

  SmallVector<AllocaInst *, 4> OverAligned;
  for (Instruction &I : instructions(F))
    if (auto *AI = dyn_cast<AllocaInst>(&I))
      if (AI->getAlign() > StackAlign)
        OverAligned.push_back(AI);

  for (AllocaInst *AI : OverAligned) {
    const uint64_t A = AI->getAlign().value();
    Instruction *Pinned = nullptr; // a user that needs the alloca itself
    bool IsCatchObject = false;
    for (User *U : AI->users()) {
      if (auto *CP = dyn_cast<CatchPadInst>(U)) {
        Pinned = CP;
        IsCatchObject = true;
      } else if (auto *II = dyn_cast<IntrinsicInst>(U);
                 II && II->getIntrinsicID() == Intrinsic::localescape &&
                 !Pinned) {
        Pinned = II;
      }
    }
    if (Pinned) {
      F.getContext().diagnose(DiagnosticInfoUnsupported(
          F,
          "Windows C++ exception handling on ARM: " +
              Twine(IsCatchObject ? "the catch object"
                                  : "a local passed to llvm.localescape") +
              " is " + Twine(A) + "-byte aligned, but the stack is only " +
              Twine(StackAlign.value()) +
              "-byte aligned and it cannot be realigned",
          Pinned->getDebugLoc()));
      continue;
    }

    // An over-sized block with the stack alignment, and the object at the
    // first suitably aligned address inside it: Raw + (A - StackAlign) rounded
    // down to a multiple of A lies in [Raw, Raw + A - StackAlign].
    IRBuilder<> B(AI);
    Type *IntPtrTy = DL.getIntPtrType(AI->getType());
    const uint64_t Slack = A - StackAlign.value();
    Value *Bytes = ConstantInt::get(
        IntPtrTy, DL.getTypeAllocSize(AI->getAllocatedType()).getFixedValue());
    if (AI->isArrayAllocation())
      Bytes =
          B.CreateMul(B.CreateZExtOrTrunc(AI->getArraySize(), IntPtrTy), Bytes);
    Bytes = B.CreateAdd(Bytes, ConstantInt::get(IntPtrTy, Slack));
    AllocaInst *Raw =
        B.CreateAlloca(B.getInt8Ty(), AI->getAddressSpace(), Bytes);
    Raw->setAlignment(StackAlign);
    Raw->takeName(AI);
    Value *Bumped = B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), Raw, Slack);
    Value *Aligned = B.CreateIntrinsic(
        Intrinsic::ptrmask, {AI->getType(), IntPtrTy},
        {Bumped, ConstantInt::getSigned(IntPtrTy, -static_cast<int64_t>(A))});

    // Lifetime markers must name an alloca; they cover the whole block.
    for (Use &U : make_early_inc_range(AI->uses()))
      if (auto *II = dyn_cast<IntrinsicInst>(U.getUser()))
        if (II->isLifetimeStartOrEnd())
          U.set(Raw);
    AI->replaceAllUsesWith(Aligned);
    AI->eraseFromParent();
  }

  F.addFnAttr("no-realign-stack");
  return true;
}
