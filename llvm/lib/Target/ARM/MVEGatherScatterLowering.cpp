//===- ARMGatherScatterLowering.cpp - Gather/Scatter lowering -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// This pass custom lowers llvm.gather and llvm.scatter instructions to
/// arm.mve.gather and arm.mve.scatter intrinsics, optimising the code to
/// produce a better final result as we go.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMBaseInstrInfo.h"
#include "ARMSubtarget.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsARM.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cassert>

using namespace llvm;

#define DEBUG_TYPE "mve-gather-scatter-lowering"

cl::opt<bool> EnableMaskedGatherScatters(
    "enable-arm-maskedgatscat", cl::Hidden, cl::init(false),
    cl::desc("Enable the generation of masked gathers and scatters"));

namespace {

class MVEGatherScatterLowering : public FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid

  explicit MVEGatherScatterLowering() : FunctionPass(ID) {
    initializeMVEGatherScatterLoweringPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;

  StringRef getPassName() const override {
    return "MVE gather/scatter lowering";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<TargetPassConfig>();
    FunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

char MVEGatherScatterLowering::ID = 0;

INITIALIZE_PASS(MVEGatherScatterLowering, DEBUG_TYPE,
                "MVE gather/scattering lowering pass", false, false)

Pass *llvm::createMVEGatherScatterLoweringPass() {
  return new MVEGatherScatterLowering();
}

static bool isLegalTypeAndAlignment(unsigned NumElements, unsigned ElemSize,
                                    unsigned Alignment) {
  // Do only allow non-extending v4i32 gathers for now
  return NumElements == 4 && ElemSize == 32 && Alignment >= 4;
}

static bool LowerGather(IntrinsicInst *I) {
  using namespace PatternMatch;
  LLVM_DEBUG(dbgs() << "masked gathers: checking transform preconditions\n");

  // @llvm.masked.gather.*(Ptrs, alignment, Mask, Src0)
  // Attempt to turn the masked gather in I into a MVE intrinsic
  // Potentially optimising the addressing modes as we do so.
  Type *Ty = I->getType();
  Value *Ptr = I->getArgOperand(0);
  unsigned Alignment = cast<ConstantInt>(I->getArgOperand(1))->getZExtValue();
  Value *Mask = I->getArgOperand(2);
  Value *PassThru = I->getArgOperand(3);

  // Check this is a valid gather with correct alignment
  if (!isLegalTypeAndAlignment(Ty->getVectorNumElements(),
                               Ty->getScalarSizeInBits(), Alignment)) {
    LLVM_DEBUG(dbgs() << "masked gathers: instruction does not have valid "
                      << "alignment or vector type \n");
    return false;
  }

  IRBuilder<> Builder(I->getContext());
  Builder.SetInsertPoint(I);
  Builder.SetCurrentDebugLocation(I->getDebugLoc());

  Value *Load = nullptr;
  // Look through bitcast instruction if #elements is the same
  if (auto *BitCast = dyn_cast<BitCastInst>(Ptr)) {
    Type *BCTy = BitCast->getType();
    Type *BCSrcTy = BitCast->getOperand(0)->getType();
    if (BCTy->getVectorNumElements() == BCSrcTy->getVectorNumElements()) {
      LLVM_DEBUG(dbgs() << "masked gathers: looking through bitcast\n");
      Ptr = BitCast->getOperand(0);
    }
  }
  assert(Ptr->getType()->isVectorTy() && "Unexpected pointer type");

  if (Ty->getVectorNumElements() != 4)
    // Can't build an intrinsic for this
    return false;
  if (match(Mask, m_One()))
    Load = Builder.CreateIntrinsic(Intrinsic::arm_mve_vldr_gather_base,
                                   {Ty, Ptr->getType()},
                                   {Ptr, Builder.getInt32(0)});
  else
    Load =
        Builder.CreateIntrinsic(Intrinsic::arm_mve_vldr_gather_base_predicated,
                                {Ty, Ptr->getType(), Mask->getType()},
                                {Ptr, Builder.getInt32(0), Mask});

  if (!isa<UndefValue>(PassThru) && !match(PassThru, m_Zero())) {
    LLVM_DEBUG(dbgs() << "masked gathers: found non-trivial passthru - "
                      << "creating select\n");
    Load = Builder.CreateSelect(Mask, Load, PassThru);
  }

  LLVM_DEBUG(dbgs() << "masked gathers: successfully built masked gather\n");
  I->replaceAllUsesWith(Load);
  I->eraseFromParent();
  return true;
}

bool MVEGatherScatterLowering::runOnFunction(Function &F) {
  if (!EnableMaskedGatherScatters)
    return false;
  auto &TPC = getAnalysis<TargetPassConfig>();
  auto &TM = TPC.getTM<TargetMachine>();
  auto *ST = &TM.getSubtarget<ARMSubtarget>(F);
  if (!ST->hasMVEIntegerOps())
    return false;
  SmallVector<IntrinsicInst *, 4> Gathers;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I);
      if (II && II->getIntrinsicID() == Intrinsic::masked_gather)
        Gathers.push_back(II);
    }
  }

  if (Gathers.empty())
    return false;

  for (IntrinsicInst *I : Gathers)
    LowerGather(I);

  return true;
}
