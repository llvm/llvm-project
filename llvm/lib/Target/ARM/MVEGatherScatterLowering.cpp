//===- MVEGatherScatterLowering.cpp - Gather/Scatter lowering -------------===//
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
#include "llvm/InitializePasses.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsARM.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
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

private:
  // Check this is a valid gather with correct alignment
  bool isLegalTypeAndAlignment(unsigned NumElements, unsigned ElemSize,
                               unsigned Alignment);
  // Check whether Ptr is hidden behind a bitcast and look through it
  void lookThroughBitcast(Value *&Ptr);
  // Check for a getelementptr and deduce base and offsets from it, on success
  // returning the base directly and the offsets indirectly using the Offsets
  // argument
  Value *checkGEP(Value *&Offsets, Type *Ty, Value *Ptr, IRBuilder<> Builder);

  bool lowerGather(IntrinsicInst *I);
  // Create a gather from a base + vector of offsets
  Value *tryCreateMaskedGatherOffset(IntrinsicInst *I, Value *Ptr,
                                     IRBuilder<> Builder);
  // Create a gather from a vector of pointers
  Value *tryCreateMaskedGatherBase(IntrinsicInst *I, Value *Ptr,
                                   IRBuilder<> Builder);
};

} // end anonymous namespace

char MVEGatherScatterLowering::ID = 0;

INITIALIZE_PASS(MVEGatherScatterLowering, DEBUG_TYPE,
                "MVE gather/scattering lowering pass", false, false)

Pass *llvm::createMVEGatherScatterLoweringPass() {
  return new MVEGatherScatterLowering();
}

bool MVEGatherScatterLowering::isLegalTypeAndAlignment(unsigned NumElements,
                                                       unsigned ElemSize,
                                                       unsigned Alignment) {
  // Do only allow non-extending gathers for now
  if (((NumElements == 4 && ElemSize == 32) ||
       (NumElements == 8 && ElemSize == 16) ||
       (NumElements == 16 && ElemSize == 8)) &&
      ElemSize / 8 <= Alignment)
    return true;
  LLVM_DEBUG(dbgs() << "masked gathers: instruction does not have valid "
                    << "alignment or vector type \n");
  return false;
}

Value *MVEGatherScatterLowering::checkGEP(Value *&Offsets, Type *Ty, Value *Ptr,
                                          IRBuilder<> Builder) {
  GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Ptr);
  if (!GEP) {
    LLVM_DEBUG(dbgs() << "masked gathers: no getelementpointer found\n");
    return nullptr;
  }
  LLVM_DEBUG(dbgs() << "masked gathers: getelementpointer found. Loading"
                    << " from base + vector of offsets\n");
  Value *GEPPtr = GEP->getPointerOperand();
  if (GEPPtr->getType()->isVectorTy()) {
    LLVM_DEBUG(dbgs() << "masked gathers: gather from a vector of pointers"
                      << " hidden behind a getelementptr currently not"
                      << " supported. Expanding.\n");
    return nullptr;
  }
  if (GEP->getNumOperands() != 2) {
    LLVM_DEBUG(dbgs() << "masked gathers: getelementptr with too many"
                      << " operands. Expanding.\n");
    return nullptr;
  }
  Offsets = GEP->getOperand(1);
  // SExt offsets inside masked gathers are not permitted by the architecture;
  // we therefore can't fold them
  if (ZExtInst *ZextOffs = dyn_cast<ZExtInst>(Offsets))
    Offsets = ZextOffs->getOperand(0);
  Type *OffsType = VectorType::getInteger(cast<VectorType>(Ty));
  // If the offset we found does not have the type the intrinsic expects,
  // i.e., the same type as the gather itself, we need to convert it (only i
  // types) or fall back to expanding the gather
  if (OffsType != Offsets->getType()) {
    if (OffsType->getScalarSizeInBits() >
        Offsets->getType()->getScalarSizeInBits()) {
      LLVM_DEBUG(dbgs() << "masked gathers: extending offsets\n");
      Offsets = Builder.CreateZExt(Offsets, OffsType, "");
    } else {
      LLVM_DEBUG(dbgs() << "masked gathers: no correct offset type. Can't"
                        << " create masked gather\n");
      return nullptr;
    }
  }
  // If none of the checks failed, return the gep's base pointer
  return GEPPtr;
}

void MVEGatherScatterLowering::lookThroughBitcast(Value *&Ptr) {
  // Look through bitcast instruction if #elements is the same
  if (auto *BitCast = dyn_cast<BitCastInst>(Ptr)) {
    Type *BCTy = BitCast->getType();
    Type *BCSrcTy = BitCast->getOperand(0)->getType();
    if (BCTy->getVectorNumElements() == BCSrcTy->getVectorNumElements()) {
      LLVM_DEBUG(dbgs() << "masked gathers: looking through bitcast\n");
      Ptr = BitCast->getOperand(0);
    }
  }
}

bool MVEGatherScatterLowering::lowerGather(IntrinsicInst *I) {
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

  if (!isLegalTypeAndAlignment(Ty->getVectorNumElements(),
                               Ty->getScalarSizeInBits(), Alignment))
    return false;
  lookThroughBitcast(Ptr);
  assert(Ptr->getType()->isVectorTy() && "Unexpected pointer type");

  IRBuilder<> Builder(I->getContext());
  Builder.SetInsertPoint(I);
  Builder.SetCurrentDebugLocation(I->getDebugLoc());
  Value *Load = tryCreateMaskedGatherOffset(I, Ptr, Builder);
  if (!Load)
    Load = tryCreateMaskedGatherBase(I, Ptr, Builder);
  if (!Load)
    return false;

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

Value *MVEGatherScatterLowering::tryCreateMaskedGatherBase(
    IntrinsicInst *I, Value *Ptr, IRBuilder<> Builder) {
  using namespace PatternMatch;
  LLVM_DEBUG(dbgs() << "masked gathers: loading from vector of pointers\n");
  Type *Ty = I->getType();
  if (Ty->getVectorNumElements() != 4)
    // Can't build an intrinsic for this
    return nullptr;
  Value *Mask = I->getArgOperand(2);
  if (match(Mask, m_One()))
    return Builder.CreateIntrinsic(Intrinsic::arm_mve_vldr_gather_base,
                                   {Ty, Ptr->getType()},
                                   {Ptr, Builder.getInt32(0)});
  else
    return Builder.CreateIntrinsic(
        Intrinsic::arm_mve_vldr_gather_base_predicated,
        {Ty, Ptr->getType(), Mask->getType()},
        {Ptr, Builder.getInt32(0), Mask});
}

Value *MVEGatherScatterLowering::tryCreateMaskedGatherOffset(
    IntrinsicInst *I, Value *Ptr, IRBuilder<> Builder) {
  using namespace PatternMatch;
  Type *Ty = I->getType();
  Value *Offsets;
  Value *BasePtr = checkGEP(Offsets, Ty, Ptr, Builder);
  if (!BasePtr)
    return nullptr;

  unsigned Scale;
  int GEPElemSize =
      BasePtr->getType()->getPointerElementType()->getPrimitiveSizeInBits();
  int ResultElemSize = Ty->getScalarSizeInBits();
  // This can be a 32bit load scaled by 4, a 16bit load scaled by 2, or a
  // 8bit, 16bit or 32bit load scaled by 1
  if (GEPElemSize == 32 && ResultElemSize == 32) {
    Scale = 2;
  } else if (GEPElemSize == 16 && ResultElemSize == 16) {
    Scale = 1;
  } else if (GEPElemSize == 8) {
    Scale = 0;
  } else {
    LLVM_DEBUG(dbgs() << "masked gathers: incorrect scale for load. Can't"
                      << " create masked gather\n");
    return nullptr;
  }

  Value *Mask = I->getArgOperand(2);
  if (!match(Mask, m_One()))
    return Builder.CreateIntrinsic(
        Intrinsic::arm_mve_vldr_gather_offset_predicated,
        {Ty, BasePtr->getType(), Offsets->getType(), Mask->getType()},
        {BasePtr, Offsets, Builder.getInt32(Ty->getScalarSizeInBits()),
         Builder.getInt32(Scale), Builder.getInt32(1), Mask});
  else
    return Builder.CreateIntrinsic(
        Intrinsic::arm_mve_vldr_gather_offset,
        {Ty, BasePtr->getType(), Offsets->getType()},
        {BasePtr, Offsets, Builder.getInt32(Ty->getScalarSizeInBits()),
         Builder.getInt32(Scale), Builder.getInt32(1)});
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
    lowerGather(I);

  return true;
}
