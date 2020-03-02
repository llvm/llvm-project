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
  Value *checkGEP(Value *&Offsets, Type *Ty, Value *Ptr, IRBuilder<> &Builder);
  // Compute the scale of this gather/scatter instruction
  int computeScale(unsigned GEPElemSize, unsigned MemoryElemSize);

  bool lowerGather(IntrinsicInst *I);
  // Create a gather from a base + vector of offsets
  Value *tryCreateMaskedGatherOffset(IntrinsicInst *I, Value *Ptr,
                                     Instruction *&Root, IRBuilder<> &Builder);
  // Create a gather from a vector of pointers
  Value *tryCreateMaskedGatherBase(IntrinsicInst *I, Value *Ptr,
                                   IRBuilder<> &Builder);

  bool lowerScatter(IntrinsicInst *I);
  // Create a scatter to a base + vector of offsets
  Value *tryCreateMaskedScatterOffset(IntrinsicInst *I, Value *Ptr,
                                      IRBuilder<> &Builder);
  // Create a scatter to a vector of pointers
  Value *tryCreateMaskedScatterBase(IntrinsicInst *I, Value *Ptr,
                                    IRBuilder<> &Builder);
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
  if (((NumElements == 4 &&
        (ElemSize == 32 || ElemSize == 16 || ElemSize == 8)) ||
       (NumElements == 8 && (ElemSize == 16 || ElemSize == 8)) ||
       (NumElements == 16 && ElemSize == 8)) &&
      ElemSize / 8 <= Alignment)
    return true;
  LLVM_DEBUG(dbgs() << "masked gathers/scatters: instruction does not have "
                    << "valid alignment or vector type \n");
  return false;
}

Value *MVEGatherScatterLowering::checkGEP(Value *&Offsets, Type *Ty, Value *Ptr,
                                          IRBuilder<> &Builder) {
  GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Ptr);
  if (!GEP) {
    LLVM_DEBUG(
        dbgs() << "masked gathers/scatters: no getelementpointer found\n");
    return nullptr;
  }
  LLVM_DEBUG(dbgs() << "masked gathers/scatters: getelementpointer found."
                    << " Looking at intrinsic for base + vector of offsets\n");
  Value *GEPPtr = GEP->getPointerOperand();
  if (GEPPtr->getType()->isVectorTy()) {
    return nullptr;
  }
  if (GEP->getNumOperands() != 2) {
    LLVM_DEBUG(dbgs() << "masked gathers/scatters: getelementptr with too many"
                      << " operands. Expanding.\n");
    return nullptr;
  }
  Offsets = GEP->getOperand(1);
  // Paranoid check whether the number of parallel lanes is the same
  assert(Ty->getVectorNumElements() ==
         Offsets->getType()->getVectorNumElements());
  // Only <N x i32> offsets can be integrated into an arm gather, any smaller
  // type would have to be sign extended by the gep - and arm gathers can only
  // zero extend. Additionally, the offsets do have to originate from a zext of
  // a vector with element types smaller or equal the type of the gather we're
  // looking at
  if (Offsets->getType()->getScalarSizeInBits() != 32)
    return nullptr;
  if (ZExtInst *ZextOffs = dyn_cast<ZExtInst>(Offsets))
    Offsets = ZextOffs->getOperand(0);
  else if (!(Offsets->getType()->getVectorNumElements() == 4 &&
             Offsets->getType()->getScalarSizeInBits() == 32))
    return nullptr;

  if (Ty != Offsets->getType()) {
    if ((Ty->getScalarSizeInBits() <
         Offsets->getType()->getScalarSizeInBits())) {
      LLVM_DEBUG(dbgs() << "masked gathers/scatters: no correct offset type."
                        << " Can't create intrinsic.\n");
      return nullptr;
    } else {
      Offsets = Builder.CreateZExt(
          Offsets, VectorType::getInteger(cast<VectorType>(Ty)));
    }
  }
  // If none of the checks failed, return the gep's base pointer
  LLVM_DEBUG(dbgs() << "masked gathers/scatters: found correct offsets\n");
  return GEPPtr;
}

void MVEGatherScatterLowering::lookThroughBitcast(Value *&Ptr) {
  // Look through bitcast instruction if #elements is the same
  if (auto *BitCast = dyn_cast<BitCastInst>(Ptr)) {
    Type *BCTy = BitCast->getType();
    Type *BCSrcTy = BitCast->getOperand(0)->getType();
    if (BCTy->getVectorNumElements() == BCSrcTy->getVectorNumElements()) {
      LLVM_DEBUG(
          dbgs() << "masked gathers/scatters: looking through bitcast\n");
      Ptr = BitCast->getOperand(0);
    }
  }
}

int MVEGatherScatterLowering::computeScale(unsigned GEPElemSize,
                                           unsigned MemoryElemSize) {
  // This can be a 32bit load/store scaled by 4, a 16bit load/store scaled by 2,
  // or a 8bit, 16bit or 32bit load/store scaled by 1
  if (GEPElemSize == 32 && MemoryElemSize == 32)
    return 2;
  else if (GEPElemSize == 16 && MemoryElemSize == 16)
    return 1;
  else if (GEPElemSize == 8)
    return 0;
  LLVM_DEBUG(dbgs() << "masked gathers/scatters: incorrect scale. Can't "
                    << "create intrinsic\n");
  return -1;
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

  Instruction *Root = I;
  Value *Load = tryCreateMaskedGatherOffset(I, Ptr, Root, Builder);
  if (!Load)
    Load = tryCreateMaskedGatherBase(I, Ptr, Builder);
  if (!Load)
    return false;

  if (!isa<UndefValue>(PassThru) && !match(PassThru, m_Zero())) {
    LLVM_DEBUG(dbgs() << "masked gathers: found non-trivial passthru - "
                      << "creating select\n");
    Load = Builder.CreateSelect(Mask, Load, PassThru);
  }

  Root->replaceAllUsesWith(Load);
  Root->eraseFromParent();
  if (Root != I)
    // If this was an extending gather, we need to get rid of the sext/zext
    // sext/zext as well as of the gather itself
    I->eraseFromParent();
  LLVM_DEBUG(dbgs() << "masked gathers: successfully built masked gather\n");
  return true;
}

Value *MVEGatherScatterLowering::tryCreateMaskedGatherBase(
    IntrinsicInst *I, Value *Ptr, IRBuilder<> &Builder) {
  using namespace PatternMatch;
  Type *Ty = I->getType();
  LLVM_DEBUG(dbgs() << "masked gathers: loading from vector of pointers\n");
  if (Ty->getVectorNumElements() != 4 || Ty->getScalarSizeInBits() != 32)
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
    IntrinsicInst *I, Value *Ptr, Instruction *&Root, IRBuilder<> &Builder) {
  using namespace PatternMatch;

  Type *OriginalTy = I->getType();
  Type *ResultTy = OriginalTy;

  unsigned Unsigned = 1;
  // The size of the gather was already checked in isLegalTypeAndAlignment;
  // if it was not a full vector width an appropriate extend should follow.
  auto *Extend = Root;
  if (OriginalTy->getPrimitiveSizeInBits() < 128) {
    // Only transform gathers with exactly one use
    if (!I->hasOneUse())
      return nullptr;

    // The correct root to replace is the not the CallInst itself, but the
    // instruction which extends it
    Extend = cast<Instruction>(*I->users().begin());
    if (isa<SExtInst>(Extend)) {
      Unsigned = 0;
    } else if (!isa<ZExtInst>(Extend)) {
      LLVM_DEBUG(dbgs() << "masked gathers: extend needed but not provided. "
                        << "Expanding\n");
      return nullptr;
    }
    LLVM_DEBUG(dbgs() << "masked gathers: found an extending gather\n");
    ResultTy = Extend->getType();
    // The final size of the gather must be a full vector width
    if (ResultTy->getPrimitiveSizeInBits() != 128) {
      LLVM_DEBUG(dbgs() << "masked gathers: extending from the wrong type. "
                        << "Expanding\n");
      return nullptr;
    }
  }

  Value *Offsets;
  Value *BasePtr = checkGEP(Offsets, ResultTy, Ptr, Builder);
  if (!BasePtr)
    return nullptr;

  int Scale = computeScale(
      BasePtr->getType()->getPointerElementType()->getPrimitiveSizeInBits(),
      OriginalTy->getScalarSizeInBits());
  if (Scale == -1)
    return nullptr;
  Root = Extend;

  Value *Mask = I->getArgOperand(2);
  if (!match(Mask, m_One()))
    return Builder.CreateIntrinsic(
        Intrinsic::arm_mve_vldr_gather_offset_predicated,
        {ResultTy, BasePtr->getType(), Offsets->getType(), Mask->getType()},
        {BasePtr, Offsets, Builder.getInt32(OriginalTy->getScalarSizeInBits()),
         Builder.getInt32(Scale), Builder.getInt32(Unsigned), Mask});
  else
    return Builder.CreateIntrinsic(
        Intrinsic::arm_mve_vldr_gather_offset,
        {ResultTy, BasePtr->getType(), Offsets->getType()},
        {BasePtr, Offsets, Builder.getInt32(OriginalTy->getScalarSizeInBits()),
         Builder.getInt32(Scale), Builder.getInt32(Unsigned)});
}

bool MVEGatherScatterLowering::lowerScatter(IntrinsicInst *I) {
  using namespace PatternMatch;
  LLVM_DEBUG(dbgs() << "masked scatters: checking transform preconditions\n");

  // @llvm.masked.scatter.*(data, ptrs, alignment, mask)
  // Attempt to turn the masked scatter in I into a MVE intrinsic
  // Potentially optimising the addressing modes as we do so.
  Value *Input = I->getArgOperand(0);
  Value *Ptr = I->getArgOperand(1);
  unsigned Alignment = cast<ConstantInt>(I->getArgOperand(2))->getZExtValue();
  Type *Ty = Input->getType();

  if (!isLegalTypeAndAlignment(Ty->getVectorNumElements(),
                               Ty->getScalarSizeInBits(), Alignment))
    return false;
  lookThroughBitcast(Ptr);
  assert(Ptr->getType()->isVectorTy() && "Unexpected pointer type");

  IRBuilder<> Builder(I->getContext());
  Builder.SetInsertPoint(I);
  Builder.SetCurrentDebugLocation(I->getDebugLoc());

  Value *Store = tryCreateMaskedScatterOffset(I, Ptr, Builder);
  if (!Store)
    Store = tryCreateMaskedScatterBase(I, Ptr, Builder);
  if (!Store)
    return false;

  LLVM_DEBUG(dbgs() << "masked scatters: successfully built masked scatter\n");
  I->replaceAllUsesWith(Store);
  I->eraseFromParent();
  return true;
}

Value *MVEGatherScatterLowering::tryCreateMaskedScatterBase(
    IntrinsicInst *I, Value *Ptr, IRBuilder<> &Builder) {
  using namespace PatternMatch;
  Value *Input = I->getArgOperand(0);
  Value *Mask = I->getArgOperand(3);
  Type *Ty = Input->getType();
  // Only QR variants allow truncating
  if (!(Ty->getVectorNumElements() == 4 && Ty->getScalarSizeInBits() == 32)) {
    // Can't build an intrinsic for this
    return nullptr;
  }
  //  int_arm_mve_vstr_scatter_base(_predicated) addr, offset, data(, mask)
  LLVM_DEBUG(dbgs() << "masked scatters: storing to a vector of pointers\n");
  if (match(Mask, m_One()))
    return Builder.CreateIntrinsic(Intrinsic::arm_mve_vstr_scatter_base,
                                   {Ptr->getType(), Input->getType()},
                                   {Ptr, Builder.getInt32(0), Input});
  else
    return Builder.CreateIntrinsic(
        Intrinsic::arm_mve_vstr_scatter_base_predicated,
        {Ptr->getType(), Input->getType(), Mask->getType()},
        {Ptr, Builder.getInt32(0), Input, Mask});
}

Value *MVEGatherScatterLowering::tryCreateMaskedScatterOffset(
    IntrinsicInst *I, Value *Ptr, IRBuilder<> &Builder) {
  using namespace PatternMatch;
  Value *Input = I->getArgOperand(0);
  Value *Mask = I->getArgOperand(3);
  Type *InputTy = Input->getType();
  Type *MemoryTy = InputTy;
  LLVM_DEBUG(dbgs() << "masked scatters: getelementpointer found. Storing"
                    << " to base + vector of offsets\n");
  // If the input has been truncated, try to integrate that trunc into the
  // scatter instruction (we don't care about alignment here)
  if (TruncInst *Trunc = dyn_cast<TruncInst>(Input)) {
    Value *PreTrunc = Trunc->getOperand(0);
    Type *PreTruncTy = PreTrunc->getType();
    if (PreTruncTy->getPrimitiveSizeInBits() == 128) {
      Input = PreTrunc;
      InputTy = PreTruncTy;
    }
  }
  if (InputTy->getPrimitiveSizeInBits() != 128) {
    LLVM_DEBUG(
        dbgs() << "masked scatters: cannot create scatters for non-standard"
               << " input types. Expanding.\n");
    return nullptr;
  }

  Value *Offsets;
  Value *BasePtr = checkGEP(Offsets, InputTy, Ptr, Builder);
  if (!BasePtr)
    return nullptr;
  int Scale = computeScale(
      BasePtr->getType()->getPointerElementType()->getPrimitiveSizeInBits(),
      MemoryTy->getScalarSizeInBits());
  if (Scale == -1)
    return nullptr;

  if (!match(Mask, m_One()))
    return Builder.CreateIntrinsic(
        Intrinsic::arm_mve_vstr_scatter_offset_predicated,
        {BasePtr->getType(), Offsets->getType(), Input->getType(),
         Mask->getType()},
        {BasePtr, Offsets, Input,
         Builder.getInt32(MemoryTy->getScalarSizeInBits()),
         Builder.getInt32(Scale), Mask});
  else
    return Builder.CreateIntrinsic(
        Intrinsic::arm_mve_vstr_scatter_offset,
        {BasePtr->getType(), Offsets->getType(), Input->getType()},
        {BasePtr, Offsets, Input,
         Builder.getInt32(MemoryTy->getScalarSizeInBits()),
         Builder.getInt32(Scale)});
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
  SmallVector<IntrinsicInst *, 4> Scatters;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I);
      if (II && II->getIntrinsicID() == Intrinsic::masked_gather)
        Gathers.push_back(II);
      else if (II && II->getIntrinsicID() == Intrinsic::masked_scatter)
        Scatters.push_back(II);
    }
  }

  bool Changed = false;
  for (IntrinsicInst *I : Gathers)
    Changed |= lowerGather(I);
  for (IntrinsicInst *I : Scatters)
    Changed |= lowerScatter(I);

  return Changed;
}
