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
#include "llvm/Transforms/Utils/Local.h"
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
    AU.addRequired<LoopInfoWrapperPass>();
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

  Value *lowerGather(IntrinsicInst *I);
  // Create a gather from a base + vector of offsets
  Value *tryCreateMaskedGatherOffset(IntrinsicInst *I, Value *Ptr,
                                     Instruction *&Root, IRBuilder<> &Builder);
  // Create a gather from a vector of pointers
  Value *tryCreateMaskedGatherBase(IntrinsicInst *I, Value *Ptr,
                                   IRBuilder<> &Builder);

  Value *lowerScatter(IntrinsicInst *I);
  // Create a scatter to a base + vector of offsets
  Value *tryCreateMaskedScatterOffset(IntrinsicInst *I, Value *Offsets,
                                      IRBuilder<> &Builder);
  // Create a scatter to a vector of pointers
  Value *tryCreateMaskedScatterBase(IntrinsicInst *I, Value *Ptr,
                                    IRBuilder<> &Builder);

  // Check whether these offsets could be moved out of the loop they're in
  bool optimiseOffsets(Value *Offsets, BasicBlock *BB, LoopInfo *LI);
  // Pushes the given add out of the loop
  void pushOutAdd(PHINode *&Phi, Value *OffsSecondOperand, unsigned StartIndex);
  // Pushes the given mul out of the loop
  void pushOutMul(PHINode *&Phi, Value *IncrementPerRound,
                  Value *OffsSecondOperand, unsigned LoopIncrement,
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
  assert(cast<VectorType>(Ty)->getNumElements() ==
         cast<VectorType>(Offsets->getType())->getNumElements());
  // Only <N x i32> offsets can be integrated into an arm gather, any smaller
  // type would have to be sign extended by the gep - and arm gathers can only
  // zero extend. Additionally, the offsets do have to originate from a zext of
  // a vector with element types smaller or equal the type of the gather we're
  // looking at
  if (Offsets->getType()->getScalarSizeInBits() != 32)
    return nullptr;
  if (ZExtInst *ZextOffs = dyn_cast<ZExtInst>(Offsets))
    Offsets = ZextOffs->getOperand(0);
  else if (!(cast<VectorType>(Offsets->getType())->getNumElements() == 4 &&
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
    auto *BCTy = cast<VectorType>(BitCast->getType());
    auto *BCSrcTy = cast<VectorType>(BitCast->getOperand(0)->getType());
    if (BCTy->getNumElements() == BCSrcTy->getNumElements()) {
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

Value *MVEGatherScatterLowering::lowerGather(IntrinsicInst *I) {
  using namespace PatternMatch;
  LLVM_DEBUG(dbgs() << "masked gathers: checking transform preconditions\n");

  // @llvm.masked.gather.*(Ptrs, alignment, Mask, Src0)
  // Attempt to turn the masked gather in I into a MVE intrinsic
  // Potentially optimising the addressing modes as we do so.
  auto *Ty = cast<VectorType>(I->getType());
  Value *Ptr = I->getArgOperand(0);
  unsigned Alignment = cast<ConstantInt>(I->getArgOperand(1))->getZExtValue();
  Value *Mask = I->getArgOperand(2);
  Value *PassThru = I->getArgOperand(3);

  if (!isLegalTypeAndAlignment(Ty->getNumElements(), Ty->getScalarSizeInBits(),
                               Alignment))
    return nullptr;
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
    return nullptr;

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
  return Load;
}

Value *MVEGatherScatterLowering::tryCreateMaskedGatherBase(IntrinsicInst *I,
                                                           Value *Ptr,
                                                           IRBuilder<> &Builder) {
  using namespace PatternMatch;
  auto *Ty = cast<VectorType>(I->getType());
  LLVM_DEBUG(dbgs() << "masked gathers: loading from vector of pointers\n");
  if (Ty->getNumElements() != 4 || Ty->getScalarSizeInBits() != 32)
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

    // The correct root to replace is not the CallInst itself, but the
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

Value *MVEGatherScatterLowering::lowerScatter(IntrinsicInst *I) {
  using namespace PatternMatch;
  LLVM_DEBUG(dbgs() << "masked scatters: checking transform preconditions\n");

  // @llvm.masked.scatter.*(data, ptrs, alignment, mask)
  // Attempt to turn the masked scatter in I into a MVE intrinsic
  // Potentially optimising the addressing modes as we do so.
  Value *Input = I->getArgOperand(0);
  Value *Ptr = I->getArgOperand(1);
  unsigned Alignment = cast<ConstantInt>(I->getArgOperand(2))->getZExtValue();
  auto *Ty = cast<VectorType>(Input->getType());

  if (!isLegalTypeAndAlignment(Ty->getNumElements(), Ty->getScalarSizeInBits(),
                               Alignment))
    return nullptr;

  lookThroughBitcast(Ptr);
  assert(Ptr->getType()->isVectorTy() && "Unexpected pointer type");

  IRBuilder<> Builder(I->getContext());
  Builder.SetInsertPoint(I);
  Builder.SetCurrentDebugLocation(I->getDebugLoc());

  Value *Store = tryCreateMaskedScatterOffset(I, Ptr, Builder);
  if (!Store)
    Store = tryCreateMaskedScatterBase(I, Ptr, Builder);
  if (!Store)
    return nullptr;

  LLVM_DEBUG(dbgs() << "masked scatters: successfully built masked scatter\n");
  I->replaceAllUsesWith(Store);
  I->eraseFromParent();
  return Store;
}

Value *MVEGatherScatterLowering::tryCreateMaskedScatterBase(
    IntrinsicInst *I, Value *Ptr, IRBuilder<> &Builder) {
  using namespace PatternMatch;
  Value *Input = I->getArgOperand(0);
  Value *Mask = I->getArgOperand(3);
  auto *Ty = cast<VectorType>(Input->getType());
  // Only QR variants allow truncating
  if (!(Ty->getNumElements() == 4 && Ty->getScalarSizeInBits() == 32)) {
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

void MVEGatherScatterLowering::pushOutAdd(PHINode *&Phi,
                                          Value *OffsSecondOperand,
                                          unsigned StartIndex) {
  LLVM_DEBUG(dbgs() << "masked gathers/scatters: optimising add instruction\n");
  Instruction *InsertionPoint;
  if (isa<Instruction>(OffsSecondOperand))
    InsertionPoint = &cast<Instruction>(OffsSecondOperand)->getParent()->back();
  else
    InsertionPoint =
        &cast<Instruction>(Phi->getIncomingBlock(StartIndex)->back());
  // Initialize the phi with a vector that contains a sum of the constants
  Instruction *NewIndex = BinaryOperator::Create(
      Instruction::Add, Phi->getIncomingValue(StartIndex), OffsSecondOperand,
      "PushedOutAdd", InsertionPoint);
  unsigned IncrementIndex = StartIndex == 0 ? 1 : 0;

  // Order such that start index comes first (this reduces mov's)
  Phi->addIncoming(NewIndex, Phi->getIncomingBlock(StartIndex));
  Phi->addIncoming(Phi->getIncomingValue(IncrementIndex),
                   Phi->getIncomingBlock(IncrementIndex));
  Phi->removeIncomingValue(IncrementIndex);
  Phi->removeIncomingValue(StartIndex);
}

void MVEGatherScatterLowering::pushOutMul(PHINode *&Phi,
                                          Value *IncrementPerRound,
                                          Value *OffsSecondOperand,
                                          unsigned LoopIncrement,
                                          IRBuilder<> &Builder) {
  LLVM_DEBUG(dbgs() << "masked gathers/scatters: optimising mul instruction\n");

  // Create a new scalar add outside of the loop and transform it to a splat
  // by which loop variable can be incremented
  Instruction *InsertionPoint;
  if (isa<Instruction>(OffsSecondOperand))
    InsertionPoint = &cast<Instruction>(OffsSecondOperand)->getParent()->back();
  else
    InsertionPoint = &cast<Instruction>(
        Phi->getIncomingBlock(LoopIncrement == 1 ? 0 : 1)->back());

  // Create a new index
  Value *StartIndex = BinaryOperator::Create(
      Instruction::Mul, Phi->getIncomingValue(LoopIncrement == 1 ? 0 : 1),
      OffsSecondOperand, "PushedOutMul", InsertionPoint);

  Instruction *Product =
      BinaryOperator::Create(Instruction::Mul, IncrementPerRound,
                             OffsSecondOperand, "Product", InsertionPoint);
  // Increment NewIndex by Product instead of the multiplication
  Instruction *NewIncrement = BinaryOperator::Create(
      Instruction::Add, Phi, Product, "IncrementPushedOutMul",
      cast<Instruction>(Phi->getIncomingBlock(LoopIncrement)->back())
          .getPrevNode());

  Phi->addIncoming(StartIndex,
                   Phi->getIncomingBlock(LoopIncrement == 1 ? 0 : 1));
  Phi->addIncoming(NewIncrement, Phi->getIncomingBlock(LoopIncrement));
  Phi->removeIncomingValue((unsigned)0);
  Phi->removeIncomingValue((unsigned)0);
  return;
}

// Return true if the given intrinsic is a gather or scatter
bool isGatherScatter(IntrinsicInst *IntInst) {
  if (IntInst == nullptr)
    return false;
  unsigned IntrinsicID = IntInst->getIntrinsicID();
  return (IntrinsicID == Intrinsic::masked_gather ||
          IntrinsicID == Intrinsic::arm_mve_vldr_gather_base ||
          IntrinsicID == Intrinsic::arm_mve_vldr_gather_base_predicated ||
          IntrinsicID == Intrinsic::arm_mve_vldr_gather_base_wb ||
          IntrinsicID == Intrinsic::arm_mve_vldr_gather_base_wb_predicated ||
          IntrinsicID == Intrinsic::arm_mve_vldr_gather_offset ||
          IntrinsicID == Intrinsic::arm_mve_vldr_gather_offset_predicated ||
          IntrinsicID == Intrinsic::masked_scatter ||
          IntrinsicID == Intrinsic::arm_mve_vstr_scatter_base ||
          IntrinsicID == Intrinsic::arm_mve_vstr_scatter_base_predicated ||
          IntrinsicID == Intrinsic::arm_mve_vstr_scatter_base_wb ||
          IntrinsicID == Intrinsic::arm_mve_vstr_scatter_base_wb_predicated ||
          IntrinsicID == Intrinsic::arm_mve_vstr_scatter_offset ||
          IntrinsicID == Intrinsic::arm_mve_vstr_scatter_offset_predicated);
}

// Check whether all usages of this instruction are as offsets of
// gathers/scatters or simple arithmetics only used by gathers/scatters
bool hasAllGatScatUsers(Instruction *I) {
  if (I->hasNUses(0)) {
    return false;
  }
  bool Gatscat = true;
  for (User *U : I->users()) {
    if (!isa<Instruction>(U))
      return false;
    if (isa<GetElementPtrInst>(U) ||
        isGatherScatter(dyn_cast<IntrinsicInst>(U))) {
      return Gatscat;
    } else {
      unsigned OpCode = cast<Instruction>(U)->getOpcode();
      if ((OpCode == Instruction::Add || OpCode == Instruction::Mul) &&
          hasAllGatScatUsers(cast<Instruction>(U))) {
        continue;
      }
      return false;
    }
  }
  return Gatscat;
}

bool MVEGatherScatterLowering::optimiseOffsets(Value *Offsets, BasicBlock *BB,
                                               LoopInfo *LI) {
  LLVM_DEBUG(dbgs() << "masked gathers/scatters: trying to optimize\n");
  // Optimise the addresses of gathers/scatters by moving invariant
  // calculations out of the loop
  if (!isa<Instruction>(Offsets))
    return false;
  Instruction *Offs = cast<Instruction>(Offsets);
  if (Offs->getOpcode() != Instruction::Add &&
      Offs->getOpcode() != Instruction::Mul)
    return false;
  Loop *L = LI->getLoopFor(BB);
  if (L == nullptr)
    return false;
  if (!Offs->hasOneUse()) {
    if (!hasAllGatScatUsers(Offs))
      return false;
  }

  // Find out which, if any, operand of the instruction
  // is a phi node
  PHINode *Phi;
  int OffsSecondOp;
  if (isa<PHINode>(Offs->getOperand(0))) {
    Phi = cast<PHINode>(Offs->getOperand(0));
    OffsSecondOp = 1;
  } else if (isa<PHINode>(Offs->getOperand(1))) {
    Phi = cast<PHINode>(Offs->getOperand(1));
    OffsSecondOp = 0;
  } else {
    bool Changed = true;
    if (isa<Instruction>(Offs->getOperand(0)) &&
        L->contains(cast<Instruction>(Offs->getOperand(0))))
      Changed |= optimiseOffsets(Offs->getOperand(0), BB, LI);
    if (isa<Instruction>(Offs->getOperand(1)) &&
        L->contains(cast<Instruction>(Offs->getOperand(1))))
      Changed |= optimiseOffsets(Offs->getOperand(1), BB, LI);
    if (!Changed) {
      return false;
    } else {
      if (isa<PHINode>(Offs->getOperand(0))) {
        Phi = cast<PHINode>(Offs->getOperand(0));
        OffsSecondOp = 1;
      } else if (isa<PHINode>(Offs->getOperand(1))) {
        Phi = cast<PHINode>(Offs->getOperand(1));
        OffsSecondOp = 0;
      } else {
        return false;
      }
    }
  }
  // A phi node we want to perform this function on should be from the
  // loop header, and shouldn't have more than 2 incoming values
  if (Phi->getParent() != L->getHeader() ||
      Phi->getNumIncomingValues() != 2)
    return false;

  // The phi must be an induction variable
  Instruction *Op;
  int IncrementingBlock = -1;

  for (int i = 0; i < 2; i++)
    if ((Op = dyn_cast<Instruction>(Phi->getIncomingValue(i))) != nullptr)
      if (Op->getOpcode() == Instruction::Add &&
          (Op->getOperand(0) == Phi || Op->getOperand(1) == Phi))
        IncrementingBlock = i;
  if (IncrementingBlock == -1)
    return false;

  Instruction *IncInstruction =
      cast<Instruction>(Phi->getIncomingValue(IncrementingBlock));

  // If the phi is not used by anything else, we can just adapt it when
  // replacing the instruction; if it is, we'll have to duplicate it
  PHINode *NewPhi;
  Value *IncrementPerRound = IncInstruction->getOperand(
      (IncInstruction->getOperand(0) == Phi) ? 1 : 0);

  // Get the value that is added to/multiplied with the phi
  Value *OffsSecondOperand = Offs->getOperand(OffsSecondOp);

  if (IncrementPerRound->getType() != OffsSecondOperand->getType())
    // Something has gone wrong, abort
    return false;

  // Only proceed if the increment per round is a constant or an instruction
  // which does not originate from within the loop
  if (!isa<Constant>(IncrementPerRound) &&
      !(isa<Instruction>(IncrementPerRound) &&
        !L->contains(cast<Instruction>(IncrementPerRound))))
    return false;

  if (Phi->getNumUses() == 2) {
    // No other users -> reuse existing phi (One user is the instruction
    // we're looking at, the other is the phi increment)
    if (IncInstruction->getNumUses() != 1) {
      // If the incrementing instruction does have more users than
      // our phi, we need to copy it
      IncInstruction = BinaryOperator::Create(
          Instruction::BinaryOps(IncInstruction->getOpcode()), Phi,
          IncrementPerRound, "LoopIncrement", IncInstruction);
      Phi->setIncomingValue(IncrementingBlock, IncInstruction);
    }
    NewPhi = Phi;
  } else {
    // There are other users -> create a new phi
    NewPhi = PHINode::Create(Phi->getType(), 0, "NewPhi", Phi);
    std::vector<Value *> Increases;
    // Copy the incoming values of the old phi
    NewPhi->addIncoming(Phi->getIncomingValue(IncrementingBlock == 1 ? 0 : 1),
                        Phi->getIncomingBlock(IncrementingBlock == 1 ? 0 : 1));
    IncInstruction = BinaryOperator::Create(
        Instruction::BinaryOps(IncInstruction->getOpcode()), NewPhi,
        IncrementPerRound, "LoopIncrement", IncInstruction);
    NewPhi->addIncoming(IncInstruction,
                        Phi->getIncomingBlock(IncrementingBlock));
    IncrementingBlock = 1;
  }

  IRBuilder<> Builder(BB->getContext());
  Builder.SetInsertPoint(Phi);
  Builder.SetCurrentDebugLocation(Offs->getDebugLoc());

  switch (Offs->getOpcode()) {
  case Instruction::Add:
    pushOutAdd(NewPhi, OffsSecondOperand, IncrementingBlock == 1 ? 0 : 1);
    break;
  case Instruction::Mul:
    pushOutMul(NewPhi, IncrementPerRound, OffsSecondOperand, IncrementingBlock,
               Builder);
    break;
  default:
    return false;
  }
  LLVM_DEBUG(
      dbgs() << "masked gathers/scatters: simplified loop variable add/mul\n");

  // The instruction has now been "absorbed" into the phi value
  Offs->replaceAllUsesWith(NewPhi);
  if (Offs->hasNUses(0))
    Offs->eraseFromParent();
  // Clean up the old increment in case it's unused because we built a new
  // one
  if (IncInstruction->hasNUses(0))
    IncInstruction->eraseFromParent();

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
  SmallVector<IntrinsicInst *, 4> Scatters;
  LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

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
  for (unsigned i = 0; i < Gathers.size(); i++) {
    IntrinsicInst *I = Gathers[i];
    if (isa<GetElementPtrInst>(I->getArgOperand(0)))
      optimiseOffsets(cast<Instruction>(I->getArgOperand(0))->getOperand(1),
                      I->getParent(), &LI);
    Value *L = lowerGather(I);
    if (L == nullptr)
      continue;
    // Get rid of any now dead instructions
    SimplifyInstructionsInBlock(cast<Instruction>(L)->getParent());
    Changed = true;
  }

  for (unsigned i = 0; i < Scatters.size(); i++) {
    IntrinsicInst *I = Scatters[i];
    if (isa<GetElementPtrInst>(I->getArgOperand(1)))
      optimiseOffsets(cast<Instruction>(I->getArgOperand(1))->getOperand(1),
                      I->getParent(), &LI);
    Value *S = lowerScatter(I);
    if (S == nullptr)
      continue;
    // Get rid of any now dead instructions
    SimplifyInstructionsInBlock(cast<Instruction>(S)->getParent());
    Changed = true;
  }
  return Changed;
}
