//===--------- VectorBuilderTest.cpp - VectorBuilder unit tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/VectorBuilder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

static unsigned VectorNumElements = 8;

class VectorBuilderTest : public testing::Test {
protected:
  LLVMContext Context;

  VectorBuilderTest() : Context() {}

  std::unique_ptr<Module> createBuilderModule(Function *&Func, BasicBlock *&BB,
                                              Value *&Mask, Value *&EVL) {
    auto Mod = std::make_unique<Module>("TestModule", Context);
    auto *Int32Ty = Type::getInt32Ty(Context);
    auto *Mask8Ty =
        FixedVectorType::get(Type::getInt1Ty(Context), VectorNumElements);
    auto *VoidFuncTy =
        FunctionType::get(Type::getVoidTy(Context), {Mask8Ty, Int32Ty}, false);
    Func =
        Function::Create(VoidFuncTy, GlobalValue::ExternalLinkage, "bla", *Mod);
    Mask = Func->getArg(0);
    EVL = Func->getArg(1);
    BB = BasicBlock::Create(Context, "entry", Func);

    return Mod;
  }
};

/// Check that creating binary arithmetic VP intrinsics works.
TEST_F(VectorBuilderTest, TestCreateBinaryInstructions) {
  Function *F;
  BasicBlock *BB;
  Value *Mask, *EVL;
  auto Mod = createBuilderModule(F, BB, Mask, EVL);

  IRBuilder<> Builder(BB);
  VectorBuilder VBuild(Builder);
  VBuild.setMask(Mask).setEVL(EVL);

  auto *FloatVecTy =
      FixedVectorType::get(Type::getFloatTy(Context), VectorNumElements);
  auto *IntVecTy =
      FixedVectorType::get(Type::getInt32Ty(Context), VectorNumElements);

#define HANDLE_BINARY_INST(NUM, OPCODE, INSTCLASS)                             \
  {                                                                            \
    auto VPID = VPIntrinsic::getForOpcode(Instruction::OPCODE);                \
    bool IsFP = (#INSTCLASS)[0] == 'F';                                        \
    auto *ValueTy = IsFP ? FloatVecTy : IntVecTy;                              \
    Value *Op = UndefValue::get(ValueTy);                                      \
    auto *I = VBuild.createVectorInstruction(Instruction::OPCODE, ValueTy,     \
                                             {Op, Op});                        \
    ASSERT_TRUE(isa<VPIntrinsic>(I));                                          \
    auto *VPIntrin = cast<VPIntrinsic>(I);                                     \
    ASSERT_EQ(VPIntrin->getIntrinsicID(), VPID);                               \
    ASSERT_EQ(VPIntrin->getMaskParam(), Mask);                                 \
    ASSERT_EQ(VPIntrin->getVectorLengthParam(), EVL);                          \
  }
#include "llvm/IR/Instruction.def"
}

static bool isAllTrueMask(Value *Val, unsigned NumElements) {
  auto *ConstMask = dyn_cast<Constant>(Val);
  if (!ConstMask)
    return false;

  // Structure check.
  if (!ConstMask->isAllOnesValue())
    return false;

  // Type check.
  auto *MaskVecTy = cast<FixedVectorType>(ConstMask->getType());
  if (MaskVecTy->getNumElements() != NumElements)
    return false;

  return MaskVecTy->getElementType()->isIntegerTy(1);
}

/// Check that creating binary arithmetic VP intrinsics works.
TEST_F(VectorBuilderTest, TestCreateBinaryInstructions_FixedVector_NoMask) {
  Function *F;
  BasicBlock *BB;
  Value *Mask, *EVL;
  auto Mod = createBuilderModule(F, BB, Mask, EVL);

  IRBuilder<> Builder(BB);
  VectorBuilder VBuild(Builder);
  VBuild.setEVL(EVL).setStaticVL(VectorNumElements);

  auto *FloatVecTy =
      FixedVectorType::get(Type::getFloatTy(Context), VectorNumElements);
  auto *IntVecTy =
      FixedVectorType::get(Type::getInt32Ty(Context), VectorNumElements);

#define HANDLE_BINARY_INST(NUM, OPCODE, INSTCLASS)                             \
  {                                                                            \
    auto VPID = VPIntrinsic::getForOpcode(Instruction::OPCODE);                \
    bool IsFP = (#INSTCLASS)[0] == 'F';                                        \
    Type *ValueTy = IsFP ? FloatVecTy : IntVecTy;                              \
    Value *Op = UndefValue::get(ValueTy);                                      \
    auto *I = VBuild.createVectorInstruction(Instruction::OPCODE, ValueTy,     \
                                             {Op, Op});                        \
    ASSERT_TRUE(isa<VPIntrinsic>(I));                                          \
    auto *VPIntrin = cast<VPIntrinsic>(I);                                     \
    ASSERT_EQ(VPIntrin->getIntrinsicID(), VPID);                               \
    ASSERT_TRUE(isAllTrueMask(VPIntrin->getMaskParam(), VectorNumElements));   \
    ASSERT_EQ(VPIntrin->getVectorLengthParam(), EVL);                          \
  }
#include "llvm/IR/Instruction.def"
}

static bool isLegalConstEVL(Value *Val, unsigned ExpectedEVL) {
  auto *ConstEVL = dyn_cast<ConstantInt>(Val);
  if (!ConstEVL)
    return false;

  // Value check.
  if (ConstEVL->getZExtValue() != ExpectedEVL)
    return false;

  // Type check.
  return ConstEVL->getType()->isIntegerTy(32);
}

/// Check that creating binary arithmetic VP intrinsics works.
TEST_F(VectorBuilderTest, TestCreateBinaryInstructions_FixedVector_NoEVL) {
  Function *F;
  BasicBlock *BB;
  Value *Mask, *EVL;
  auto Mod = createBuilderModule(F, BB, Mask, EVL);

  IRBuilder<> Builder(BB);
  VectorBuilder VBuild(Builder);
  VBuild.setMask(Mask).setStaticVL(VectorNumElements);

  auto *FloatVecTy =
      FixedVectorType::get(Type::getFloatTy(Context), VectorNumElements);
  auto *IntVecTy =
      FixedVectorType::get(Type::getInt32Ty(Context), VectorNumElements);

#define HANDLE_BINARY_INST(NUM, OPCODE, INSTCLASS)                             \
  {                                                                            \
    auto VPID = VPIntrinsic::getForOpcode(Instruction::OPCODE);                \
    bool IsFP = (#INSTCLASS)[0] == 'F';                                        \
    Type *ValueTy = IsFP ? FloatVecTy : IntVecTy;                              \
    Value *Op = UndefValue::get(ValueTy);                                      \
    auto *I = VBuild.createVectorInstruction(Instruction::OPCODE, ValueTy,     \
                                             {Op, Op});                        \
    ASSERT_TRUE(isa<VPIntrinsic>(I));                                          \
    auto *VPIntrin = cast<VPIntrinsic>(I);                                     \
    ASSERT_EQ(VPIntrin->getIntrinsicID(), VPID);                               \
    ASSERT_EQ(VPIntrin->getMaskParam(), Mask);                                 \
    ASSERT_TRUE(                                                               \
        isLegalConstEVL(VPIntrin->getVectorLengthParam(), VectorNumElements)); \
  }
#include "llvm/IR/Instruction.def"
}

/// Check that creating binary arithmetic VP intrinsics works.
TEST_F(VectorBuilderTest,
       TestCreateBinaryInstructions_FixedVector_NoMask_NoEVL) {
  Function *F;
  BasicBlock *BB;
  Value *Mask, *EVL;
  auto Mod = createBuilderModule(F, BB, Mask, EVL);

  IRBuilder<> Builder(BB);
  VectorBuilder VBuild(Builder);
  VBuild.setStaticVL(VectorNumElements);

  auto *FloatVecTy =
      FixedVectorType::get(Type::getFloatTy(Context), VectorNumElements);
  auto *IntVecTy =
      FixedVectorType::get(Type::getInt32Ty(Context), VectorNumElements);

#define HANDLE_BINARY_INST(NUM, OPCODE, INSTCLASS)                             \
  {                                                                            \
    auto VPID = VPIntrinsic::getForOpcode(Instruction::OPCODE);                \
    bool IsFP = (#INSTCLASS)[0] == 'F';                                        \
    Type *ValueTy = IsFP ? FloatVecTy : IntVecTy;                              \
    Value *Op = UndefValue::get(ValueTy);                                      \
    auto *I = VBuild.createVectorInstruction(Instruction::OPCODE, ValueTy,     \
                                             {Op, Op});                        \
    ASSERT_TRUE(isa<VPIntrinsic>(I));                                          \
    auto *VPIntrin = cast<VPIntrinsic>(I);                                     \
    ASSERT_EQ(VPIntrin->getIntrinsicID(), VPID);                               \
    ASSERT_TRUE(isAllTrueMask(VPIntrin->getMaskParam(), VectorNumElements));   \
    ASSERT_TRUE(                                                               \
        isLegalConstEVL(VPIntrin->getVectorLengthParam(), VectorNumElements)); \
  }
#include "llvm/IR/Instruction.def"
}
/// Check that creating vp.load/vp.store works.
TEST_F(VectorBuilderTest, TestCreateLoadStore) {
  Function *F;
  BasicBlock *BB;
  Value *Mask, *EVL;
  auto Mod = createBuilderModule(F, BB, Mask, EVL);

  IRBuilder<> Builder(BB);
  VectorBuilder VBuild(Builder);
  VBuild.setMask(Mask).setEVL(EVL);

  auto *FloatVecTy =
      FixedVectorType::get(Type::getFloatTy(Context), VectorNumElements);

  Value *FloatVecPtr = UndefValue::get(Builder.getPtrTy(0));
  Value *FloatVec = UndefValue::get(FloatVecTy);

  // vp.load
  auto LoadVPID = VPIntrinsic::getForOpcode(Instruction::Load);
  auto *LoadIntrin = VBuild.createVectorInstruction(Instruction::Load,
                                                    FloatVecTy, {FloatVecPtr});
  ASSERT_TRUE(isa<VPIntrinsic>(LoadIntrin));
  auto *VPLoad = cast<VPIntrinsic>(LoadIntrin);
  ASSERT_EQ(VPLoad->getIntrinsicID(), LoadVPID);
  ASSERT_EQ(VPLoad->getMemoryPointerParam(), FloatVecPtr);

  // vp.store
  auto *VoidTy = Builder.getVoidTy();
  auto StoreVPID = VPIntrinsic::getForOpcode(Instruction::Store);
  auto *StoreIntrin = VBuild.createVectorInstruction(Instruction::Store, VoidTy,
                                                     {FloatVec, FloatVecPtr});
  ASSERT_TRUE(isa<VPIntrinsic>(LoadIntrin));
  auto *VPStore = cast<VPIntrinsic>(StoreIntrin);
  ASSERT_EQ(VPStore->getIntrinsicID(), StoreVPID);
  ASSERT_EQ(VPStore->getMemoryPointerParam(), FloatVecPtr);
  ASSERT_EQ(VPStore->getMemoryDataParam(), FloatVec);
}

/// Check that the SilentlyReturnNone error handling mode works.
TEST_F(VectorBuilderTest, TestFail_SilentlyReturnNone) {
  Function *F;
  BasicBlock *BB;
  Value *Mask, *EVL;
  auto Mod = createBuilderModule(F, BB, Mask, EVL);

  IRBuilder<> Builder(BB);
  auto *VoidTy = Builder.getVoidTy();
  VectorBuilder VBuild(Builder, VectorBuilder::Behavior::SilentlyReturnNone);
  VBuild.setMask(Mask).setEVL(EVL);
  auto *Val = VBuild.createVectorInstruction(Instruction::Br, VoidTy, {});
  ASSERT_EQ(Val, nullptr);
}

/// Check that the ReportAndFail error handling mode aborts as advertised.
TEST_F(VectorBuilderTest, TestFail_ReportAndAbort) {
  Function *F;
  BasicBlock *BB;
  Value *Mask, *EVL;
  auto Mod = createBuilderModule(F, BB, Mask, EVL);

  IRBuilder<> Builder(BB);
  auto *VoidTy = Builder.getVoidTy();
  VectorBuilder VBuild(Builder, VectorBuilder::Behavior::ReportAndAbort);
  VBuild.setMask(Mask).setEVL(EVL);
  ASSERT_DEATH({ VBuild.createVectorInstruction(Instruction::Br, VoidTy, {}); },
               "No VPIntrinsic for this opcode");
}

} // end anonymous namespace
