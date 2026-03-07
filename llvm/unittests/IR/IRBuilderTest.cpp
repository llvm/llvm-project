//===- llvm/unittest/IR/IRBuilderTest.cpp - IRBuilder tests ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/IRBuilder.h"
#include "llvm/Analysis/InstSimplifyFolder.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <string>
#include <type_traits>

using namespace llvm;
using namespace PatternMatch;
using ::testing::UnorderedElementsAre;

namespace {

class IRBuilderTest : public testing::Test {
protected:
  void SetUp() override {
    M.reset(new Module("MyModule", Ctx));
    FunctionType *FTy = FunctionType::get(Type::getVoidTy(Ctx),
                                          /*isVarArg=*/false);
    F = Function::Create(FTy, Function::ExternalLinkage, "", M.get());
    BB = BasicBlock::Create(Ctx, "", F);
    GV = new GlobalVariable(*M, Type::getFloatTy(Ctx), true,
                            GlobalValue::ExternalLinkage, nullptr);
  }

  void TearDown() override {
    BB = nullptr;
    M.reset();
  }

  LLVMContext Ctx;
  std::unique_ptr<Module> M;
  Function *F;
  BasicBlock *BB;
  GlobalVariable *GV;
};

TEST_F(IRBuilderTest, Intrinsics) {
  IRBuilder<> Builder(BB);
  Value *V;
  Instruction *I;
  Value *Result;
  IntrinsicInst *II;

  V = Builder.CreateLoad(GV->getValueType(), GV);
  I = cast<Instruction>(Builder.CreateFAdd(V, V));
  I->setHasNoInfs(true);
  I->setHasNoNaNs(false);

  Result = Builder.CreateMinNum(V, V);
  II = cast<IntrinsicInst>(Result);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::minnum);

  Result = Builder.CreateMaxNum(V, V);
  II = cast<IntrinsicInst>(Result);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::maxnum);

  Result = Builder.CreateMinimum(V, V);
  II = cast<IntrinsicInst>(Result);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::minimum);

  Result = Builder.CreateMaximum(V, V);
  II = cast<IntrinsicInst>(Result);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::maximum);

  Result = Builder.CreateIntrinsic(Intrinsic::readcyclecounter,
                                   ArrayRef<Type *>{}, {});
  II = cast<IntrinsicInst>(Result);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::readcyclecounter);

  Result = Builder.CreateIntrinsic(Intrinsic::readcyclecounter, {});
  II = cast<IntrinsicInst>(Result);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::readcyclecounter);

  Result = Builder.CreateUnaryIntrinsic(Intrinsic::fabs, V);
  II = cast<IntrinsicInst>(Result);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::fabs);
  EXPECT_FALSE(II->hasNoInfs());
  EXPECT_FALSE(II->hasNoNaNs());

  Result = Builder.CreateUnaryIntrinsic(Intrinsic::fabs, V, I);
  II = cast<IntrinsicInst>(Result);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::fabs);
  EXPECT_TRUE(II->hasNoInfs());
  EXPECT_FALSE(II->hasNoNaNs());

  Result = Builder.CreateBinaryIntrinsic(Intrinsic::pow, V, V);
  II = cast<IntrinsicInst>(Result);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::pow);
  EXPECT_FALSE(II->hasNoInfs());
  EXPECT_FALSE(II->hasNoNaNs());

  Result = Builder.CreateBinaryIntrinsic(Intrinsic::pow, V, V, I);
  II = cast<IntrinsicInst>(Result);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::pow);
  EXPECT_TRUE(II->hasNoInfs());
  EXPECT_FALSE(II->hasNoNaNs());

  Result = Builder.CreateFMA(V, V, V);
  II = cast<IntrinsicInst>(Result);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::fma);
  EXPECT_FALSE(II->hasNoInfs());
  EXPECT_FALSE(II->hasNoNaNs());

  Result = Builder.CreateFMA(V, V, V, I);
  II = cast<IntrinsicInst>(Result);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::fma);
  EXPECT_TRUE(II->hasNoInfs());
  EXPECT_FALSE(II->hasNoNaNs());

  Result = Builder.CreateFMA(V, V, V, FastMathFlags::getFast());
  II = cast<IntrinsicInst>(Result);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::fma);
  EXPECT_TRUE(II->hasNoInfs());
  EXPECT_TRUE(II->hasNoNaNs());

  Result = Builder.CreateUnaryIntrinsic(Intrinsic::roundeven, V);
  II = cast<IntrinsicInst>(Result);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::roundeven);
  EXPECT_FALSE(II->hasNoInfs());
  EXPECT_FALSE(II->hasNoNaNs());

  Result = Builder.CreateIntrinsic(
      Intrinsic::set_rounding,
      {Builder.getInt32(static_cast<uint32_t>(RoundingMode::TowardZero))});
  II = cast<IntrinsicInst>(Result);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::set_rounding);
}

TEST_F(IRBuilderTest, IntrinsicMangling) {
  IRBuilder<> Builder(BB);
  Type *VoidTy = Builder.getVoidTy();
  Type *Int64Ty = Builder.getInt64Ty();
  Value *Int64Val = Builder.getInt64(0);
  Value *DoubleVal = PoisonValue::get(Builder.getDoubleTy());
  CallInst *Call;

  // Mangled return type, no arguments.
  Call = Builder.CreateIntrinsic(Int64Ty, Intrinsic::coro_size, {});
  EXPECT_EQ(Call->getCalledFunction()->getName(), "llvm.coro.size.i64");

  // Void return type, mangled argument type.
  Call =
      Builder.CreateIntrinsic(VoidTy, Intrinsic::set_loop_iterations, Int64Val);
  EXPECT_EQ(Call->getCalledFunction()->getName(),
            "llvm.set.loop.iterations.i64");

  // Mangled return type and argument type.
  Call = Builder.CreateIntrinsic(Int64Ty, Intrinsic::lround, DoubleVal);
  EXPECT_EQ(Call->getCalledFunction()->getName(), "llvm.lround.i64.f64");
}

TEST_F(IRBuilderTest, IntrinsicsWithScalableVectors) {
  IRBuilder<> Builder(BB);
  CallInst *Call;
  FunctionType *FTy;

  // Test scalable flag isn't dropped for intrinsic that is explicitly defined
  // with scalable vectors, e.g. LLVMType<nxv4i32>.
  Type *SrcVecTy = VectorType::get(Builder.getHalfTy(), 8, true);
  Type *DstVecTy = VectorType::get(Builder.getInt32Ty(), 4, true);
  Type *PredTy = VectorType::get(Builder.getInt1Ty(), 4, true);

  SmallVector<Value *, 3> Args;
  Args.push_back(UndefValue::get(DstVecTy));
  Args.push_back(UndefValue::get(PredTy));
  Args.push_back(UndefValue::get(SrcVecTy));

  Call = Builder.CreateIntrinsic(Intrinsic::aarch64_sve_fcvtzs_i32f16, Args,
                                 nullptr, "aarch64.sve.fcvtzs.i32f16");
  FTy = Call->getFunctionType();
  EXPECT_EQ(FTy->getReturnType(), DstVecTy);
  for (unsigned i = 0; i != Args.size(); ++i)
    EXPECT_EQ(FTy->getParamType(i), Args[i]->getType());

  // Test scalable flag isn't dropped for intrinsic defined with
  // LLVMScalarOrSameVectorWidth.

  Type *VecTy = VectorType::get(Builder.getInt32Ty(), 4, true);
  Type *PtrToVecTy = Builder.getPtrTy();
  PredTy = VectorType::get(Builder.getInt1Ty(), 4, true);

  Args.clear();
  Args.push_back(UndefValue::get(PtrToVecTy));
  Args.push_back(UndefValue::get(PredTy));
  Args.push_back(UndefValue::get(VecTy));

  Call = Builder.CreateIntrinsic(Intrinsic::masked_load, {VecTy, PtrToVecTy},
                                 Args, nullptr, "masked.load");
  FTy = Call->getFunctionType();
  EXPECT_EQ(FTy->getReturnType(), VecTy);
  for (unsigned i = 0; i != Args.size(); ++i)
    EXPECT_EQ(FTy->getParamType(i), Args[i]->getType());
}

TEST_F(IRBuilderTest, CreateStepVector) {
  IRBuilder<> Builder(BB);

  // Fixed width vectors
  Type *DstVecTy = VectorType::get(Builder.getInt32Ty(), 4, false);
  Value *StepVec = Builder.CreateStepVector(DstVecTy);
  EXPECT_TRUE(isa<Constant>(StepVec));
  EXPECT_EQ(StepVec->getType(), DstVecTy);

  const auto *VectorValue = cast<Constant>(StepVec);
  for (unsigned i = 0; i < 4; i++) {
    EXPECT_TRUE(isa<ConstantInt>(VectorValue->getAggregateElement(i)));
    ConstantInt *El = cast<ConstantInt>(VectorValue->getAggregateElement(i));
    EXPECT_EQ(El->getValue(), i);
  }

  // Scalable vectors
  DstVecTy = VectorType::get(Builder.getInt32Ty(), 4, true);
  StepVec = Builder.CreateStepVector(DstVecTy);
  EXPECT_TRUE(isa<CallInst>(StepVec));
  CallInst *Call = cast<CallInst>(StepVec);
  FunctionType *FTy = Call->getFunctionType();
  EXPECT_EQ(FTy->getReturnType(), DstVecTy);
  EXPECT_EQ(Call->getIntrinsicID(), Intrinsic::stepvector);
}

TEST_F(IRBuilderTest, CreateStepVectorI3) {
  IRBuilder<> Builder(BB);

  // Scalable vectors
  Type *DstVecTy = VectorType::get(IntegerType::get(Ctx, 3), 2, true);
  Type *VecI8Ty = VectorType::get(Builder.getInt8Ty(), 2, true);
  Value *StepVec = Builder.CreateStepVector(DstVecTy);
  EXPECT_TRUE(isa<TruncInst>(StepVec));
  TruncInst *Trunc = cast<TruncInst>(StepVec);
  EXPECT_EQ(Trunc->getDestTy(), DstVecTy);
  EXPECT_EQ(Trunc->getSrcTy(), VecI8Ty);
  EXPECT_TRUE(isa<CallInst>(Trunc->getOperand(0)));

  CallInst *Call = cast<CallInst>(Trunc->getOperand(0));
  FunctionType *FTy = Call->getFunctionType();
  EXPECT_EQ(FTy->getReturnType(), VecI8Ty);
  EXPECT_EQ(Call->getIntrinsicID(), Intrinsic::stepvector);
}

TEST_F(IRBuilderTest, CreateVectorSpliceLeft) {
  IRBuilder<> Builder(BB);

  // Fixed width vectors with constant offsets
  Type *FixedTy = VectorType::get(Builder.getInt32Ty(), 4, false);
  Value *FixedVec = Builder.CreateLoad(FixedTy, GV);
  Value *Shuffle = Builder.CreateVectorSpliceLeft(FixedVec, FixedVec, 1);
  EXPECT_TRUE(
      match(Shuffle, m_Shuffle(m_Specific(FixedVec), m_Specific(FixedVec),
                               m_SpecificMask({1, 2, 3, 4}))));

  Value *Offset = Builder.CreateLoad(Builder.getInt32Ty(), GV);
  Value *FixedSplice =
      Builder.CreateVectorSpliceLeft(FixedVec, FixedVec, Offset);
  EXPECT_TRUE(match(FixedSplice, m_Intrinsic<Intrinsic::vector_splice_left>(
                                     m_Specific(FixedVec), m_Specific(FixedVec),
                                     m_Specific(Offset))));

  Type *ScalableTy = VectorType::get(Builder.getInt32Ty(), 4, true);
  Value *ScalableVec = Builder.CreateLoad(ScalableTy, GV);
  Value *ScalableSplice =
      Builder.CreateVectorSpliceLeft(ScalableVec, ScalableVec, Offset);
  EXPECT_TRUE(
      match(ScalableSplice, m_Intrinsic<Intrinsic::vector_splice_left>(
                                m_Specific(ScalableVec),
                                m_Specific(ScalableVec), m_Specific(Offset))));
}

TEST_F(IRBuilderTest, CreateVectorSpliceRight) {
  IRBuilder<> Builder(BB);

  // Fixed width vectors with constant offsets
  Type *FixedTy = VectorType::get(Builder.getInt32Ty(), 4, false);
  Value *FixedVec = Builder.CreateLoad(FixedTy, GV);
  Value *Shuffle = Builder.CreateVectorSpliceRight(FixedVec, FixedVec, 1);
  EXPECT_TRUE(
      match(Shuffle, m_Shuffle(m_Specific(FixedVec), m_Specific(FixedVec),
                               m_SpecificMask({3, 4, 5, 6}))));

  Value *Offset = Builder.CreateLoad(Builder.getInt32Ty(), GV);
  Value *FixedSplice =
      Builder.CreateVectorSpliceRight(FixedVec, FixedVec, Offset);
  EXPECT_TRUE(match(FixedSplice, m_Intrinsic<Intrinsic::vector_splice_right>(
                                     m_Specific(FixedVec), m_Specific(FixedVec),
                                     m_Specific(Offset))));

  Type *ScalableTy = VectorType::get(Builder.getInt32Ty(), 4, true);
  Value *ScalableVec = Builder.CreateLoad(ScalableTy, GV);
  Value *ScalableSplice =
      Builder.CreateVectorSpliceRight(ScalableVec, ScalableVec, Offset);
  EXPECT_TRUE(
      match(ScalableSplice, m_Intrinsic<Intrinsic::vector_splice_right>(
                                m_Specific(ScalableVec),
                                m_Specific(ScalableVec), m_Specific(Offset))));
}

TEST_F(IRBuilderTest, ConstrainedFP) {
  IRBuilder<> Builder(BB);
  Value *V;
  Value *VDouble;
  Value *VInt;
  CallInst *Call;
  IntrinsicInst *II;
  GlobalVariable *GVDouble = new GlobalVariable(*M, Type::getDoubleTy(Ctx),
                            true, GlobalValue::ExternalLinkage, nullptr);

  V = Builder.CreateLoad(GV->getValueType(), GV);
  VDouble = Builder.CreateLoad(GVDouble->getValueType(), GVDouble);

  // See if we get constrained intrinsics instead of non-constrained
  // instructions.
  Builder.setIsFPConstrained(true);
  auto Parent = BB->getParent();
  Parent->addFnAttr(Attribute::StrictFP);

  V = Builder.CreateFAdd(V, V);
  ASSERT_TRUE(isa<IntrinsicInst>(V));
  II = cast<IntrinsicInst>(V);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::experimental_constrained_fadd);

  V = Builder.CreateFSub(V, V);
  ASSERT_TRUE(isa<IntrinsicInst>(V));
  II = cast<IntrinsicInst>(V);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::experimental_constrained_fsub);

  V = Builder.CreateFMul(V, V);
  ASSERT_TRUE(isa<IntrinsicInst>(V));
  II = cast<IntrinsicInst>(V);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::experimental_constrained_fmul);
  
  V = Builder.CreateFDiv(V, V);
  ASSERT_TRUE(isa<IntrinsicInst>(V));
  II = cast<IntrinsicInst>(V);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::experimental_constrained_fdiv);
  
  V = Builder.CreateFRem(V, V);
  ASSERT_TRUE(isa<IntrinsicInst>(V));
  II = cast<IntrinsicInst>(V);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::experimental_constrained_frem);

  V = Builder.CreateFMA(V, V, V);
  ASSERT_TRUE(isa<IntrinsicInst>(V));
  II = cast<IntrinsicInst>(V);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::experimental_constrained_fma);

  VInt = Builder.CreateFPToUI(VDouble, Builder.getInt32Ty());
  ASSERT_TRUE(isa<IntrinsicInst>(VInt));
  II = cast<IntrinsicInst>(VInt);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::experimental_constrained_fptoui);

  VInt = Builder.CreateFPToSI(VDouble, Builder.getInt32Ty());
  ASSERT_TRUE(isa<IntrinsicInst>(VInt));
  II = cast<IntrinsicInst>(VInt);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::experimental_constrained_fptosi);

  VDouble = Builder.CreateUIToFP(VInt, Builder.getDoubleTy());
  ASSERT_TRUE(isa<IntrinsicInst>(VDouble));
  II = cast<IntrinsicInst>(VDouble);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::experimental_constrained_uitofp);

  VDouble = Builder.CreateSIToFP(VInt, Builder.getDoubleTy());
  ASSERT_TRUE(isa<IntrinsicInst>(VDouble));
  II = cast<IntrinsicInst>(VDouble);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::experimental_constrained_sitofp);

  V = Builder.CreateFPTrunc(VDouble, Type::getFloatTy(Ctx));
  ASSERT_TRUE(isa<IntrinsicInst>(V));
  II = cast<IntrinsicInst>(V);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::experimental_constrained_fptrunc);

  VDouble = Builder.CreateFPExt(V, Type::getDoubleTy(Ctx));
  ASSERT_TRUE(isa<IntrinsicInst>(VDouble));
  II = cast<IntrinsicInst>(VDouble);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::experimental_constrained_fpext);

  // Verify attributes on the call are created automatically.
  AttributeSet CallAttrs = II->getAttributes().getFnAttrs();
  EXPECT_EQ(CallAttrs.hasAttribute(Attribute::StrictFP), true);

  // Verify attributes on the containing function are created when requested.
  Builder.setConstrainedFPFunctionAttr();
  AttributeList Attrs = BB->getParent()->getAttributes();
  AttributeSet FnAttrs = Attrs.getFnAttrs();
  EXPECT_EQ(FnAttrs.hasAttribute(Attribute::StrictFP), true);

  // Verify the codepaths for setting and overriding the default metadata.
  V = Builder.CreateFAdd(V, V);
  ASSERT_TRUE(isa<ConstrainedFPIntrinsic>(V));
  auto *CII = cast<ConstrainedFPIntrinsic>(V);
  EXPECT_EQ(fp::ebStrict, CII->getExceptionBehavior());
  EXPECT_EQ(RoundingMode::Dynamic, CII->getRoundingMode());

  Builder.setDefaultConstrainedExcept(fp::ebIgnore);
  Builder.setDefaultConstrainedRounding(RoundingMode::TowardPositive);
  V = Builder.CreateFAdd(V, V);
  CII = cast<ConstrainedFPIntrinsic>(V);
  EXPECT_EQ(fp::ebIgnore, CII->getExceptionBehavior());
  EXPECT_EQ(CII->getRoundingMode(), RoundingMode::TowardPositive);

  Builder.setDefaultConstrainedExcept(fp::ebIgnore);
  Builder.setDefaultConstrainedRounding(RoundingMode::NearestTiesToEven);
  V = Builder.CreateFAdd(V, V);
  CII = cast<ConstrainedFPIntrinsic>(V);
  EXPECT_EQ(fp::ebIgnore, CII->getExceptionBehavior());
  EXPECT_EQ(RoundingMode::NearestTiesToEven, CII->getRoundingMode());

  Builder.setDefaultConstrainedExcept(fp::ebMayTrap);
  Builder.setDefaultConstrainedRounding(RoundingMode::TowardNegative);
  V = Builder.CreateFAdd(V, V);
  CII = cast<ConstrainedFPIntrinsic>(V);
  EXPECT_EQ(fp::ebMayTrap, CII->getExceptionBehavior());
  EXPECT_EQ(RoundingMode::TowardNegative, CII->getRoundingMode());

  Builder.setDefaultConstrainedExcept(fp::ebStrict);
  Builder.setDefaultConstrainedRounding(RoundingMode::TowardZero);
  V = Builder.CreateFAdd(V, V);
  CII = cast<ConstrainedFPIntrinsic>(V);
  EXPECT_EQ(fp::ebStrict, CII->getExceptionBehavior());
  EXPECT_EQ(RoundingMode::TowardZero, CII->getRoundingMode());

  Builder.setDefaultConstrainedExcept(fp::ebIgnore);
  Builder.setDefaultConstrainedRounding(RoundingMode::Dynamic);
  V = Builder.CreateFAdd(V, V);
  CII = cast<ConstrainedFPIntrinsic>(V);
  EXPECT_EQ(fp::ebIgnore, CII->getExceptionBehavior());
  EXPECT_EQ(RoundingMode::Dynamic, CII->getRoundingMode());

  // Now override the defaults.
  Call = Builder.CreateConstrainedFPBinOp(
        Intrinsic::experimental_constrained_fadd, V, V, nullptr, "", nullptr,
        RoundingMode::TowardNegative, fp::ebMayTrap);
  CII = cast<ConstrainedFPIntrinsic>(Call);
  EXPECT_EQ(CII->getIntrinsicID(), Intrinsic::experimental_constrained_fadd);
  EXPECT_EQ(fp::ebMayTrap, CII->getExceptionBehavior());
  EXPECT_EQ(RoundingMode::TowardNegative, CII->getRoundingMode());

  // Same as previous test for CreateConstrainedFPIntrinsic
  Call = Builder.CreateConstrainedFPIntrinsic(
      Intrinsic::experimental_constrained_fadd, {V->getType()}, {V, V}, nullptr,
      "", nullptr, RoundingMode::TowardNegative, fp::ebMayTrap);
  CII = cast<ConstrainedFPIntrinsic>(Call);
  EXPECT_EQ(CII->getIntrinsicID(), Intrinsic::experimental_constrained_fadd);
  EXPECT_EQ(fp::ebMayTrap, CII->getExceptionBehavior());
  EXPECT_EQ(RoundingMode::TowardNegative, CII->getRoundingMode());

  Builder.CreateRetVoid();
  EXPECT_FALSE(verifyModule(*M));
}

TEST_F(IRBuilderTest, ConstrainedFPIntrinsics) {
  IRBuilder<> Builder(BB);
  Value *V;
  Value *VDouble;
  ConstrainedFPIntrinsic *CII;
  GlobalVariable *GVDouble = new GlobalVariable(
      *M, Type::getDoubleTy(Ctx), true, GlobalValue::ExternalLinkage, nullptr);
  VDouble = Builder.CreateLoad(GVDouble->getValueType(), GVDouble);

  Builder.setDefaultConstrainedExcept(fp::ebStrict);
  Builder.setDefaultConstrainedRounding(RoundingMode::TowardZero);
  Function *Fn = Intrinsic::getOrInsertDeclaration(
      M.get(), Intrinsic::experimental_constrained_roundeven,
      {Type::getDoubleTy(Ctx)});
  V = Builder.CreateConstrainedFPCall(Fn, { VDouble });
  CII = cast<ConstrainedFPIntrinsic>(V);
  EXPECT_EQ(Intrinsic::experimental_constrained_roundeven, CII->getIntrinsicID());
  EXPECT_EQ(fp::ebStrict, CII->getExceptionBehavior());
}

TEST_F(IRBuilderTest, ConstrainedFPFunctionCall) {
  IRBuilder<> Builder(BB);

  // Create an empty constrained FP function.
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(Ctx),
                                        /*isVarArg=*/false);
  Function *Callee =
      Function::Create(FTy, Function::ExternalLinkage, "", M.get());
  BasicBlock *CalleeBB = BasicBlock::Create(Ctx, "", Callee);
  IRBuilder<> CalleeBuilder(CalleeBB);
  CalleeBuilder.setIsFPConstrained(true);
  CalleeBuilder.setConstrainedFPFunctionAttr();
  CalleeBuilder.CreateRetVoid();

  // Now call the empty constrained FP function.
  Builder.setIsFPConstrained(true);
  Builder.setConstrainedFPFunctionAttr();
  CallInst *FCall = Builder.CreateCall(Callee, {});

  // Check the attributes to verify the strictfp attribute is on the call.
  EXPECT_TRUE(
      FCall->getAttributes().getFnAttrs().hasAttribute(Attribute::StrictFP));

  Builder.CreateRetVoid();
  EXPECT_FALSE(verifyModule(*M));
}

TEST_F(IRBuilderTest, Lifetime) {
  IRBuilder<> Builder(BB);
  AllocaInst *Var1 = Builder.CreateAlloca(Builder.getInt8Ty());
  AllocaInst *Var2 = Builder.CreateAlloca(Builder.getInt32Ty());
  AllocaInst *Var3 = Builder.CreateAlloca(Builder.getInt8Ty(),
                                          Builder.getInt32(123));

  CallInst *Start1 = Builder.CreateLifetimeStart(Var1);
  CallInst *Start2 = Builder.CreateLifetimeStart(Var2);
  CallInst *Start3 = Builder.CreateLifetimeStart(Var3);

  EXPECT_EQ(Start1->getArgOperand(0), Var1);
  EXPECT_EQ(Start2->getArgOperand(0), Var2);
  EXPECT_EQ(Start3->getArgOperand(0), Var3);

  Value *End1 = Builder.CreateLifetimeEnd(Var1);
  Builder.CreateLifetimeEnd(Var2);
  Builder.CreateLifetimeEnd(Var3);

  IntrinsicInst *II_Start1 = dyn_cast<IntrinsicInst>(Start1);
  IntrinsicInst *II_End1 = dyn_cast<IntrinsicInst>(End1);
  ASSERT_TRUE(II_Start1 != nullptr);
  EXPECT_EQ(II_Start1->getIntrinsicID(), Intrinsic::lifetime_start);
  ASSERT_TRUE(II_End1 != nullptr);
  EXPECT_EQ(II_End1->getIntrinsicID(), Intrinsic::lifetime_end);
}

TEST_F(IRBuilderTest, CreateCondBr) {
  IRBuilder<> Builder(BB);
  BasicBlock *TBB = BasicBlock::Create(Ctx, "", F);
  BasicBlock *FBB = BasicBlock::Create(Ctx, "", F);

  CondBrInst *BI = Builder.CreateCondBr(Builder.getTrue(), TBB, FBB);
  Instruction *TI = BB->getTerminator();
  EXPECT_EQ(BI, TI);
  EXPECT_EQ(2u, TI->getNumSuccessors());
  EXPECT_EQ(TBB, TI->getSuccessor(0));
  EXPECT_EQ(FBB, TI->getSuccessor(1));

  BI->eraseFromParent();
  MDNode *Weights = MDBuilder(Ctx).createBranchWeights(42, 13);
  BI = Builder.CreateCondBr(Builder.getTrue(), TBB, FBB, Weights);
  TI = BB->getTerminator();
  EXPECT_EQ(BI, TI);
  EXPECT_EQ(2u, TI->getNumSuccessors());
  EXPECT_EQ(TBB, TI->getSuccessor(0));
  EXPECT_EQ(FBB, TI->getSuccessor(1));
  EXPECT_EQ(Weights, TI->getMetadata(LLVMContext::MD_prof));
}

TEST_F(IRBuilderTest, LandingPadName) {
  IRBuilder<> Builder(BB);
  LandingPadInst *LP = Builder.CreateLandingPad(Builder.getInt32Ty(), 0, "LP");
  EXPECT_EQ(LP->getName(), "LP");
}

TEST_F(IRBuilderTest, DataLayout) {
  std::unique_ptr<Module> M(new Module("test", Ctx));
  M->setDataLayout("e-n32");
  EXPECT_TRUE(M->getDataLayout().isLegalInteger(32));
  M->setDataLayout("e");
  EXPECT_FALSE(M->getDataLayout().isLegalInteger(32));
}

TEST_F(IRBuilderTest, GetIntTy) {
  IRBuilder<> Builder(BB);
  IntegerType *Ty1 = Builder.getInt1Ty();
  EXPECT_EQ(Ty1, IntegerType::get(Ctx, 1));

  const DataLayout &DL = M->getDataLayout();
  IntegerType *IntPtrTy = Builder.getIntPtrTy(DL);
  unsigned IntPtrBitSize = DL.getPointerSizeInBits(0);
  EXPECT_EQ(IntPtrTy, IntegerType::get(Ctx, IntPtrBitSize));
}

TEST_F(IRBuilderTest, UnaryOperators) {
  IRBuilder<NoFolder> Builder(BB);
  Value *V = Builder.CreateLoad(GV->getValueType(), GV);

  // Test CreateUnOp(X)
  Value *U = Builder.CreateUnOp(Instruction::FNeg, V);
  ASSERT_TRUE(isa<Instruction>(U));
  ASSERT_TRUE(isa<FPMathOperator>(U));
  ASSERT_TRUE(isa<UnaryOperator>(U));
  ASSERT_FALSE(isa<BinaryOperator>(U));

  // Test CreateFNegFMF(X)
  Instruction *I = cast<Instruction>(U);
  I->setHasNoSignedZeros(true);
  I->setHasNoNaNs(true);
  Value *VFMF = Builder.CreateFNegFMF(V, I);
  Instruction *IFMF = cast<Instruction>(VFMF);
  EXPECT_TRUE(IFMF->hasNoSignedZeros());
  EXPECT_TRUE(IFMF->hasNoNaNs());
  EXPECT_FALSE(IFMF->hasAllowReassoc());
}

TEST_F(IRBuilderTest, FastMathFlags) {
  IRBuilder<> Builder(BB);
  Value *F, *FC;
  Instruction *FDiv, *FAdd, *FCmp, *FCall, *FNeg, *FSub, *FMul, *FRem;

  F = Builder.CreateLoad(GV->getValueType(), GV);
  F = Builder.CreateFAdd(F, F);

  EXPECT_FALSE(Builder.getFastMathFlags().any());
  ASSERT_TRUE(isa<Instruction>(F));
  FAdd = cast<Instruction>(F);
  EXPECT_FALSE(FAdd->hasNoNaNs());

  FastMathFlags FMF;
  Builder.setFastMathFlags(FMF);

  // By default, no flags are set.
  F = Builder.CreateFAdd(F, F);
  EXPECT_FALSE(Builder.getFastMathFlags().any());
  ASSERT_TRUE(isa<Instruction>(F));
  FAdd = cast<Instruction>(F);
  EXPECT_FALSE(FAdd->hasNoNaNs());
  EXPECT_FALSE(FAdd->hasNoInfs());
  EXPECT_FALSE(FAdd->hasNoSignedZeros());
  EXPECT_FALSE(FAdd->hasAllowReciprocal());
  EXPECT_FALSE(FAdd->hasAllowContract());
  EXPECT_FALSE(FAdd->hasAllowReassoc());
  EXPECT_FALSE(FAdd->hasApproxFunc());

  // Set all flags in the instruction.
  FAdd->setFast(true);
  EXPECT_TRUE(FAdd->hasNoNaNs());
  EXPECT_TRUE(FAdd->hasNoInfs());
  EXPECT_TRUE(FAdd->hasNoSignedZeros());
  EXPECT_TRUE(FAdd->hasAllowReciprocal());
  EXPECT_TRUE(FAdd->hasAllowContract());
  EXPECT_TRUE(FAdd->hasAllowReassoc());
  EXPECT_TRUE(FAdd->hasApproxFunc());

  // All flags are set in the builder.
  FMF.setFast();
  Builder.setFastMathFlags(FMF);

  F = Builder.CreateFAdd(F, F);
  EXPECT_TRUE(Builder.getFastMathFlags().any());
  EXPECT_TRUE(Builder.getFastMathFlags().all());
  ASSERT_TRUE(isa<Instruction>(F));
  FAdd = cast<Instruction>(F);
  EXPECT_TRUE(FAdd->hasNoNaNs());
  EXPECT_TRUE(FAdd->isFast());

  // Now, try it with CreateBinOp
  F = Builder.CreateBinOp(Instruction::FAdd, F, F);
  EXPECT_TRUE(Builder.getFastMathFlags().any());
  ASSERT_TRUE(isa<Instruction>(F));
  FAdd = cast<Instruction>(F);
  EXPECT_TRUE(FAdd->hasNoNaNs());
  EXPECT_TRUE(FAdd->isFast());

  F = Builder.CreateFDiv(F, F);
  EXPECT_TRUE(Builder.getFastMathFlags().all());
  ASSERT_TRUE(isa<Instruction>(F));
  FDiv = cast<Instruction>(F);
  EXPECT_TRUE(FDiv->hasAllowReciprocal());

  // Clear all FMF in the builder.
  Builder.clearFastMathFlags();

  F = Builder.CreateFDiv(F, F);
  ASSERT_TRUE(isa<Instruction>(F));
  FDiv = cast<Instruction>(F);
  EXPECT_FALSE(FDiv->hasAllowReciprocal());
 
  // Try individual flags.
  FMF.clear();
  FMF.setAllowReciprocal();
  Builder.setFastMathFlags(FMF);

  F = Builder.CreateFDiv(F, F);
  EXPECT_TRUE(Builder.getFastMathFlags().any());
  EXPECT_TRUE(Builder.getFastMathFlags().AllowReciprocal);
  ASSERT_TRUE(isa<Instruction>(F));
  FDiv = cast<Instruction>(F);
  EXPECT_TRUE(FDiv->hasAllowReciprocal());

  Builder.clearFastMathFlags();

  FC = Builder.CreateFCmpOEQ(F, F);
  ASSERT_TRUE(isa<Instruction>(FC));
  FCmp = cast<Instruction>(FC);
  EXPECT_FALSE(FCmp->hasAllowReciprocal());

  FMF.clear();
  FMF.setAllowReciprocal();
  Builder.setFastMathFlags(FMF);

  FC = Builder.CreateFCmpOEQ(F, F);
  EXPECT_TRUE(Builder.getFastMathFlags().any());
  EXPECT_TRUE(Builder.getFastMathFlags().AllowReciprocal);
  ASSERT_TRUE(isa<Instruction>(FC));
  FCmp = cast<Instruction>(FC);
  EXPECT_TRUE(FCmp->hasAllowReciprocal());

  Builder.clearFastMathFlags();

  // Test FP-contract
  FC = Builder.CreateFAdd(F, F);
  ASSERT_TRUE(isa<Instruction>(FC));
  FAdd = cast<Instruction>(FC);
  EXPECT_FALSE(FAdd->hasAllowContract());

  FMF.clear();
  FMF.setAllowContract(true);
  Builder.setFastMathFlags(FMF);

  FC = Builder.CreateFAdd(F, F);
  EXPECT_TRUE(Builder.getFastMathFlags().any());
  EXPECT_TRUE(Builder.getFastMathFlags().AllowContract);
  ASSERT_TRUE(isa<Instruction>(FC));
  FAdd = cast<Instruction>(FC);
  EXPECT_TRUE(FAdd->hasAllowContract());

  FMF.setApproxFunc();
  Builder.clearFastMathFlags();
  Builder.setFastMathFlags(FMF);
  // Now 'aml' and 'contract' are set.
  F = Builder.CreateFMul(F, F);
  FAdd = cast<Instruction>(F);
  EXPECT_TRUE(FAdd->hasApproxFunc());
  EXPECT_TRUE(FAdd->hasAllowContract());
  EXPECT_FALSE(FAdd->hasAllowReassoc());
  
  FMF.setAllowReassoc();
  Builder.clearFastMathFlags();
  Builder.setFastMathFlags(FMF);
  // Now 'aml' and 'contract' and 'reassoc' are set.
  F = Builder.CreateFMul(F, F);
  FAdd = cast<Instruction>(F);
  EXPECT_TRUE(FAdd->hasApproxFunc());
  EXPECT_TRUE(FAdd->hasAllowContract());
  EXPECT_TRUE(FAdd->hasAllowReassoc());

  // Test a call with FMF.
  auto CalleeTy = FunctionType::get(Type::getFloatTy(Ctx),
                                    /*isVarArg=*/false);
  auto Callee =
      Function::Create(CalleeTy, Function::ExternalLinkage, "", M.get());

  FCall = Builder.CreateCall(Callee, {});
  EXPECT_FALSE(FCall->hasNoNaNs());

  Function *V =
      Function::Create(CalleeTy, Function::ExternalLinkage, "", M.get());
  FCall = Builder.CreateCall(V, {});
  EXPECT_FALSE(FCall->hasNoNaNs());

  FMF.clear();
  FMF.setNoNaNs();
  Builder.setFastMathFlags(FMF);

  FCall = Builder.CreateCall(Callee, {});
  EXPECT_TRUE(Builder.getFastMathFlags().any());
  EXPECT_TRUE(Builder.getFastMathFlags().NoNaNs);
  EXPECT_TRUE(FCall->hasNoNaNs());

  FCall = Builder.CreateCall(V, {});
  EXPECT_TRUE(Builder.getFastMathFlags().any());
  EXPECT_TRUE(Builder.getFastMathFlags().NoNaNs);
  EXPECT_TRUE(FCall->hasNoNaNs());

  Builder.clearFastMathFlags();

  // To test a copy, make sure that a '0' and a '1' change state.
  F = Builder.CreateFDiv(F, F);
  ASSERT_TRUE(isa<Instruction>(F));
  FDiv = cast<Instruction>(F);
  EXPECT_FALSE(FDiv->getFastMathFlags().any());
  FDiv->setHasAllowReciprocal(true);
  FAdd->setHasAllowReciprocal(false);
  FAdd->setHasNoNaNs(true);
  FDiv->copyFastMathFlags(FAdd);
  EXPECT_TRUE(FDiv->hasNoNaNs());
  EXPECT_FALSE(FDiv->hasAllowReciprocal());

  // Test that CreateF*FMF functions copy flags from the source instruction
  // instead of using the builder default.
  Instruction *const FMFSource = FAdd;
  EXPECT_FALSE(Builder.getFastMathFlags().noNaNs());
  EXPECT_TRUE(FMFSource->hasNoNaNs());

  F = Builder.CreateFNegFMF(F, FMFSource);
  ASSERT_TRUE(isa<Instruction>(F));
  FNeg = cast<Instruction>(F);
  EXPECT_TRUE(FNeg->hasNoNaNs());
  F = Builder.CreateFAddFMF(F, F, FMFSource);
  ASSERT_TRUE(isa<Instruction>(F));
  FAdd = cast<Instruction>(F);
  EXPECT_TRUE(FAdd->hasNoNaNs());
  F = Builder.CreateFSubFMF(F, F, FMFSource);
  ASSERT_TRUE(isa<Instruction>(F));
  FSub = cast<Instruction>(F);
  EXPECT_TRUE(FSub->hasNoNaNs());
  F = Builder.CreateFMulFMF(F, F, FMFSource);
  ASSERT_TRUE(isa<Instruction>(F));
  FMul = cast<Instruction>(F);
  EXPECT_TRUE(FMul->hasNoNaNs());
  F = Builder.CreateFDivFMF(F, F, FMFSource);
  ASSERT_TRUE(isa<Instruction>(F));
  FDiv = cast<Instruction>(F);
  EXPECT_TRUE(FDiv->hasNoNaNs());
  F = Builder.CreateFRemFMF(F, F, FMFSource);
  ASSERT_TRUE(isa<Instruction>(F));
  FRem = cast<Instruction>(F);
  EXPECT_TRUE(FRem->hasNoNaNs());
}

TEST_F(IRBuilderTest, WrapFlags) {
  IRBuilder<NoFolder> Builder(BB);

  // Test instructions.
  GlobalVariable *G = new GlobalVariable(*M, Builder.getInt32Ty(), true,
                                         GlobalValue::ExternalLinkage, nullptr);
  Value *V = Builder.CreateLoad(G->getValueType(), G);
  EXPECT_TRUE(
      cast<BinaryOperator>(Builder.CreateNSWAdd(V, V))->hasNoSignedWrap());
  EXPECT_TRUE(
      cast<BinaryOperator>(Builder.CreateNSWMul(V, V))->hasNoSignedWrap());
  EXPECT_TRUE(
      cast<BinaryOperator>(Builder.CreateNSWSub(V, V))->hasNoSignedWrap());
  EXPECT_TRUE(cast<BinaryOperator>(
                  Builder.CreateShl(V, V, "", /* NUW */ false, /* NSW */ true))
                  ->hasNoSignedWrap());

  EXPECT_TRUE(
      cast<BinaryOperator>(Builder.CreateNUWAdd(V, V))->hasNoUnsignedWrap());
  EXPECT_TRUE(
      cast<BinaryOperator>(Builder.CreateNUWMul(V, V))->hasNoUnsignedWrap());
  EXPECT_TRUE(
      cast<BinaryOperator>(Builder.CreateNUWSub(V, V))->hasNoUnsignedWrap());
  EXPECT_TRUE(cast<BinaryOperator>(
                  Builder.CreateShl(V, V, "", /* NUW */ true, /* NSW */ false))
                  ->hasNoUnsignedWrap());

  // Test operators created with constants.
  Constant *C = Builder.getInt32(42);
  EXPECT_TRUE(cast<OverflowingBinaryOperator>(Builder.CreateNSWAdd(C, C))
                  ->hasNoSignedWrap());
  EXPECT_TRUE(cast<OverflowingBinaryOperator>(Builder.CreateNSWSub(C, C))
                  ->hasNoSignedWrap());
  EXPECT_TRUE(cast<OverflowingBinaryOperator>(Builder.CreateNSWMul(C, C))
                  ->hasNoSignedWrap());
  EXPECT_TRUE(cast<OverflowingBinaryOperator>(
                  Builder.CreateShl(C, C, "", /* NUW */ false, /* NSW */ true))
                  ->hasNoSignedWrap());

  EXPECT_TRUE(cast<OverflowingBinaryOperator>(Builder.CreateNUWAdd(C, C))
                  ->hasNoUnsignedWrap());
  EXPECT_TRUE(cast<OverflowingBinaryOperator>(Builder.CreateNUWSub(C, C))
                  ->hasNoUnsignedWrap());
  EXPECT_TRUE(cast<OverflowingBinaryOperator>(Builder.CreateNUWMul(C, C))
                  ->hasNoUnsignedWrap());
  EXPECT_TRUE(cast<OverflowingBinaryOperator>(
                  Builder.CreateShl(C, C, "", /* NUW */ true, /* NSW */ false))
                  ->hasNoUnsignedWrap());
}

TEST_F(IRBuilderTest, RAIIHelpersTest) {
  IRBuilder<> Builder(BB);
  EXPECT_FALSE(Builder.getFastMathFlags().allowReciprocal());
  MDBuilder MDB(M->getContext());

  MDNode *FPMathA = MDB.createFPMath(0.01f);
  MDNode *FPMathB = MDB.createFPMath(0.1f);

  Builder.setDefaultFPMathTag(FPMathA);

  {
    IRBuilder<>::FastMathFlagGuard Guard(Builder);
    FastMathFlags FMF;
    FMF.setAllowReciprocal();
    Builder.setFastMathFlags(FMF);
    Builder.setDefaultFPMathTag(FPMathB);
    EXPECT_TRUE(Builder.getFastMathFlags().allowReciprocal());
    EXPECT_EQ(FPMathB, Builder.getDefaultFPMathTag());
  }

  EXPECT_FALSE(Builder.getFastMathFlags().allowReciprocal());
  EXPECT_EQ(FPMathA, Builder.getDefaultFPMathTag());

  Value *F = Builder.CreateLoad(GV->getValueType(), GV);

  {
    IRBuilder<>::InsertPointGuard Guard(Builder);
    Builder.SetInsertPoint(cast<Instruction>(F));
    EXPECT_EQ(F, &*Builder.GetInsertPoint());
  }

  EXPECT_EQ(BB->end(), Builder.GetInsertPoint());
  EXPECT_EQ(BB, Builder.GetInsertBlock());
}

TEST_F(IRBuilderTest, createFunction) {
  IRBuilder<> Builder(BB);
  DIBuilder DIB(*M);
  auto File = DIB.createFile("error.swift", "/");
  auto CU = DIB.createCompileUnit(DISourceLanguageName(dwarf::DW_LANG_Swift),
                                  File, "swiftc", true, "", 0);
  auto Type = DIB.createSubroutineType(DIB.getOrCreateTypeArray({}));
  auto NoErr = DIB.createFunction(
      CU, "noerr", "", File, 1, Type, 1, DINode::FlagZero,
      DISubprogram::SPFlagDefinition | DISubprogram::SPFlagOptimized);
  EXPECT_TRUE(!NoErr->getThrownTypes());
  auto Int = DIB.createBasicType("Int", 64, dwarf::DW_ATE_signed);
  auto Error = DIB.getOrCreateArray({Int});
  auto Err = DIB.createFunction(
      CU, "err", "", File, 1, Type, 1, DINode::FlagZero,
      DISubprogram::SPFlagDefinition | DISubprogram::SPFlagOptimized, nullptr,
      nullptr, Error.get());
  EXPECT_TRUE(Err->getThrownTypes().get() == Error.get());
  DIB.finalize();
}

TEST_F(IRBuilderTest, DIBuilder) {
  auto GetLastDbgRecord = [](const Instruction *I) -> DbgRecord * {
    if (I->getDbgRecordRange().empty())
      return nullptr;
    return &*std::prev(I->getDbgRecordRange().end());
  };

  auto ExpectOrder = [&](DbgInstPtr First, BasicBlock::iterator Second) {
    EXPECT_TRUE(isa<DbgRecord *>(First));
    EXPECT_FALSE(Second->getDbgRecordRange().empty());
    EXPECT_EQ(GetLastDbgRecord(&*Second), cast<DbgRecord *>(First));
  };

  auto RunTest = [&]() {
    IRBuilder<> Builder(BB);
    DIBuilder DIB(*M);
    auto File = DIB.createFile("F.CBL", "/");
    auto CU = DIB.createCompileUnit(
        DISourceLanguageName(dwarf::DW_LANG_Cobol74),
        DIB.createFile("F.CBL", "/"), "llvm-cobol74", true, "", 0);
    auto Type = DIB.createSubroutineType(DIB.getOrCreateTypeArray({}));
    auto SP = DIB.createFunction(
        CU, "foo", "", File, 1, Type, 1, DINode::FlagZero,
        DISubprogram::SPFlagDefinition | DISubprogram::SPFlagOptimized);
    F->setSubprogram(SP);
    AllocaInst *I = Builder.CreateAlloca(Builder.getInt8Ty());
    auto BarSP = DIB.createFunction(
        CU, "bar", "", File, 1, Type, 1, DINode::FlagZero,
        DISubprogram::SPFlagDefinition | DISubprogram::SPFlagOptimized);
    auto BarScope = DIB.createLexicalBlockFile(BarSP, File, 0);
    I->setDebugLoc(DILocation::get(Ctx, 2, 0, BarScope));

    // Create another instruction so that there's one before the alloca we're
    // inserting debug intrinsics before, to make end-checking easier.
    I = Builder.CreateAlloca(Builder.getInt1Ty());

    // Label metadata and records
    // --------------------------
    DILocation *LabelLoc = DILocation::get(Ctx, 1, 0, BarScope);
    DILabel *AlwaysPreserveLabel = DIB.createLabel(
        BarScope, "meles_meles", File, 1, /*Column*/ 0, /*IsArtificial*/ false,
        /*CoroSuspendIdx*/ std::nullopt, /*AlwaysPreserve*/ true);
    DILabel *Label = DIB.createLabel(
        BarScope, "badger", File, 1, /*Column*/ 0, /*IsArtificial*/ false,
        /*CoroSuspendIdx*/ std::nullopt, /*AlwaysPreserve*/ false);

    { /* dbg.label | DbgLabelRecord */
      // Insert before I and check order.
      ExpectOrder(DIB.insertLabel(Label, LabelLoc, I->getIterator()),
                  I->getIterator());

      // We should be able to insert at the end of the block, even if there's
      // no terminator yet. Note that in RemoveDIs mode this record won't get
      // inserted into the block untill another instruction is added.
      DbgInstPtr LabelRecord = DIB.insertLabel(Label, LabelLoc, BB->end());
      // Specifically do not insert a terminator, to check this works. `I`
      // should have absorbed the DbgLabelRecord in the new debug info mode.
      I = Builder.CreateAlloca(Builder.getInt32Ty());
      ExpectOrder(LabelRecord, I->getIterator());
    }

    // Variable metadata and records
    // -----------------------------
    DILocation *VarLoc = DILocation::get(Ctx, 2, 0, BarScope);
    auto *IntType = DIB.createBasicType("int", 32, dwarf::DW_ATE_signed);
    DILocalVariable *VarX =
        DIB.createAutoVariable(BarSP, "X", File, 2, IntType, true);
    DILocalVariable *VarY =
        DIB.createAutoVariable(BarSP, "Y", File, 2, IntType, true);
    { /* dbg.value | DbgVariableRecord::Value */
      ExpectOrder(DIB.insertDbgValueIntrinsic(I, VarX, DIB.createExpression(),
                                              VarLoc, I->getIterator()),
                  I->getIterator());
      // Check inserting at end of the block works as with labels.
      DbgInstPtr VarXValue = DIB.insertDbgValueIntrinsic(
          I, VarX, DIB.createExpression(), VarLoc, BB);
      I = Builder.CreateAlloca(Builder.getInt32Ty());
      ExpectOrder(VarXValue, I->getIterator());
      EXPECT_EQ(BB->getTrailingDbgRecords(), nullptr);
    }
    { /* dbg.declare | DbgVariableRecord::Declare */
      ExpectOrder(DIB.insertDeclare(I, VarY, DIB.createExpression(), VarLoc,
                                    I->getIterator()),
                  I->getIterator());
      // Check inserting at end of the block works as with labels.
      DbgInstPtr VarYDeclare =
          DIB.insertDeclare(I, VarY, DIB.createExpression(), VarLoc, BB);
      I = Builder.CreateAlloca(Builder.getInt32Ty());
      ExpectOrder(VarYDeclare, I->getIterator());
      EXPECT_EQ(BB->getTrailingDbgRecords(), nullptr);
    }
    { /* dbg.assign | DbgVariableRecord::Assign */
      I = Builder.CreateAlloca(Builder.getInt32Ty());
      I->setMetadata(LLVMContext::MD_DIAssignID, DIAssignID::getDistinct(Ctx));
      // DbgAssign interface is slightly different - it always inserts after the
      // linked instr. Check we can do this with no instruction to insert
      // before.
      DbgInstPtr VarXAssign =
          DIB.insertDbgAssign(I, I, VarX, DIB.createExpression(), I,
                              DIB.createExpression(), VarLoc);
      I = Builder.CreateAlloca(Builder.getInt32Ty());
      ExpectOrder(VarXAssign, I->getIterator());
      EXPECT_EQ(BB->getTrailingDbgRecords(), nullptr);
    }

    Builder.CreateRet(nullptr);
    DIB.finalize();
    // Check the labels are not/are added to Bar's retainedNodes array
    // (AlwaysPreserve).
    EXPECT_EQ(find(BarSP->getRetainedNodes(), Label),
              BarSP->getRetainedNodes().end());
    EXPECT_NE(find(BarSP->getRetainedNodes(), AlwaysPreserveLabel),
              BarSP->getRetainedNodes().end());
    EXPECT_NE(find(BarSP->getRetainedNodes(), VarX),
              BarSP->getRetainedNodes().end());
    EXPECT_NE(find(BarSP->getRetainedNodes(), VarY),
              BarSP->getRetainedNodes().end());
    EXPECT_TRUE(verifyModule(*M));
  };

  RunTest();
  TearDown();
}

TEST_F(IRBuilderTest, createArtificialSubprogram) {
  IRBuilder<> Builder(BB);
  DIBuilder DIB(*M);
  auto File = DIB.createFile("main.c", "/");
  auto CU = DIB.createCompileUnit(DISourceLanguageName(dwarf::DW_LANG_C), File,
                                  "clang",
                                  /*isOptimized=*/true, /*Flags=*/"",
                                  /*Runtime Version=*/0);
  auto Type = DIB.createSubroutineType(DIB.getOrCreateTypeArray({}));
  auto SP = DIB.createFunction(
      CU, "foo", /*LinkageName=*/"", File,
      /*LineNo=*/1, Type, /*ScopeLine=*/2, DINode::FlagZero,
      DISubprogram::SPFlagDefinition | DISubprogram::SPFlagOptimized);
  EXPECT_TRUE(SP->isDistinct());

  F->setSubprogram(SP);
  AllocaInst *I = Builder.CreateAlloca(Builder.getInt8Ty());
  ReturnInst *R = Builder.CreateRetVoid();
  I->setDebugLoc(DILocation::get(Ctx, 3, 2, SP));
  R->setDebugLoc(DILocation::get(Ctx, 4, 2, SP));
  DIB.finalize();
  EXPECT_FALSE(verifyModule(*M));

  Function *G = Function::Create(F->getFunctionType(),
                                 Function::ExternalLinkage, "", M.get());
  BasicBlock *GBB = BasicBlock::Create(Ctx, "", G);
  Builder.SetInsertPoint(GBB);
  I->removeFromParent();
  Builder.Insert(I);
  Builder.CreateRetVoid();
  EXPECT_FALSE(verifyModule(*M));

  DISubprogram *GSP = DIBuilder::createArtificialSubprogram(F->getSubprogram());
  EXPECT_EQ(SP->getFile(), GSP->getFile());
  EXPECT_EQ(SP->getType(), GSP->getType());
  EXPECT_EQ(SP->getLine(), GSP->getLine());
  EXPECT_EQ(SP->getScopeLine(), GSP->getScopeLine());
  EXPECT_TRUE(GSP->isDistinct());

  G->setSubprogram(GSP);
  EXPECT_TRUE(verifyModule(*M));

  auto *InlinedAtNode =
      DILocation::getDistinct(Ctx, GSP->getScopeLine(), 0, GSP);
  DebugLoc DL = I->getDebugLoc();
  DenseMap<const MDNode *, MDNode *> IANodes;
  auto IA = DebugLoc::appendInlinedAt(DL, InlinedAtNode, Ctx, IANodes);
  auto NewDL =
      DILocation::get(Ctx, DL.getLine(), DL.getCol(), DL.getScope(), IA);
  I->setDebugLoc(NewDL);
  EXPECT_FALSE(verifyModule(*M));

  EXPECT_EQ("foo", SP->getName());
  EXPECT_EQ("foo", GSP->getName());
  EXPECT_FALSE(SP->isArtificial());
  EXPECT_TRUE(GSP->isArtificial());
}

// Check that we can add debug info to an existing DICompileUnit.
TEST_F(IRBuilderTest, appendDebugInfo) {
  IRBuilder<> Builder(BB);
  Builder.CreateRetVoid();
  EXPECT_FALSE(verifyModule(*M));

  auto GetNames = [](DICompileUnit *CU) {
    SmallVector<StringRef> Names;
    for (auto *ET : CU->getEnumTypes())
      Names.push_back(ET->getName());
    for (auto *RT : CU->getRetainedTypes())
      Names.push_back(RT->getName());
    for (auto *GV : CU->getGlobalVariables())
      Names.push_back(GV->getVariable()->getName());
    for (auto *IE : CU->getImportedEntities())
      Names.push_back(IE->getName());
    for (auto *Node : CU->getMacros())
      if (auto *MN = dyn_cast_or_null<DIMacro>(Node))
        Names.push_back(MN->getName());
    return Names;
  };

  DICompileUnit *CU;
  {
    DIBuilder DIB(*M);
    auto *File = DIB.createFile("main.c", "/");
    CU = DIB.createCompileUnit(DISourceLanguageName(dwarf::DW_LANG_C), File,
                               "clang",
                               /*isOptimized=*/true, /*Flags=*/"",
                               /*Runtime Version=*/0);
    auto *ByteTy = DIB.createBasicType("byte0", 8, dwarf::DW_ATE_signed);
    DIB.createEnumerationType(CU, "ET0", File, /*LineNo=*/0, /*SizeInBits=*/8,
                              /*AlignInBits=*/8, /*Elements=*/{}, ByteTy);
    DIB.retainType(ByteTy);
    DIB.createGlobalVariableExpression(CU, "GV0", /*LinkageName=*/"", File,
                                       /*LineNo=*/1, ByteTy,
                                       /*IsLocalToUnit=*/true);
    DIB.createImportedDeclaration(CU, nullptr, File, /*LineNo=*/2, "IM0");
    DIB.createMacro(nullptr, /*LineNo=*/0, dwarf::DW_MACINFO_define, "M0");
    DIB.finalize();
  }
  EXPECT_FALSE(verifyModule(*M));
  EXPECT_THAT(GetNames(CU),
              UnorderedElementsAre("ET0", "byte0", "GV0", "IM0", "M0"));

  {
    DIBuilder DIB(*M, true, CU);
    auto *File = CU->getFile();
    auto *ByteTy = DIB.createBasicType("byte1", 8, dwarf::DW_ATE_signed);
    DIB.createEnumerationType(CU, "ET1", File, /*LineNo=*/0,
                              /*SizeInBits=*/8, /*AlignInBits=*/8,
                              /*Elements=*/{}, ByteTy);
    DIB.retainType(ByteTy);
    DIB.createGlobalVariableExpression(CU, "GV1", /*LinkageName=*/"", File,
                                       /*LineNo=*/1, ByteTy,
                                       /*IsLocalToUnit=*/true);
    DIB.createImportedDeclaration(CU, nullptr, File, /*LineNo=*/2, "IM1");
    DIB.createMacro(nullptr, /*LineNo=*/0, dwarf::DW_MACINFO_define, "M1");
    DIB.finalize();
  }
  EXPECT_FALSE(verifyModule(*M));
  EXPECT_THAT(GetNames(CU),
              UnorderedElementsAre("ET0", "byte0", "GV0", "IM0", "M0", "ET1",
                                   "byte1", "GV1", "IM1", "M1"));
}

TEST_F(IRBuilderTest, InsertExtractElement) {
  IRBuilder<> Builder(BB);

  auto VecTy = FixedVectorType::get(Builder.getInt64Ty(), 4);
  auto Elt1 = Builder.getInt64(-1);
  auto Elt2 = Builder.getInt64(-2);
  Value *Vec = Builder.CreateInsertElement(VecTy, Elt1, Builder.getInt8(1));
  Vec = Builder.CreateInsertElement(Vec, Elt2, 2);
  auto X1 = Builder.CreateExtractElement(Vec, 1);
  auto X2 = Builder.CreateExtractElement(Vec, Builder.getInt32(2));
  EXPECT_EQ(Elt1, X1);
  EXPECT_EQ(Elt2, X2);
}

TEST_F(IRBuilderTest, CreateGlobalString) {
  IRBuilder<> Builder(BB);

  auto String1a = Builder.CreateGlobalString("TestString", "String1a");
  auto String1b = Builder.CreateGlobalString("TestString", "String1b", 0);
  auto String2 = Builder.CreateGlobalString("TestString", "String2", 1);
  auto String3 = Builder.CreateGlobalString("TestString", "String3", 2);

  EXPECT_TRUE(String1a->getType()->getPointerAddressSpace() == 0);
  EXPECT_TRUE(String1b->getType()->getPointerAddressSpace() == 0);
  EXPECT_TRUE(String2->getType()->getPointerAddressSpace() == 1);
  EXPECT_TRUE(String3->getType()->getPointerAddressSpace() == 2);
}

TEST_F(IRBuilderTest, DebugLoc) {
  auto CalleeTy = FunctionType::get(Type::getVoidTy(Ctx),
                                    /*isVarArg=*/false);
  auto Callee =
      Function::Create(CalleeTy, Function::ExternalLinkage, "", M.get());

  DIBuilder DIB(*M);
  auto File = DIB.createFile("tmp.cpp", "/");
  auto CU =
      DIB.createCompileUnit(DISourceLanguageName(dwarf::DW_LANG_C_plus_plus_11),
                            DIB.createFile("tmp.cpp", "/"), "", true, "", 0);
  auto SPType = DIB.createSubroutineType(DIB.getOrCreateTypeArray({}));
  auto SP =
      DIB.createFunction(CU, "foo", "foo", File, 1, SPType, 1, DINode::FlagZero,
                         DISubprogram::SPFlagDefinition);
  DebugLoc DL1 = DILocation::get(Ctx, 2, 0, SP);
  DebugLoc DL2 = DILocation::get(Ctx, 3, 0, SP);

  auto BB2 = BasicBlock::Create(Ctx, "bb2", F);
  auto Br = UncondBrInst::Create(BB2, BB);
  Br->setDebugLoc(DL1);

  IRBuilder<> Builder(Ctx);
  Builder.SetInsertPoint(Br);
  EXPECT_EQ(DL1, Builder.getCurrentDebugLocation());
  auto Call1 = Builder.CreateCall(Callee, {});
  EXPECT_EQ(DL1, Call1->getDebugLoc());

  Call1->setDebugLoc(DL2);
  Builder.SetInsertPoint(Call1->getParent(), Call1->getIterator());
  EXPECT_EQ(DL2, Builder.getCurrentDebugLocation());
  auto Call2 = Builder.CreateCall(Callee, {});
  EXPECT_EQ(DL2, Call2->getDebugLoc());

  DIB.finalize();
}

TEST_F(IRBuilderTest, DIImportedEntity) {
  IRBuilder<> Builder(BB);
  DIBuilder DIB(*M);
  auto F = DIB.createFile("F.CBL", "/");
  auto CU = DIB.createCompileUnit(DISourceLanguageName(dwarf::DW_LANG_Cobol74),
                                  F, "llvm-cobol74", true, "", 0);
  MDTuple *Elements = MDTuple::getDistinct(Ctx, {});

  DIB.createImportedDeclaration(CU, nullptr, F, 1);
  DIB.createImportedDeclaration(CU, nullptr, F, 1);
  DIB.createImportedModule(CU, (DIImportedEntity *)nullptr, F, 2);
  DIB.createImportedModule(CU, (DIImportedEntity *)nullptr, F, 2);
  DIB.createImportedModule(CU, (DIImportedEntity *)nullptr, F, 2, Elements);
  DIB.createImportedModule(CU, (DIImportedEntity *)nullptr, F, 2, Elements);
  DIB.finalize();
  EXPECT_TRUE(verifyModule(*M));
  EXPECT_TRUE(CU->getImportedEntities().size() == 3);
}

//  0: #define M0 V0          <-- command line definition
//  0: main.c                 <-- main file
//     3:   #define M1 V1     <-- M1 definition in main.c
//     5:   #include "file.h" <-- inclusion of file.h from main.c
//          1: #define M2     <-- M2 definition in file.h with no value
//     7:   #undef M1 V1      <-- M1 un-definition in main.c
TEST_F(IRBuilderTest, DIBuilderMacro) {
  IRBuilder<> Builder(BB);
  DIBuilder DIB(*M);
  auto File1 = DIB.createFile("main.c", "/");
  auto File2 = DIB.createFile("file.h", "/");
  auto CU = DIB.createCompileUnit(DISourceLanguageName(dwarf::DW_LANG_C),
                                  DIB.createFile("main.c", "/"), "llvm-c", true,
                                  "", 0);
  auto MDef0 =
      DIB.createMacro(nullptr, 0, dwarf::DW_MACINFO_define, "M0", "V0");
  auto TMF1 = DIB.createTempMacroFile(nullptr, 0, File1);
  auto MDef1 = DIB.createMacro(TMF1, 3, dwarf::DW_MACINFO_define, "M1", "V1");
  auto TMF2 = DIB.createTempMacroFile(TMF1, 5, File2);
  auto MDef2 = DIB.createMacro(TMF2, 1, dwarf::DW_MACINFO_define, "M2");
  auto MUndef1 = DIB.createMacro(TMF1, 7, dwarf::DW_MACINFO_undef, "M1");

  EXPECT_EQ(dwarf::DW_MACINFO_define, MDef1->getMacinfoType());
  EXPECT_EQ(3u, MDef1->getLine());
  EXPECT_EQ("M1", MDef1->getName());
  EXPECT_EQ("V1", MDef1->getValue());

  EXPECT_EQ(dwarf::DW_MACINFO_undef, MUndef1->getMacinfoType());
  EXPECT_EQ(7u, MUndef1->getLine());
  EXPECT_EQ("M1", MUndef1->getName());
  EXPECT_EQ("", MUndef1->getValue());

  EXPECT_EQ(dwarf::DW_MACINFO_start_file, TMF2->getMacinfoType());
  EXPECT_EQ(5u, TMF2->getLine());
  EXPECT_EQ(File2, TMF2->getFile());

  DIB.finalize();

  SmallVector<Metadata *, 4> Elements;
  Elements.push_back(MDef2);
  auto MF2 = DIMacroFile::get(Ctx, dwarf::DW_MACINFO_start_file, 5, File2,
                              DIB.getOrCreateMacroArray(Elements));

  Elements.clear();
  Elements.push_back(MDef1);
  Elements.push_back(MF2);
  Elements.push_back(MUndef1);
  auto MF1 = DIMacroFile::get(Ctx, dwarf::DW_MACINFO_start_file, 0, File1,
                              DIB.getOrCreateMacroArray(Elements));

  Elements.clear();
  Elements.push_back(MDef0);
  Elements.push_back(MF1);
  auto MN0 = MDTuple::get(Ctx, Elements);
  EXPECT_EQ(MN0, CU->getRawMacros());

  Elements.clear();
  Elements.push_back(MDef1);
  Elements.push_back(MF2);
  Elements.push_back(MUndef1);
  auto MN1 = MDTuple::get(Ctx, Elements);
  EXPECT_EQ(MN1, MF1->getRawElements());

  Elements.clear();
  Elements.push_back(MDef2);
  auto MN2 = MDTuple::get(Ctx, Elements);
  EXPECT_EQ(MN2, MF2->getRawElements());
  EXPECT_TRUE(verifyModule(*M));
}

TEST_F(IRBuilderTest, NoFolderNames) {
  IRBuilder<NoFolder> Builder(BB);
  auto *Add =
      Builder.CreateAdd(Builder.getInt32(1), Builder.getInt32(2), "add");
  EXPECT_EQ(Add->getName(), "add");
}

TEST_F(IRBuilderTest, CTAD) {
  struct TestInserter : public IRBuilderDefaultInserter {
    TestInserter() = default;
  };
  InstSimplifyFolder Folder(M->getDataLayout());

  IRBuilder Builder1(Ctx, Folder, TestInserter());
  static_assert(std::is_same_v<decltype(Builder1),
                               IRBuilder<InstSimplifyFolder, TestInserter>>);
  IRBuilder Builder2(Ctx);
  static_assert(std::is_same_v<decltype(Builder2), IRBuilder<>>);
  IRBuilder Builder3(BB, Folder);
  static_assert(
      std::is_same_v<decltype(Builder3), IRBuilder<InstSimplifyFolder>>);
  IRBuilder Builder4(BB);
  static_assert(std::is_same_v<decltype(Builder4), IRBuilder<>>);
  // The block BB is empty, so don't test this one.
  // IRBuilder Builder5(BB->getTerminator());
  // static_assert(std::is_same_v<decltype(Builder5), IRBuilder<>>);
  IRBuilder Builder6(BB, BB->end(), Folder);
  static_assert(
      std::is_same_v<decltype(Builder6), IRBuilder<InstSimplifyFolder>>);
  IRBuilder Builder7(BB, BB->end());
  static_assert(std::is_same_v<decltype(Builder7), IRBuilder<>>);
}

TEST_F(IRBuilderTest, finalizeSubprogram) {
  IRBuilder<> Builder(BB);
  DIBuilder DIB(*M);
  auto File = DIB.createFile("main.c", "/");
  auto CU = DIB.createCompileUnit(
      DISourceLanguageName(dwarf::DW_LANG_C_plus_plus), File, "clang",
      /*isOptimized=*/true, /*Flags=*/"",
      /*Runtime Version=*/0);
  auto FuncType = DIB.createSubroutineType(DIB.getOrCreateTypeArray({}));
  auto FooSP = DIB.createFunction(
      CU, "foo", /*LinkageName=*/"", File,
      /*LineNo=*/1, FuncType, /*ScopeLine=*/2, DINode::FlagZero,
      DISubprogram::SPFlagDefinition | DISubprogram::SPFlagOptimized);

  F->setSubprogram(FooSP);
  AllocaInst *I = Builder.CreateAlloca(Builder.getInt8Ty());
  ReturnInst *R = Builder.CreateRetVoid();
  I->setDebugLoc(DILocation::get(Ctx, 3, 2, FooSP));
  R->setDebugLoc(DILocation::get(Ctx, 4, 2, FooSP));

  auto BarSP = DIB.createFunction(
      CU, "bar", /*LinkageName=*/"", File,
      /*LineNo=*/1, FuncType, /*ScopeLine=*/2, DINode::FlagZero,
      DISubprogram::SPFlagDefinition | DISubprogram::SPFlagOptimized);

  // Create a temporary structure in scope of FooSP.
  llvm::TempDIType ForwardDeclaredType =
      llvm::TempDIType(DIB.createReplaceableCompositeType(
          llvm::dwarf::DW_TAG_structure_type, "MyType", FooSP, File, 0, 0, 8, 8,
          {}, "UniqueIdentifier"));

  // Instantiate the real structure in scope of BarSP.
  DICompositeType *Type = DIB.createStructType(
      BarSP, "MyType", File, 0, 8, 8, {}, {}, {}, 0, {}, "UniqueIdentifier");
  // Replace the temporary type with the real type.
  DIB.replaceTemporary(std::move(ForwardDeclaredType), Type);

  DIB.finalize();
  EXPECT_FALSE(verifyModule(*M));

  // After finalization, MyType should appear in retainedNodes of BarSP,
  // not in FooSP's.
  EXPECT_EQ(BarSP->getRetainedNodes().size(), 1u);
  EXPECT_EQ(BarSP->getRetainedNodes()[0], Type);
  EXPECT_TRUE(FooSP->getRetainedNodes().empty());
}

TEST_F(IRBuilderTest, CreateAggregateRet) {
  IRBuilder<> Builder(BB);
  // Terminate the function/block created in SetUp.
  Builder.CreateRetVoid();

  Type *AggType =
      StructType::create(Ctx, {Builder.getInt8Ty(), Builder.getInt64Ty()});
  ConstantInt *RV0 = Builder.getInt8(5);
  ConstantInt *RV1 = Builder.getInt64(55);

  FunctionType *FTy = FunctionType::get(AggType, /*isVarArg=*/false);

  Function *F1 =
      Function::Create(FTy, Function::ExternalLinkage, "F2", M.get());
  BasicBlock *CalleeBB = BasicBlock::Create(Ctx, "", F1);
  IRBuilder<> CalleeBuilder(CalleeBB);
  CalleeBuilder.CreateAggregateRet({RV0, RV1});

  EXPECT_FALSE(verifyModule(*M));
}

// ============================================================
// Tests for IRBuilderBase::CreateLayoutReinterpretCast
// ============================================================

struct DLConfig {
  bool BigEndian;
  unsigned AS0PtrBits; ///< pointer size for address space 0 (32 or 64)
};

void PrintTo(const DLConfig &C, std::ostream *OS) {
  *OS << (C.BigEndian ? "BE" : "LE") << C.AS0PtrBits;
}

/// Fixture that sets a concrete DataLayout on the shared module so that
/// aggregate-field offsets and pointer sizes are well-defined.
///
/// Parameterized over endianness and AS0 pointer width.  AS1 is always
/// 32-bit and AS2 is always 64-bit so that the pointer-cast tests have
/// a fixed reference for "diff-size" and "same-size" variants.
class LayoutReinterpretCastTest
    : public IRBuilderTest,
      public ::testing::WithParamInterface<DLConfig> {
protected:
  bool isBigEndian() const { return GetParam().BigEndian; }
  unsigned as0PtrBits() const { return GetParam().AS0PtrBits; }

  void SetUp() override {
    IRBuilderTest::SetUp();
    unsigned PS = as0PtrBits();
    std::string DL = isBigEndian() ? "E" : "e";
    DL += "-p:" + std::to_string(PS) + ":" + std::to_string(PS);
    DL += "-p1:32:32-p2:64:64"
          "-i8:8:8-i16:16:16-i32:32:32-i64:64:64"
          "-f32:32:32-f64:64:64-n8:16:32:64";
    M->setDataLayout(DL);
    IRBuilder<>(BB).CreateRetVoid();
  }

  /// Verify the module and check that the instructions printed before the
  /// trailing "  ret void" end with \p Expected.  Pass an empty string to
  /// skip the suffix check and only verify the module.
  void nameBB() {
    size_t Idx = 0;
    for (auto &I : *BB) {
      ++Idx;
      if (!I.getType()->isVoidTy() && !I.hasName())
        I.setName(BB->getName() + "." + Twine(Idx));
    }
  }

  void checkBB(StringRef Expected) {
    EXPECT_FALSE(verifyModule(*M, &errs()));
    std::string S;
    raw_string_ostream OS(S);
    BB->print(OS);
    // Strip the shared "  ret void\n" trailer so callers only supply the
    // instructions of interest.
    StringRef Actual(S);
    const StringRef RetVoid = "  ret void\n";
    if (Actual.ends_with(RetVoid))
      Actual = Actual.drop_back(RetVoid.size());
    if (Expected.empty())
      EXPECT_TRUE(Actual.trim().empty())
          << "Expected no instructions before ret:\n"
          << Expected << "\nActual BB:\n"
          << S;
    else
      EXPECT_TRUE(Actual.ends_with(Expected))
          << "Expected BB suffix (before ret void):\n"
          << Expected << "\nActual BB:\n"
          << S;
  }

  /// Build a non-constant value of type Ty so that instructions emitted by
  /// CreateLayoutReinterpretCast are visible (not folded away).
  Value *makeVal(IRBuilderBase &B, Type *Ty) { return B.CreateLoad(Ty, GV); }
};

// ---------------------------------------------------------------------------
// Fast path: leaf-to-leaf with matching bitwidth
// ---------------------------------------------------------------------------

/// i32 → float: single bitcast, no integer accumulation.
TEST_P(LayoutReinterpretCastTest, I32ToFloat) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *I32Ty = B.getInt32Ty();
  Type *F32Ty = B.getFloatTy();
  Value *Src = makeVal(B, I32Ty);
  Value *Result = B.CreateLayoutReinterpretCast(Src, F32Ty);
  EXPECT_EQ(Result->getType(), F32Ty);
  EXPECT_TRUE(isa<BitCastInst>(Result));
  checkBB("  %2 = bitcast i32 %1 to float\n");
}

/// float → i32: single bitcast.
TEST_P(LayoutReinterpretCastTest, FloatToI32) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *F32Ty = B.getFloatTy();
  Type *I32Ty = B.getInt32Ty();
  Value *Src = makeVal(B, F32Ty);
  Value *Result = B.CreateLayoutReinterpretCast(Src, I32Ty);
  EXPECT_EQ(Result->getType(), I32Ty);
  EXPECT_TRUE(isa<BitCastInst>(Result));
  checkBB("  %2 = bitcast float %1 to i32\n");
}

/// Same source and destination type
TEST_P(LayoutReinterpretCastTest, SameType) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *I32Ty = B.getInt32Ty();
  Value *Src = makeVal(B, I32Ty);
  Value *Result = B.CreateLayoutReinterpretCast(Src, I32Ty);
  EXPECT_EQ(Result, Src);
  checkBB("  %1 = load i32, ptr @0, align 4\n");
}

/// {ptr, i32} → {ptr, i32} → {{ptr, i32}} → {ptr, i32} → {{ptr, i32}, {ptr,
/// i32}}
TEST_P(LayoutReinterpretCastTest, PtrStruct_SameType) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *PtrTy = B.getPtrTy(0);
  StructType *STy = StructType::get(PtrTy, B.getInt32Ty());
  Value *Src = makeVal(B, STy);
  Value *Result = B.CreateLayoutReinterpretCast(Src, STy);
  EXPECT_EQ(Result->getType(), STy);
  EXPECT_TRUE(isa<LoadInst>(Result));
  if (as0PtrBits() == 64)
    checkBB("  %1 = load { ptr, i32 }, ptr @0, align 8\n");
  else
    checkBB("  %1 = load { ptr, i32 }, ptr @0, align 4\n");

  StructType *STyW = StructType::get(STy);
  Result = B.CreateLayoutReinterpretCast(Result, STyW);
  EXPECT_EQ(Result->getType(), STyW);
  EXPECT_TRUE(isa<InsertValueInst>(Result));
  Result = B.CreateLayoutReinterpretCast(Result, STy);
  EXPECT_EQ(Result->getType(), STy);
  EXPECT_TRUE(isa<ExtractValueInst>(Result));
  StructType *STyW2 = StructType::get(STy, STy);
  Result = B.CreateLayoutReinterpretCast(Result, STyW2);
  EXPECT_EQ(Result->getType(), STyW2);
  EXPECT_TRUE(isa<InsertValueInst>(Result));
  checkBB("  %2 = insertvalue { { ptr, i32 } } poison, { ptr, i32 } %1, 0\n"
          "  %3 = extractvalue { { ptr, i32 } } %2, 0\n"
          "  %4 = insertvalue { { ptr, i32 }, { ptr, i32 } } poison, { ptr, "
          "i32 } %3, 0\n");
}

// ---------------------------------------------------------------------------
// Pointer casts
// ---------------------------------------------------------------------------

/// ptr (AS0) → i64: single ptrtoint when AS0=64, ptrtoint+zext when AS0=32.
TEST_P(LayoutReinterpretCastTest, PtrToI64) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *PtrTy = B.getPtrTy(0);
  Type *I64Ty = B.getInt64Ty();
  Value *Src = makeVal(B, PtrTy);
  Value *Result = B.CreateLayoutReinterpretCast(Src, I64Ty);
  EXPECT_EQ(Result->getType(), I64Ty);
  if (as0PtrBits() == 64) {
    EXPECT_TRUE(isa<PtrToIntInst>(Result));
    checkBB("  %2 = ptrtoint ptr %1 to i64\n");
  } else if (!isBigEndian()) {
    EXPECT_TRUE(isa<ZExtInst>(Result));
    checkBB("  %2 = ptrtoint ptr %1 to i32\n"
            "  %3 = zext i32 %2 to i64\n");
  } else {
    EXPECT_TRUE(isa<BinaryOperator>(Result));
    checkBB("  %2 = ptrtoint ptr %1 to i32\n"
            "  %3 = zext i32 %2 to i64\n"
            "  %4 = shl i64 %3, 32\n");
  }
}

/// i64 → ptr (AS0): single inttoptr when AS0=64, trunc+inttoptr when AS0=32.
TEST_P(LayoutReinterpretCastTest, I64ToPtr) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *I64Ty = B.getInt64Ty();
  Type *PtrTy = B.getPtrTy(0);
  Value *Src = makeVal(B, I64Ty);
  Value *Result = B.CreateLayoutReinterpretCast(Src, PtrTy);
  EXPECT_EQ(Result->getType(), PtrTy);
  EXPECT_TRUE(isa<IntToPtrInst>(Result));
  if (as0PtrBits() == 64) {
    checkBB("  %2 = inttoptr i64 %1 to ptr\n");
  } else if (!isBigEndian()) {
    checkBB("  %2 = trunc i64 %1 to i32\n"
            "  %3 = inttoptr i32 %2 to ptr\n");
  } else {
    checkBB("  %2 = lshr i64 %1, 32\n"
            "  %3 = trunc i64 %2 to i32\n"
            "  %4 = inttoptr i32 %3 to ptr\n");
  }
}

/// ptr (AS0) → float: ptrtoint to integer, trunc if needed, bitcast to float.
TEST_P(LayoutReinterpretCastTest, PtrToFloat) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *PtrTy = B.getPtrTy(0);
  Type *F32Ty = B.getFloatTy();
  Value *Src = makeVal(B, PtrTy);
  Value *Result = B.CreateLayoutReinterpretCast(Src, F32Ty);
  EXPECT_EQ(Result->getType(), F32Ty);
  if (as0PtrBits() == 64) {
    if (!isBigEndian()) {
      checkBB("  %2 = ptrtoint ptr %1 to i64\n"
              "  %3 = trunc i64 %2 to i32\n"
              "  %4 = bitcast i32 %3 to float\n");
    } else {
      checkBB("  %2 = ptrtoint ptr %1 to i64\n"
              "  %3 = lshr i64 %2, 32\n"
              "  %4 = trunc i64 %3 to i32\n"
              "  %5 = bitcast i32 %4 to float\n");
    }
  } else {
    checkBB("  %2 = ptrtoint ptr %1 to i32\n"
            "  %3 = bitcast i32 %2 to float\n");
  }
}

/// ptr (AS0) → ptr (AS2, always 64-bit): always uses ptrtoint/inttoptr.
TEST_P(LayoutReinterpretCastTest, PtrSameSize) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *PtrAS0 = B.getPtrTy(0);
  Type *PtrAS2 = B.getPtrTy(2); // always 64-bit
  Value *Src = makeVal(B, PtrAS0);
  Value *Result = B.CreateLayoutReinterpretCast(Src, PtrAS2);
  EXPECT_EQ(Result->getType(), PtrAS2);
  EXPECT_TRUE(isa<IntToPtrInst>(Result));
  if (as0PtrBits() == 64) {
    checkBB("  %2 = ptrtoint ptr %1 to i64\n"
            "  %3 = inttoptr i64 %2 to ptr addrspace(2)\n");
  } else if (!isBigEndian()) {
    checkBB("  %2 = ptrtoint ptr %1 to i32\n"
            "  %3 = zext i32 %2 to i64\n"
            "  %4 = inttoptr i64 %3 to ptr addrspace(2)\n");
  } else {
    checkBB("  %2 = ptrtoint ptr %1 to i32\n"
            "  %3 = zext i32 %2 to i64\n"
            "  %4 = shl i64 %3, 32\n"
            "  %5 = inttoptr i64 %4 to ptr addrspace(2)\n");
  }
}

/// ptr (AS0) → ptr (AS1, always 32-bit): ptrtoint/trunc/inttoptr when AS0 >
/// AS1; ptrtoint/inttoptr when both are 32-bit.
TEST_P(LayoutReinterpretCastTest, PtrDiffSize) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *PtrAS0 = B.getPtrTy(0);
  Type *PtrAS1 = B.getPtrTy(1); // always 32-bit
  Value *Src = makeVal(B, PtrAS0);
  Value *Result = B.CreateLayoutReinterpretCast(Src, PtrAS1);
  EXPECT_EQ(Result->getType(), PtrAS1);
  if (as0PtrBits() == 64 && !isBigEndian()) {
    EXPECT_TRUE(isa<IntToPtrInst>(Result));
    checkBB("  %2 = ptrtoint ptr %1 to i64\n"
            "  %3 = trunc i64 %2 to i32\n"
            "  %4 = inttoptr i32 %3 to ptr addrspace(1)\n");
  } else if (as0PtrBits() == 64) {
    EXPECT_TRUE(isa<IntToPtrInst>(Result));
    checkBB("  %2 = ptrtoint ptr %1 to i64\n"
            "  %3 = lshr i64 %2, 32\n"
            "  %4 = trunc i64 %3 to i32\n"
            "  %5 = inttoptr i32 %4 to ptr addrspace(1)\n");
  } else {
    EXPECT_FALSE(isa<AddrSpaceCastInst>(Result));
    EXPECT_TRUE(isa<IntToPtrInst>(Result));
    checkBB("  %2 = ptrtoint ptr %1 to i32\n"
            "  %3 = inttoptr i32 %2 to ptr addrspace(1)\n");
  }
}

// ---------------------------------------------------------------------------
// Struct ↔ integer
// ---------------------------------------------------------------------------

/// {i32, i32} → i64: two i32 halves are OR-assembled into an i64.
TEST_P(LayoutReinterpretCastTest, StructI32x2_ToI64) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  StructType *STy = StructType::get(B.getInt32Ty(), B.getInt32Ty());
  Type *I64Ty = B.getInt64Ty();
  Value *Src = makeVal(B, STy);
  Value *Result = B.CreateLayoutReinterpretCast(Src, I64Ty);
  EXPECT_EQ(Result->getType(), I64Ty);
  if (!isBigEndian()) {
    checkBB("  %2 = extractvalue { i32, i32 } %1, 0\n"
            "  %3 = zext i32 %2 to i64\n"
            "  %4 = extractvalue { i32, i32 } %1, 1\n"
            "  %5 = zext i32 %4 to i64\n"
            "  %6 = shl i64 %5, 32\n"
            "  %7 = or i64 %3, %6\n");
  } else {
    checkBB("  %2 = extractvalue { i32, i32 } %1, 0\n"
            "  %3 = zext i32 %2 to i64\n"
            "  %4 = shl i64 %3, 32\n"
            "  %5 = extractvalue { i32, i32 } %1, 1\n"
            "  %6 = zext i32 %5 to i64\n"
            "  %7 = or i64 %4, %6\n");
  }
}

/// i64 → {i32, i32}: i64 is split into two i32 fields.
TEST_P(LayoutReinterpretCastTest, I64_ToStructI32x2) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *I64Ty = B.getInt64Ty();
  StructType *STy = StructType::get(B.getInt32Ty(), B.getInt32Ty());
  Value *Src = makeVal(B, I64Ty);
  Value *Result = B.CreateLayoutReinterpretCast(Src, STy);
  EXPECT_EQ(Result->getType(), STy);
  if (!isBigEndian()) {
    checkBB("  %2 = trunc i64 %1 to i32\n"
            "  %3 = insertvalue { i32, i32 } poison, i32 %2, 0\n"
            "  %4 = lshr i64 %1, 32\n"
            "  %5 = trunc i64 %4 to i32\n"
            "  %6 = insertvalue { i32, i32 } %3, i32 %5, 1\n");
  } else {
    checkBB("  %2 = lshr i64 %1, 32\n"
            "  %3 = trunc i64 %2 to i32\n"
            "  %4 = insertvalue { i32, i32 } poison, i32 %3, 0\n"
            "  %5 = trunc i64 %1 to i32\n"
            "  %6 = insertvalue { i32, i32 } %4, i32 %5, 1\n");
  }
}

/// {i16, i16} → i32: two 16-bit halves zext-shifted into a 32-bit integer.
TEST_P(LayoutReinterpretCastTest, StructI16x2_ToI32) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  StructType *STy = StructType::get(B.getInt16Ty(), B.getInt16Ty());
  Type *I32Ty = B.getInt32Ty();
  Value *Src = makeVal(B, STy);
  Value *Result = B.CreateLayoutReinterpretCast(Src, I32Ty);
  EXPECT_EQ(Result->getType(), I32Ty);
  if (!isBigEndian()) {
    checkBB("  %2 = extractvalue { i16, i16 } %1, 0\n"
            "  %3 = zext i16 %2 to i32\n"
            "  %4 = extractvalue { i16, i16 } %1, 1\n"
            "  %5 = zext i16 %4 to i32\n"
            "  %6 = shl i32 %5, 16\n"
            "  %7 = or i32 %3, %6\n");
  } else {
    checkBB("  %2 = extractvalue { i16, i16 } %1, 0\n"
            "  %3 = zext i16 %2 to i32\n"
            "  %4 = shl i32 %3, 16\n"
            "  %5 = extractvalue { i16, i16 } %1, 1\n"
            "  %6 = zext i16 %5 to i32\n"
            "  %7 = or i32 %4, %6\n");
  }
}

// ---------------------------------------------------------------------------
// Vector ↔ scalar / struct
// ---------------------------------------------------------------------------

/// <4 x i8> → i32: four bytes OR-assembled into a 32-bit integer.
TEST_P(LayoutReinterpretCastTest, Vec4xi8_ToI32) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *VTy = FixedVectorType::get(B.getInt8Ty(), 4);
  Type *I32Ty = B.getInt32Ty();
  Value *Src = makeVal(B, VTy);
  Value *Result = B.CreateLayoutReinterpretCast(Src, I32Ty);
  EXPECT_EQ(Result->getType(), I32Ty);
  checkBB("  %2 = bitcast <4 x i8> %1 to i32\n");
}

/// i32 → <4 x i8>: i32 is split into four byte-wide vector elements.
TEST_P(LayoutReinterpretCastTest, I32_ToVec4xi8) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *I32Ty = B.getInt32Ty();
  Type *VTy = FixedVectorType::get(B.getInt8Ty(), 4);
  Value *Src = makeVal(B, I32Ty);
  Value *Result = B.CreateLayoutReinterpretCast(Src, VTy);
  EXPECT_EQ(Result->getType(), VTy);
  checkBB("  %2 = bitcast i32 %1 to <4 x i8>\n");
}

// ---------------------------------------------------------------------------
// Sub-byte element vector (<4 x i4>) casts
// ---------------------------------------------------------------------------

/// <4 x i4> → i16: same 16-bit reinterpretation via bitcast.
TEST_P(LayoutReinterpretCastTest, Vec4xi4_ToI16) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *VTy = FixedVectorType::get(Type::getIntNTy(Ctx, 4), 4);
  Type *I16Ty = B.getInt16Ty();
  Value *Src = makeVal(B, VTy);
  Value *Result = B.CreateLayoutReinterpretCast(Src, I16Ty);
  EXPECT_EQ(Result->getType(), I16Ty);
  EXPECT_TRUE(isa<BitCastInst>(Result));
  checkBB("  %2 = bitcast <4 x i4> %1 to i16\n");
}

/// i16 → <4 x i4>: same 16-bit reinterpretation via bitcast.
TEST_P(LayoutReinterpretCastTest, I16_ToVec4xi4) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *I16Ty = B.getInt16Ty();
  Type *VTy = FixedVectorType::get(Type::getIntNTy(Ctx, 4), 4);
  Value *Src = makeVal(B, I16Ty);
  Value *Result = B.CreateLayoutReinterpretCast(Src, VTy);
  EXPECT_EQ(Result->getType(), VTy);
  EXPECT_TRUE(isa<BitCastInst>(Result));
  checkBB("  %2 = bitcast i16 %1 to <4 x i4>\n");
}

/// <4 x i4> → <2 x i8>: both 16-bit vectors, single bitcast.
TEST_P(LayoutReinterpretCastTest, Vec4xi4_ToVec2xi8) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *Src4xi4 = FixedVectorType::get(Type::getIntNTy(Ctx, 4), 4);
  Type *Dst2xi8 = FixedVectorType::get(B.getInt8Ty(), 2);
  Value *Src = makeVal(B, Src4xi4);
  Value *Result = B.CreateLayoutReinterpretCast(Src, Dst2xi8);
  EXPECT_EQ(Result->getType(), Dst2xi8);
  EXPECT_TRUE(isa<BitCastInst>(Result));
  checkBB("  %2 = bitcast <4 x i4> %1 to <2 x i8>\n");
}

/// <2 x i8> → <4 x i4>: both 16-bit vectors, single bitcast.
TEST_P(LayoutReinterpretCastTest, Vec2xi8_ToVec4xi4) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *Src2xi8 = FixedVectorType::get(B.getInt8Ty(), 2);
  Type *Dst4xi4 = FixedVectorType::get(Type::getIntNTy(Ctx, 4), 4);
  Value *Src = makeVal(B, Src2xi8);
  Value *Result = B.CreateLayoutReinterpretCast(Src, Dst4xi4);
  EXPECT_EQ(Result->getType(), Dst4xi4);
  EXPECT_TRUE(isa<BitCastInst>(Result));
  checkBB("  %2 = bitcast <2 x i8> %1 to <4 x i4>\n");
}

/// <4 x i4> → [2 x i8]
TEST_P(LayoutReinterpretCastTest, Vec4xi4_ToArr2xi8) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *VTy = FixedVectorType::get(Type::getIntNTy(Ctx, 4), 4);
  Type *ArrTy = ArrayType::get(B.getInt8Ty(), 2);
  Value *Src = makeVal(B, VTy);
  Value *Result = B.CreateLayoutReinterpretCast(Src, ArrTy);
  EXPECT_EQ(Result->getType(), ArrTy);
  checkBB("  %2 = bitcast <4 x i4> %1 to <2 x i8>\n"
          "  %3 = extractelement <2 x i8> %2, i64 0\n"
          "  %4 = insertvalue [2 x i8] poison, i8 %3, 0\n"
          "  %5 = bitcast <4 x i4> %1 to <2 x i8>\n"
          "  %6 = extractelement <2 x i8> %5, i64 1\n"
          "  %7 = insertvalue [2 x i8] %4, i8 %6, 1\n");
}

/// [2 x i8] → <4 x i4>
TEST_P(LayoutReinterpretCastTest, Arr2xi8_ToVec4xi4) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *ArrTy = ArrayType::get(B.getInt8Ty(), 2);
  Type *VTy = FixedVectorType::get(Type::getIntNTy(Ctx, 4), 4);
  Value *Src = makeVal(B, ArrTy);
  Value *Result = B.CreateLayoutReinterpretCast(Src, VTy);
  EXPECT_EQ(Result->getType(), VTy);
  checkBB("  %2 = extractvalue [2 x i8] %1, 0\n"
          "  %3 = insertelement <2 x i8> poison, i8 %2, i64 0\n"
          "  %4 = extractvalue [2 x i8] %1, 1\n"
          "  %5 = insertelement <2 x i8> %3, i8 %4, i64 1\n"
          "  %6 = bitcast <2 x i8> %5 to <4 x i4>\n");
}

/// {float, float} → <2 x float>
TEST_P(LayoutReinterpretCastTest, StructFloatx2_ToVec2xFloat) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  StructType *STy = StructType::get(B.getFloatTy(), B.getFloatTy());
  Type *VTy = FixedVectorType::get(B.getFloatTy(), 2);
  Value *Src = makeVal(B, STy);
  Value *Result = B.CreateLayoutReinterpretCast(Src, VTy);
  EXPECT_EQ(Result->getType(), VTy);
  // Vector fast path: each float struct field aligns exactly with one vector
  // element, so insertelement is used directly instead of integer accumulation.
  checkBB("  %2 = extractvalue { float, float } %1, 0\n"
          "  %3 = insertelement <2 x float> poison, float %2, i64 0\n"
          "  %4 = extractvalue { float, float } %1, 1\n"
          "  %5 = insertelement <2 x float> %3, float %4, i64 1\n");
}

/// [i16, i16, i32] → <2 x float>
TEST_P(LayoutReinterpretCastTest, StructMixedAAB_ToVec2xFloat) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  StructType *STy =
      StructType::get(B.getInt16Ty(), B.getInt16Ty(), B.getInt32Ty());
  Type *VTy = FixedVectorType::get(B.getInt32Ty(), 2);
  Value *Src = makeVal(B, STy);
  Value *Result = B.CreateLayoutReinterpretCast(Src, VTy);
  EXPECT_EQ(Result->getType(), VTy);
  checkBB("  %2 = extractvalue { i16, i16, i32 } %1, 0\n"
          "  %3 = insertelement <4 x i16> poison, i16 %2, i64 0\n"
          "  %4 = extractvalue { i16, i16, i32 } %1, 1\n"
          "  %5 = insertelement <4 x i16> %3, i16 %4, i64 1\n"
          "  %6 = extractvalue { i16, i16, i32 } %1, 2\n"
          "  %7 = bitcast <4 x i16> %5 to <2 x i32>\n"
          "  %8 = insertelement <2 x i32> %7, i32 %6, i64 1\n");
}

/// [i32, i16, i16] → <2 x float>
TEST_P(LayoutReinterpretCastTest, StructMixedBAA_ToVec2xFloat) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  StructType *STy =
      StructType::get(B.getInt32Ty(), B.getInt16Ty(), B.getInt16Ty());
  Type *VTy = FixedVectorType::get(B.getInt32Ty(), 2);
  Value *Src = makeVal(B, STy);
  Value *Result = B.CreateLayoutReinterpretCast(Src, VTy);
  EXPECT_EQ(Result->getType(), VTy);
  checkBB("  %2 = extractvalue { i32, i16, i16 } %1, 0\n"
          "  %3 = insertelement <2 x i32> poison, i32 %2, i64 0\n"
          "  %4 = extractvalue { i32, i16, i16 } %1, 1\n"
          "  %5 = bitcast <2 x i32> %3 to <4 x i16>\n"
          "  %6 = insertelement <4 x i16> %5, i16 %4, i64 2\n"
          "  %7 = extractvalue { i32, i16, i16 } %1, 2\n"
          "  %8 = insertelement <4 x i16> %6, i16 %7, i64 3\n"
          "  %9 = bitcast <4 x i16> %8 to <2 x i32>\n");
}

/// <2 x float> → <3 x float>
TEST_P(LayoutReinterpretCastTest, VecFloatCast_2_3) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *STy = FixedVectorType::get(B.getFloatTy(), 2);
  Type *DTy = FixedVectorType::get(B.getFloatTy(), 3);
  Value *Src = makeVal(B, STy);
  Value *Result = B.CreateLayoutReinterpretCast(Src, DTy);
  EXPECT_EQ(Result->getType(), DTy);
  checkBB("  %2 = shufflevector <2 x float> %1, <2 x float> poison, <3 x i32> "
          "<i32 0, i32 1, i32 poison>\n");
}

/// [<2 x float>, float] → <3 x float>
TEST_P(LayoutReinterpretCastTest, VecFloatCast_21_3) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *DTy = FixedVectorType::get(B.getFloatTy(), 3);
  StructType *STy =
      StructType::get(FixedVectorType::get(B.getFloatTy(), 2), B.getFloatTy());
  Value *Src = makeVal(B, STy);
  Value *Result = B.CreateLayoutReinterpretCast(Src, DTy);
  EXPECT_EQ(Result->getType(), DTy);
  checkBB("  %2 = extractvalue { <2 x float>, float } %1, 0\n"
          "  %3 = shufflevector <2 x float> %2, <2 x float> poison, <3 x i32> "
          "<i32 0, i32 1, i32 poison>\n"
          "  %4 = extractvalue { <2 x float>, float } %1, 1\n"
          "  %5 = insertelement <3 x float> %3, float %4, i64 2\n");
}

/// <3 x float> → <2 x float>
TEST_P(LayoutReinterpretCastTest, VecFloatCast_3_2) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *DTy = FixedVectorType::get(B.getFloatTy(), 2);
  Type *STy = FixedVectorType::get(B.getFloatTy(), 3);
  Value *Src = makeVal(B, STy);
  Value *Result = B.CreateLayoutReinterpretCast(Src, DTy);
  EXPECT_EQ(Result->getType(), DTy);
  checkBB("  %2 = shufflevector <3 x float> %1, <3 x float> poison, <2 x i32> "
          "<i32 0, i32 1>\n");
}

/// <3 x float> → [<2 x float>, float]
TEST_P(LayoutReinterpretCastTest, VecFloatCast_3_21) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  StructType *DTy =
      StructType::get(FixedVectorType::get(B.getFloatTy(), 2), B.getFloatTy());
  Type *STy = FixedVectorType::get(B.getFloatTy(), 3);
  Value *Src = makeVal(B, STy);
  Value *Result = B.CreateLayoutReinterpretCast(Src, DTy);
  EXPECT_EQ(Result->getType(), DTy);
  checkBB(
      "  %2 = shufflevector <3 x float> %1, <3 x float> poison, <2 x i32> <i32 "
      "0, i32 1>\n"
      "  %3 = insertvalue { <2 x float>, float } poison, <2 x float> %2, 0\n"
      "  %4 = extractelement <3 x float> %1, i64 2\n"
      "  %5 = insertvalue { <2 x float>, float } %3, float %4, 1\n");
}

/// {<2 x float>, <2 x float>} → {<2 x i32>, <2 x i32>}: element-type
/// reinterpret within a same-shape struct; each field is an independent
/// bitcast with no cross-field data movement.
TEST_P(LayoutReinterpretCastTest, StructVec2Float_ToStructVec2I32) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *V2F = FixedVectorType::get(B.getFloatTy(), 2);
  Type *V2I = FixedVectorType::get(B.getInt32Ty(), 2);
  StructType *STy = StructType::get(V2F, V2F);
  StructType *DTy = StructType::get(V2I, V2I);
  Value *Src = makeVal(B, STy);
  Value *Result = B.CreateLayoutReinterpretCast(Src, DTy);
  EXPECT_EQ(Result->getType(), DTy);
  checkBB(
      "  %2 = extractvalue { <2 x float>, <2 x float> } %1, 0\n"
      "  %3 = bitcast <2 x float> %2 to <2 x i32>\n"
      "  %4 = insertvalue { <2 x i32>, <2 x i32> } poison, <2 x i32> %3, 0\n"
      "  %5 = extractvalue { <2 x float>, <2 x float> } %1, 1\n"
      "  %6 = bitcast <2 x float> %5 to <2 x i32>\n"
      "  %7 = insertvalue { <2 x i32>, <2 x i32> } %4, <2 x i32> %6, 1\n");
}

/// {<4 x i32>, <2 x i32>} → {<2 x i32>, <2 x i32>, <2 x i32>}: split the
/// 128-bit first field into two 64-bit fields; third dst field = second src
/// field.  Field boundaries cross, requiring shuffles or extracts.
TEST_P(LayoutReinterpretCastTest, StructVec4_SplitTo3Vec2) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *V2 = FixedVectorType::get(B.getInt32Ty(), 2);
  Type *V4 = FixedVectorType::get(B.getInt32Ty(), 4);
  StructType *STy = StructType::get(V4, V2);
  StructType *DTy = StructType::get(V2, V2, V2);
  Value *Src = makeVal(B, STy);
  Value *Result = B.CreateLayoutReinterpretCast(Src, DTy);
  EXPECT_EQ(Result->getType(), DTy);
  checkBB("  %2 = extractvalue { <4 x i32>, <2 x i32> } %1, 0\n"
          "  %3 = shufflevector <4 x i32> %2, <4 x i32> poison, <2 x i32> <i32 "
          "0, i32 1>\n"
          "  %4 = insertvalue { <2 x i32>, <2 x i32>, <2 x i32> } poison, <2 x "
          "i32> %3, 0\n"
          "  %5 = shufflevector <4 x i32> %2, <4 x i32> poison, <2 x i32> <i32 "
          "2, i32 3>\n"
          "  %6 = insertvalue { <2 x i32>, <2 x i32>, <2 x i32> } %4, <2 x "
          "i32> %5, 1\n"
          "  %7 = extractvalue { <4 x i32>, <2 x i32> } %1, 1\n"
          "  %8 = insertvalue { <2 x i32>, <2 x i32>, <2 x i32> } %6, <2 x "
          "i32> %7, 2\n");
}

/// {<2 x i32>, <2 x i32>, <2 x i32>} → {<4 x i32>, <2 x i32>}: merge the
/// first two 64-bit fields into a single 128-bit field; second dst field =
/// third src field.  Reverse of StructVec4_SplitTo3Vec2.
TEST_P(LayoutReinterpretCastTest, Struct3Vec2_MergeToVec4) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *V2 = FixedVectorType::get(B.getInt32Ty(), 2);
  Type *V4 = FixedVectorType::get(B.getInt32Ty(), 4);
  StructType *STy = StructType::get(V2, V2, V2);
  StructType *DTy = StructType::get(V4, V2);
  Value *Src = makeVal(B, STy);
  Value *Result = B.CreateLayoutReinterpretCast(Src, DTy);
  EXPECT_EQ(Result->getType(), DTy);
  checkBB(
      "  %2 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32> } %1, 0\n"
      "  %3 = shufflevector <2 x i32> %2, <2 x i32> poison, <4 x i32> <i32 0, "
      "i32 1, i32 poison, i32 poison>\n"
      "  %4 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32> } %1, 1\n"
      "  %5 = shufflevector <2 x i32> %4, <2 x i32> poison, <4 x i32> <i32 "
      "poison, i32 poison, i32 0, i32 1>\n"
      "  %6 = shufflevector <4 x i32> %3, <4 x i32> %5, <4 x i32> <i32 0, i32 "
      "1, i32 6, i32 7>\n"
      "  %7 = insertvalue { <4 x i32>, <2 x i32> } poison, <4 x i32> %6, 0\n"
      "  %8 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32> } %1, 2\n"
      "  %9 = insertvalue { <4 x i32>, <2 x i32> } %7, <2 x i32> %8, 1\n");
}

/// {<2 x float>, float, float} → {float, float, <2 x float>}: both structs
/// are 128 bits but with different field positions.  The vector occupies bits
/// 0-63 in the source and bits 64-127 in the destination, so every field
/// crosses a source/destination boundary.
TEST_P(LayoutReinterpretCastTest, StructVec2FF_ToStructFFVec2) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *F32 = B.getFloatTy();
  Type *V2F = FixedVectorType::get(F32, 2);
  StructType *STy = StructType::get(V2F, F32, F32);
  StructType *DTy = StructType::get(F32, F32, V2F);
  Value *Src = makeVal(B, STy);
  Value *Result = B.CreateLayoutReinterpretCast(Src, DTy);
  EXPECT_EQ(Result->getType(), DTy);
  checkBB(
      "  %2 = extractvalue { <2 x float>, float, float } %1, 0\n"
      "  %3 = extractelement <2 x float> %2, i64 0\n"
      "  %4 = insertvalue { float, float, <2 x float> } poison, float %3, 0\n"
      "  %5 = extractelement <2 x float> %2, i64 1\n"
      "  %6 = insertvalue { float, float, <2 x float> } %4, float %5, 1\n"
      "  %7 = extractvalue { <2 x float>, float, float } %1, 1\n"
      "  %8 = insertelement <2 x float> poison, float %7, i64 0\n"
      "  %9 = extractvalue { <2 x float>, float, float } %1, 2\n"
      "  %10 = insertelement <2 x float> %8, float %9, i64 1\n"
      "  %11 = insertvalue { float, float, <2 x float> } %6, <2 x float> %10, "
      "2\n");
}

/// [2 x <3 x i32>] → <3 x i64>: array-of-odd-vector to packed vector.
/// The array element stride is 16 bytes (alloc size of <3 x i32> with 128-bit
/// alignment), so element[1] starts at byte offset 16 even though it only
/// holds 12 bytes of data.  The destination i64 elements cross those alloc
/// boundaries, giving the implementation a non-trivial stitching problem.
TEST_P(LayoutReinterpretCastTest, ArrOf3xi32_ToVec3xi64) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *V3I32 = FixedVectorType::get(B.getInt32Ty(), 3);
  Type *ArrTy = ArrayType::get(V3I32, 2);
  Type *DTy = FixedVectorType::get(B.getInt64Ty(), 3);
  Value *Src = makeVal(B, ArrTy);
  Value *Result = B.CreateLayoutReinterpretCast(Src, DTy);
  EXPECT_EQ(Result->getType(), DTy);
  if (!isBigEndian())
    checkBB("  %2 = extractvalue [2 x <3 x i32>] %1, 0\n"
            "  %3 = shufflevector <3 x i32> %2, <3 x i32> poison, <6 x i32> "
            "<i32 0, i32 1, i32 2, i32 poison, i32 poison, i32 poison>\n"
            "  %4 = extractvalue [2 x <3 x i32>] %1, 1\n"
            "  %5 = bitcast <3 x i32> %4 to i96\n"
            "  %6 = trunc i96 %5 to i64\n"
            "  %7 = bitcast <6 x i32> %3 to <3 x i64>\n"
            "  %8 = insertelement <3 x i64> %7, i64 %6, i64 2\n");
  else
    checkBB("  %2 = extractvalue [2 x <3 x i32>] %1, 0\n"
            "  %3 = shufflevector <3 x i32> %2, <3 x i32> poison, <6 x i32> "
            "<i32 0, i32 1, i32 2, i32 poison, i32 poison, i32 poison>\n"
            "  %4 = extractvalue [2 x <3 x i32>] %1, 1\n"
            "  %5 = bitcast <3 x i32> %4 to i96\n"
            "  %6 = lshr i96 %5, 32\n"
            "  %7 = trunc i96 %6 to i64\n"
            "  %8 = bitcast <6 x i32> %3 to <3 x i64>\n"
            "  %9 = insertelement <3 x i64> %8, i64 %7, i64 2\n");
}

/// <5 x i32> → <3 x i64>: 160 source bits into 192 destination bits.
/// The first four i32 lanes fill the first two i64 slots completely; the fifth
/// i32 fills only the low half of the third i64, leaving the high 32 bits as
/// poison.
TEST_P(LayoutReinterpretCastTest, Vec5xi32_ToVec3xi64) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *STy = FixedVectorType::get(B.getInt32Ty(), 5);
  Type *DTy = FixedVectorType::get(B.getInt64Ty(), 3);
  Value *Src = makeVal(B, STy);
  Value *Result = B.CreateLayoutReinterpretCast(Src, DTy);
  EXPECT_EQ(Result->getType(), DTy);
  checkBB("  %2 = shufflevector <5 x i32> %1, <5 x i32> poison, <6 x i32> <i32 "
          "0, i32 1, i32 2, i32 3, i32 4, i32 poison>\n"
          "  %3 = bitcast <6 x i32> %2 to <3 x i64>\n");
}

/// <4 x i16> → <3 x i32>: src bitcast to i32 view, then scatter first 2 of 3.
TEST_P(LayoutReinterpretCastTest, Vec4xi16_ToVec3xi32) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *STy = FixedVectorType::get(B.getInt16Ty(), 4);
  Type *DTy = FixedVectorType::get(B.getInt32Ty(), 3);
  Value *Src = makeVal(B, STy);
  Value *Result = B.CreateLayoutReinterpretCast(Src, DTy);
  EXPECT_EQ(Result->getType(), DTy);
  checkBB("  %2 = bitcast <4 x i16> %1 to <2 x i32>\n"
          "  %3 = shufflevector <2 x i32> %2, <2 x i32> poison, <3 x i32> <i32 "
          "0, i32 1, i32 poison>\n");
}

/// <3 x i32> → <4 x i16>: src bitcast to i16 view, then extract first 4 of 6.
TEST_P(LayoutReinterpretCastTest, Vec3xi32_ToVec4xi16) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *STy = FixedVectorType::get(B.getInt32Ty(), 3);
  Type *DTy = FixedVectorType::get(B.getInt16Ty(), 4);
  Value *Src = makeVal(B, STy);
  Value *Result = B.CreateLayoutReinterpretCast(Src, DTy);
  EXPECT_EQ(Result->getType(), DTy);
  checkBB("  %2 = bitcast <3 x i32> %1 to <6 x i16>\n"
          "  %3 = shufflevector <6 x i16> %2, <6 x i16> poison, <4 x i32> <i32 "
          "0, i32 1, i32 2, i32 3>\n");
}

// ---------------------------------------------------------------------------
// Mixed / nested aggregates
// ---------------------------------------------------------------------------

/// {i32, [2 x i16]} → i64: struct with a nested array.
/// Three leaves (i32, i16, i16) are OR-assembled into i64.
TEST_P(LayoutReinterpretCastTest, MixedStruct_ToI64) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *ArrayTy = ArrayType::get(B.getInt16Ty(), 2);
  StructType *STy = StructType::get(B.getInt32Ty(), ArrayTy);
  Type *I64Ty = B.getInt64Ty();
  Value *Src = makeVal(B, STy);
  Value *Result = B.CreateLayoutReinterpretCast(Src, I64Ty);
  EXPECT_EQ(Result->getType(), I64Ty);
  if (!isBigEndian()) {
    checkBB("  %2 = extractvalue { i32, [2 x i16] } %1, 0\n"
            "  %3 = zext i32 %2 to i64\n"
            "  %4 = extractvalue { i32, [2 x i16] } %1, 1\n"
            "  %5 = extractvalue [2 x i16] %4, 0\n"
            "  %6 = zext i16 %5 to i64\n"
            "  %7 = shl i64 %6, 32\n"
            "  %8 = or i64 %3, %7\n"
            "  %9 = extractvalue [2 x i16] %4, 1\n"
            "  %10 = zext i16 %9 to i64\n"
            "  %11 = shl i64 %10, 48\n"
            "  %12 = or i64 %8, %11\n");
  } else {
    checkBB("  %2 = extractvalue { i32, [2 x i16] } %1, 0\n"
            "  %3 = zext i32 %2 to i64\n"
            "  %4 = shl i64 %3, 32\n"
            "  %5 = extractvalue { i32, [2 x i16] } %1, 1\n"
            "  %6 = extractvalue [2 x i16] %5, 0\n"
            "  %7 = zext i16 %6 to i64\n"
            "  %8 = shl i64 %7, 16\n"
            "  %9 = or i64 %4, %8\n"
            "  %10 = extractvalue [2 x i16] %5, 1\n"
            "  %11 = zext i16 %10 to i64\n"
            "  %12 = or i64 %9, %11\n");
  }
}

/// {i32, i32} → {i16, i16, i16, i16}: cross-struct layout reinterpret.
/// Destination has twice as many leaves, each half the width of a source leaf.
TEST_P(LayoutReinterpretCastTest, CrossStructLayout) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *I16Ty = B.getInt16Ty();
  Type *I32Ty = B.getInt32Ty();
  StructType *SrcTy = StructType::get(I32Ty, I32Ty);
  StructType *DstTy = StructType::get(I16Ty, I16Ty, I16Ty, I16Ty);
  Value *Src = makeVal(B, SrcTy);
  Value *Result = B.CreateLayoutReinterpretCast(Src, DstTy);
  EXPECT_EQ(Result->getType(), DstTy);
  if (!isBigEndian()) {
    checkBB("  %2 = extractvalue { i32, i32 } %1, 0\n"
            "  %3 = trunc i32 %2 to i16\n"
            "  %4 = insertvalue { i16, i16, i16, i16 } poison, i16 %3, 0\n"
            "  %5 = lshr i32 %2, 16\n"
            "  %6 = trunc i32 %5 to i16\n"
            "  %7 = insertvalue { i16, i16, i16, i16 } %4, i16 %6, 1\n"
            "  %8 = extractvalue { i32, i32 } %1, 1\n"
            "  %9 = trunc i32 %8 to i16\n"
            "  %10 = insertvalue { i16, i16, i16, i16 } %7, i16 %9, 2\n"
            "  %11 = lshr i32 %8, 16\n"
            "  %12 = trunc i32 %11 to i16\n"
            "  %13 = insertvalue { i16, i16, i16, i16 } %10, i16 %12, 3\n");
  } else {
    checkBB("  %2 = extractvalue { i32, i32 } %1, 0\n"
            "  %3 = lshr i32 %2, 16\n"
            "  %4 = trunc i32 %3 to i16\n"
            "  %5 = insertvalue { i16, i16, i16, i16 } poison, i16 %4, 0\n"
            "  %6 = trunc i32 %2 to i16\n"
            "  %7 = insertvalue { i16, i16, i16, i16 } %5, i16 %6, 1\n"
            "  %8 = extractvalue { i32, i32 } %1, 1\n"
            "  %9 = lshr i32 %8, 16\n"
            "  %10 = trunc i32 %9 to i16\n"
            "  %11 = insertvalue { i16, i16, i16, i16 } %7, i16 %10, 2\n"
            "  %12 = trunc i32 %8 to i16\n"
            "  %13 = insertvalue { i16, i16, i16, i16 } %11, i16 %12, 3\n");
  }
}

// ---------------------------------------------------------------------------
// Struct padding
// ---------------------------------------------------------------------------

/// {i8, i32} (with 3-byte natural padding after i8) → i64.
/// The two data leaves (i8, i32) must be placed at the correct bit offsets;
/// padding bits are not touched and remain as the zero base of the accumulator.
TEST_P(LayoutReinterpretCastTest, PaddedStruct_ToI64) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  // {i8, i32} natural layout: i8@bit0, padding bits 8-31, i32@bit32.
  StructType *STy = StructType::get(B.getInt8Ty(), B.getInt32Ty());
  Type *I64Ty = B.getInt64Ty();
  Value *Src = makeVal(B, STy);
  Value *Result = B.CreateLayoutReinterpretCast(Src, I64Ty);
  EXPECT_EQ(Result->getType(), I64Ty);
  if (!isBigEndian()) {
    checkBB("  %2 = extractvalue { i8, i32 } %1, 0\n"
            "  %3 = zext i8 %2 to i64\n"
            "  %4 = extractvalue { i8, i32 } %1, 1\n"
            "  %5 = zext i32 %4 to i64\n"
            "  %6 = shl i64 %5, 32\n"
            "  %7 = or i64 %3, %6\n");
  } else {
    checkBB("  %2 = extractvalue { i8, i32 } %1, 0\n"
            "  %3 = zext i8 %2 to i64\n"
            "  %4 = shl i64 %3, 56\n"
            "  %5 = extractvalue { i8, i32 } %1, 1\n"
            "  %6 = zext i32 %5 to i64\n"
            "  %7 = or i64 %4, %6\n");
  }
}

/// i64 → {i8, i32} (with 3-byte padding).
TEST_P(LayoutReinterpretCastTest, I64_ToPaddedStruct) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  StructType *STy = StructType::get(B.getInt8Ty(), B.getInt32Ty());
  Type *I64Ty = B.getInt64Ty();
  Value *Src = makeVal(B, I64Ty);
  Value *Result = B.CreateLayoutReinterpretCast(Src, STy);
  EXPECT_EQ(Result->getType(), STy);
  if (!isBigEndian()) {
    checkBB("  %2 = trunc i64 %1 to i8\n"
            "  %3 = insertvalue { i8, i32 } poison, i8 %2, 0\n"
            "  %4 = lshr i64 %1, 32\n"
            "  %5 = trunc i64 %4 to i32\n"
            "  %6 = insertvalue { i8, i32 } %3, i32 %5, 1\n");
  } else {
    checkBB("  %2 = lshr i64 %1, 56\n"
            "  %3 = trunc i64 %2 to i8\n"
            "  %4 = insertvalue { i8, i32 } poison, i8 %3, 0\n"
            "  %5 = trunc i64 %1 to i32\n"
            "  %6 = insertvalue { i8, i32 } %4, i32 %5, 1\n");
  }
}

// ---------------------------------------------------------------------------
// Bit offsets
// ---------------------------------------------------------------------------

/// DstOffset=1: read bits [8..23] of an i32, produce an i16.
TEST_P(LayoutReinterpretCastTest, SrcBitOffset) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *I32Ty = B.getInt32Ty();
  Type *I16Ty = B.getInt16Ty();
  Value *Src = makeVal(B, I32Ty);
  Value *Result = B.CreateLayoutReinterpretCast(Src, I16Ty, /*SrcOffset=*/0,
                                                /*DstOffset=*/1);
  EXPECT_EQ(Result->getType(), I16Ty);
  EXPECT_TRUE(isa<TruncInst>(Result));

  Result = B.CreateLayoutReinterpretCast(Src, I16Ty, /*SrcOffset=*/0,
                                         /*DstOffset=*/64);
  EXPECT_EQ(Result->getType(), I16Ty);
  EXPECT_TRUE(isa<PoisonValue>(Result));

  // Symmetric shift for either of BE or LE
  checkBB("  %2 = lshr i32 %1, 8\n"
          "  %3 = trunc i32 %2 to i16\n");
}

/// SrcOffset=2: place an i16 value into bits [16..31] of an i32.
TEST_P(LayoutReinterpretCastTest, DstBitOffset) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *I16Ty = B.getInt16Ty();
  Type *I32Ty = B.getInt32Ty();
  Value *Src = makeVal(B, I16Ty);
  Value *Result = B.CreateLayoutReinterpretCast(Src, I32Ty, /*SrcOffset=*/2,
                                                /*DstOffset=*/0);
  EXPECT_EQ(Result->getType(), I32Ty);
  if (!isBigEndian())
    EXPECT_TRUE(isa<BinaryOperator>(Result));
  else
    EXPECT_TRUE(isa<ZExtInst>(Result));

  Result = B.CreateLayoutReinterpretCast(Src, I32Ty, /*SrcOffset=*/64,
                                         /*DstOffset=*/0);
  EXPECT_EQ(Result->getType(), I32Ty);
  EXPECT_TRUE(isa<PoisonValue>(Result));

  if (!isBigEndian())
    checkBB("  %2 = zext i16 %1 to i32\n"
            "  %3 = shl i32 %2, 16\n");
  else
    checkBB("  %2 = zext i16 %1 to i32\n");
}

// ---------------------------------------------------------------------------
// Size mismatch
// ---------------------------------------------------------------------------

/// Src (i64) larger than Dst (i16): extra source bits are dropped via trunc.
TEST_P(LayoutReinterpretCastTest, SrcLargerThanDst) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Value *Src = makeVal(B, B.getInt64Ty());
  Value *Result = B.CreateLayoutReinterpretCast(Src, B.getInt16Ty());
  EXPECT_EQ(Result->getType(), B.getInt16Ty());
  if (!isBigEndian())
    checkBB("  %2 = trunc i64 %1 to i16\n");
  else
    checkBB("  %2 = lshr i64 %1, 48\n"
            "  %3 = trunc i64 %2 to i16\n");
}

/// Dst (i32) larger than Src (i16): source is zero-extended to fill Dst.
TEST_P(LayoutReinterpretCastTest, DstLargerThanSrc) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Value *Src = makeVal(B, B.getInt16Ty());
  Value *Result = B.CreateLayoutReinterpretCast(Src, B.getInt32Ty());
  EXPECT_EQ(Result->getType(), B.getInt32Ty());
  if (!isBigEndian())
    checkBB("  %2 = zext i16 %1 to i32\n");
  else
    checkBB("  %2 = zext i16 %1 to i32\n"
            "  %3 = shl i32 %2, 16\n");
}

// ---------------------------------------------------------------------------
// Empty aggregate
// ---------------------------------------------------------------------------

/// Empty struct {} → i32: no source leaves
TEST_P(LayoutReinterpretCastTest, EmptyStruct_ToI32) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  StructType *EmptyTy = StructType::get(Ctx, {});
  Value *Src = makeVal(B, EmptyTy);
  Value *Result = B.CreateLayoutReinterpretCast(Src, B.getInt32Ty());
  EXPECT_TRUE(isa<PoisonValue>(Result));
  checkBB("  %1 = load {}, ptr @0, align 1\n");
}

/// i32 → empty struct {}: no destination leaves
TEST_P(LayoutReinterpretCastTest, I32_ToEmptyStruct) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  StructType *EmptyTy = StructType::get(Ctx, {});
  Value *Src = makeVal(B, B.getInt32Ty());
  Value *Result = B.CreateLayoutReinterpretCast(Src, EmptyTy);
  EXPECT_TRUE(isa<PoisonValue>(Result));
  checkBB("  %1 = load i32, ptr @0, align 4\n");
}

// ---------------------------------------------------------------------------
// Constant folding
// ---------------------------------------------------------------------------

/// A constant source produces a constant result with ConstantFolder.
TEST_P(LayoutReinterpretCastTest, ConstantSource_I32ToFloat_Folds) {
  IRBuilder<> B(BB->getTerminator());
  // 0x3F800000 = IEEE 1.0f
  Constant *Src = ConstantInt::get(B.getInt32Ty(), 0x3F800000u);
  Value *Result = B.CreateLayoutReinterpretCast(Src, B.getFloatTy());
  EXPECT_TRUE(isa<ConstantFP>(Result));
  checkBB("");
}

TEST_P(LayoutReinterpretCastTest, ConstantSource_StructJoin_Folds) {
  IRBuilder<> B(BB->getTerminator());
  Type *I32Ty = B.getInt32Ty();
  StructType *STy = StructType::get(I32Ty, I32Ty);
  Constant *Src = ConstantStruct::get(
      STy, {ConstantInt::get(I32Ty, 1), ConstantInt::get(I32Ty, 2)});
  Value *Result = B.CreateLayoutReinterpretCast(Src, B.getInt64Ty());
  EXPECT_TRUE(isa<ConstantInt>(Result));
  checkBB("");
}

// ---------------------------------------------------------------------------
// Round-trip
// ---------------------------------------------------------------------------

/// Cast A→B then B→A on a constant input; the round-trip must reproduce the
/// original value in all non-padding bits.  Verified by constant folding.
TEST_P(LayoutReinterpretCastTest, RoundTrip_I32ToFloat_ToI32) {
  IRBuilder<> B(BB->getTerminator());
  Type *I32Ty = B.getInt32Ty();
  Type *F32Ty = B.getFloatTy();
  Constant *Orig = ConstantInt::get(I32Ty, 0xDEADBEEFu);
  Value *Mid = B.CreateLayoutReinterpretCast(Orig, F32Ty);
  Value *Back = B.CreateLayoutReinterpretCast(Mid, I32Ty);
  EXPECT_EQ(Back, Orig) << "round-trip i32→float→i32 must be identity";
  checkBB("");
}

TEST_P(LayoutReinterpretCastTest, RoundTrip_StructJoinSplit) {
  IRBuilder<> B(BB->getTerminator());
  Type *I32Ty = B.getInt32Ty();
  StructType *STy = StructType::get(I32Ty, I32Ty);
  Type *I64Ty = B.getInt64Ty();
  Constant *Orig =
      ConstantStruct::get(STy, {ConstantInt::get(I32Ty, 0xAAAAAAAAu),
                                ConstantInt::get(I32Ty, 0x55555555u)});
  Value *Mid = B.CreateLayoutReinterpretCast(Orig, I64Ty);
  Value *Back = B.CreateLayoutReinterpretCast(Mid, STy);
  EXPECT_EQ(Back->getType(), STy);
  checkBB("");
}

// ---------------------------------------------------------------------------
// Pointer struct with addrspacecast
// ---------------------------------------------------------------------------

/// {ptr addrspace(1), i32} → {ptr addrspace(2), i32}
TEST_P(LayoutReinterpretCastTest, PtrStructSameSize) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *PtrAS0 = B.getPtrTy(0);
  Type *PtrAS2 = B.getPtrTy(2); // always 64-bit
  Type *I32Ty = B.getInt32Ty();
  StructType *SrcTy = StructType::get(PtrAS0, I32Ty);
  StructType *DstTy = StructType::get(PtrAS2, I32Ty);
  Value *Src = makeVal(B, SrcTy);
  Value *Result = B.CreateLayoutReinterpretCast(Src, DstTy);
  EXPECT_EQ(Result->getType(), DstTy);
  if (as0PtrBits() == 64) {
    checkBB("  %2 = extractvalue { ptr, i32 } %1, 0\n"
            "  %3 = ptrtoint ptr %2 to i64\n"
            "  %4 = inttoptr i64 %3 to ptr addrspace(2)\n"
            "  %5 = insertvalue { ptr addrspace(2), i32 } poison, ptr "
            "addrspace(2) %4, 0\n"
            "  %6 = extractvalue { ptr, i32 } %1, 1\n"
            "  %7 = insertvalue { ptr addrspace(2), i32 } %5, i32 %6, 1\n");
  } else if (!isBigEndian()) {
    checkBB("  %2 = extractvalue { ptr, i32 } %1, 0\n"
            "  %3 = ptrtoint ptr %2 to i32\n"
            "  %4 = zext i32 %3 to i64\n"
            "  %5 = extractvalue { ptr, i32 } %1, 1\n"
            "  %6 = zext i32 %5 to i64\n"
            "  %7 = shl i64 %6, 32\n"
            "  %8 = or i64 %4, %7\n"
            "  %9 = inttoptr i64 %8 to ptr addrspace(2)\n"
            "  %10 = insertvalue { ptr addrspace(2), i32 } poison, ptr "
            "addrspace(2) %9, 0\n");
  } else {
    checkBB("  %2 = extractvalue { ptr, i32 } %1, 0\n"
            "  %3 = ptrtoint ptr %2 to i32\n"
            "  %4 = zext i32 %3 to i64\n"
            "  %5 = shl i64 %4, 32\n"
            "  %6 = extractvalue { ptr, i32 } %1, 1\n"
            "  %7 = zext i32 %6 to i64\n"
            "  %8 = or i64 %5, %7\n"
            "  %9 = inttoptr i64 %8 to ptr addrspace(2)\n"
            "  %10 = insertvalue { ptr addrspace(2), i32 } poison, ptr "
            "addrspace(2) %9, 0\n");
  }
}

TEST_P(LayoutReinterpretCastTest, Nested) {
  IRBuilder<NoFolder> B(BB->getTerminator());
  Type *I8Ty = B.getInt8Ty();
  Type *I16Ty = ArrayType::get(I8Ty, 2);
  StructType *STyL = StructType::get(
      I16Ty, StructType::get(
                 I16Ty, StructType::get(I16Ty, StructType::get(I16Ty, I8Ty))));
  StructType *STyR = StructType::get(
      StructType::get(StructType::get(StructType::get(I8Ty, I16Ty), I16Ty),
                      I16Ty),
      I16Ty);
  Value *Src = makeVal(B, STyL);
  Value *Result = B.CreateLayoutReinterpretCast(Src, STyR);
  EXPECT_EQ(Result->getType(), STyR);
  BB->setName("LtoR");
  nameBB();
  checkBB(
      "  %LtoR.2 = extractvalue { [2 x i8], { [2 x i8], { [2 x i8], { [2 x "
      "i8], i8 } } } } %LtoR.1, 0\n"
      "  %LtoR.3 = extractvalue [2 x i8] %LtoR.2, 0\n"
      "  %LtoR.4 = insertvalue { i8, [2 x i8] } poison, i8 %LtoR.3, 0\n"
      "  %LtoR.5 = extractvalue [2 x i8] %LtoR.2, 1\n"
      "  %LtoR.6 = insertvalue [2 x i8] poison, i8 %LtoR.5, 0\n"
      "  %LtoR.7 = extractvalue { [2 x i8], { [2 x i8], { [2 x i8], { [2 x "
      "i8], i8 } } } } %LtoR.1, 1\n"
      "  %LtoR.8 = extractvalue { [2 x i8], { [2 x i8], { [2 x i8], i8 } } } "
      "%LtoR.7, 0\n"
      "  %LtoR.9 = extractvalue [2 x i8] %LtoR.8, 0\n"
      "  %LtoR.10 = insertvalue [2 x i8] %LtoR.6, i8 %LtoR.9, 1\n"
      "  %LtoR.11 = extractvalue [2 x i8] %LtoR.8, 1\n"
      "  %LtoR.12 = insertvalue { i8, [2 x i8] } %LtoR.4, [2 x i8] %LtoR.10, "
      "1\n"
      "  %LtoR.13 = insertvalue { { i8, [2 x i8] }, [2 x i8] } poison, { i8, "
      "[2 x i8] } %LtoR.12, 0\n"
      "  %LtoR.14 = insertvalue [2 x i8] poison, i8 %LtoR.11, 0\n"
      "  %LtoR.15 = extractvalue { [2 x i8], { [2 x i8], { [2 x i8], i8 } } } "
      "%LtoR.7, 1\n"
      "  %LtoR.16 = extractvalue { [2 x i8], { [2 x i8], i8 } } %LtoR.15, 0\n"
      "  %LtoR.17 = extractvalue [2 x i8] %LtoR.16, 0\n"
      "  %LtoR.18 = insertvalue [2 x i8] %LtoR.14, i8 %LtoR.17, 1\n"
      "  %LtoR.19 = extractvalue [2 x i8] %LtoR.16, 1\n"
      "  %LtoR.20 = insertvalue { { i8, [2 x i8] }, [2 x i8] } %LtoR.13, [2 x "
      "i8] %LtoR.18, 1\n"
      "  %LtoR.21 = insertvalue { { { i8, [2 x i8] }, [2 x i8] }, [2 x i8] } "
      "poison, { { i8, [2 x i8] }, [2 x i8] } %LtoR.20, 0\n"
      "  %LtoR.22 = insertvalue [2 x i8] poison, i8 %LtoR.19, 0\n"
      "  %LtoR.23 = extractvalue { [2 x i8], { [2 x i8], i8 } } %LtoR.15, 1\n"
      "  %LtoR.24 = extractvalue { [2 x i8], i8 } %LtoR.23, 0\n"
      "  %LtoR.25 = extractvalue [2 x i8] %LtoR.24, 0\n"
      "  %LtoR.26 = insertvalue [2 x i8] %LtoR.22, i8 %LtoR.25, 1\n"
      "  %LtoR.27 = extractvalue [2 x i8] %LtoR.24, 1\n"
      "  %LtoR.28 = insertvalue { { { i8, [2 x i8] }, [2 x i8] }, [2 x i8] } "
      "%LtoR.21, [2 x i8] %LtoR.26, 1\n"
      "  %LtoR.29 = insertvalue { { { { i8, [2 x i8] }, [2 x i8] }, [2 x i8] "
      "}, [2 x i8] } poison, { { { i8, [2 x i8] }, [2 x i8] }, [2 x i8] } "
      "%LtoR.28, 0\n"
      "  %LtoR.30 = insertvalue [2 x i8] poison, i8 %LtoR.27, 0\n"
      "  %LtoR.31 = extractvalue { [2 x i8], i8 } %LtoR.23, 1\n"
      "  %LtoR.32 = insertvalue [2 x i8] %LtoR.30, i8 %LtoR.31, 1\n"
      "  %LtoR.33 = insertvalue { { { { i8, [2 x i8] }, [2 x i8] }, [2 x i8] "
      "}, [2 x i8] } %LtoR.29, [2 x i8] %LtoR.32, 1\n");

  BB = BasicBlock::Create(Ctx, "RtoL", F);
  B.SetInsertPoint(BB);
  Src = makeVal(B, STyR);
  Result = B.CreateLayoutReinterpretCast(Src, STyL);
  EXPECT_EQ(Result->getType(), STyL);
  B.CreateRetVoid();
  nameBB();
  checkBB(
      "  %RtoL.2 = extractvalue { { { { i8, [2 x i8] }, [2 x i8] }, [2 x i8] "
      "}, [2 x i8] } %RtoL.1, 0\n"
      "  %RtoL.3 = extractvalue { { { i8, [2 x i8] }, [2 x i8] }, [2 x i8] } "
      "%RtoL.2, 0\n"
      "  %RtoL.4 = extractvalue { { i8, [2 x i8] }, [2 x i8] } %RtoL.3, 0\n"
      "  %RtoL.5 = extractvalue { i8, [2 x i8] } %RtoL.4, 0\n"
      "  %RtoL.6 = insertvalue [2 x i8] poison, i8 %RtoL.5, 0\n"
      "  %RtoL.7 = extractvalue { i8, [2 x i8] } %RtoL.4, 1\n"
      "  %RtoL.8 = extractvalue [2 x i8] %RtoL.7, 0\n"
      "  %RtoL.9 = insertvalue [2 x i8] %RtoL.6, i8 %RtoL.8, 1\n"
      "  %RtoL.10 = extractvalue [2 x i8] %RtoL.7, 1\n"
      "  %RtoL.11 = insertvalue { [2 x i8], { [2 x i8], { [2 x i8], { [2 x "
      "i8], i8 } } } } poison, [2 x i8] %RtoL.9, 0\n"
      "  %RtoL.12 = insertvalue [2 x i8] poison, i8 %RtoL.10, 0\n"
      "  %RtoL.13 = extractvalue { { i8, [2 x i8] }, [2 x i8] } %RtoL.3, 1\n"
      "  %RtoL.14 = extractvalue [2 x i8] %RtoL.13, 0\n"
      "  %RtoL.15 = insertvalue [2 x i8] %RtoL.12, i8 %RtoL.14, 1\n"
      "  %RtoL.16 = extractvalue [2 x i8] %RtoL.13, 1\n"
      "  %RtoL.17 = insertvalue { [2 x i8], { [2 x i8], { [2 x i8], i8 } } } "
      "poison, [2 x i8] %RtoL.15, 0\n"
      "  %RtoL.18 = insertvalue [2 x i8] poison, i8 %RtoL.16, 0\n"
      "  %RtoL.19 = extractvalue { { { i8, [2 x i8] }, [2 x i8] }, [2 x i8] } "
      "%RtoL.2, 1\n"
      "  %RtoL.20 = extractvalue [2 x i8] %RtoL.19, 0\n"
      "  %RtoL.21 = insertvalue [2 x i8] %RtoL.18, i8 %RtoL.20, 1\n"
      "  %RtoL.22 = extractvalue [2 x i8] %RtoL.19, 1\n"
      "  %RtoL.23 = insertvalue { [2 x i8], { [2 x i8], i8 } } poison, [2 x "
      "i8] %RtoL.21, 0\n"
      "  %RtoL.24 = insertvalue [2 x i8] poison, i8 %RtoL.22, 0\n"
      "  %RtoL.25 = extractvalue { { { { i8, [2 x i8] }, [2 x i8] }, [2 x i8] "
      "}, [2 x i8] } %RtoL.1, 1\n"
      "  %RtoL.26 = extractvalue [2 x i8] %RtoL.25, 0\n"
      "  %RtoL.27 = insertvalue [2 x i8] %RtoL.24, i8 %RtoL.26, 1\n"
      "  %RtoL.28 = extractvalue [2 x i8] %RtoL.25, 1\n"
      "  %RtoL.29 = insertvalue { [2 x i8], i8 } poison, [2 x i8] %RtoL.27, 0\n"
      "  %RtoL.30 = insertvalue { [2 x i8], i8 } %RtoL.29, i8 %RtoL.28, 1\n"
      "  %RtoL.31 = insertvalue { [2 x i8], { [2 x i8], i8 } } %RtoL.23, { [2 "
      "x i8], i8 } %RtoL.30, 1\n"
      "  %RtoL.32 = insertvalue { [2 x i8], { [2 x i8], { [2 x i8], i8 } } } "
      "%RtoL.17, { [2 x i8], { [2 x i8], i8 } } %RtoL.31, 1\n"
      "  %RtoL.33 = insertvalue { [2 x i8], { [2 x i8], { [2 x i8], { [2 x "
      "i8], i8 } } } } %RtoL.11, { [2 x i8], { [2 x i8], { [2 x i8], i8 } } } "
      "%RtoL.32, 1\n");
}

INSTANTIATE_TEST_SUITE_P(
    LayoutVariants, LayoutReinterpretCastTest,
    ::testing::Values(DLConfig{false, 64}, DLConfig{false, 32},
                      DLConfig{true, 64}, DLConfig{true, 32}),
    [](const ::testing::TestParamInfo<DLConfig> &Info) {
      return std::string(Info.param.BigEndian ? "BE" : "LE") +
             std::to_string(Info.param.AS0PtrBits);
    });

} // namespace