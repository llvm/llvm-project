//===- TypesTest.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/SandboxIR/SandboxIR.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

struct SandboxTypeTest : public testing::Test {
  LLVMContext C;
  std::unique_ptr<Module> M;

  void parseIR(LLVMContext &C, const char *IR) {
    SMDiagnostic Err;
    M = parseAssemblyString(IR, Err, C);
    if (!M)
      Err.print("SandboxTypeTest", errs());
  }
  BasicBlock *getBasicBlockByName(Function &F, StringRef Name) {
    for (BasicBlock &BB : F)
      if (BB.getName() == Name)
        return &BB;
    llvm_unreachable("Expected to find basic block!");
  }
};

TEST_F(SandboxTypeTest, Type) {
  parseIR(C, R"IR(
define void @foo(i32 %v0) {
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  sandboxir::Type *I32Ty = F->getArg(0)->getType();

  auto *LLVMInt32Ty = llvm::Type::getInt32Ty(C);
  auto *LLVMFloatTy = llvm::Type::getFloatTy(C);
  auto *LLVMInt8Ty = llvm::Type::getInt8Ty(C);

  auto *Int32Ty = Ctx.getType(LLVMInt32Ty);
  auto *FloatTy = Ctx.getType(LLVMFloatTy);

  // Check print().
  std::string Buff1;
  raw_string_ostream BS1(Buff1);
  Int32Ty->print(BS1, /*IsForDebug=*/true, /*NoDetails=*/false);
  std::string Buff2;
  raw_string_ostream BS2(Buff2);
  LLVMInt32Ty->print(BS2, /*IsForDebug=*/true, /*NoDetails=*/false);
  EXPECT_EQ(Buff1, Buff2);

  // Check getContext().
  EXPECT_EQ(&I32Ty->getContext(), &Ctx);
  // Check that Ctx.getType(nullptr) == nullptr.
  EXPECT_EQ(Ctx.getType(nullptr), nullptr);

#define CHK(LLVMCreate, SBCheck)                                               \
  Ctx.getType(llvm::Type::LLVMCreate(C))->SBCheck()
  // Check isVoidTy().
  EXPECT_TRUE(Ctx.getType(llvm::Type::getVoidTy(C))->isVoidTy());
  EXPECT_TRUE(CHK(getVoidTy, isVoidTy));
  // Check isHalfTy().
  EXPECT_TRUE(CHK(getHalfTy, isHalfTy));
  // Check isBFloatTy().
  EXPECT_TRUE(CHK(getBFloatTy, isBFloatTy));
  // Check is16bitFPTy().
  EXPECT_TRUE(CHK(getHalfTy, is16bitFPTy));
  // Check isFloatTy().
  EXPECT_TRUE(CHK(getFloatTy, isFloatTy));
  // Check isDoubleTy().
  EXPECT_TRUE(CHK(getDoubleTy, isDoubleTy));
  // Check isX86_FP80Ty().
  EXPECT_TRUE(CHK(getX86_FP80Ty, isX86_FP80Ty));
  // Check isFP128Ty().
  EXPECT_TRUE(CHK(getFP128Ty, isFP128Ty));
  // Check isPPC_FP128Ty().
  EXPECT_TRUE(CHK(getPPC_FP128Ty, isPPC_FP128Ty));
  // Check isIEEELikeFPTy().
  EXPECT_TRUE(CHK(getFloatTy, isIEEELikeFPTy));
  // Check isFloatingPointTy().
  EXPECT_TRUE(CHK(getFloatTy, isFloatingPointTy));
  EXPECT_TRUE(CHK(getDoubleTy, isFloatingPointTy));
  // Check isMultiUnitFPType().
  EXPECT_TRUE(CHK(getPPC_FP128Ty, isMultiUnitFPType));
  EXPECT_FALSE(CHK(getFloatTy, isMultiUnitFPType));
  // Check getFltSemantics().
  EXPECT_EQ(&sandboxir::Type::getFloatTy(Ctx)->getFltSemantics(),
            &llvm::Type::getFloatTy(C)->getFltSemantics());
  // Check isX86_AMXTy().
  EXPECT_TRUE(CHK(getX86_AMXTy, isX86_AMXTy));
  // Check isTargetExtTy().
  EXPECT_TRUE(Ctx.getType(llvm::TargetExtType::get(C, "foo"))->isTargetExtTy());
  // Check isScalableTargetExtTy().
  EXPECT_TRUE(Ctx.getType(llvm::TargetExtType::get(C, "aarch64.svcount"))
                  ->isScalableTargetExtTy());
  // Check isScalableTy().
  EXPECT_TRUE(Ctx.getType(llvm::ScalableVectorType::get(LLVMInt32Ty, 2u))
                  ->isScalableTy());
  // Check isFPOrFPVectorTy().
  EXPECT_TRUE(CHK(getFloatTy, isFPOrFPVectorTy));
  EXPECT_FALSE(CHK(getInt32Ty, isFPOrFPVectorTy));
  // Check isLabelTy().
  EXPECT_TRUE(CHK(getLabelTy, isLabelTy));
  // Check isMetadataTy().
  EXPECT_TRUE(CHK(getMetadataTy, isMetadataTy));
  // Check isTokenTy().
  EXPECT_TRUE(CHK(getTokenTy, isTokenTy));
  // Check isIntegerTy().
  EXPECT_TRUE(CHK(getInt32Ty, isIntegerTy));
  EXPECT_FALSE(CHK(getFloatTy, isIntegerTy));
  // Check isIntegerTy(Bitwidth).
  EXPECT_TRUE(LLVMInt32Ty->isIntegerTy(32u));
  EXPECT_FALSE(LLVMInt32Ty->isIntegerTy(31u));
  EXPECT_FALSE(Ctx.getType(llvm::Type::getFloatTy(C))->isIntegerTy(32u));
  // Check isIntOrIntVectorTy().
  EXPECT_TRUE(LLVMInt32Ty->isIntOrIntVectorTy());
  EXPECT_TRUE(Ctx.getType(llvm::FixedVectorType::get(LLVMInt32Ty, 8))
                  ->isIntOrIntVectorTy());
  EXPECT_FALSE(Ctx.getType(LLVMFloatTy)->isIntOrIntVectorTy());
  EXPECT_FALSE(Ctx.getType(llvm::FixedVectorType::get(LLVMFloatTy, 8))
                   ->isIntOrIntVectorTy());
  // Check isIntOrPtrTy().
  EXPECT_TRUE(Int32Ty->isIntOrPtrTy());
  EXPECT_TRUE(Ctx.getType(llvm::PointerType::get(C, 0u))->isIntOrPtrTy());
  EXPECT_FALSE(FloatTy->isIntOrPtrTy());
  // Check isFunctionTy().
  EXPECT_TRUE(Ctx.getType(llvm::FunctionType::get(LLVMInt32Ty, {}, false))
                  ->isFunctionTy());
  // Check isStructTy().
  EXPECT_TRUE(Ctx.getType(llvm::StructType::get(C))->isStructTy());
  // Check isArrayTy().
  EXPECT_TRUE(Ctx.getType(llvm::ArrayType::get(LLVMInt32Ty, 10))->isArrayTy());
  // Check isPointerTy().
  EXPECT_TRUE(Ctx.getType(llvm::PointerType::get(C, 0u))->isPointerTy());
  // Check isPtrOrPtrVectroTy().
  EXPECT_TRUE(
      Ctx.getType(llvm::FixedVectorType::get(llvm::PointerType::get(C, 0u), 8u))
          ->isPtrOrPtrVectorTy());
  // Check isVectorTy().
  EXPECT_TRUE(
      Ctx.getType(llvm::FixedVectorType::get(LLVMInt32Ty, 8u))->isVectorTy());
  // Check canLosslesslyBitCastTo().
  auto *VecTy32x4 = Ctx.getType(llvm::FixedVectorType::get(LLVMInt32Ty, 4u));
  auto *VecTy32x2 = Ctx.getType(llvm::FixedVectorType::get(LLVMInt32Ty, 2u));
  auto *VecTy8x16 = Ctx.getType(llvm::FixedVectorType::get(LLVMInt8Ty, 16u));
  EXPECT_TRUE(VecTy32x4->canLosslesslyBitCastTo(VecTy8x16));
  EXPECT_FALSE(VecTy32x4->canLosslesslyBitCastTo(VecTy32x2));
  // Check isEmptyTy().
  EXPECT_TRUE(Ctx.getType(llvm::StructType::get(C))->isEmptyTy());
  // Check isFirstClassType().
  EXPECT_TRUE(Int32Ty->isFirstClassType());
  // Check isSingleValueType().
  EXPECT_TRUE(Int32Ty->isSingleValueType());
  // Check isAggregateType().
  EXPECT_FALSE(Int32Ty->isAggregateType());
  // Check isSized().
  SmallPtrSet<sandboxir::Type *, 1> Visited;
  EXPECT_TRUE(Int32Ty->isSized(&Visited));
  // Check getPrimitiveSizeInBits().
  EXPECT_EQ(VecTy32x2->getPrimitiveSizeInBits(), 32u * 2);
  // Check getScalarSizeInBits().
  EXPECT_EQ(VecTy32x2->getScalarSizeInBits(), 32u);
  // Check getFPMantissaWidth().
  EXPECT_EQ(FloatTy->getFPMantissaWidth(), LLVMFloatTy->getFPMantissaWidth());
  // Check isIEEE().
  EXPECT_EQ(FloatTy->isIEEE(), LLVMFloatTy->isIEEE());
  // Check getScalarType().
  EXPECT_EQ(
      Ctx.getType(llvm::FixedVectorType::get(LLVMInt32Ty, 8u))->getScalarType(),
      Int32Ty);

#define CHK_GET(TY)                                                            \
  EXPECT_EQ(Ctx.getType(llvm::Type::get##TY##Ty(C)),                           \
            sandboxir::Type::get##TY##Ty(Ctx))
  // Check getInt64Ty().
  CHK_GET(Int64);
  // Check getInt32Ty().
  CHK_GET(Int32);
  // Check getInt16Ty().
  CHK_GET(Int16);
  // Check getInt8Ty().
  CHK_GET(Int8);
  // Check getInt1Ty().
  CHK_GET(Int1);
  // Check getDoubleTy().
  CHK_GET(Double);
  // Check getFloatTy().
  CHK_GET(Float);
}

TEST_F(SandboxTypeTest, PointerType) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  // Check classof(), creation.
  auto *PtrTy = cast<sandboxir::PointerType>(F->getArg(0)->getType());
  // Check get(ElementType, AddressSpace).
  auto *NewPtrTy =
      sandboxir::PointerType::get(sandboxir::Type::getInt32Ty(Ctx), 0u);
  EXPECT_EQ(NewPtrTy, PtrTy);
  // Check get(Ctx, AddressSpace).
  auto *NewPtrTy2 = sandboxir::PointerType::get(Ctx, 0u);
  EXPECT_EQ(NewPtrTy2, PtrTy);
}

TEST_F(SandboxTypeTest, ArrayType) {
  parseIR(C, R"IR(
define void @foo([2 x i8] %v0) {
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  // Check classof(), creation.
  [[maybe_unused]] auto *ArrayTy =
      cast<sandboxir::ArrayType>(F->getArg(0)->getType());
  // Check get().
  auto *NewArrayTy =
      sandboxir::ArrayType::get(sandboxir::Type::getInt8Ty(Ctx), 2u);
  EXPECT_EQ(NewArrayTy, ArrayTy);
}

TEST_F(SandboxTypeTest, StructType) {
  parseIR(C, R"IR(
define void @foo({i32, i8} %v0) {
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *Int32Ty = sandboxir::Type::getInt32Ty(Ctx);
  auto *Int8Ty = sandboxir::Type::getInt8Ty(Ctx);
  // Check classof(), creation.
  [[maybe_unused]] auto *StructTy =
      cast<sandboxir::StructType>(F->getArg(0)->getType());
  // Check get().
  auto *NewStructTy = sandboxir::StructType::get(Ctx, {Int32Ty, Int8Ty});
  EXPECT_EQ(NewStructTy, StructTy);
  // Check get(Packed).
  auto *NewStructTyPacked =
      sandboxir::StructType::get(Ctx, {Int32Ty, Int8Ty}, /*Packed=*/true);
  EXPECT_NE(NewStructTyPacked, StructTy);
  EXPECT_TRUE(NewStructTyPacked->isPacked());
}

TEST_F(SandboxTypeTest, VectorType) {
  parseIR(C, R"IR(
define void @foo(<4 x i16> %vi0, <4 x float> %vf1, i8 %i0) {
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  // Check classof(), creation, accessors
  auto *VecTy = cast<sandboxir::VectorType>(F->getArg(0)->getType());
  EXPECT_TRUE(VecTy->getElementType()->isIntegerTy(16));
  EXPECT_EQ(VecTy->getElementCount(), ElementCount::getFixed(4));

  // get(ElementType, NumElements, Scalable)
  EXPECT_EQ(sandboxir::VectorType::get(sandboxir::Type::getInt16Ty(Ctx), 4,
                                       /*Scalable=*/false),
            F->getArg(0)->getType());
  // get(ElementType, Other)
  EXPECT_EQ(sandboxir::VectorType::get(
                sandboxir::Type::getInt16Ty(Ctx),
                cast<sandboxir::VectorType>(F->getArg(0)->getType())),
            F->getArg(0)->getType());
  auto *FVecTy = cast<sandboxir::VectorType>(F->getArg(1)->getType());
  EXPECT_TRUE(FVecTy->getElementType()->isFloatTy());
  // getInteger
  auto *IVecTy = sandboxir::VectorType::getInteger(FVecTy);
  EXPECT_TRUE(IVecTy->getElementType()->isIntegerTy(32));
  EXPECT_EQ(IVecTy->getElementCount(), FVecTy->getElementCount());
  // getExtendedElementCountVectorType
  auto *ExtVecTy = sandboxir::VectorType::getExtendedElementVectorType(IVecTy);
  EXPECT_TRUE(ExtVecTy->getElementType()->isIntegerTy(64));
  EXPECT_EQ(ExtVecTy->getElementCount(), VecTy->getElementCount());
  // getTruncatedElementVectorType
  auto *TruncVecTy =
      sandboxir::VectorType::getTruncatedElementVectorType(IVecTy);
  EXPECT_TRUE(TruncVecTy->getElementType()->isIntegerTy(16));
  EXPECT_EQ(TruncVecTy->getElementCount(), VecTy->getElementCount());
  // getSubdividedVectorType
  auto *SubVecTy = sandboxir::VectorType::getSubdividedVectorType(VecTy, 1);
  EXPECT_TRUE(SubVecTy->getElementType()->isIntegerTy(8));
  EXPECT_EQ(SubVecTy->getElementCount(), ElementCount::getFixed(8));
  // getHalfElementsVectorType
  auto *HalfVecTy = sandboxir::VectorType::getHalfElementsVectorType(VecTy);
  EXPECT_TRUE(HalfVecTy->getElementType()->isIntegerTy(16));
  EXPECT_EQ(HalfVecTy->getElementCount(), ElementCount::getFixed(2));
  // getDoubleElementsVectorType
  auto *DoubleVecTy = sandboxir::VectorType::getDoubleElementsVectorType(VecTy);
  EXPECT_TRUE(DoubleVecTy->getElementType()->isIntegerTy(16));
  EXPECT_EQ(DoubleVecTy->getElementCount(), ElementCount::getFixed(8));
  // isValidElementType
  auto *I8Type = F->getArg(2)->getType();
  EXPECT_TRUE(I8Type->isIntegerTy());
  EXPECT_TRUE(sandboxir::VectorType::isValidElementType(I8Type));
  EXPECT_FALSE(sandboxir::VectorType::isValidElementType(FVecTy));
}

TEST_F(SandboxTypeTest, FunctionType) {
  parseIR(C, R"IR(
define void @foo() {
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  // Check classof(), creation.
  [[maybe_unused]] auto *FTy =
      cast<sandboxir::FunctionType>(F->getFunctionType());
}

TEST_F(SandboxTypeTest, IntegerType) {
  parseIR(C, R"IR(
define void @foo(i32 %v0) {
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  // Check classof(), creation.
  auto *Int32Ty = cast<sandboxir::IntegerType>(F->getArg(0)->getType());
  // Check get().
  auto *NewInt32Ty = sandboxir::IntegerType::get(Ctx, 32u);
  EXPECT_EQ(NewInt32Ty, Int32Ty);
}
