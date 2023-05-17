//===- VectorUtilsTest.cpp - VectorUtils tests ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/VectorUtils.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/KnownBits.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class VectorUtilsTest : public testing::Test {
protected:
  void parseAssembly(const char *Assembly) {
    SMDiagnostic Error;
    M = parseAssemblyString(Assembly, Error, Context);

    std::string errMsg;
    raw_string_ostream os(errMsg);
    Error.print("", os);

    // A failure here means that the test itself is buggy.
    if (!M)
      report_fatal_error(Twine(os.str()));

    Function *F = M->getFunction("test");
    if (F == nullptr)
      report_fatal_error("Test must have a function named @test");

    A = nullptr;
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      if (I->hasName()) {
        if (I->getName() == "A")
          A = &*I;
      }
    }
    if (A == nullptr)
      report_fatal_error("@test must have an instruction %A");
  }

  LLVMContext Context;
  std::unique_ptr<Module> M;
  Instruction *A;
};

struct BasicTest : public testing::Test {
  LLVMContext Ctx;
  std::unique_ptr<Module> M;
  Function *F;
  BasicBlock *BB;
  IRBuilder<NoFolder> IRB;

  BasicTest()
      : M(new Module("VectorUtils", Ctx)),
        F(Function::Create(
            FunctionType::get(Type::getVoidTy(Ctx), /* IsVarArg */ false),
            Function::ExternalLinkage, "f", M.get())),
        BB(BasicBlock::Create(Ctx, "entry", F)), IRB(BB) {}
};


} // namespace

TEST_F(BasicTest, isSplat) {
  Value *UndefVec = UndefValue::get(FixedVectorType::get(IRB.getInt8Ty(), 4));
  EXPECT_TRUE(isSplatValue(UndefVec));

  Constant *UndefScalar = UndefValue::get(IRB.getInt8Ty());
  EXPECT_FALSE(isSplatValue(UndefScalar));

  Constant *ScalarC = IRB.getInt8(42);
  EXPECT_FALSE(isSplatValue(ScalarC));

  Constant *OtherScalarC = IRB.getInt8(-42);
  Constant *NonSplatC = ConstantVector::get({ScalarC, OtherScalarC});
  EXPECT_FALSE(isSplatValue(NonSplatC));

  Value *SplatC = IRB.CreateVectorSplat(5, ScalarC);
  EXPECT_TRUE(isSplatValue(SplatC));

  Value *SplatC_SVE =
      IRB.CreateVectorSplat(ElementCount::getScalable(5), ScalarC);
  EXPECT_TRUE(isSplatValue(SplatC_SVE));

  // FIXME: Constant splat analysis does not allow undef elements.
  Constant *SplatWithUndefC = ConstantVector::get({ScalarC, UndefScalar});
  EXPECT_FALSE(isSplatValue(SplatWithUndefC));
}

TEST_F(BasicTest, narrowShuffleMaskElts) {
  SmallVector<int, 16> ScaledMask;
  narrowShuffleMaskElts(1, {3,2,0,-2}, ScaledMask);
  EXPECT_EQ(ArrayRef(ScaledMask), ArrayRef({3, 2, 0, -2}));
  narrowShuffleMaskElts(4, {3,2,0,-1}, ScaledMask);
  EXPECT_EQ(ArrayRef(ScaledMask), ArrayRef({12, 13, 14, 15, 8, 9, 10, 11, 0, 1,
                                            2, 3, -1, -1, -1, -1}));
}

TEST_F(BasicTest, widenShuffleMaskElts) {
  SmallVector<int, 16> WideMask;
  SmallVector<int, 16> NarrowMask;

  // scale == 1 is a copy
  EXPECT_TRUE(widenShuffleMaskElts(1, {3,2,0,-1}, WideMask));
  EXPECT_EQ(ArrayRef(WideMask), ArrayRef({3, 2, 0, -1}));

  // back to original mask
  narrowShuffleMaskElts(1, ArrayRef(WideMask), NarrowMask);
  EXPECT_EQ(ArrayRef(NarrowMask), ArrayRef({3, 2, 0, -1}));

  // can't widen non-consecutive 3/2
  EXPECT_FALSE(widenShuffleMaskElts(2, {3,2,0,-1}, WideMask));

  // can't widen if not evenly divisible
  EXPECT_FALSE(widenShuffleMaskElts(2, {0,1,2}, WideMask));

  // can always widen identity to single element
  EXPECT_TRUE(widenShuffleMaskElts(3, {0,1,2}, WideMask));
  EXPECT_EQ(ArrayRef(WideMask), ArrayRef({0}));

  // back to original mask
  narrowShuffleMaskElts(3, ArrayRef(WideMask), NarrowMask);
  EXPECT_EQ(ArrayRef(NarrowMask), ArrayRef({0, 1, 2}));

  // groups of 4 must be consecutive/undef
  EXPECT_TRUE(widenShuffleMaskElts(4, {12,13,14,15,8,9,10,11,0,1,2,3,-1,-1,-1,-1}, WideMask));
  EXPECT_EQ(ArrayRef(WideMask), ArrayRef({3, 2, 0, -1}));

  // back to original mask
  narrowShuffleMaskElts(4, ArrayRef(WideMask), NarrowMask);
  EXPECT_EQ(ArrayRef(NarrowMask), ArrayRef({12, 13, 14, 15, 8, 9, 10, 11, 0, 1,
                                            2, 3, -1, -1, -1, -1}));

  // groups of 2 must be consecutive/undef
  EXPECT_FALSE(widenShuffleMaskElts(2, {12,12,14,15,8,9,10,11,0,1,2,3,-1,-1,-1,-1}, WideMask));

  // groups of 3 must be consecutive/undef
  EXPECT_TRUE(widenShuffleMaskElts(3, {6,7,8,0,1,2,-1,-1,-1}, WideMask));
  EXPECT_EQ(ArrayRef(WideMask), ArrayRef({2, 0, -1}));

  // back to original mask
  narrowShuffleMaskElts(3, ArrayRef(WideMask), NarrowMask);
  EXPECT_EQ(ArrayRef(NarrowMask), ArrayRef({6, 7, 8, 0, 1, 2, -1, -1, -1}));

  // groups of 3 must be consecutive/undef (partial undefs are not ok)
  EXPECT_FALSE(widenShuffleMaskElts(3, {-1,7,8,0,-1,2,-1,-1,-1}, WideMask));

  // negative indexes must match across a wide element
  EXPECT_FALSE(widenShuffleMaskElts(2, {-1,-2,-1,-1}, WideMask));

  // negative indexes must match across a wide element
  EXPECT_TRUE(widenShuffleMaskElts(2, {-2,-2,-3,-3}, WideMask));
  EXPECT_EQ(ArrayRef(WideMask), ArrayRef({-2, -3}));
}

TEST_F(BasicTest, getShuffleMaskWithWidestElts) {
  SmallVector<int, 16> WideMask;

  // can not widen anything here.
  getShuffleMaskWithWidestElts({3, 2, 0, -1}, WideMask);
  EXPECT_EQ(ArrayRef(WideMask), ArrayRef({3, 2, 0, -1}));

  // can't widen non-consecutive 3/2
  getShuffleMaskWithWidestElts({3, 2, 0, -1}, WideMask);
  EXPECT_EQ(ArrayRef(WideMask), ArrayRef({3, 2, 0, -1}));

  // can always widen identity to single element
  getShuffleMaskWithWidestElts({0, 1, 2}, WideMask);
  EXPECT_EQ(ArrayRef(WideMask), ArrayRef({0}));

  // groups of 4 must be consecutive/undef
  getShuffleMaskWithWidestElts(
      {12, 13, 14, 15, 8, 9, 10, 11, 0, 1, 2, 3, -1, -1, -1, -1}, WideMask);
  EXPECT_EQ(ArrayRef(WideMask), ArrayRef({3, 2, 0, -1}));

  // groups of 2 must be consecutive/undef
  getShuffleMaskWithWidestElts(
      {12, 12, 14, 15, 8, 9, 10, 11, 0, 1, 2, 3, -1, -1, -1, -1}, WideMask);
  EXPECT_EQ(ArrayRef(WideMask), ArrayRef({12, 12, 14, 15, 8, 9, 10, 11, 0, 1, 2,
                                          3, -1, -1, -1, -1}));

  // groups of 3 must be consecutive/undef
  getShuffleMaskWithWidestElts({6, 7, 8, 0, 1, 2, -1, -1, -1}, WideMask);
  EXPECT_EQ(ArrayRef(WideMask), ArrayRef({2, 0, -1}));

  // groups of 3 must be consecutive/undef (partial undefs are not ok)
  getShuffleMaskWithWidestElts({-1, 7, 8, 0, -1, 2, -1, -1, -1}, WideMask);
  EXPECT_EQ(ArrayRef(WideMask), ArrayRef({-1, 7, 8, 0, -1, 2, -1, -1, -1}));

  // negative indexes must match across a wide element
  getShuffleMaskWithWidestElts({-1, -2, -1, -1}, WideMask);
  EXPECT_EQ(ArrayRef(WideMask), ArrayRef({-1, -2, -1, -1}));

  // negative indexes must match across a wide element
  getShuffleMaskWithWidestElts({-2, -2, -3, -3}, WideMask);
  EXPECT_EQ(ArrayRef(WideMask), ArrayRef({-2, -3}));
}

TEST_F(BasicTest, getShuffleDemandedElts) {
  APInt LHS, RHS;

  // broadcast zero
  EXPECT_TRUE(getShuffleDemandedElts(4, {0, 0, 0, 0}, APInt(4,0xf), LHS, RHS));
  EXPECT_EQ(LHS.getZExtValue(), 0x1U);
  EXPECT_EQ(RHS.getZExtValue(), 0x0U);

  // broadcast zero (with non-permitted undefs)
  EXPECT_FALSE(getShuffleDemandedElts(2, {0, -1}, APInt(2, 0x3), LHS, RHS));

  // broadcast zero (with permitted undefs)
  EXPECT_TRUE(getShuffleDemandedElts(3, {0, 0, -1}, APInt(3, 0x7), LHS, RHS, true));
  EXPECT_EQ(LHS.getZExtValue(), 0x1U);
  EXPECT_EQ(RHS.getZExtValue(), 0x0U);

  // broadcast one in demanded
  EXPECT_TRUE(getShuffleDemandedElts(4, {1, 1, 1, -1}, APInt(4, 0x7), LHS, RHS));
  EXPECT_EQ(LHS.getZExtValue(), 0x2U);
  EXPECT_EQ(RHS.getZExtValue(), 0x0U);

  // broadcast 7 in demanded
  EXPECT_TRUE(getShuffleDemandedElts(4, {7, 0, 7, 7}, APInt(4, 0xd), LHS, RHS));
  EXPECT_EQ(LHS.getZExtValue(), 0x0U);
  EXPECT_EQ(RHS.getZExtValue(), 0x8U);

  // general test
  EXPECT_TRUE(getShuffleDemandedElts(4, {4, 2, 7, 3}, APInt(4, 0xf), LHS, RHS));
  EXPECT_EQ(LHS.getZExtValue(), 0xcU);
  EXPECT_EQ(RHS.getZExtValue(), 0x9U);
}

TEST_F(BasicTest, getSplatIndex) {
  EXPECT_EQ(getSplatIndex({0,0,0}), 0);
  EXPECT_EQ(getSplatIndex({1,0,0}), -1);     // no splat
  EXPECT_EQ(getSplatIndex({0,1,1}), -1);     // no splat
  EXPECT_EQ(getSplatIndex({42,42,42}), 42);  // array size is independent of splat index
  EXPECT_EQ(getSplatIndex({42,42,-1}), 42);  // ignore negative
  EXPECT_EQ(getSplatIndex({-1,42,-1}), 42);  // ignore negatives
  EXPECT_EQ(getSplatIndex({-4,42,-42}), 42); // ignore all negatives
  EXPECT_EQ(getSplatIndex({-4,-1,-42}), -1); // all negative values map to -1
}

TEST_F(VectorUtilsTest, isSplatValue_00) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i8> %x) {\n"
      "  %A = shufflevector <2 x i8> %x, <2 x i8> undef, <2 x i32> zeroinitializer\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_TRUE(isSplatValue(A));
}

TEST_F(VectorUtilsTest, isSplatValue_00_index0) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i8> %x) {\n"
      "  %A = shufflevector <2 x i8> %x, <2 x i8> undef, <2 x i32> zeroinitializer\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_TRUE(isSplatValue(A, 0));
}

TEST_F(VectorUtilsTest, isSplatValue_00_index1) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i8> %x) {\n"
      "  %A = shufflevector <2 x i8> %x, <2 x i8> undef, <2 x i32> zeroinitializer\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_FALSE(isSplatValue(A, 1));
}

TEST_F(VectorUtilsTest, isSplatValue_11) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i8> %x) {\n"
      "  %A = shufflevector <2 x i8> %x, <2 x i8> undef, <2 x i32> <i32 1, i32 1>\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_TRUE(isSplatValue(A));
}

TEST_F(VectorUtilsTest, isSplatValue_11_index0) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i8> %x) {\n"
      "  %A = shufflevector <2 x i8> %x, <2 x i8> undef, <2 x i32> <i32 1, i32 1>\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_FALSE(isSplatValue(A, 0));
}

TEST_F(VectorUtilsTest, isSplatValue_11_index1) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i8> %x) {\n"
      "  %A = shufflevector <2 x i8> %x, <2 x i8> undef, <2 x i32> <i32 1, i32 1>\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_TRUE(isSplatValue(A, 1));
}

TEST_F(VectorUtilsTest, isSplatValue_01) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i8> %x) {\n"
      "  %A = shufflevector <2 x i8> %x, <2 x i8> undef, <2 x i32> <i32 0, i32 1>\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_FALSE(isSplatValue(A));
}

TEST_F(VectorUtilsTest, isSplatValue_01_index0) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i8> %x) {\n"
      "  %A = shufflevector <2 x i8> %x, <2 x i8> undef, <2 x i32> <i32 0, i32 1>\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_FALSE(isSplatValue(A, 0));
}

TEST_F(VectorUtilsTest, isSplatValue_01_index1) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i8> %x) {\n"
      "  %A = shufflevector <2 x i8> %x, <2 x i8> undef, <2 x i32> <i32 0, i32 1>\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_FALSE(isSplatValue(A, 1));
}

// FIXME: Allow undef matching with Constant (mask) splat analysis.

TEST_F(VectorUtilsTest, isSplatValue_0u) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i8> %x) {\n"
      "  %A = shufflevector <2 x i8> %x, <2 x i8> undef, <2 x i32> <i32 0, i32 undef>\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_FALSE(isSplatValue(A));
}

// FIXME: Allow undef matching with Constant (mask) splat analysis.

TEST_F(VectorUtilsTest, isSplatValue_0u_index0) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i8> %x) {\n"
      "  %A = shufflevector <2 x i8> %x, <2 x i8> undef, <2 x i32> <i32 0, i32 undef>\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_FALSE(isSplatValue(A, 0));
}

TEST_F(VectorUtilsTest, isSplatValue_0u_index1) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i8> %x) {\n"
      "  %A = shufflevector <2 x i8> %x, <2 x i8> undef, <2 x i32> <i32 0, i32 undef>\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_FALSE(isSplatValue(A, 1));
}

TEST_F(VectorUtilsTest, isSplatValue_Binop) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i8> %x) {\n"
      "  %v0 = shufflevector <2 x i8> %x, <2 x i8> undef, <2 x i32> <i32 0, i32 0>\n"
      "  %v1 = shufflevector <2 x i8> %x, <2 x i8> undef, <2 x i32> <i32 1, i32 1>\n"
      "  %A = udiv <2 x i8> %v0, %v1\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_TRUE(isSplatValue(A));
}

TEST_F(VectorUtilsTest, isSplatValue_Binop_index0) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i8> %x) {\n"
      "  %v0 = shufflevector <2 x i8> %x, <2 x i8> undef, <2 x i32> <i32 0, i32 0>\n"
      "  %v1 = shufflevector <2 x i8> %x, <2 x i8> undef, <2 x i32> <i32 1, i32 1>\n"
      "  %A = udiv <2 x i8> %v0, %v1\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_FALSE(isSplatValue(A, 0));
}

TEST_F(VectorUtilsTest, isSplatValue_Binop_index1) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i8> %x) {\n"
      "  %v0 = shufflevector <2 x i8> %x, <2 x i8> undef, <2 x i32> <i32 0, i32 0>\n"
      "  %v1 = shufflevector <2 x i8> %x, <2 x i8> undef, <2 x i32> <i32 1, i32 1>\n"
      "  %A = udiv <2 x i8> %v0, %v1\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_FALSE(isSplatValue(A, 1));
}

TEST_F(VectorUtilsTest, isSplatValue_Binop_ConstantOp0) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i8> %x) {\n"
      "  %v1 = shufflevector <2 x i8> %x, <2 x i8> undef, <2 x i32> <i32 1, i32 1>\n"
      "  %A = ashr <2 x i8> <i8 42, i8 42>, %v1\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_TRUE(isSplatValue(A));
}

TEST_F(VectorUtilsTest, isSplatValue_Binop_ConstantOp0_index0) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i8> %x) {\n"
      "  %v1 = shufflevector <2 x i8> %x, <2 x i8> undef, <2 x i32> <i32 1, i32 1>\n"
      "  %A = ashr <2 x i8> <i8 42, i8 42>, %v1\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_FALSE(isSplatValue(A, 0));
}

TEST_F(VectorUtilsTest, isSplatValue_Binop_ConstantOp0_index1) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i8> %x) {\n"
      "  %v1 = shufflevector <2 x i8> %x, <2 x i8> undef, <2 x i32> <i32 1, i32 1>\n"
      "  %A = ashr <2 x i8> <i8 42, i8 42>, %v1\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_TRUE(isSplatValue(A, 1));
}

TEST_F(VectorUtilsTest, isSplatValue_Binop_Not_Op0) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i8> %x) {\n"
      "  %v0 = shufflevector <2 x i8> %x, <2 x i8> undef, <2 x i32> <i32 1, i32 0>\n"
      "  %v1 = shufflevector <2 x i8> %x, <2 x i8> undef, <2 x i32> <i32 1, i32 1>\n"
      "  %A = add <2 x i8> %v0, %v1\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_FALSE(isSplatValue(A));
}

TEST_F(VectorUtilsTest, isSplatValue_Binop_Not_Op1) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i8> %x) {\n"
      "  %v0 = shufflevector <2 x i8> %x, <2 x i8> undef, <2 x i32> <i32 1, i32 1>\n"
      "  %v1 = shufflevector <2 x i8> %x, <2 x i8> undef, <2 x i32> <i32 0, i32 1>\n"
      "  %A = shl <2 x i8> %v0, %v1\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_FALSE(isSplatValue(A));
}

TEST_F(VectorUtilsTest, isSplatValue_Select) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i1> %x, <2 x i8> %y, <2 x i8> %z) {\n"
      "  %v0 = shufflevector <2 x i1> %x, <2 x i1> undef, <2 x i32> <i32 1, i32 1>\n"
      "  %v1 = shufflevector <2 x i8> %y, <2 x i8> undef, <2 x i32> <i32 0, i32 0>\n"
      "  %v2 = shufflevector <2 x i8> %z, <2 x i8> undef, <2 x i32> <i32 1, i32 1>\n"
      "  %A = select <2 x i1> %v0, <2 x i8> %v1, <2 x i8> %v2\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_TRUE(isSplatValue(A));
}

TEST_F(VectorUtilsTest, isSplatValue_Select_ConstantOp) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i1> %x, <2 x i8> %y, <2 x i8> %z) {\n"
      "  %v0 = shufflevector <2 x i1> %x, <2 x i1> undef, <2 x i32> <i32 1, i32 1>\n"
      "  %v2 = shufflevector <2 x i8> %z, <2 x i8> undef, <2 x i32> <i32 1, i32 1>\n"
      "  %A = select <2 x i1> %v0, <2 x i8> <i8 42, i8 42>, <2 x i8> %v2\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_TRUE(isSplatValue(A));
}

TEST_F(VectorUtilsTest, isSplatValue_Select_NotCond) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i1> %x, <2 x i8> %y, <2 x i8> %z) {\n"
      "  %v1 = shufflevector <2 x i8> %y, <2 x i8> undef, <2 x i32> <i32 0, i32 0>\n"
      "  %v2 = shufflevector <2 x i8> %z, <2 x i8> undef, <2 x i32> <i32 1, i32 1>\n"
      "  %A = select <2 x i1> %x, <2 x i8> %v1, <2 x i8> %v2\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_FALSE(isSplatValue(A));
}

TEST_F(VectorUtilsTest, isSplatValue_Select_NotOp1) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i1> %x, <2 x i8> %y, <2 x i8> %z) {\n"
      "  %v0 = shufflevector <2 x i1> %x, <2 x i1> undef, <2 x i32> <i32 1, i32 1>\n"
      "  %v2 = shufflevector <2 x i8> %z, <2 x i8> undef, <2 x i32> <i32 1, i32 1>\n"
      "  %A = select <2 x i1> %v0, <2 x i8> %y, <2 x i8> %v2\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_FALSE(isSplatValue(A));
}

TEST_F(VectorUtilsTest, isSplatValue_Select_NotOp2) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i1> %x, <2 x i8> %y, <2 x i8> %z) {\n"
      "  %v0 = shufflevector <2 x i1> %x, <2 x i1> undef, <2 x i32> <i32 1, i32 1>\n"
      "  %v1 = shufflevector <2 x i8> %y, <2 x i8> undef, <2 x i32> <i32 0, i32 0>\n"
      "  %A = select <2 x i1> %v0, <2 x i8> %v1, <2 x i8> %z\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_FALSE(isSplatValue(A));
}

TEST_F(VectorUtilsTest, isSplatValue_SelectBinop) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i1> %x, <2 x i8> %y, <2 x i8> %z) {\n"
      "  %v0 = shufflevector <2 x i1> %x, <2 x i1> undef, <2 x i32> <i32 1, i32 1>\n"
      "  %v1 = shufflevector <2 x i8> %y, <2 x i8> undef, <2 x i32> <i32 0, i32 0>\n"
      "  %v2 = shufflevector <2 x i8> %z, <2 x i8> undef, <2 x i32> <i32 1, i32 1>\n"
      "  %bo = xor <2 x i8> %v1, %v2\n"
      "  %A = select <2 x i1> %v0, <2 x i8> %bo, <2 x i8> %v2\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_TRUE(isSplatValue(A));
}

TEST_F(VectorUtilsTest, getSplatValueElt0) {
  parseAssembly(
      "define <2 x i8> @test(i8 %x) {\n"
      "  %ins = insertelement <2 x i8> undef, i8 %x, i32 0\n"
      "  %A = shufflevector <2 x i8> %ins, <2 x i8> undef, <2 x i32> zeroinitializer\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_EQ(getSplatValue(A)->getName(), "x");
}

TEST_F(VectorUtilsTest, getSplatValueEltMismatch) {
  parseAssembly(
      "define <2 x i8> @test(i8 %x) {\n"
      "  %ins = insertelement <2 x i8> undef, i8 %x, i32 1\n"
      "  %A = shufflevector <2 x i8> %ins, <2 x i8> undef, <2 x i32> zeroinitializer\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_EQ(getSplatValue(A), nullptr);
}

// TODO: This is a splat, but we don't recognize it.

TEST_F(VectorUtilsTest, getSplatValueElt1) {
  parseAssembly(
      "define <2 x i8> @test(i8 %x) {\n"
      "  %ins = insertelement <2 x i8> undef, i8 %x, i32 1\n"
      "  %A = shufflevector <2 x i8> %ins, <2 x i8> undef, <2 x i32> <i32 1, i32 1>\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_EQ(getSplatValue(A), nullptr);
}

////////////////////////////////////////////////////////////////////////////////
// VFShape API tests.
////////////////////////////////////////////////////////////////////////////////

class VFShapeAPITest : public testing::Test {
protected:
  void SetUp() override {
    M = parseAssemblyString(IR, Err, Ctx);
    // Get the only call instruction in the block, which is the first
    // instruction.
    CI = dyn_cast<CallInst>(&*(instructions(M->getFunction("f")).begin()));
  }

  const char *IR = "define i32 @f(i32 %a, i64 %b, double %c) {\n"
                   " %1 = call i32 @g(i32 %a, i64 %b, double %c)\n"
                   "  ret i32 %1\n"
                   "}\n"
                   "declare i32 @g(i32, i64, double)\n";
  LLVMContext Ctx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M;
  CallInst *CI;
  // Dummy shape with no parameters, overwritten by buildShape when invoked.
  VFShape Shape = {/*VF*/ ElementCount::getFixed(2), /*Parameters*/ {}};
  VFShape Expected;
  SmallVector<VFParameter, 8> &ExpectedParams = Expected.Parameters;

  void buildShape(ElementCount VF, bool HasGlobalPred) {
    Shape = VFShape::get(*CI, VF, HasGlobalPred);
  }

  bool validParams(ArrayRef<VFParameter> Parameters) {
    Shape.Parameters =
        SmallVector<VFParameter, 8>(Parameters.begin(), Parameters.end());
    return Shape.hasValidParameterList();
  }
};

TEST_F(VFShapeAPITest, API_buildVFShape) {
  buildShape(/*VF*/ ElementCount::getFixed(2), /*HasGlobalPred*/ false);
  Expected = {/*VF*/ ElementCount::getFixed(2), /*Parameters*/ {
                  {0, VFParamKind::Vector},
                  {1, VFParamKind::Vector},
                  {2, VFParamKind::Vector},
              }};
  EXPECT_EQ(Shape, Expected);

  buildShape(/*VF*/ ElementCount::getFixed(4), /*HasGlobalPred*/ true);
  Expected = {/*VF*/ ElementCount::getFixed(4), /*Parameters*/ {
                  {0, VFParamKind::Vector},
                  {1, VFParamKind::Vector},
                  {2, VFParamKind::Vector},
                  {3, VFParamKind::GlobalPredicate},
              }};
  EXPECT_EQ(Shape, Expected);

  buildShape(/*VF*/ ElementCount::getScalable(16), /*HasGlobalPred*/ false);
  Expected = {/*VF*/ ElementCount::getScalable(16), /*Parameters*/ {
                  {0, VFParamKind::Vector},
                  {1, VFParamKind::Vector},
                  {2, VFParamKind::Vector},
              }};
  EXPECT_EQ(Shape, Expected);
}

TEST_F(VFShapeAPITest, API_getScalarShape) {
  buildShape(/*VF*/ ElementCount::getFixed(1), /*HasGlobalPred*/ false);
  EXPECT_EQ(VFShape::getScalarShape(*CI), Shape);
}

TEST_F(VFShapeAPITest, API_getVectorizedFunction) {
  VFShape ScalarShape = VFShape::getScalarShape(*CI);
  EXPECT_EQ(VFDatabase(*CI).getVectorizedFunction(ScalarShape),
            M->getFunction("g"));

  buildShape(/*VF*/ ElementCount::getScalable(1), /*HasGlobalPred*/ false);
  EXPECT_EQ(VFDatabase(*CI).getVectorizedFunction(Shape), nullptr);
  buildShape(/*VF*/ ElementCount::getFixed(1), /*HasGlobalPred*/ true);
  EXPECT_EQ(VFDatabase(*CI).getVectorizedFunction(Shape), nullptr);
  buildShape(/*VF*/ ElementCount::getScalable(1), /*HasGlobalPred*/ true);
  EXPECT_EQ(VFDatabase(*CI).getVectorizedFunction(Shape), nullptr);
}

TEST_F(VFShapeAPITest, API_updateVFShape) {

  buildShape(/*VF*/ ElementCount::getFixed(2), /*HasGlobalPred*/ false);
  Shape.updateParam({0 /*Pos*/, VFParamKind::OMP_Linear, 1, Align(4)});
  Expected = {/*VF*/ ElementCount::getFixed(2), /*Parameters*/ {
                  {0, VFParamKind::OMP_Linear, 1, Align(4)},
                  {1, VFParamKind::Vector},
                  {2, VFParamKind::Vector},
              }};
  EXPECT_EQ(Shape, Expected);

  // From this point on, we update only the parameters of the VFShape,
  // so we update only the reference of the expected Parameters.
  Shape.updateParam({1 /*Pos*/, VFParamKind::OMP_Uniform});
  ExpectedParams = {
      {0, VFParamKind::OMP_Linear, 1, Align(4)},
      {1, VFParamKind::OMP_Uniform},
      {2, VFParamKind::Vector},
  };
  EXPECT_EQ(Shape, Expected);

  Shape.updateParam({2 /*Pos*/, VFParamKind::OMP_LinearRefPos, 1});
  ExpectedParams = {
      {0, VFParamKind::OMP_Linear, 1, Align(4)},
      {1, VFParamKind::OMP_Uniform},
      {2, VFParamKind::OMP_LinearRefPos, 1},
  };
  EXPECT_EQ(Shape, Expected);
}

TEST_F(VFShapeAPITest, API_updateVFShape_GlobalPredicate) {

  buildShape(/*VF*/ ElementCount::getScalable(2), /*HasGlobalPred*/ true);
  Shape.updateParam({1 /*Pos*/, VFParamKind::OMP_Uniform});
  Expected = {/*VF*/ ElementCount::getScalable(2),
              /*Parameters*/ {{0, VFParamKind::Vector},
                              {1, VFParamKind::OMP_Uniform},
                              {2, VFParamKind::Vector},
                              {3, VFParamKind::GlobalPredicate}}};
  EXPECT_EQ(Shape, Expected);
}

TEST_F(VFShapeAPITest, Parameters_Valid) {
  // ParamPos in order.
  EXPECT_TRUE(validParams({{0, VFParamKind::Vector}}));
  EXPECT_TRUE(
      validParams({{0, VFParamKind::Vector}, {1, VFParamKind::Vector}}));
  EXPECT_TRUE(validParams({{0, VFParamKind::Vector},
                           {1, VFParamKind::Vector},
                           {2, VFParamKind::Vector}}));

  // GlocalPredicate is unique.
  EXPECT_TRUE(validParams({{0, VFParamKind::Vector},
                           {1, VFParamKind::Vector},
                           {2, VFParamKind::Vector},
                           {3, VFParamKind::GlobalPredicate}}));

  EXPECT_TRUE(validParams({{0, VFParamKind::Vector},
                           {1, VFParamKind::GlobalPredicate},
                           {2, VFParamKind::Vector}}));
}

TEST_F(VFShapeAPITest, Parameters_ValidOpenMPLinear) {
// Valid linear constant step (>0).
#define __BUILD_PARAMETERS(Kind, Val)                                          \
  {                                                                            \
    { 0, Kind, Val }                                                           \
  }
  EXPECT_TRUE(validParams(__BUILD_PARAMETERS(VFParamKind::OMP_Linear, 1)));
  EXPECT_TRUE(validParams(__BUILD_PARAMETERS(VFParamKind::OMP_LinearRef, 2)));
  EXPECT_TRUE(validParams(__BUILD_PARAMETERS(VFParamKind::OMP_LinearVal, 4)));
  EXPECT_TRUE(validParams(__BUILD_PARAMETERS(VFParamKind::OMP_LinearUVal, 33)));
#undef __BUILD_PARAMETERS

// Valid linear runtime step (the step parameter is marked uniform).
#define __BUILD_PARAMETERS(Kind)                                               \
  {                                                                            \
    {0, VFParamKind::OMP_Uniform}, {1, VFParamKind::Vector}, { 2, Kind, 0 }    \
  }
  EXPECT_TRUE(validParams(__BUILD_PARAMETERS(VFParamKind::OMP_LinearPos)));
  EXPECT_TRUE(validParams(__BUILD_PARAMETERS(VFParamKind::OMP_LinearRefPos)));
  EXPECT_TRUE(validParams(__BUILD_PARAMETERS(VFParamKind::OMP_LinearValPos)));
  EXPECT_TRUE(validParams(__BUILD_PARAMETERS(VFParamKind::OMP_LinearUValPos)));
#undef __BUILD_PARAMETERS
}

TEST_F(VFShapeAPITest, Parameters_Invalid) {
#ifndef NDEBUG
  // Wrong order is checked by an assertion: make sure that the
  // assertion is not removed.
  EXPECT_DEATH(validParams({{1, VFParamKind::Vector}}),
               "Broken parameter list.");
  EXPECT_DEATH(
      validParams({{1, VFParamKind::Vector}, {0, VFParamKind::Vector}}),
      "Broken parameter list.");
#endif

  // GlobalPredicate is not unique
  EXPECT_FALSE(validParams({{0, VFParamKind::Vector},
                            {1, VFParamKind::GlobalPredicate},
                            {2, VFParamKind::GlobalPredicate}}));
  EXPECT_FALSE(validParams({{0, VFParamKind::GlobalPredicate},
                            {1, VFParamKind::Vector},
                            {2, VFParamKind::GlobalPredicate}}));
}

TEST_F(VFShapeAPITest, Parameters_InvalidOpenMPLinear) {
// Compile time linear steps must be non-zero (compile time invariant).
#define __BUILD_PARAMETERS(Kind)                                               \
  {                                                                            \
    { 0, Kind, 0 }                                                             \
  }
  EXPECT_FALSE(validParams(__BUILD_PARAMETERS(VFParamKind::OMP_Linear)));
  EXPECT_FALSE(validParams(__BUILD_PARAMETERS(VFParamKind::OMP_LinearRef)));
  EXPECT_FALSE(validParams(__BUILD_PARAMETERS(VFParamKind::OMP_LinearVal)));
  EXPECT_FALSE(validParams(__BUILD_PARAMETERS(VFParamKind::OMP_LinearUVal)));
#undef __BUILD_PARAMETERS

// The step of a runtime linear parameter must be marked
// as uniform (runtime invariant).
#define __BUILD_PARAMETERS(Kind)                                               \
  {                                                                            \
    {0, VFParamKind::OMP_Uniform}, {1, VFParamKind::Vector}, { 2, Kind, 1 }    \
  }
  EXPECT_FALSE(validParams(__BUILD_PARAMETERS(VFParamKind::OMP_LinearPos)));
  EXPECT_FALSE(validParams(__BUILD_PARAMETERS(VFParamKind::OMP_LinearRefPos)));
  EXPECT_FALSE(validParams(__BUILD_PARAMETERS(VFParamKind::OMP_LinearValPos)));
  EXPECT_FALSE(validParams(__BUILD_PARAMETERS(VFParamKind::OMP_LinearUValPos)));
#undef __BUILD_PARAMETERS

// The linear step parameter can't point at itself.
#define __BUILD_PARAMETERS(Kind)                                               \
  {                                                                            \
    {0, VFParamKind::Vector}, {1, VFParamKind::Vector}, { 2, Kind, 2 }         \
  }
  EXPECT_FALSE(validParams(__BUILD_PARAMETERS(VFParamKind::OMP_LinearPos)));
  EXPECT_FALSE(validParams(__BUILD_PARAMETERS(VFParamKind::OMP_LinearRefPos)));
  EXPECT_FALSE(validParams(__BUILD_PARAMETERS(VFParamKind::OMP_LinearValPos)));
  EXPECT_FALSE(validParams(__BUILD_PARAMETERS(VFParamKind::OMP_LinearUValPos)));
#undef __BUILD_PARAMETERS

// Linear parameter (runtime) is out of range.
#define __BUILD_PARAMETERS(Kind)                                               \
  {                                                                            \
    {0, VFParamKind::Vector}, {1, VFParamKind::Vector}, { 2, Kind, 3 }         \
  }
  EXPECT_FALSE(validParams(__BUILD_PARAMETERS(VFParamKind::OMP_LinearPos)));
  EXPECT_FALSE(validParams(__BUILD_PARAMETERS(VFParamKind::OMP_LinearRefPos)));
  EXPECT_FALSE(validParams(__BUILD_PARAMETERS(VFParamKind::OMP_LinearValPos)));
  EXPECT_FALSE(validParams(__BUILD_PARAMETERS(VFParamKind::OMP_LinearUValPos)));
#undef __BUILD_PARAMETERS
}
