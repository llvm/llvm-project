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
      report_fatal_error(os.str());

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
  Value *UndefVec = UndefValue::get(VectorType::get(IRB.getInt8Ty(), 4));
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

  // FIXME: Constant splat analysis does not allow undef elements.
  Constant *SplatWithUndefC = ConstantVector::get({ScalarC, UndefScalar});
  EXPECT_FALSE(isSplatValue(SplatWithUndefC));
}

TEST_F(VectorUtilsTest, isSplatValue_00) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i8> %x) {\n"
      "  %A = shufflevector <2 x i8> %x, <2 x i8> undef, <2 x i32> zeroinitializer\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_TRUE(isSplatValue(A));
}

TEST_F(VectorUtilsTest, isSplatValue_11) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i8> %x) {\n"
      "  %A = shufflevector <2 x i8> %x, <2 x i8> undef, <2 x i32> <i32 1, i32 1>\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_TRUE(isSplatValue(A));
}

TEST_F(VectorUtilsTest, isSplatValue_01) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i8> %x) {\n"
      "  %A = shufflevector <2 x i8> %x, <2 x i8> undef, <2 x i32> <i32 0, i32 1>\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_FALSE(isSplatValue(A));
}

// FIXME: Constant (mask) splat analysis does not allow undef elements.

TEST_F(VectorUtilsTest, isSplatValue_0u) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i8> %x) {\n"
      "  %A = shufflevector <2 x i8> %x, <2 x i8> undef, <2 x i32> <i32 0, i32 undef>\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_FALSE(isSplatValue(A));
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

TEST_F(VectorUtilsTest, isSplatValue_Binop_ConstantOp0) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i8> %x) {\n"
      "  %v1 = shufflevector <2 x i8> %x, <2 x i8> undef, <2 x i32> <i32 1, i32 1>\n"
      "  %A = ashr <2 x i8> <i8 42, i8 42>, %v1\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_TRUE(isSplatValue(A));
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
