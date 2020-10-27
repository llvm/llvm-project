//===- ValueTrackingTest.cpp - ValueTracking tests ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

static Instruction &findInstructionByName(Function *F, StringRef Name) {
  for (Instruction &I : instructions(F))
    if (I.getName() == Name)
      return I;

  llvm_unreachable("Expected value not found");
}

class ValueTrackingTest : public testing::Test {
protected:
  std::unique_ptr<Module> parseModule(StringRef Assembly) {
    SMDiagnostic Error;
    std::unique_ptr<Module> M = parseAssemblyString(Assembly, Error, Context);

    std::string errMsg;
    raw_string_ostream os(errMsg);
    Error.print("", os);
    EXPECT_TRUE(M) << os.str();

    return M;
  }

  void parseAssembly(StringRef Assembly) {
    M = parseModule(Assembly);
    ASSERT_TRUE(M);

    F = M->getFunction("test");
    ASSERT_TRUE(F) << "Test must have a function @test";
    if (!F)
      return;

    A = &findInstructionByName(F, "A");
    ASSERT_TRUE(A) << "@test must have an instruction %A";
  }

  LLVMContext Context;
  std::unique_ptr<Module> M;
  Function *F = nullptr;
  Instruction *A = nullptr;
};

class MatchSelectPatternTest : public ValueTrackingTest {
protected:
  void expectPattern(const SelectPatternResult &P) {
    Value *LHS, *RHS;
    Instruction::CastOps CastOp;
    SelectPatternResult R = matchSelectPattern(A, LHS, RHS, &CastOp);
    EXPECT_EQ(P.Flavor, R.Flavor);
    EXPECT_EQ(P.NaNBehavior, R.NaNBehavior);
    EXPECT_EQ(P.Ordered, R.Ordered);
  }
};

class ComputeKnownBitsTest : public ValueTrackingTest {
protected:
  void expectKnownBits(uint64_t Zero, uint64_t One) {
    auto Known = computeKnownBits(A, M->getDataLayout());
    ASSERT_FALSE(Known.hasConflict());
    EXPECT_EQ(Known.One.getZExtValue(), One);
    EXPECT_EQ(Known.Zero.getZExtValue(), Zero);
  }
};

}

TEST_F(MatchSelectPatternTest, SimpleFMin) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp ult float %a, 5.0\n"
      "  %A = select i1 %1, float %a, float 5.0\n"
      "  ret float %A\n"
      "}\n");
  expectPattern({SPF_FMINNUM, SPNB_RETURNS_NAN, false});
}

TEST_F(MatchSelectPatternTest, SimpleFMax) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp ogt float %a, 5.0\n"
      "  %A = select i1 %1, float %a, float 5.0\n"
      "  ret float %A\n"
      "}\n");
  expectPattern({SPF_FMAXNUM, SPNB_RETURNS_OTHER, true});
}

TEST_F(MatchSelectPatternTest, SwappedFMax) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp olt float 5.0, %a\n"
      "  %A = select i1 %1, float %a, float 5.0\n"
      "  ret float %A\n"
      "}\n");
  expectPattern({SPF_FMAXNUM, SPNB_RETURNS_OTHER, false});
}

TEST_F(MatchSelectPatternTest, SwappedFMax2) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp olt float %a, 5.0\n"
      "  %A = select i1 %1, float 5.0, float %a\n"
      "  ret float %A\n"
      "}\n");
  expectPattern({SPF_FMAXNUM, SPNB_RETURNS_NAN, false});
}

TEST_F(MatchSelectPatternTest, SwappedFMax3) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp ult float %a, 5.0\n"
      "  %A = select i1 %1, float 5.0, float %a\n"
      "  ret float %A\n"
      "}\n");
  expectPattern({SPF_FMAXNUM, SPNB_RETURNS_OTHER, true});
}

TEST_F(MatchSelectPatternTest, FastFMin) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp nnan olt float %a, 5.0\n"
      "  %A = select i1 %1, float %a, float 5.0\n"
      "  ret float %A\n"
      "}\n");
  expectPattern({SPF_FMINNUM, SPNB_RETURNS_ANY, false});
}

TEST_F(MatchSelectPatternTest, FMinConstantZero) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp ole float %a, 0.0\n"
      "  %A = select i1 %1, float %a, float 0.0\n"
      "  ret float %A\n"
      "}\n");
  // This shouldn't be matched, as %a could be -0.0.
  expectPattern({SPF_UNKNOWN, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, FMinConstantZeroNsz) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp nsz ole float %a, 0.0\n"
      "  %A = select i1 %1, float %a, float 0.0\n"
      "  ret float %A\n"
      "}\n");
  // But this should be, because we've ignored signed zeroes.
  expectPattern({SPF_FMINNUM, SPNB_RETURNS_OTHER, true});
}

TEST_F(MatchSelectPatternTest, FMinMismatchConstantZero1) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp olt float -0.0, %a\n"
      "  %A = select i1 %1, float 0.0, float %a\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_FMINNUM, SPNB_RETURNS_NAN, true});
}

TEST_F(MatchSelectPatternTest, FMinMismatchConstantZero2) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp ogt float %a, -0.0\n"
      "  %A = select i1 %1, float 0.0, float %a\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_FMINNUM, SPNB_RETURNS_NAN, false});
}

TEST_F(MatchSelectPatternTest, FMinMismatchConstantZero3) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp olt float 0.0, %a\n"
      "  %A = select i1 %1, float -0.0, float %a\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_FMINNUM, SPNB_RETURNS_NAN, true});
}

TEST_F(MatchSelectPatternTest, FMinMismatchConstantZero4) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp ogt float %a, 0.0\n"
      "  %A = select i1 %1, float -0.0, float %a\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_FMINNUM, SPNB_RETURNS_NAN, false});
}

TEST_F(MatchSelectPatternTest, FMinMismatchConstantZero5) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp ogt float -0.0, %a\n"
      "  %A = select i1 %1, float %a, float 0.0\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_FMINNUM, SPNB_RETURNS_OTHER, false});
}

TEST_F(MatchSelectPatternTest, FMinMismatchConstantZero6) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp olt float %a, -0.0\n"
      "  %A = select i1 %1, float %a, float 0.0\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_FMINNUM, SPNB_RETURNS_OTHER, true});
}

TEST_F(MatchSelectPatternTest, FMinMismatchConstantZero7) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp ogt float 0.0, %a\n"
      "  %A = select i1 %1, float %a, float -0.0\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_FMINNUM, SPNB_RETURNS_OTHER, false});
}

TEST_F(MatchSelectPatternTest, FMinMismatchConstantZero8) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp olt float %a, 0.0\n"
      "  %A = select i1 %1, float %a, float -0.0\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_FMINNUM, SPNB_RETURNS_OTHER, true});
}

TEST_F(MatchSelectPatternTest, FMaxMismatchConstantZero1) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp ogt float -0.0, %a\n"
      "  %A = select i1 %1, float 0.0, float %a\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_FMAXNUM, SPNB_RETURNS_NAN, true});
}

TEST_F(MatchSelectPatternTest, FMaxMismatchConstantZero2) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp olt float %a, -0.0\n"
      "  %A = select i1 %1, float 0.0, float %a\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_FMAXNUM, SPNB_RETURNS_NAN, false});
}

TEST_F(MatchSelectPatternTest, FMaxMismatchConstantZero3) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp ogt float 0.0, %a\n"
      "  %A = select i1 %1, float -0.0, float %a\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_FMAXNUM, SPNB_RETURNS_NAN, true});
}

TEST_F(MatchSelectPatternTest, FMaxMismatchConstantZero4) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp olt float %a, 0.0\n"
      "  %A = select i1 %1, float -0.0, float %a\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_FMAXNUM, SPNB_RETURNS_NAN, false});
}

TEST_F(MatchSelectPatternTest, FMaxMismatchConstantZero5) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp olt float -0.0, %a\n"
      "  %A = select i1 %1, float %a, float 0.0\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_FMAXNUM, SPNB_RETURNS_OTHER, false});
}

TEST_F(MatchSelectPatternTest, FMaxMismatchConstantZero6) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp ogt float %a, -0.0\n"
      "  %A = select i1 %1, float %a, float 0.0\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_FMAXNUM, SPNB_RETURNS_OTHER, true});
}

TEST_F(MatchSelectPatternTest, FMaxMismatchConstantZero7) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp olt float 0.0, %a\n"
      "  %A = select i1 %1, float %a, float -0.0\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_FMAXNUM, SPNB_RETURNS_OTHER, false});
}

TEST_F(MatchSelectPatternTest, FMaxMismatchConstantZero8) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp ogt float %a, 0.0\n"
      "  %A = select i1 %1, float %a, float -0.0\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_FMAXNUM, SPNB_RETURNS_OTHER, true});
}

TEST_F(MatchSelectPatternTest, FMinMismatchConstantZeroVecUndef) {
  parseAssembly(
      "define <2 x float> @test(<2 x float> %a) {\n"
      "  %1 = fcmp ogt <2 x float> %a, <float -0.0, float -0.0>\n"
      "  %A = select <2 x i1> %1, <2 x float> <float undef, float 0.0>, <2 x float> %a\n"
      "  ret <2 x float> %A\n"
      "}\n");
  // An undef in a vector constant can not be back-propagated for this analysis.
  expectPattern({SPF_UNKNOWN, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, FMaxMismatchConstantZeroVecUndef) {
  parseAssembly(
      "define <2 x float> @test(<2 x float> %a) {\n"
      "  %1 = fcmp ogt <2 x float> %a, zeroinitializer\n"
      "  %A = select <2 x i1> %1, <2 x float> %a, <2 x float> <float -0.0, float undef>\n"
      "  ret <2 x float> %A\n"
      "}\n");
  // An undef in a vector constant can not be back-propagated for this analysis.
  expectPattern({SPF_UNKNOWN, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, VectorFMinimum) {
  parseAssembly(
      "define <4 x float> @test(<4 x float> %a) {\n"
      "  %1 = fcmp ule <4 x float> %a, \n"
      "    <float 5.0, float 5.0, float 5.0, float 5.0>\n"
      "  %A = select <4 x i1> %1, <4 x float> %a,\n"
      "     <4 x float> <float 5.0, float 5.0, float 5.0, float 5.0>\n"
      "  ret <4 x float> %A\n"
      "}\n");
  // Check that pattern matching works on vectors where each lane has the same
  // unordered pattern.
  expectPattern({SPF_FMINNUM, SPNB_RETURNS_NAN, false});
}

TEST_F(MatchSelectPatternTest, VectorFMinOtherOrdered) {
  parseAssembly(
      "define <4 x float> @test(<4 x float> %a) {\n"
      "  %1 = fcmp ole <4 x float> %a, \n"
      "    <float 5.0, float 5.0, float 5.0, float 5.0>\n"
      "  %A = select <4 x i1> %1, <4 x float> %a,\n"
      "     <4 x float> <float 5.0, float 5.0, float 5.0, float 5.0>\n"
      "  ret <4 x float> %A\n"
      "}\n");
  // Check that pattern matching works on vectors where each lane has the same
  // ordered pattern.
  expectPattern({SPF_FMINNUM, SPNB_RETURNS_OTHER, true});
}

TEST_F(MatchSelectPatternTest, VectorNotFMinimum) {
  parseAssembly(
      "define <4 x float> @test(<4 x float> %a) {\n"
      "  %1 = fcmp ule <4 x float> %a, \n"
      "    <float 5.0, float 0x7ff8000000000000, float 5.0, float 5.0>\n"
      "  %A = select <4 x i1> %1, <4 x float> %a,\n"
      "     <4 x float> <float 5.0, float 0x7ff8000000000000, float 5.0, float "
      "5.0>\n"
      "  ret <4 x float> %A\n"
      "}\n");
  // The lane that contains a NaN (0x7ff80...) behaves like a
  // non-NaN-propagating min and the other lines behave like a NaN-propagating
  // min, so check that neither is returned.
  expectPattern({SPF_UNKNOWN, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, VectorNotFMinZero) {
  parseAssembly(
      "define <4 x float> @test(<4 x float> %a) {\n"
      "  %1 = fcmp ule <4 x float> %a, \n"
      "    <float 5.0, float -0.0, float 5.0, float 5.0>\n"
      "  %A = select <4 x i1> %1, <4 x float> %a,\n"
      "     <4 x float> <float 5.0, float 0.0, float 5.0, float 5.0>\n"
      "  ret <4 x float> %A\n"
      "}\n");
  // Always selects the second lane of %a if it is positive or negative zero, so
  // this is stricter than a min.
  expectPattern({SPF_UNKNOWN, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, DoubleCastU) {
  parseAssembly(
      "define i32 @test(i8 %a, i8 %b) {\n"
      "  %1 = icmp ult i8 %a, %b\n"
      "  %2 = zext i8 %a to i32\n"
      "  %3 = zext i8 %b to i32\n"
      "  %A = select i1 %1, i32 %2, i32 %3\n"
      "  ret i32 %A\n"
      "}\n");
  // We should be able to look through the situation where we cast both operands
  // to the select.
  expectPattern({SPF_UMIN, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, DoubleCastS) {
  parseAssembly(
      "define i32 @test(i8 %a, i8 %b) {\n"
      "  %1 = icmp slt i8 %a, %b\n"
      "  %2 = sext i8 %a to i32\n"
      "  %3 = sext i8 %b to i32\n"
      "  %A = select i1 %1, i32 %2, i32 %3\n"
      "  ret i32 %A\n"
      "}\n");
  // We should be able to look through the situation where we cast both operands
  // to the select.
  expectPattern({SPF_SMIN, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, DoubleCastBad) {
  parseAssembly(
      "define i32 @test(i8 %a, i8 %b) {\n"
      "  %1 = icmp ult i8 %a, %b\n"
      "  %2 = zext i8 %a to i32\n"
      "  %3 = sext i8 %b to i32\n"
      "  %A = select i1 %1, i32 %2, i32 %3\n"
      "  ret i32 %A\n"
      "}\n");
  // The cast types here aren't the same, so we cannot match an UMIN.
  expectPattern({SPF_UNKNOWN, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, NotNotSMin) {
  parseAssembly(
      "define i8 @test(i8 %a, i8 %b) {\n"
      "  %cmp = icmp sgt i8 %a, %b\n"
      "  %an = xor i8 %a, -1\n"
      "  %bn = xor i8 %b, -1\n"
      "  %A = select i1 %cmp, i8 %an, i8 %bn\n"
      "  ret i8 %A\n"
      "}\n");
  expectPattern({SPF_SMIN, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, NotNotSMinSwap) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i8> %a, <2 x i8> %b) {\n"
      "  %cmp = icmp slt <2 x i8> %a, %b\n"
      "  %an = xor <2 x i8> %a, <i8 -1, i8-1>\n"
      "  %bn = xor <2 x i8> %b, <i8 -1, i8-1>\n"
      "  %A = select <2 x i1> %cmp, <2 x i8> %bn, <2 x i8> %an\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  expectPattern({SPF_SMIN, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, NotNotSMax) {
  parseAssembly(
      "define i8 @test(i8 %a, i8 %b) {\n"
      "  %cmp = icmp slt i8 %a, %b\n"
      "  %an = xor i8 %a, -1\n"
      "  %bn = xor i8 %b, -1\n"
      "  %A = select i1 %cmp, i8 %an, i8 %bn\n"
      "  ret i8 %A\n"
      "}\n");
  expectPattern({SPF_SMAX, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, NotNotSMaxSwap) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i8> %a, <2 x i8> %b) {\n"
      "  %cmp = icmp sgt <2 x i8> %a, %b\n"
      "  %an = xor <2 x i8> %a, <i8 -1, i8-1>\n"
      "  %bn = xor <2 x i8> %b, <i8 -1, i8-1>\n"
      "  %A = select <2 x i1> %cmp, <2 x i8> %bn, <2 x i8> %an\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  expectPattern({SPF_SMAX, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, NotNotUMin) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i8> %a, <2 x i8> %b) {\n"
      "  %cmp = icmp ugt <2 x i8> %a, %b\n"
      "  %an = xor <2 x i8> %a, <i8 -1, i8-1>\n"
      "  %bn = xor <2 x i8> %b, <i8 -1, i8-1>\n"
      "  %A = select <2 x i1> %cmp, <2 x i8> %an, <2 x i8> %bn\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  expectPattern({SPF_UMIN, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, NotNotUMinSwap) {
  parseAssembly(
      "define i8 @test(i8 %a, i8 %b) {\n"
      "  %cmp = icmp ult i8 %a, %b\n"
      "  %an = xor i8 %a, -1\n"
      "  %bn = xor i8 %b, -1\n"
      "  %A = select i1 %cmp, i8 %bn, i8 %an\n"
      "  ret i8 %A\n"
      "}\n");
  expectPattern({SPF_UMIN, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, NotNotUMax) {
  parseAssembly(
      "define <2 x i8> @test(<2 x i8> %a, <2 x i8> %b) {\n"
      "  %cmp = icmp ult <2 x i8> %a, %b\n"
      "  %an = xor <2 x i8> %a, <i8 -1, i8-1>\n"
      "  %bn = xor <2 x i8> %b, <i8 -1, i8-1>\n"
      "  %A = select <2 x i1> %cmp, <2 x i8> %an, <2 x i8> %bn\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  expectPattern({SPF_UMAX, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, NotNotUMaxSwap) {
  parseAssembly(
      "define i8 @test(i8 %a, i8 %b) {\n"
      "  %cmp = icmp ugt i8 %a, %b\n"
      "  %an = xor i8 %a, -1\n"
      "  %bn = xor i8 %b, -1\n"
      "  %A = select i1 %cmp, i8 %bn, i8 %an\n"
      "  ret i8 %A\n"
      "}\n");
  expectPattern({SPF_UMAX, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, NotNotEq) {
  parseAssembly(
      "define i8 @test(i8 %a, i8 %b) {\n"
      "  %cmp = icmp eq i8 %a, %b\n"
      "  %an = xor i8 %a, -1\n"
      "  %bn = xor i8 %b, -1\n"
      "  %A = select i1 %cmp, i8 %bn, i8 %an\n"
      "  ret i8 %A\n"
      "}\n");
  expectPattern({SPF_UNKNOWN, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, NotNotNe) {
  parseAssembly(
      "define i8 @test(i8 %a, i8 %b) {\n"
      "  %cmp = icmp ne i8 %a, %b\n"
      "  %an = xor i8 %a, -1\n"
      "  %bn = xor i8 %b, -1\n"
      "  %A = select i1 %cmp, i8 %bn, i8 %an\n"
      "  ret i8 %A\n"
      "}\n");
  expectPattern({SPF_UNKNOWN, SPNB_NA, false});
}

TEST(ValueTracking, GuaranteedToTransferExecutionToSuccessor) {
  StringRef Assembly =
      "declare void @nounwind_readonly(i32*) nounwind readonly "
      "declare void @nounwind_argmemonly(i32*) nounwind argmemonly "
      "declare void @throws_but_readonly(i32*) readonly "
      "declare void @throws_but_argmemonly(i32*) argmemonly "
      "declare void @nounwind_willreturn(i32*) nounwind willreturn"
      " "
      "declare void @unknown(i32*) "
      " "
      "define void @f(i32* %p) { "
      "  call void @nounwind_readonly(i32* %p) "
      "  call void @nounwind_argmemonly(i32* %p) "
      "  call void @throws_but_readonly(i32* %p) "
      "  call void @throws_but_argmemonly(i32* %p) "
      "  call void @unknown(i32* %p) nounwind readonly "
      "  call void @unknown(i32* %p) nounwind argmemonly "
      "  call void @unknown(i32* %p) readonly "
      "  call void @unknown(i32* %p) argmemonly "
      "  call void @nounwind_willreturn(i32* %p)"
      "  ret void "
      "} ";

  LLVMContext Context;
  SMDiagnostic Error;
  auto M = parseAssemblyString(Assembly, Error, Context);
  assert(M && "Bad assembly?");

  auto *F = M->getFunction("f");
  assert(F && "Bad assembly?");

  auto &BB = F->getEntryBlock();
  bool ExpectedAnswers[] = {
      true,  // call void @nounwind_readonly(i32* %p)
      true,  // call void @nounwind_argmemonly(i32* %p)
      false, // call void @throws_but_readonly(i32* %p)
      false, // call void @throws_but_argmemonly(i32* %p)
      true,  // call void @unknown(i32* %p) nounwind readonly
      true,  // call void @unknown(i32* %p) nounwind argmemonly
      false, // call void @unknown(i32* %p) readonly
      false, // call void @unknown(i32* %p) argmemonly
      true,  // call void @nounwind_willreturn(i32* %p)
      false, // ret void
  };

  int Index = 0;
  for (auto &I : BB) {
    EXPECT_EQ(isGuaranteedToTransferExecutionToSuccessor(&I),
              ExpectedAnswers[Index])
        << "Incorrect answer at instruction " << Index << " = " << I;
    Index++;
  }
}

TEST_F(ValueTrackingTest, ComputeNumSignBits_PR32045) {
  parseAssembly(
      "define i32 @test(i32 %a) {\n"
      "  %A = ashr i32 %a, -1\n"
      "  ret i32 %A\n"
      "}\n");
  EXPECT_EQ(ComputeNumSignBits(A, M->getDataLayout()), 1u);
}

// No guarantees for canonical IR in this analysis, so this just bails out.
TEST_F(ValueTrackingTest, ComputeNumSignBits_Shuffle) {
  parseAssembly(
      "define <2 x i32> @test() {\n"
      "  %A = shufflevector <2 x i32> undef, <2 x i32> undef, <2 x i32> <i32 0, i32 0>\n"
      "  ret <2 x i32> %A\n"
      "}\n");
  EXPECT_EQ(ComputeNumSignBits(A, M->getDataLayout()), 1u);
}

// No guarantees for canonical IR in this analysis, so a shuffle element that
// references an undef value means this can't return any extra information.
TEST_F(ValueTrackingTest, ComputeNumSignBits_Shuffle2) {
  parseAssembly(
      "define <2 x i32> @test(<2 x i1> %x) {\n"
      "  %sext = sext <2 x i1> %x to <2 x i32>\n"
      "  %A = shufflevector <2 x i32> %sext, <2 x i32> undef, <2 x i32> <i32 0, i32 2>\n"
      "  ret <2 x i32> %A\n"
      "}\n");
  EXPECT_EQ(ComputeNumSignBits(A, M->getDataLayout()), 1u);
}

TEST(ValueTracking, propagatesPoison) {
  std::string AsmHead = "declare i32 @g(i32)\n"
                        "define void @f(i32 %x, i32 %y, float %fx, float %fy, "
                        "i1 %cond, i8* %p) {\n";
  std::string AsmTail = "  ret void\n}";
  // (propagates poison?, IR instruction)
  SmallVector<std::pair<bool, std::string>, 32> Data = {
      {true, "add i32 %x, %y"},
      {true, "add nsw nuw i32 %x, %y"},
      {true, "ashr i32 %x, %y"},
      {true, "lshr exact i32 %x, 31"},
      {true, "fcmp oeq float %fx, %fy"},
      {true, "icmp eq i32 %x, %y"},
      {true, "getelementptr i8, i8* %p, i32 %x"},
      {true, "getelementptr inbounds i8, i8* %p, i32 %x"},
      {true, "bitcast float %fx to i32"},
      {false, "select i1 %cond, i32 %x, i32 %y"},
      {false, "freeze i32 %x"},
      {true, "udiv i32 %x, %y"},
      {true, "urem i32 %x, %y"},
      {true, "sdiv exact i32 %x, %y"},
      {true, "srem i32 %x, %y"},
      {false, "call i32 @g(i32 %x)"}};

  std::string AssemblyStr = AsmHead;
  for (auto &Itm : Data)
    AssemblyStr += Itm.second + "\n";
  AssemblyStr += AsmTail;

  LLVMContext Context;
  SMDiagnostic Error;
  auto M = parseAssemblyString(AssemblyStr, Error, Context);
  assert(M && "Bad assembly?");

  auto *F = M->getFunction("f");
  assert(F && "Bad assembly?");

  auto &BB = F->getEntryBlock();

  int Index = 0;
  for (auto &I : BB) {
    if (isa<ReturnInst>(&I))
      break;
    EXPECT_EQ(propagatesPoison(&I), Data[Index].first)
        << "Incorrect answer at instruction " << Index << " = " << I;
    Index++;
  }
}

TEST(ValueTracking, canCreatePoisonOrUndef) {
  std::string AsmHead =
      "declare i32 @g(i32)\n"
      "define void @f(i32 %x, i32 %y, float %fx, float %fy, i1 %cond, "
      "<4 x i32> %vx, <4 x i32> %vx2, <vscale x 4 x i32> %svx, i8* %p) {\n";
  std::string AsmTail = "  ret void\n}";
  // (can create poison?, can create undef?, IR instruction)
  SmallVector<std::pair<std::pair<bool, bool>, std::string>, 32> Data = {
      {{false, false}, "add i32 %x, %y"},
      {{true, false}, "add nsw nuw i32 %x, %y"},
      {{true, false}, "shl i32 %x, %y"},
      {{true, false}, "shl <4 x i32> %vx, %vx2"},
      {{true, false}, "shl nsw i32 %x, %y"},
      {{true, false}, "shl nsw <4 x i32> %vx, <i32 0, i32 1, i32 2, i32 3>"},
      {{false, false}, "shl i32 %x, 31"},
      {{true, false}, "shl i32 %x, 32"},
      {{false, false}, "shl <4 x i32> %vx, <i32 0, i32 1, i32 2, i32 3>"},
      {{true, false}, "shl <4 x i32> %vx, <i32 0, i32 1, i32 2, i32 32>"},
      {{true, false}, "ashr i32 %x, %y"},
      {{true, false}, "ashr exact i32 %x, %y"},
      {{false, false}, "ashr i32 %x, 31"},
      {{true, false}, "ashr exact i32 %x, 31"},
      {{false, false}, "ashr <4 x i32> %vx, <i32 0, i32 1, i32 2, i32 3>"},
      {{true, false}, "ashr <4 x i32> %vx, <i32 0, i32 1, i32 2, i32 32>"},
      {{true, false}, "ashr exact <4 x i32> %vx, <i32 0, i32 1, i32 2, i32 3>"},
      {{true, false}, "lshr i32 %x, %y"},
      {{true, false}, "lshr exact i32 %x, 31"},
      {{false, false}, "udiv i32 %x, %y"},
      {{true, false}, "udiv exact i32 %x, %y"},
      {{false, false}, "getelementptr i8, i8* %p, i32 %x"},
      {{true, false}, "getelementptr inbounds i8, i8* %p, i32 %x"},
      {{true, false}, "fneg nnan float %fx"},
      {{false, false}, "fneg float %fx"},
      {{false, false}, "fadd float %fx, %fy"},
      {{true, false}, "fadd nnan float %fx, %fy"},
      {{false, false}, "urem i32 %x, %y"},
      {{true, false}, "fptoui float %fx to i32"},
      {{true, false}, "fptosi float %fx to i32"},
      {{false, false}, "bitcast float %fx to i32"},
      {{false, false}, "select i1 %cond, i32 %x, i32 %y"},
      {{true, false}, "select nnan i1 %cond, float %fx, float %fy"},
      {{true, false}, "extractelement <4 x i32> %vx, i32 %x"},
      {{false, false}, "extractelement <4 x i32> %vx, i32 3"},
      {{true, false}, "extractelement <vscale x 4 x i32> %svx, i32 4"},
      {{true, false}, "insertelement <4 x i32> %vx, i32 %x, i32 %y"},
      {{false, false}, "insertelement <4 x i32> %vx, i32 %x, i32 3"},
      {{true, false}, "insertelement <vscale x 4 x i32> %svx, i32 %x, i32 4"},
      {{false, false}, "freeze i32 %x"},
      {{false, false},
       "shufflevector <4 x i32> %vx, <4 x i32> %vx2, "
       "<4 x i32> <i32 0, i32 1, i32 2, i32 3>"},
      {{false, true},
       "shufflevector <4 x i32> %vx, <4 x i32> %vx2, "
       "<4 x i32> <i32 0, i32 1, i32 2, i32 undef>"},
      {{false, true},
       "shufflevector <vscale x 4 x i32> %svx, "
       "<vscale x 4 x i32> %svx, <vscale x 4 x i32> undef"},
      {{true, false}, "call i32 @g(i32 %x)"},
      {{false, false}, "call noundef i32 @g(i32 %x)"},
      {{true, false}, "fcmp nnan oeq float %fx, %fy"},
      {{false, false}, "fcmp oeq float %fx, %fy"}};

  std::string AssemblyStr = AsmHead;
  for (auto &Itm : Data)
    AssemblyStr += Itm.second + "\n";
  AssemblyStr += AsmTail;

  LLVMContext Context;
  SMDiagnostic Error;
  auto M = parseAssemblyString(AssemblyStr, Error, Context);
  assert(M && "Bad assembly?");

  auto *F = M->getFunction("f");
  assert(F && "Bad assembly?");

  auto &BB = F->getEntryBlock();

  int Index = 0;
  for (auto &I : BB) {
    if (isa<ReturnInst>(&I))
      break;
    bool Poison = Data[Index].first.first;
    bool Undef = Data[Index].first.second;
    EXPECT_EQ(canCreatePoison(cast<Operator>(&I)), Poison)
        << "Incorrect answer of canCreatePoison at instruction " << Index
        << " = " << I;
    EXPECT_EQ(canCreateUndefOrPoison(cast<Operator>(&I)), Undef || Poison)
        << "Incorrect answer of canCreateUndef at instruction " << Index
        << " = " << I;
    Index++;
  }
}

TEST_F(ComputeKnownBitsTest, ComputeKnownBits) {
  parseAssembly(
      "define i32 @test(i32 %a, i32 %b) {\n"
      "  %ash = mul i32 %a, 8\n"
      "  %aad = add i32 %ash, 7\n"
      "  %aan = and i32 %aad, 4095\n"
      "  %bsh = shl i32 %b, 4\n"
      "  %bad = or i32 %bsh, 6\n"
      "  %ban = and i32 %bad, 4095\n"
      "  %A = mul i32 %aan, %ban\n"
      "  ret i32 %A\n"
      "}\n");
  expectKnownBits(/*zero*/ 4278190085u, /*one*/ 10u);
}

TEST_F(ComputeKnownBitsTest, ComputeKnownMulBits) {
  parseAssembly(
      "define i32 @test(i32 %a, i32 %b) {\n"
      "  %aa = shl i32 %a, 5\n"
      "  %bb = shl i32 %b, 5\n"
      "  %aaa = or i32 %aa, 24\n"
      "  %bbb = or i32 %bb, 28\n"
      "  %A = mul i32 %aaa, %bbb\n"
      "  ret i32 %A\n"
      "}\n");
  expectKnownBits(/*zero*/ 95u, /*one*/ 32u);
}

TEST_F(ComputeKnownBitsTest, KnownNonZeroShift) {
  // %q is known nonzero without known bits.
  // Because %q is nonzero, %A[0] is known to be zero.
  parseAssembly(
      "define i8 @test(i8 %p, i8* %pq) {\n"
      "  %q = load i8, i8* %pq, !range !0\n"
      "  %A = shl i8 %p, %q\n"
      "  ret i8 %A\n"
      "}\n"
      "!0 = !{ i8 1, i8 5 }\n");
  expectKnownBits(/*zero*/ 1u, /*one*/ 0u);
}

TEST_F(ComputeKnownBitsTest, ComputeKnownFshl) {
  // fshl(....1111....0000, 00..1111........, 6)
  // = 11....000000..11
  parseAssembly(
      "define i16 @test(i16 %a, i16 %b) {\n"
      "  %aa = shl i16 %a, 4\n"
      "  %bb = lshr i16 %b, 2\n"
      "  %aaa = or i16 %aa, 3840\n"
      "  %bbb = or i16 %bb, 3840\n"
      "  %A = call i16 @llvm.fshl.i16(i16 %aaa, i16 %bbb, i16 6)\n"
      "  ret i16 %A\n"
      "}\n"
      "declare i16 @llvm.fshl.i16(i16, i16, i16)\n");
  expectKnownBits(/*zero*/ 1008u, /*one*/ 49155u);
}

TEST_F(ComputeKnownBitsTest, ComputeKnownFshr) {
  // fshr(....1111....0000, 00..1111........, 26)
  // = 11....000000..11
  parseAssembly(
      "define i16 @test(i16 %a, i16 %b) {\n"
      "  %aa = shl i16 %a, 4\n"
      "  %bb = lshr i16 %b, 2\n"
      "  %aaa = or i16 %aa, 3840\n"
      "  %bbb = or i16 %bb, 3840\n"
      "  %A = call i16 @llvm.fshr.i16(i16 %aaa, i16 %bbb, i16 26)\n"
      "  ret i16 %A\n"
      "}\n"
      "declare i16 @llvm.fshr.i16(i16, i16, i16)\n");
  expectKnownBits(/*zero*/ 1008u, /*one*/ 49155u);
}

TEST_F(ComputeKnownBitsTest, ComputeKnownFshlZero) {
  // fshl(....1111....0000, 00..1111........, 0)
  // = ....1111....0000
  parseAssembly(
      "define i16 @test(i16 %a, i16 %b) {\n"
      "  %aa = shl i16 %a, 4\n"
      "  %bb = lshr i16 %b, 2\n"
      "  %aaa = or i16 %aa, 3840\n"
      "  %bbb = or i16 %bb, 3840\n"
      "  %A = call i16 @llvm.fshl.i16(i16 %aaa, i16 %bbb, i16 0)\n"
      "  ret i16 %A\n"
      "}\n"
      "declare i16 @llvm.fshl.i16(i16, i16, i16)\n");
  expectKnownBits(/*zero*/ 15u, /*one*/ 3840u);
}

TEST_F(ComputeKnownBitsTest, ComputeKnownUAddSatLeadingOnes) {
  // uadd.sat(1111...1, ........)
  // = 1111....
  parseAssembly(
      "define i8 @test(i8 %a, i8 %b) {\n"
      "  %aa = or i8 %a, 241\n"
      "  %A = call i8 @llvm.uadd.sat.i8(i8 %aa, i8 %b)\n"
      "  ret i8 %A\n"
      "}\n"
      "declare i8 @llvm.uadd.sat.i8(i8, i8)\n");
  expectKnownBits(/*zero*/ 0u, /*one*/ 240u);
}

TEST_F(ComputeKnownBitsTest, ComputeKnownUAddSatOnesPreserved) {
  // uadd.sat(00...011, .1...110)
  // = .......1
  parseAssembly(
      "define i8 @test(i8 %a, i8 %b) {\n"
      "  %aa = or i8 %a, 3\n"
      "  %aaa = and i8 %aa, 59\n"
      "  %bb = or i8 %b, 70\n"
      "  %bbb = and i8 %bb, 254\n"
      "  %A = call i8 @llvm.uadd.sat.i8(i8 %aaa, i8 %bbb)\n"
      "  ret i8 %A\n"
      "}\n"
      "declare i8 @llvm.uadd.sat.i8(i8, i8)\n");
  expectKnownBits(/*zero*/ 0u, /*one*/ 1u);
}

TEST_F(ComputeKnownBitsTest, ComputeKnownUSubSatLHSLeadingZeros) {
  // usub.sat(0000...0, ........)
  // = 0000....
  parseAssembly(
      "define i8 @test(i8 %a, i8 %b) {\n"
      "  %aa = and i8 %a, 14\n"
      "  %A = call i8 @llvm.usub.sat.i8(i8 %aa, i8 %b)\n"
      "  ret i8 %A\n"
      "}\n"
      "declare i8 @llvm.usub.sat.i8(i8, i8)\n");
  expectKnownBits(/*zero*/ 240u, /*one*/ 0u);
}

TEST_F(ComputeKnownBitsTest, ComputeKnownUSubSatRHSLeadingOnes) {
  // usub.sat(........, 1111...1)
  // = 0000....
  parseAssembly(
      "define i8 @test(i8 %a, i8 %b) {\n"
      "  %bb = or i8 %a, 241\n"
      "  %A = call i8 @llvm.usub.sat.i8(i8 %a, i8 %bb)\n"
      "  ret i8 %A\n"
      "}\n"
      "declare i8 @llvm.usub.sat.i8(i8, i8)\n");
  expectKnownBits(/*zero*/ 240u, /*one*/ 0u);
}

TEST_F(ComputeKnownBitsTest, ComputeKnownUSubSatZerosPreserved) {
  // usub.sat(11...011, .1...110)
  // = ......0.
  parseAssembly(
      "define i8 @test(i8 %a, i8 %b) {\n"
      "  %aa = or i8 %a, 195\n"
      "  %aaa = and i8 %aa, 251\n"
      "  %bb = or i8 %b, 70\n"
      "  %bbb = and i8 %bb, 254\n"
      "  %A = call i8 @llvm.usub.sat.i8(i8 %aaa, i8 %bbb)\n"
      "  ret i8 %A\n"
      "}\n"
      "declare i8 @llvm.usub.sat.i8(i8, i8)\n");
  expectKnownBits(/*zero*/ 2u, /*one*/ 0u);
}

TEST_F(ComputeKnownBitsTest, ComputeKnownBitsPtrToIntTrunc) {
  // ptrtoint truncates the pointer type.
  parseAssembly(
      "define void @test(i8** %p) {\n"
      "  %A = load i8*, i8** %p\n"
      "  %i = ptrtoint i8* %A to i32\n"
      "  %m = and i32 %i, 31\n"
      "  %c = icmp eq i32 %m, 0\n"
      "  call void @llvm.assume(i1 %c)\n"
      "  ret void\n"
      "}\n"
      "declare void @llvm.assume(i1)\n");
  AssumptionCache AC(*F);
  KnownBits Known = computeKnownBits(
      A, M->getDataLayout(), /* Depth */ 0, &AC, F->front().getTerminator());
  EXPECT_EQ(Known.Zero.getZExtValue(), 31u);
  EXPECT_EQ(Known.One.getZExtValue(), 0u);
}

TEST_F(ComputeKnownBitsTest, ComputeKnownBitsPtrToIntZext) {
  // ptrtoint zero extends the pointer type.
  parseAssembly(
      "define void @test(i8** %p) {\n"
      "  %A = load i8*, i8** %p\n"
      "  %i = ptrtoint i8* %A to i128\n"
      "  %m = and i128 %i, 31\n"
      "  %c = icmp eq i128 %m, 0\n"
      "  call void @llvm.assume(i1 %c)\n"
      "  ret void\n"
      "}\n"
      "declare void @llvm.assume(i1)\n");
  AssumptionCache AC(*F);
  KnownBits Known = computeKnownBits(
      A, M->getDataLayout(), /* Depth */ 0, &AC, F->front().getTerminator());
  EXPECT_EQ(Known.Zero.getZExtValue(), 31u);
  EXPECT_EQ(Known.One.getZExtValue(), 0u);
}

class IsBytewiseValueTest : public ValueTrackingTest,
                            public ::testing::WithParamInterface<
                                std::pair<const char *, const char *>> {
protected:
};

const std::pair<const char *, const char *> IsBytewiseValueTests[] = {
    {
        "i8 0",
        "i48* null",
    },
    {
        "i8 undef",
        "i48* undef",
    },
    {
        "i8 0",
        "i8 zeroinitializer",
    },
    {
        "i8 0",
        "i8 0",
    },
    {
        "i8 -86",
        "i8 -86",
    },
    {
        "i8 -1",
        "i8 -1",
    },
    {
        "i8 undef",
        "i16 undef",
    },
    {
        "i8 0",
        "i16 0",
    },
    {
        "",
        "i16 7",
    },
    {
        "i8 -86",
        "i16 -21846",
    },
    {
        "i8 -1",
        "i16 -1",
    },
    {
        "i8 0",
        "i48 0",
    },
    {
        "i8 -1",
        "i48 -1",
    },
    {
        "i8 0",
        "i49 0",
    },
    {
        "",
        "i49 -1",
    },
    {
        "i8 0",
        "half 0xH0000",
    },
    {
        "i8 -85",
        "half 0xHABAB",
    },
    {
        "i8 0",
        "float 0.0",
    },
    {
        "i8 -1",
        "float 0xFFFFFFFFE0000000",
    },
    {
        "i8 0",
        "double 0.0",
    },
    {
        "i8 -15",
        "double 0xF1F1F1F1F1F1F1F1",
    },
    {
        "i8 undef",
        "i16* undef",
    },
    {
        "i8 0",
        "i16* inttoptr (i64 0 to i16*)",
    },
    {
        "i8 -1",
        "i16* inttoptr (i64 -1 to i16*)",
    },
    {
        "i8 -86",
        "i16* inttoptr (i64 -6148914691236517206 to i16*)",
    },
    {
        "",
        "i16* inttoptr (i48 -1 to i16*)",
    },
    {
        "i8 -1",
        "i16* inttoptr (i96 -1 to i16*)",
    },
    {
        "i8 undef",
        "[0 x i8] zeroinitializer",
    },
    {
        "i8 undef",
        "[0 x i8] undef",
    },
    {
        "i8 undef",
        "[5 x [0 x i8]] zeroinitializer",
    },
    {
        "i8 undef",
        "[5 x [0 x i8]] undef",
    },
    {
        "i8 0",
        "[6 x i8] zeroinitializer",
    },
    {
        "i8 undef",
        "[6 x i8] undef",
    },
    {
        "i8 1",
        "[5 x i8] [i8 1, i8 1, i8 1, i8 1, i8 1]",
    },
    {
        "",
        "[5 x i64] [i64 1, i64 1, i64 1, i64 1, i64 1]",
    },
    {
        "i8 -1",
        "[5 x i64] [i64 -1, i64 -1, i64 -1, i64 -1, i64 -1]",
    },
    {
        "",
        "[4 x i8] [i8 1, i8 2, i8 1, i8 1]",
    },
    {
        "i8 1",
        "[4 x i8] [i8 1, i8 undef, i8 1, i8 1]",
    },
    {
        "i8 0",
        "<6 x i8> zeroinitializer",
    },
    {
        "i8 undef",
        "<6 x i8> undef",
    },
    {
        "i8 1",
        "<5 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1>",
    },
    {
        "",
        "<5 x i64> <i64 1, i64 1, i64 1, i64 1, i64 1>",
    },
    {
        "i8 -1",
        "<5 x i64> <i64 -1, i64 -1, i64 -1, i64 -1, i64 -1>",
    },
    {
        "",
        "<4 x i8> <i8 1, i8 1, i8 2, i8 1>",
    },
    {
        "i8 5",
        "<2 x i8> < i8 5, i8 undef >",
    },
    {
        "i8 0",
        "[2 x [2 x i16]] zeroinitializer",
    },
    {
        "i8 undef",
        "[2 x [2 x i16]] undef",
    },
    {
        "i8 -86",
        "[2 x [2 x i16]] [[2 x i16] [i16 -21846, i16 -21846], "
        "[2 x i16] [i16 -21846, i16 -21846]]",
    },
    {
        "",
        "[2 x [2 x i16]] [[2 x i16] [i16 -21846, i16 -21846], "
        "[2 x i16] [i16 -21836, i16 -21846]]",
    },
    {
        "i8 undef",
        "{ } zeroinitializer",
    },
    {
        "i8 undef",
        "{ } undef",
    },
    {
        "i8 undef",
        "{ {}, {} } zeroinitializer",
    },
    {
        "i8 undef",
        "{ {}, {} } undef",
    },
    {
        "i8 0",
        "{i8, i64, i16*} zeroinitializer",
    },
    {
        "i8 undef",
        "{i8, i64, i16*} undef",
    },
    {
        "i8 -86",
        "{i8, i64, i16*} {i8 -86, i64 -6148914691236517206, i16* undef}",
    },
    {
        "",
        "{i8, i64, i16*} {i8 86, i64 -6148914691236517206, i16* undef}",
    },
};

INSTANTIATE_TEST_CASE_P(IsBytewiseValueParamTests, IsBytewiseValueTest,
                        ::testing::ValuesIn(IsBytewiseValueTests),);

TEST_P(IsBytewiseValueTest, IsBytewiseValue) {
  auto M = parseModule(std::string("@test = global ") + GetParam().second);
  GlobalVariable *GV = dyn_cast<GlobalVariable>(M->getNamedValue("test"));
  Value *Actual = isBytewiseValue(GV->getInitializer(), M->getDataLayout());
  std::string Buff;
  raw_string_ostream S(Buff);
  if (Actual)
    S << *Actual;
  EXPECT_EQ(GetParam().first, S.str());
}

TEST_F(ValueTrackingTest, ComputeConstantRange) {
  {
    // Assumptions:
    //  * stride >= 5
    //  * stride < 10
    //
    // stride = [5, 10)
    auto M = parseModule(R"(
  declare void @llvm.assume(i1)

  define i32 @test(i32 %stride) {
    %gt = icmp uge i32 %stride, 5
    call void @llvm.assume(i1 %gt)
    %lt = icmp ult i32 %stride, 10
    call void @llvm.assume(i1 %lt)
    %stride.plus.one = add nsw nuw i32 %stride, 1
    ret i32 %stride.plus.one
  })");
    Function *F = M->getFunction("test");

    AssumptionCache AC(*F);
    Value *Stride = &*F->arg_begin();
    ConstantRange CR1 = computeConstantRange(Stride, true, &AC, nullptr);
    EXPECT_TRUE(CR1.isFullSet());

    Instruction *I = &findInstructionByName(F, "stride.plus.one");
    ConstantRange CR2 = computeConstantRange(Stride, true, &AC, I);
    EXPECT_EQ(5, CR2.getLower());
    EXPECT_EQ(10, CR2.getUpper());
  }

  {
    // Assumptions:
    //  * stride >= 5
    //  * stride < 200
    //  * stride == 99
    //
    // stride = [99, 100)
    auto M = parseModule(R"(
  declare void @llvm.assume(i1)

  define i32 @test(i32 %stride) {
    %gt = icmp uge i32 %stride, 5
    call void @llvm.assume(i1 %gt)
    %lt = icmp ult i32 %stride, 200
    call void @llvm.assume(i1 %lt)
    %eq = icmp eq i32 %stride, 99
    call void @llvm.assume(i1 %eq)
    %stride.plus.one = add nsw nuw i32 %stride, 1
    ret i32 %stride.plus.one
  })");
    Function *F = M->getFunction("test");

    AssumptionCache AC(*F);
    Value *Stride = &*F->arg_begin();
    Instruction *I = &findInstructionByName(F, "stride.plus.one");
    ConstantRange CR = computeConstantRange(Stride, true, &AC, I);
    EXPECT_EQ(99, *CR.getSingleElement());
  }

  {
    // Assumptions:
    //  * stride >= 5
    //  * stride >= 50
    //  * stride < 100
    //  * stride < 200
    //
    // stride = [50, 100)
    auto M = parseModule(R"(
  declare void @llvm.assume(i1)

  define i32 @test(i32 %stride, i1 %cond) {
    %gt = icmp uge i32 %stride, 5
    call void @llvm.assume(i1 %gt)
    %gt.2 = icmp uge i32 %stride, 50
    call void @llvm.assume(i1 %gt.2)
    br i1 %cond, label %bb1, label %bb2

  bb1:
    %lt = icmp ult i32 %stride, 200
    call void @llvm.assume(i1 %lt)
    %lt.2 = icmp ult i32 %stride, 100
    call void @llvm.assume(i1 %lt.2)
    %stride.plus.one = add nsw nuw i32 %stride, 1
    ret i32 %stride.plus.one

  bb2:
    ret i32 0
  })");
    Function *F = M->getFunction("test");

    AssumptionCache AC(*F);
    Value *Stride = &*F->arg_begin();
    Instruction *GT2 = &findInstructionByName(F, "gt.2");
    ConstantRange CR = computeConstantRange(Stride, true, &AC, GT2);
    EXPECT_EQ(5, CR.getLower());
    EXPECT_EQ(0, CR.getUpper());

    Instruction *I = &findInstructionByName(F, "stride.plus.one");
    ConstantRange CR2 = computeConstantRange(Stride, true, &AC, I);
    EXPECT_EQ(50, CR2.getLower());
    EXPECT_EQ(100, CR2.getUpper());
  }

  {
    // Assumptions:
    //  * stride > 5
    //  * stride < 5
    //
    // stride = empty range, as the assumptions contradict each other.
    auto M = parseModule(R"(
  declare void @llvm.assume(i1)

  define i32 @test(i32 %stride, i1 %cond) {
    %gt = icmp ugt i32 %stride, 5
    call void @llvm.assume(i1 %gt)
    %lt = icmp ult i32 %stride, 5
    call void @llvm.assume(i1 %lt)
    %stride.plus.one = add nsw nuw i32 %stride, 1
    ret i32 %stride.plus.one
  })");
    Function *F = M->getFunction("test");

    AssumptionCache AC(*F);
    Value *Stride = &*F->arg_begin();

    Instruction *I = &findInstructionByName(F, "stride.plus.one");
    ConstantRange CR = computeConstantRange(Stride, true, &AC, I);
    EXPECT_TRUE(CR.isEmptySet());
  }

  {
    // Assumptions:
    //  * x.1 >= 5
    //  * x.2 < x.1
    //
    // stride = [0, 5)
    auto M = parseModule(R"(
  declare void @llvm.assume(i1)

  define i32 @test(i32 %x.1, i32 %x.2) {
    %gt = icmp uge i32 %x.1, 5
    call void @llvm.assume(i1 %gt)
    %lt = icmp ult i32 %x.2, %x.1
    call void @llvm.assume(i1 %lt)
    %stride.plus.one = add nsw nuw i32 %x.1, 1
    ret i32 %stride.plus.one
  })");
    Function *F = M->getFunction("test");

    AssumptionCache AC(*F);
    Value *X2 = &*std::next(F->arg_begin());

    Instruction *I = &findInstructionByName(F, "stride.plus.one");
    ConstantRange CR1 = computeConstantRange(X2, true, &AC, I);
    EXPECT_EQ(0, CR1.getLower());
    EXPECT_EQ(5, CR1.getUpper());

    // Check the depth cutoff results in a conservative result (full set) by
    // passing Depth == MaxDepth == 6.
    ConstantRange CR2 = computeConstantRange(X2, true, &AC, I, 6);
    EXPECT_TRUE(CR2.isFullSet());
  }
}
