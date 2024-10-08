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
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Utils/Local.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

static Instruction *findInstructionByNameOrNull(Function *F, StringRef Name) {
  for (Instruction &I : instructions(F))
    if (I.getName() == Name)
      return &I;

  return nullptr;
}

static Instruction &findInstructionByName(Function *F, StringRef Name) {
  auto *I = findInstructionByNameOrNull(F, Name);
  if (I)
    return *I;

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
    EXPECT_TRUE(M) << errMsg;

    return M;
  }

  void parseAssembly(StringRef Assembly) {
    M = parseModule(Assembly);
    ASSERT_TRUE(M);

    F = M->getFunction("test");
    ASSERT_TRUE(F) << "Test must have a function @test";
    if (!F)
      return;

    A = findInstructionByNameOrNull(F, "A");
    ASSERT_TRUE(A) << "@test must have an instruction %A";
    A2 = findInstructionByNameOrNull(F, "A2");
    A3 = findInstructionByNameOrNull(F, "A3");
    A4 = findInstructionByNameOrNull(F, "A4");
    A5 = findInstructionByNameOrNull(F, "A5");
    A6 = findInstructionByNameOrNull(F, "A6");
    A7 = findInstructionByNameOrNull(F, "A7");

    CxtI = findInstructionByNameOrNull(F, "CxtI");
    CxtI2 = findInstructionByNameOrNull(F, "CxtI2");
    CxtI3 = findInstructionByNameOrNull(F, "CxtI3");
  }

  LLVMContext Context;
  std::unique_ptr<Module> M;
  Function *F = nullptr;
  Instruction *A = nullptr;
  // Instructions (optional)
  Instruction *A2 = nullptr, *A3 = nullptr, *A4 = nullptr, *A5 = nullptr,
              *A6 = nullptr, *A7 = nullptr;

  // Context instructions (optional)
  Instruction *CxtI = nullptr, *CxtI2 = nullptr, *CxtI3 = nullptr;
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

class ComputeKnownFPClassTest : public ValueTrackingTest {
protected:
  void expectKnownFPClass(unsigned KnownTrue, std::optional<bool> SignBitKnown,
                          Instruction *TestVal = nullptr) {
    if (!TestVal)
      TestVal = A;

    KnownFPClass Known = computeKnownFPClass(TestVal, M->getDataLayout());
    EXPECT_EQ(KnownTrue, Known.KnownFPClasses);
    EXPECT_EQ(SignBitKnown, Known.SignBit);
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
  expectPattern({SPF_UNKNOWN, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, FMinMismatchConstantZero2) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp ogt float %a, -0.0\n"
      "  %A = select i1 %1, float 0.0, float %a\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_UNKNOWN, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, FMinMismatchConstantZero3) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp olt float 0.0, %a\n"
      "  %A = select i1 %1, float -0.0, float %a\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_UNKNOWN, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, FMinMismatchConstantZero4) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp ogt float %a, 0.0\n"
      "  %A = select i1 %1, float -0.0, float %a\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_UNKNOWN, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, FMinMismatchConstantZero5) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp ogt float -0.0, %a\n"
      "  %A = select i1 %1, float %a, float 0.0\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_UNKNOWN, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, FMinMismatchConstantZero6) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp olt float %a, -0.0\n"
      "  %A = select i1 %1, float %a, float 0.0\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_UNKNOWN, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, FMinMismatchConstantZero7) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp ogt float 0.0, %a\n"
      "  %A = select i1 %1, float %a, float -0.0\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_UNKNOWN, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, FMinMismatchConstantZero8) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp olt float %a, 0.0\n"
      "  %A = select i1 %1, float %a, float -0.0\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_UNKNOWN, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, FMaxMismatchConstantZero1) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp ogt float -0.0, %a\n"
      "  %A = select i1 %1, float 0.0, float %a\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_UNKNOWN, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, FMaxMismatchConstantZero2) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp olt float %a, -0.0\n"
      "  %A = select i1 %1, float 0.0, float %a\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_UNKNOWN, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, FMaxMismatchConstantZero3) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp ogt float 0.0, %a\n"
      "  %A = select i1 %1, float -0.0, float %a\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_UNKNOWN, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, FMaxMismatchConstantZero4) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp olt float %a, 0.0\n"
      "  %A = select i1 %1, float -0.0, float %a\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_UNKNOWN, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, FMaxMismatchConstantZero5) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp olt float -0.0, %a\n"
      "  %A = select i1 %1, float %a, float 0.0\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_UNKNOWN, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, FMaxMismatchConstantZero6) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp ogt float %a, -0.0\n"
      "  %A = select i1 %1, float %a, float 0.0\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_UNKNOWN, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, FMaxMismatchConstantZero7) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp olt float 0.0, %a\n"
      "  %A = select i1 %1, float %a, float -0.0\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_UNKNOWN, SPNB_NA, false});
}

TEST_F(MatchSelectPatternTest, FMaxMismatchConstantZero8) {
  parseAssembly(
      "define float @test(float %a) {\n"
      "  %1 = fcmp ogt float %a, 0.0\n"
      "  %A = select i1 %1, float %a, float -0.0\n"
      "  ret float %A\n"
      "}\n");
  // The sign of zero doesn't matter in fcmp.
  expectPattern({SPF_UNKNOWN, SPNB_NA, false});
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
      "declare void @nounwind_readonly(ptr) nounwind readonly "
      "declare void @nounwind_argmemonly(ptr) nounwind argmemonly "
      "declare void @nounwind_willreturn(ptr) nounwind willreturn "
      "declare void @throws_but_readonly(ptr) readonly "
      "declare void @throws_but_argmemonly(ptr) argmemonly "
      "declare void @throws_but_willreturn(ptr) willreturn "
      " "
      "declare void @unknown(ptr) "
      " "
      "define void @f(ptr %p) { "
      "  call void @nounwind_readonly(ptr %p) "
      "  call void @nounwind_argmemonly(ptr %p) "
      "  call void @nounwind_willreturn(ptr %p)"
      "  call void @throws_but_readonly(ptr %p) "
      "  call void @throws_but_argmemonly(ptr %p) "
      "  call void @throws_but_willreturn(ptr %p) "
      "  call void @unknown(ptr %p) nounwind readonly "
      "  call void @unknown(ptr %p) nounwind argmemonly "
      "  call void @unknown(ptr %p) nounwind willreturn "
      "  call void @unknown(ptr %p) readonly "
      "  call void @unknown(ptr %p) argmemonly "
      "  call void @unknown(ptr %p) willreturn "
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
      false, // call void @nounwind_readonly(ptr %p)
      false, // call void @nounwind_argmemonly(ptr %p)
      true,  // call void @nounwind_willreturn(ptr %p)
      false, // call void @throws_but_readonly(ptr %p)
      false, // call void @throws_but_argmemonly(ptr %p)
      false, // call void @throws_but_willreturn(ptr %p)
      false, // call void @unknown(ptr %p) nounwind readonly
      false, // call void @unknown(ptr %p) nounwind argmemonly
      true,  // call void @unknown(ptr %p) nounwind willreturn
      false, // call void @unknown(ptr %p) readonly
      false, // call void @unknown(ptr %p) argmemonly
      false, // call void @unknown(ptr %p) willreturn
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
  EXPECT_EQ(ComputeNumSignBits(A, M->getDataLayout()), 32u);
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

TEST_F(ValueTrackingTest, impliesPoisonTest_Identity) {
  parseAssembly("define void @test(i32 %x, i32 %y) {\n"
                "  %A = add i32 %x, %y\n"
                "  ret void\n"
                "}");
  EXPECT_TRUE(impliesPoison(A, A));
}

TEST_F(ValueTrackingTest, impliesPoisonTest_ICmp) {
  parseAssembly("define void @test(i32 %x) {\n"
                "  %A2 = icmp eq i32 %x, 0\n"
                "  %A = icmp eq i32 %x, 1\n"
                "  ret void\n"
                "}");
  EXPECT_TRUE(impliesPoison(A2, A));
}

TEST_F(ValueTrackingTest, impliesPoisonTest_ICmpUnknown) {
  parseAssembly("define void @test(i32 %x, i32 %y) {\n"
                "  %A2 = icmp eq i32 %x, %y\n"
                "  %A = icmp eq i32 %x, 1\n"
                "  ret void\n"
                "}");
  EXPECT_FALSE(impliesPoison(A2, A));
}

TEST_F(ValueTrackingTest, impliesPoisonTest_AddNswOkay) {
  parseAssembly("define void @test(i32 %x) {\n"
                "  %A2 = add nsw i32 %x, 1\n"
                "  %A = add i32 %A2, 1\n"
                "  ret void\n"
                "}");
  EXPECT_TRUE(impliesPoison(A2, A));
}

TEST_F(ValueTrackingTest, impliesPoisonTest_AddNswOkay2) {
  parseAssembly("define void @test(i32 %x) {\n"
                "  %A2 = add i32 %x, 1\n"
                "  %A = add nsw i32 %A2, 1\n"
                "  ret void\n"
                "}");
  EXPECT_TRUE(impliesPoison(A2, A));
}

TEST_F(ValueTrackingTest, impliesPoisonTest_AddNsw) {
  parseAssembly("define void @test(i32 %x) {\n"
                "  %A2 = add nsw i32 %x, 1\n"
                "  %A = add i32 %x, 1\n"
                "  ret void\n"
                "}");
  EXPECT_FALSE(impliesPoison(A2, A));
}

TEST_F(ValueTrackingTest, impliesPoisonTest_Cmp) {
  parseAssembly("define void @test(i32 %x, i32 %y, i1 %c) {\n"
                "  %A2 = icmp eq i32 %x, %y\n"
                "  %A0 = icmp ult i32 %x, %y\n"
                "  %A = or i1 %A0, %c\n"
                "  ret void\n"
                "}");
  EXPECT_TRUE(impliesPoison(A2, A));
}

TEST_F(ValueTrackingTest, impliesPoisonTest_FCmpFMF) {
  parseAssembly("define void @test(float %x, float %y, i1 %c) {\n"
                "  %A2 = fcmp nnan oeq float %x, %y\n"
                "  %A0 = fcmp olt float %x, %y\n"
                "  %A = or i1 %A0, %c\n"
                "  ret void\n"
                "}");
  EXPECT_FALSE(impliesPoison(A2, A));
}

TEST_F(ValueTrackingTest, impliesPoisonTest_AddSubSameOps) {
  parseAssembly("define void @test(i32 %x, i32 %y, i1 %c) {\n"
                "  %A2 = add i32 %x, %y\n"
                "  %A = sub i32 %x, %y\n"
                "  ret void\n"
                "}");
  EXPECT_TRUE(impliesPoison(A2, A));
}

TEST_F(ValueTrackingTest, impliesPoisonTest_MaskCmp) {
  parseAssembly("define void @test(i32 %x, i32 %y, i1 %c) {\n"
                "  %M2 = and i32 %x, 7\n"
                "  %A2 = icmp eq i32 %M2, 1\n"
                "  %M = and i32 %x, 15\n"
                "  %A = icmp eq i32 %M, 3\n"
                "  ret void\n"
                "}");
  EXPECT_TRUE(impliesPoison(A2, A));
}

TEST_F(ValueTrackingTest, ComputeNumSignBits_Shuffle_Pointers) {
  parseAssembly(
      "define <2 x ptr> @test(<2 x ptr> %x) {\n"
      "  %A = shufflevector <2 x ptr> zeroinitializer, <2 x ptr> undef, <2 x i32> zeroinitializer\n"
      "  ret <2 x ptr> %A\n"
      "}\n");
  EXPECT_EQ(ComputeNumSignBits(A, M->getDataLayout()), 64u);
}

TEST(ValueTracking, propagatesPoison) {
  std::string AsmHead =
      "declare i32 @g(i32)\n"
      "define void @f(i32 %x, i32 %y, i32 %shamt, float %fx, float %fy, "
      "i1 %cond, ptr %p) {\n";
  std::string AsmTail = "  ret void\n}";
  // (propagates poison?, IR instruction)
  SmallVector<std::tuple<bool, std::string, unsigned>, 32> Data = {
      {true, "add i32 %x, %y", 0},
      {true, "add i32 %x, %y", 1},
      {true, "add nsw nuw i32 %x, %y", 0},
      {true, "add nsw nuw i32 %x, %y", 1},
      {true, "ashr i32 %x, %y", 0},
      {true, "ashr i32 %x, %y", 1},
      {true, "lshr exact i32 %x, 31", 0},
      {true, "lshr exact i32 %x, 31", 1},
      {true, "fadd float %fx, %fy", 0},
      {true, "fadd float %fx, %fy", 1},
      {true, "fsub float %fx, %fy", 0},
      {true, "fsub float %fx, %fy", 1},
      {true, "fmul float %fx, %fy", 0},
      {true, "fmul float %fx, %fy", 1},
      {true, "fdiv float %fx, %fy", 0},
      {true, "fdiv float %fx, %fy", 1},
      {true, "frem float %fx, %fy", 0},
      {true, "frem float %fx, %fy", 1},
      {true, "fneg float %fx", 0},
      {true, "fcmp oeq float %fx, %fy", 0},
      {true, "fcmp oeq float %fx, %fy", 1},
      {true, "icmp eq i32 %x, %y", 0},
      {true, "icmp eq i32 %x, %y", 1},
      {true, "getelementptr i8, ptr %p, i32 %x", 0},
      {true, "getelementptr i8, ptr %p, i32 %x", 1},
      {true, "getelementptr inbounds i8, ptr %p, i32 %x", 0},
      {true, "getelementptr inbounds i8, ptr %p, i32 %x", 1},
      {true, "bitcast float %fx to i32", 0},
      {true, "select i1 %cond, i32 %x, i32 %y", 0},
      {false, "select i1 %cond, i32 %x, i32 %y", 1},
      {false, "select i1 %cond, i32 %x, i32 %y", 2},
      {false, "freeze i32 %x", 0},
      {true, "udiv i32 %x, %y", 0},
      {true, "udiv i32 %x, %y", 1},
      {true, "urem i32 %x, %y", 0},
      {true, "urem i32 %x, %y", 1},
      {true, "sdiv exact i32 %x, %y", 0},
      {true, "sdiv exact i32 %x, %y", 1},
      {true, "srem i32 %x, %y", 0},
      {true, "srem i32 %x, %y", 1},
      {false, "call i32 @g(i32 %x)", 0},
      {false, "call i32 @g(i32 %x)", 1},
      {true, "call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %x, i32 %y)", 0},
      {true, "call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %x, i32 %y)", 0},
      {true, "call {i32, i1} @llvm.smul.with.overflow.i32(i32 %x, i32 %y)", 0},
      {true, "call {i32, i1} @llvm.uadd.with.overflow.i32(i32 %x, i32 %y)", 0},
      {true, "call {i32, i1} @llvm.usub.with.overflow.i32(i32 %x, i32 %y)", 0},
      {true, "call {i32, i1} @llvm.umul.with.overflow.i32(i32 %x, i32 %y)", 0},
      {true, "call i32 @llvm.sadd.sat.i32(i32 %x, i32 %y)", 0},
      {true, "call i32 @llvm.ssub.sat.i32(i32 %x, i32 %y)", 0},
      {true, "call i32 @llvm.sshl.sat.i32(i32 %x, i32 %y)", 0},
      {true, "call i32 @llvm.uadd.sat.i32(i32 %x, i32 %y)", 0},
      {true, "call i32 @llvm.usub.sat.i32(i32 %x, i32 %y)", 0},
      {true, "call i32 @llvm.ushl.sat.i32(i32 %x, i32 %y)", 0},
      {true, "call i32 @llvm.ctpop.i32(i32 %x)", 0},
      {true, "call i32 @llvm.ctlz.i32(i32 %x, i1 true)", 0},
      {true, "call i32 @llvm.cttz.i32(i32 %x, i1 true)", 0},
      {true, "call i32 @llvm.abs.i32(i32 %x, i1 true)", 0},
      {true, "call i32 @llvm.smax.i32(i32 %x, i32 %y)", 0},
      {true, "call i32 @llvm.smin.i32(i32 %x, i32 %y)", 0},
      {true, "call i32 @llvm.umax.i32(i32 %x, i32 %y)", 0},
      {true, "call i32 @llvm.umin.i32(i32 %x, i32 %y)", 0},
      {true, "call i32 @llvm.bitreverse.i32(i32 %x)", 0},
      {true, "call i32 @llvm.bswap.i32(i32 %x)", 0},
      {false, "call i32 @llvm.fshl.i32(i32 %x, i32 %y, i32 %shamt)", 0},
      {false, "call i32 @llvm.fshl.i32(i32 %x, i32 %y, i32 %shamt)", 1},
      {false, "call i32 @llvm.fshl.i32(i32 %x, i32 %y, i32 %shamt)", 2},
      {false, "call i32 @llvm.fshr.i32(i32 %x, i32 %y, i32 %shamt)", 0},
      {false, "call i32 @llvm.fshr.i32(i32 %x, i32 %y, i32 %shamt)", 1},
      {false, "call i32 @llvm.fshr.i32(i32 %x, i32 %y, i32 %shamt)", 2},
      {false, "call float @llvm.sqrt.f32(float %fx)", 0},
      {false, "call float @llvm.powi.f32.i32(float %fx, i32 %x)", 0},
      {false, "call float @llvm.sin.f32(float %fx)", 0},
      {false, "call float @llvm.cos.f32(float %fx)", 0},
      {false, "call float @llvm.pow.f32(float %fx, float %fy)", 0},
      {false, "call float @llvm.exp.f32(float %fx)", 0},
      {false, "call float @llvm.exp2.f32(float %fx)", 0},
      {false, "call float @llvm.log.f32(float %fx)", 0},
      {false, "call float @llvm.log10.f32(float %fx)", 0},
      {false, "call float @llvm.log2.f32(float %fx)", 0},
      {false, "call float @llvm.fma.f32(float %fx, float %fx, float %fy)", 0},
      {false, "call float @llvm.fabs.f32(float %fx)", 0},
      {false, "call float @llvm.minnum.f32(float %fx, float %fy)", 0},
      {false, "call float @llvm.maxnum.f32(float %fx, float %fy)", 0},
      {false, "call float @llvm.minimum.f32(float %fx, float %fy)", 0},
      {false, "call float @llvm.maximum.f32(float %fx, float %fy)", 0},
      {false, "call float @llvm.copysign.f32(float %fx, float %fy)", 0},
      {false, "call float @llvm.floor.f32(float %fx)", 0},
      {false, "call float @llvm.ceil.f32(float %fx)", 0},
      {false, "call float @llvm.trunc.f32(float %fx)", 0},
      {false, "call float @llvm.rint.f32(float %fx)", 0},
      {false, "call float @llvm.nearbyint.f32(float %fx)", 0},
      {false, "call float @llvm.round.f32(float %fx)", 0},
      {false, "call float @llvm.roundeven.f32(float %fx)", 0},
      {false, "call i32 @llvm.lround.f32(float %fx)", 0},
      {false, "call i64 @llvm.llround.f32(float %fx)", 0},
      {false, "call i32 @llvm.lrint.f32(float %fx)", 0},
      {false, "call i64 @llvm.llrint.f32(float %fx)", 0},
      {false, "call float @llvm.fmuladd.f32(float %fx, float %fx, float %fy)",
       0}};

  std::string AssemblyStr = AsmHead;
  for (auto &Itm : Data)
    AssemblyStr += std::get<1>(Itm) + "\n";
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
    bool ExpectedVal = std::get<0>(Data[Index]);
    unsigned OpIdx = std::get<2>(Data[Index]);
    EXPECT_EQ(propagatesPoison(I.getOperandUse(OpIdx)), ExpectedVal)
        << "Incorrect answer at instruction " << Index << " = " << I;
    Index++;
  }
}

TEST_F(ValueTrackingTest, programUndefinedIfPoison) {
  parseAssembly("declare i32 @any_num()"
                "define void @test(i32 %mask) {\n"
                "  %A = call i32 @any_num()\n"
                "  %B = or i32 %A, %mask\n"
                "  udiv i32 1, %B"
                "  ret void\n"
                "}\n");
  // If %A was poison, udiv raises UB regardless of %mask's value
  EXPECT_EQ(programUndefinedIfPoison(A), true);
}

TEST_F(ValueTrackingTest, programUndefinedIfPoisonSelect) {
  parseAssembly("declare i32 @any_num()"
                "define void @test(i1 %Cond) {\n"
                "  %A = call i32 @any_num()\n"
                "  %B = add i32 %A, 1\n"
                "  %C = select i1 %Cond, i32 %A, i32 %B\n"
                "  udiv i32 1, %C"
                "  ret void\n"
                "}\n");
  // If A is poison, B is also poison, and therefore C is poison regardless of
  // the value of %Cond.
  EXPECT_EQ(programUndefinedIfPoison(A), true);
}

TEST_F(ValueTrackingTest, programUndefinedIfUndefOrPoison) {
  parseAssembly("declare i32 @any_num()"
                "define void @test(i32 %mask) {\n"
                "  %A = call i32 @any_num()\n"
                "  %B = or i32 %A, %mask\n"
                "  udiv i32 1, %B"
                "  ret void\n"
                "}\n");
  // If %A was undef and %mask was 1, udiv does not raise UB
  EXPECT_EQ(programUndefinedIfUndefOrPoison(A), false);
}

TEST_F(ValueTrackingTest, isGuaranteedNotToBePoison_exploitBranchCond) {
  parseAssembly("declare i1 @any_bool()"
                "define void @test(i1 %y) {\n"
                "  %A = call i1 @any_bool()\n"
                "  %cond = and i1 %A, %y\n"
                "  br i1 %cond, label %BB1, label %BB2\n"
                "BB1:\n"
                "  ret void\n"
                "BB2:\n"
                "  ret void\n"
                "}\n");
  DominatorTree DT(*F);
  for (auto &BB : *F) {
    if (&BB == &F->getEntryBlock())
      continue;

    EXPECT_EQ(isGuaranteedNotToBePoison(A, nullptr, BB.getTerminator(), &DT),
              true)
        << "isGuaranteedNotToBePoison does not hold at " << *BB.getTerminator();
  }
}

TEST_F(ValueTrackingTest, isGuaranteedNotToBePoison_phi) {
  parseAssembly("declare i32 @any_i32(i32)"
                "define void @test() {\n"
                "ENTRY:\n"
                "  br label %LOOP\n"
                "LOOP:\n"
                "  %A = phi i32 [0, %ENTRY], [%A.next, %NEXT]\n"
                "  %A.next = call i32 @any_i32(i32 %A)\n"
                "  %cond = icmp eq i32 %A.next, 0\n"
                "  br i1 %cond, label %NEXT, label %EXIT\n"
                "NEXT:\n"
                "  br label %LOOP\n"
                "EXIT:\n"
                "  ret void\n"
                "}\n");
  DominatorTree DT(*F);
  for (auto &BB : *F) {
    if (BB.getName() == "LOOP") {
      EXPECT_EQ(isGuaranteedNotToBePoison(A, nullptr, A, &DT), true)
          << "isGuaranteedNotToBePoison does not hold";
    }
  }
}

TEST_F(ValueTrackingTest, isGuaranteedNotToBeUndefOrPoison) {
  parseAssembly("declare void @f(i32 noundef)"
                "define void @test(i32 %x) {\n"
                "  %A = bitcast i32 %x to i32\n"
                "  call void @f(i32 noundef %x)\n"
                "  ret void\n"
                "}\n");
  EXPECT_EQ(isGuaranteedNotToBeUndefOrPoison(A), true);
  EXPECT_EQ(isGuaranteedNotToBeUndefOrPoison(UndefValue::get(IntegerType::get(Context, 8))), false);
  EXPECT_EQ(isGuaranteedNotToBeUndefOrPoison(PoisonValue::get(IntegerType::get(Context, 8))), false);
  EXPECT_EQ(isGuaranteedNotToBePoison(UndefValue::get(IntegerType::get(Context, 8))), true);
  EXPECT_EQ(isGuaranteedNotToBePoison(PoisonValue::get(IntegerType::get(Context, 8))), false);

  Type *Int32Ty = Type::getInt32Ty(Context);
  Constant *CU = UndefValue::get(Int32Ty);
  Constant *CP = PoisonValue::get(Int32Ty);
  Constant *C1 = ConstantInt::get(Int32Ty, 1);
  Constant *C2 = ConstantInt::get(Int32Ty, 2);

  {
    Constant *V1 = ConstantVector::get({C1, C2});
    EXPECT_TRUE(isGuaranteedNotToBeUndefOrPoison(V1));
    EXPECT_TRUE(isGuaranteedNotToBePoison(V1));
  }

  {
    Constant *V2 = ConstantVector::get({C1, CU});
    EXPECT_FALSE(isGuaranteedNotToBeUndefOrPoison(V2));
    EXPECT_TRUE(isGuaranteedNotToBePoison(V2));
  }

  {
    Constant *V3 = ConstantVector::get({C1, CP});
    EXPECT_FALSE(isGuaranteedNotToBeUndefOrPoison(V3));
    EXPECT_FALSE(isGuaranteedNotToBePoison(V3));
  }
}

TEST_F(ValueTrackingTest, isGuaranteedNotToBeUndefOrPoison_assume) {
  parseAssembly("declare i1 @f_i1()\n"
                "declare i32 @f_i32()\n"
                "declare void @llvm.assume(i1)\n"
                "define void @test() {\n"
                "  %A = call i32 @f_i32()\n"
                "  %cond = call i1 @f_i1()\n"
                "  %CxtI = add i32 0, 0\n"
                "  br i1 %cond, label %BB1, label %EXIT\n"
                "BB1:\n"
                "  %CxtI2 = add i32 0, 0\n"
                "  %cond2 = call i1 @f_i1()\n"
                "  call void @llvm.assume(i1 true) [ \"noundef\"(i32 %A) ]\n"
                "  br i1 %cond2, label %BB2, label %EXIT\n"
                "BB2:\n"
                "  %CxtI3 = add i32 0, 0\n"
                "  ret void\n"
                "EXIT:\n"
                "  ret void\n"
                "}");
  AssumptionCache AC(*F);
  DominatorTree DT(*F);
  EXPECT_FALSE(isGuaranteedNotToBeUndefOrPoison(A, &AC, CxtI, &DT));
  EXPECT_FALSE(isGuaranteedNotToBeUndefOrPoison(A, &AC, CxtI2, &DT));
  EXPECT_TRUE(isGuaranteedNotToBeUndefOrPoison(A, &AC, CxtI3, &DT));
}

TEST(ValueTracking, canCreatePoisonOrUndef) {
  std::string AsmHead =
      "@s = external dso_local global i32, align 1\n"
      "declare i32 @g(i32)\n"
      "declare {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)\n"
      "declare {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)\n"
      "declare {i32, i1} @llvm.smul.with.overflow.i32(i32 %a, i32 %b)\n"
      "declare {i32, i1} @llvm.uadd.with.overflow.i32(i32 %a, i32 %b)\n"
      "declare {i32, i1} @llvm.usub.with.overflow.i32(i32 %a, i32 %b)\n"
      "declare {i32, i1} @llvm.umul.with.overflow.i32(i32 %a, i32 %b)\n"
      "define void @f(i32 %x, i32 %y, float %fx, float %fy, i1 %cond, "
      "<4 x i32> %vx, <4 x i32> %vx2, <vscale x 4 x i32> %svx, ptr %p) {\n";
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
      {{false, false}, "getelementptr i8, ptr %p, i32 %x"},
      {{true, false}, "getelementptr inbounds i8, ptr %p, i32 %x"},
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
      {{true, false},
       "shufflevector <4 x i32> %vx, <4 x i32> %vx2, "
       "<4 x i32> <i32 0, i32 1, i32 2, i32 poison>"},
      {{true, false},
       "shufflevector <vscale x 4 x i32> %svx, "
       "<vscale x 4 x i32> %svx, <vscale x 4 x i32> poison"},
      {{true, false}, "call i32 @g(i32 %x)"},
      {{false, false}, "call noundef i32 @g(i32 %x)"},
      {{true, false}, "fcmp nnan oeq float %fx, %fy"},
      {{false, false}, "fcmp oeq float %fx, %fy"},
      {{true, false}, "ashr i32 %x, ptrtoint (ptr @s to i32)"},
      {{false, false},
       "call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %x, i32 %y)"},
      {{false, false},
       "call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %x, i32 %y)"},
      {{false, false},
       "call {i32, i1} @llvm.smul.with.overflow.i32(i32 %x, i32 %y)"},
      {{false, false},
       "call {i32, i1} @llvm.uadd.with.overflow.i32(i32 %x, i32 %y)"},
      {{false, false},
       "call {i32, i1} @llvm.usub.with.overflow.i32(i32 %x, i32 %y)"},
      {{false, false},
       "call {i32, i1} @llvm.umul.with.overflow.i32(i32 %x, i32 %y)"}};

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

TEST_F(ValueTrackingTest, computePtrAlignment) {
  parseAssembly("declare i1 @f_i1()\n"
                "declare ptr @f_i8p()\n"
                "declare void @llvm.assume(i1)\n"
                "define void @test() {\n"
                "  %A = call ptr @f_i8p()\n"
                "  %cond = call i1 @f_i1()\n"
                "  %CxtI = add i32 0, 0\n"
                "  br i1 %cond, label %BB1, label %EXIT\n"
                "BB1:\n"
                "  %CxtI2 = add i32 0, 0\n"
                "  %cond2 = call i1 @f_i1()\n"
                "  call void @llvm.assume(i1 true) [ \"align\"(ptr %A, i64 16) ]\n"
                "  br i1 %cond2, label %BB2, label %EXIT\n"
                "BB2:\n"
                "  %CxtI3 = add i32 0, 0\n"
                "  ret void\n"
                "EXIT:\n"
                "  ret void\n"
                "}");
  AssumptionCache AC(*F);
  DominatorTree DT(*F);
  const DataLayout &DL = M->getDataLayout();
  EXPECT_EQ(getKnownAlignment(A, DL, CxtI, &AC, &DT), Align(1));
  EXPECT_EQ(getKnownAlignment(A, DL, CxtI2, &AC, &DT), Align(1));
  EXPECT_EQ(getKnownAlignment(A, DL, CxtI3, &AC, &DT), Align(16));
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

TEST_F(ComputeKnownFPClassTest, SelectPos0) {
  parseAssembly(
      "define float @test(i1 %cond) {\n"
      "  %A = select i1 %cond, float 0.0, float 0.0"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(fcPosZero, false);
}

TEST_F(ComputeKnownFPClassTest, SelectNeg0) {
  parseAssembly(
      "define float @test(i1 %cond) {\n"
      "  %A = select i1 %cond, float -0.0, float -0.0"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(fcNegZero, true);
}

TEST_F(ComputeKnownFPClassTest, SelectPosOrNeg0) {
  parseAssembly(
      "define float @test(i1 %cond) {\n"
      "  %A = select i1 %cond, float 0.0, float -0.0"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(fcZero, std::nullopt);
}

TEST_F(ComputeKnownFPClassTest, SelectPosInf) {
  parseAssembly(
      "define float @test(i1 %cond) {\n"
      "  %A = select i1 %cond, float 0x7FF0000000000000, float 0x7FF0000000000000"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(fcPosInf, false);
}

TEST_F(ComputeKnownFPClassTest, SelectNegInf) {
  parseAssembly(
      "define float @test(i1 %cond) {\n"
      "  %A = select i1 %cond, float 0xFFF0000000000000, float 0xFFF0000000000000"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(fcNegInf, true);
}

TEST_F(ComputeKnownFPClassTest, SelectPosOrNegInf) {
  parseAssembly(
      "define float @test(i1 %cond) {\n"
      "  %A = select i1 %cond, float 0x7FF0000000000000, float 0xFFF0000000000000"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(fcInf, std::nullopt);
}

TEST_F(ComputeKnownFPClassTest, SelectNNaN) {
  parseAssembly(
      "define float @test(i1 %cond, float %arg0, float %arg1) {\n"
      "  %A = select nnan i1 %cond, float %arg0, float %arg1"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(~fcNan, std::nullopt);
}

TEST_F(ComputeKnownFPClassTest, SelectNInf) {
  parseAssembly(
      "define float @test(i1 %cond, float %arg0, float %arg1) {\n"
      "  %A = select ninf i1 %cond, float %arg0, float %arg1"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(~fcInf, std::nullopt);
}

TEST_F(ComputeKnownFPClassTest, SelectNNaNNInf) {
  parseAssembly(
      "define float @test(i1 %cond, float %arg0, float %arg1) {\n"
      "  %A = select nnan ninf i1 %cond, float %arg0, float %arg1"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(~(fcNan | fcInf), std::nullopt);
}

TEST_F(ComputeKnownFPClassTest, SelectNoFPClassArgUnionAll) {
  parseAssembly(
      "define float @test(i1 %cond, float nofpclass(snan ninf nsub pzero pnorm) %arg0, float nofpclass(qnan nnorm nzero psub pinf) %arg1) {\n"
      "  %A = select i1 %cond, float %arg0, float %arg1"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(fcAllFlags, std::nullopt);
}

TEST_F(ComputeKnownFPClassTest, SelectNoFPClassArgNoNan) {
  parseAssembly(
      "define float @test(i1 %cond, float nofpclass(nan) %arg0, float nofpclass(nan) %arg1) {\n"
      "  %A = select i1 %cond, float %arg0, float %arg1"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(~fcNan, std::nullopt);
}

TEST_F(ComputeKnownFPClassTest, SelectNoFPClassArgNoPInf) {
  parseAssembly(
      "define float @test(i1 %cond, float nofpclass(inf) %arg0, float nofpclass(pinf) %arg1) {\n"
      "  %A = select i1 %cond, float %arg0, float %arg1"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(~fcPosInf, std::nullopt);
}

TEST_F(ComputeKnownFPClassTest, SelectNoFPClassArgNoNInf) {
  parseAssembly(
      "define float @test(i1 %cond, float nofpclass(ninf) %arg0, float nofpclass(inf) %arg1) {\n"
      "  %A = select i1 %cond, float %arg0, float %arg1"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(~fcNegInf, std::nullopt);
}

TEST_F(ComputeKnownFPClassTest, SelectNoFPClassCallSiteNoNan) {
  parseAssembly(
      "declare float @func()\n"
      "define float @test() {\n"
      "  %A = call nofpclass(nan) float @func()\n"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(~fcNan, std::nullopt);
}

TEST_F(ComputeKnownFPClassTest, SelectNoFPClassCallSiteNoZeros) {
  parseAssembly(
      "declare float @func()\n"
      "define float @test() {\n"
      "  %A = call nofpclass(zero) float @func()\n"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(~fcZero, std::nullopt);
}

TEST_F(ComputeKnownFPClassTest, SelectNoFPClassDeclarationNoNan) {
  parseAssembly(
      "declare nofpclass(nan) float @no_nans()\n"
      "define float @test() {\n"
      "  %A = call float @no_nans()\n"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(~fcNan, std::nullopt);
}

// Check nofpclass + ninf works on a callsite
TEST_F(ComputeKnownFPClassTest, SelectNoFPClassCallSiteNoZerosNInfFlags) {
  parseAssembly(
      "declare float @func()\n"
      "define float @test() {\n"
      "  %A = call ninf nofpclass(zero) float @func()\n"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(~(fcZero | fcInf), std::nullopt);
}

TEST_F(ComputeKnownFPClassTest, FNegNInf) {
  parseAssembly(
      "define float @test(float %arg) {\n"
      "  %A = fneg ninf float %arg"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(~fcInf, std::nullopt);
}

TEST_F(ComputeKnownFPClassTest, FabsUnknown) {
  parseAssembly(
      "declare float @llvm.fabs.f32(float)"
      "define float @test(float %arg) {\n"
      "  %A = call float @llvm.fabs.f32(float %arg)"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(fcPositive | fcNan, false);
}

TEST_F(ComputeKnownFPClassTest, FNegFabsUnknown) {
  parseAssembly(
      "declare float @llvm.fabs.f32(float)"
      "define float @test(float %arg) {\n"
      "  %fabs = call float @llvm.fabs.f32(float %arg)"
      "  %A = fneg float %fabs"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(fcNegative | fcNan, true);
}

TEST_F(ComputeKnownFPClassTest, NegFabsNInf) {
  parseAssembly(
      "declare float @llvm.fabs.f32(float)"
      "define float @test(float %arg) {\n"
      "  %fabs = call ninf float @llvm.fabs.f32(float %arg)"
      "  %A = fneg float %fabs"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass((fcNegative & ~fcNegInf) | fcNan, true);
}

TEST_F(ComputeKnownFPClassTest, FNegFabsNNaN) {
  parseAssembly(
      "declare float @llvm.fabs.f32(float)"
      "define float @test(float %arg) {\n"
      "  %fabs = call nnan float @llvm.fabs.f32(float %arg)"
      "  %A = fneg float %fabs"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(fcNegative, true);
}

TEST_F(ComputeKnownFPClassTest, CopySignNNanSrc0) {
  parseAssembly(
      "declare float @llvm.fabs.f32(float)\n"
      "declare float @llvm.copysign.f32(float, float)\n"
      "define float @test(float %arg0, float %arg1) {\n"
      "  %fabs = call nnan float @llvm.fabs.f32(float %arg0)"
      "  %A = call float @llvm.copysign.f32(float %fabs, float %arg1)"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(~fcNan, std::nullopt);
}

TEST_F(ComputeKnownFPClassTest, CopySignNInfSrc0_NegSign) {
  parseAssembly(
      "declare float @llvm.log.f32(float)\n"
      "declare float @llvm.copysign.f32(float, float)\n"
      "define float @test(float %arg0, float %arg1) {\n"
      "  %ninf = call ninf float @llvm.log.f32(float %arg0)"
      "  %A = call float @llvm.copysign.f32(float %ninf, float -1.0)"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(fcNegFinite | fcNan, true);
}

TEST_F(ComputeKnownFPClassTest, CopySignNInfSrc0_PosSign) {
  parseAssembly(
      "declare float @llvm.sqrt.f32(float)\n"
      "declare float @llvm.copysign.f32(float, float)\n"
      "define float @test(float %arg0, float %arg1) {\n"
      "  %ninf = call ninf float @llvm.sqrt.f32(float %arg0)"
      "  %A = call float @llvm.copysign.f32(float %ninf, float 1.0)"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(fcPosFinite | fcNan, false);
}

TEST_F(ComputeKnownFPClassTest, UIToFP) {
  parseAssembly(
      "define float @test(i32 %arg0, i16 %arg1) {\n"
      "  %A = uitofp i32 %arg0 to float"
      "  %A2 = uitofp i16 %arg1 to half"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(fcPosFinite & ~fcSubnormal, false, A);
  expectKnownFPClass(fcPositive & ~fcSubnormal, false, A2);
}

TEST_F(ComputeKnownFPClassTest, SIToFP) {
  parseAssembly(
      "define float @test(i32 %arg0, i16 %arg1, i17 %arg2) {\n"
      "  %A = sitofp i32 %arg0 to float"
      "  %A2 = sitofp i16 %arg1 to half"
      "  %A3 = sitofp i17 %arg2 to half"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(fcFinite & ~fcNegZero & ~fcSubnormal, std::nullopt, A);
  expectKnownFPClass(fcFinite & ~fcNegZero & ~fcSubnormal, std::nullopt, A2);
  expectKnownFPClass(~(fcNan | fcNegZero | fcSubnormal), std::nullopt, A3);
}

TEST_F(ComputeKnownFPClassTest, FAdd) {
  parseAssembly(
      "define float @test(float nofpclass(nan inf) %nnan.ninf, float nofpclass(nan) %nnan, float nofpclass(qnan) %no.qnan, float %unknown) {\n"
      "  %A = fadd float %nnan, %nnan.ninf"
      "  %A2 = fadd float %nnan.ninf, %nnan"
      "  %A3 = fadd float %nnan.ninf, %unknown"
      "  %A4 = fadd float %nnan.ninf, %no.qnan"
      "  %A5 = fadd float %nnan, %nnan"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(fcFinite | fcInf, std::nullopt, A);
  expectKnownFPClass(fcFinite | fcInf, std::nullopt, A2);
  expectKnownFPClass(fcAllFlags, std::nullopt, A3);
  expectKnownFPClass(fcAllFlags, std::nullopt, A4);
  expectKnownFPClass(fcAllFlags, std::nullopt, A5);
}

TEST_F(ComputeKnownFPClassTest, FSub) {
  parseAssembly(
      "define float @test(float nofpclass(nan inf) %nnan.ninf, float nofpclass(nan) %nnan, float nofpclass(qnan) %no.qnan, float %unknown) {\n"
      "  %A = fsub float %nnan, %nnan.ninf"
      "  %A2 = fsub float %nnan.ninf, %nnan"
      "  %A3 = fsub float %nnan.ninf, %unknown"
      "  %A4 = fsub float %nnan.ninf, %no.qnan"
      "  %A5 = fsub float %nnan, %nnan"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(fcFinite | fcInf, std::nullopt, A);
  expectKnownFPClass(fcFinite | fcInf, std::nullopt, A2);
  expectKnownFPClass(fcAllFlags, std::nullopt, A3);
  expectKnownFPClass(fcAllFlags, std::nullopt, A4);
  expectKnownFPClass(fcAllFlags, std::nullopt, A5);
}

TEST_F(ComputeKnownFPClassTest, FMul) {
  parseAssembly(
      "define float @test(float nofpclass(nan inf) %nnan.ninf0, float nofpclass(nan inf) %nnan.ninf1, float nofpclass(nan) %nnan, float nofpclass(qnan) %no.qnan, float %unknown) {\n"
      "  %A = fmul float %nnan.ninf0, %nnan.ninf1"
      "  %A2 = fmul float %nnan.ninf0, %nnan"
      "  %A3 = fmul float %nnan, %nnan.ninf0"
      "  %A4 = fmul float %nnan.ninf0, %no.qnan"
      "  %A5 = fmul float %nnan, %nnan"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(fcFinite | fcInf, std::nullopt, A);
  expectKnownFPClass(fcAllFlags, std::nullopt, A2);
  expectKnownFPClass(fcAllFlags, std::nullopt, A3);
  expectKnownFPClass(fcAllFlags, std::nullopt, A4);
  expectKnownFPClass(fcPositive | fcNan, std::nullopt, A5);
}

TEST_F(ComputeKnownFPClassTest, FMulNoZero) {
  parseAssembly(
      "define float @test(float nofpclass(zero) %no.zero, float nofpclass(zero nan) %no.zero.nan0, float nofpclass(zero nan) %no.zero.nan1, float nofpclass(nzero nan) %no.negzero.nan, float nofpclass(pzero nan) %no.poszero.nan, float nofpclass(inf nan) %no.inf.nan, float nofpclass(inf) %no.inf, float nofpclass(nan) %no.nan) {\n"
      "  %A = fmul float %no.zero.nan0, %no.zero.nan1"
      "  %A2 = fmul float %no.zero, %no.zero"
      "  %A3 = fmul float %no.poszero.nan, %no.zero.nan0"
      "  %A4 = fmul float %no.nan, %no.zero"
      "  %A5 = fmul float %no.zero, %no.inf"
      "  %A6 = fmul float %no.zero.nan0, %no.nan"
      "  %A7 = fmul float %no.nan, %no.zero.nan0"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(fcFinite | fcInf, std::nullopt, A);
  expectKnownFPClass(fcPositive | fcNan, std::nullopt, A2);
  expectKnownFPClass(fcAllFlags, std::nullopt, A3);
  expectKnownFPClass(fcAllFlags, std::nullopt, A4);
  expectKnownFPClass(fcAllFlags, std::nullopt, A5);
  expectKnownFPClass(fcAllFlags, std::nullopt, A6);
  expectKnownFPClass(fcAllFlags, std::nullopt, A7);
}

TEST_F(ComputeKnownFPClassTest, Phi) {
  parseAssembly(
      "define float @test(i1 %cond, float nofpclass(nan inf) %arg0, float nofpclass(nan) %arg1) {\n"
      "entry:\n"
      "  br i1 %cond, label %bb0, label %bb1\n"
      "bb0:\n"
      "  br label %ret\n"
      "bb1:\n"
      "  br label %ret\n"
      "ret:\n"
      "  %A = phi float [ %arg0, %bb0 ],  [ %arg1, %bb1 ]\n"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(~fcNan, std::nullopt);
}

TEST_F(ComputeKnownFPClassTest, PhiKnownSignFalse) {
  parseAssembly(
      "declare float @llvm.fabs.f32(float)"
      "define float @test(i1 %cond, float nofpclass(nan) %arg0, float nofpclass(nan) %arg1) {\n"
      "entry:\n"
      "  br i1 %cond, label %bb0, label %bb1\n"
      "bb0:\n"
      "  %fabs.arg0 = call float @llvm.fabs.f32(float %arg0)\n"
      "  br label %ret\n"
      "bb1:\n"
      "  %fabs.arg1 = call float @llvm.fabs.f32(float %arg1)\n"
      "  br label %ret\n"
      "ret:\n"
      "  %A = phi float [ %fabs.arg0, %bb0 ],  [ %fabs.arg1, %bb1 ]\n"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(fcPositive, false);
}

TEST_F(ComputeKnownFPClassTest, PhiKnownSignTrue) {
  parseAssembly(
      "declare float @llvm.fabs.f32(float)"
      "define float @test(i1 %cond, float nofpclass(nan) %arg0, float %arg1) {\n"
      "entry:\n"
      "  br i1 %cond, label %bb0, label %bb1\n"
      "bb0:\n"
      "  %fabs.arg0 = call float @llvm.fabs.f32(float %arg0)\n"
      "  %fneg.fabs.arg0 = fneg float %fabs.arg0\n"
      "  br label %ret\n"
      "bb1:\n"
      "  %fabs.arg1 = call float @llvm.fabs.f32(float %arg1)\n"
      "  %fneg.fabs.arg1 = fneg float %fabs.arg1\n"
      "  br label %ret\n"
      "ret:\n"
      "  %A = phi float [ %fneg.fabs.arg0, %bb0 ],  [ %fneg.fabs.arg1, %bb1 ]\n"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(fcNegative | fcNan, true);
}

TEST_F(ComputeKnownFPClassTest, UnreachablePhi) {
  parseAssembly(
      "define float @test(float %arg) {\n"
      "entry:\n"
      "  ret float 0.0\n"
      "unreachable:\n"
      "  %A = phi float\n"
      "  ret float %A\n"
      "}\n");
  expectKnownFPClass(fcAllFlags, std::nullopt);
}

TEST_F(ComputeKnownFPClassTest, SelfPhiOnly) {
  parseAssembly(
      "define float @test(float %arg) {\n"
      "entry:\n"
      "  ret float 0.0\n"
      "loop:\n"
      "  %A = phi float [ %A, %loop ]\n"
      "  br label %loop\n"
      "}\n");
  expectKnownFPClass(fcAllFlags, std::nullopt);
}

TEST_F(ComputeKnownFPClassTest, SelfPhiFirstArg) {
  parseAssembly(
      "define float @test(i1 %cond, float nofpclass(inf) %arg) {\n"
      "entry:\n"
      "  br i1 %cond, label %loop, label %ret\n"
      "loop:\n"
      "  %A = phi float [ %arg, %entry ], [ %A, %loop ]\n"
      "  br label %loop\n"
      "ret:\n"
      "  ret float %A"
      "}\n");
  expectKnownFPClass(~fcInf, std::nullopt);
}

TEST_F(ComputeKnownFPClassTest, SelfPhiSecondArg) {
  parseAssembly(
      "define float @test(i1 %cond, float nofpclass(inf) %arg) {\n"
      "entry:\n"
      "  br i1 %cond, label %loop, label %ret\n"
      "loop:\n"
      "  %A = phi float [ %A, %loop ], [ %arg, %entry ]\n"
      "  br label %loop\n"
      "ret:\n"
      "  ret float %A"
      "}\n");
  expectKnownFPClass(~fcInf, std::nullopt);
}

TEST_F(ComputeKnownFPClassTest, CannotBeOrderedLessThanZero) {
  parseAssembly("define float @test(float %arg) {\n"
                "  %A = fmul float %arg, %arg"
                "  ret float %A\n"
                "}\n");

  Type *FPTy = Type::getDoubleTy(M->getContext());
  const DataLayout &DL = M->getDataLayout();

  EXPECT_TRUE(
      computeKnownFPClass(ConstantFP::getZero(FPTy, /*Negative=*/false), DL)
          .cannotBeOrderedLessThanZero());
  EXPECT_TRUE(
      computeKnownFPClass(ConstantFP::getZero(FPTy, /*Negative=*/true), DL)
          .cannotBeOrderedLessThanZero());

  EXPECT_TRUE(computeKnownFPClass(ConstantFP::getInfinity(FPTy, false), DL)
                  .cannotBeOrderedLessThanZero());
  EXPECT_FALSE(computeKnownFPClass(ConstantFP::getInfinity(FPTy, true), DL)
                   .cannotBeOrderedLessThanZero());

  EXPECT_TRUE(computeKnownFPClass(ConstantFP::get(FPTy, 1.0), DL)
                  .cannotBeOrderedLessThanZero());
  EXPECT_FALSE(computeKnownFPClass(ConstantFP::get(FPTy, -1.0), DL)
                   .cannotBeOrderedLessThanZero());

  EXPECT_TRUE(
      computeKnownFPClass(
          ConstantFP::get(FPTy, APFloat::getSmallest(FPTy->getFltSemantics(),
                                                     /*Negative=*/false)),
          DL)
          .cannotBeOrderedLessThanZero());
  EXPECT_FALSE(
      computeKnownFPClass(
          ConstantFP::get(FPTy, APFloat::getSmallest(FPTy->getFltSemantics(),
                                                     /*Negative=*/true)),
          DL)
          .cannotBeOrderedLessThanZero());

  EXPECT_TRUE(
      computeKnownFPClass(ConstantFP::getQNaN(FPTy, /*Negative=*/false), DL)
          .cannotBeOrderedLessThanZero());
  EXPECT_TRUE(
      computeKnownFPClass(ConstantFP::getQNaN(FPTy, /*Negative=*/true), DL)
          .cannotBeOrderedLessThanZero());
  EXPECT_TRUE(
      computeKnownFPClass(ConstantFP::getSNaN(FPTy, /*Negative=*/false), DL)
          .cannotBeOrderedLessThanZero());
  EXPECT_TRUE(
      computeKnownFPClass(ConstantFP::getSNaN(FPTy, /*Negative=*/true), DL)
          .cannotBeOrderedLessThanZero());
}

TEST_F(ComputeKnownFPClassTest, FCmpToClassTest_OrdNan) {
  parseAssembly("define i1 @test(double %arg) {\n"
                "  %A = fcmp ord double %arg, 0x7FF8000000000000"
                "  %A2 = fcmp uno double %arg, 0x7FF8000000000000"
                "  %A3 = fcmp oeq double %arg, 0x7FF8000000000000"
                "  %A4 = fcmp ueq double %arg, 0x7FF8000000000000"
                "  ret i1 %A\n"
                "}\n");

  auto [OrdVal, OrdClass] = fcmpToClassTest(
      CmpInst::FCMP_ORD, *A->getFunction(), A->getOperand(0), A->getOperand(1));
  EXPECT_EQ(A->getOperand(0), OrdVal);
  EXPECT_EQ(fcNone, OrdClass);

  auto [UnordVal, UnordClass] =
      fcmpToClassTest(CmpInst::FCMP_UNO, *A2->getFunction(), A2->getOperand(0),
                      A2->getOperand(1));
  EXPECT_EQ(A2->getOperand(0), UnordVal);
  EXPECT_EQ(fcAllFlags, UnordClass);

  auto [OeqVal, OeqClass] =
      fcmpToClassTest(CmpInst::FCMP_OEQ, *A3->getFunction(), A3->getOperand(0),
                      A3->getOperand(1));
  EXPECT_EQ(A3->getOperand(0), OeqVal);
  EXPECT_EQ(fcNone, OeqClass);

  auto [UeqVal, UeqClass] =
      fcmpToClassTest(CmpInst::FCMP_UEQ, *A3->getFunction(), A3->getOperand(0),
                      A3->getOperand(1));
  EXPECT_EQ(A3->getOperand(0), UeqVal);
  EXPECT_EQ(fcAllFlags, UeqClass);
}

TEST_F(ComputeKnownFPClassTest, FCmpToClassTest_NInf) {
  parseAssembly("define i1 @test(double %arg) {\n"
                "  %A = fcmp olt double %arg, 0xFFF0000000000000"
                "  %A2 = fcmp uge double %arg, 0xFFF0000000000000"
                "  %A3 = fcmp ogt double %arg, 0xFFF0000000000000"
                "  %A4 = fcmp ule double %arg, 0xFFF0000000000000"
                "  %A5 = fcmp oge double %arg, 0xFFF0000000000000"
                "  %A6 = fcmp ult double %arg, 0xFFF0000000000000"
                "  ret i1 %A\n"
                "}\n");

  auto [OltVal, OltClass] = fcmpToClassTest(
      CmpInst::FCMP_OLT, *A->getFunction(), A->getOperand(0), A->getOperand(1));
  EXPECT_EQ(A->getOperand(0), OltVal);
  EXPECT_EQ(fcNone, OltClass);

  auto [UgeVal, UgeClass] =
      fcmpToClassTest(CmpInst::FCMP_UGE, *A2->getFunction(), A2->getOperand(0),
                      A2->getOperand(1));
  EXPECT_EQ(A2->getOperand(0), UgeVal);
  EXPECT_EQ(fcAllFlags, UgeClass);

  auto [OgtVal, OgtClass] =
      fcmpToClassTest(CmpInst::FCMP_OGT, *A3->getFunction(), A3->getOperand(0),
                      A3->getOperand(1));
  EXPECT_EQ(A3->getOperand(0), OgtVal);
  EXPECT_EQ(~(fcNegInf | fcNan), OgtClass);

  auto [UleVal, UleClass] =
      fcmpToClassTest(CmpInst::FCMP_ULE, *A4->getFunction(), A4->getOperand(0),
                      A4->getOperand(1));
  EXPECT_EQ(A4->getOperand(0), UleVal);
  EXPECT_EQ(fcNegInf | fcNan, UleClass);

  auto [OgeVal, OgeClass] =
      fcmpToClassTest(CmpInst::FCMP_OGE, *A5->getFunction(), A5->getOperand(0),
                      A5->getOperand(1));
  EXPECT_EQ(A5->getOperand(0), OgeVal);
  EXPECT_EQ(~fcNan, OgeClass);

  auto [UltVal, UltClass] =
      fcmpToClassTest(CmpInst::FCMP_ULT, *A6->getFunction(), A6->getOperand(0),
                      A6->getOperand(1));
  EXPECT_EQ(A6->getOperand(0), UltVal);
  EXPECT_EQ(fcNan, UltClass);
}

TEST_F(ComputeKnownFPClassTest, FCmpToClassTest_FabsNInf) {
  parseAssembly("declare double @llvm.fabs.f64(double)\n"
                "define i1 @test(double %arg) {\n"
                "  %fabs.arg = call double @llvm.fabs.f64(double %arg)\n"
                "  %A = fcmp olt double %fabs.arg, 0xFFF0000000000000"
                "  %A2 = fcmp uge double %fabs.arg, 0xFFF0000000000000"
                "  %A3 = fcmp ogt double %fabs.arg, 0xFFF0000000000000"
                "  %A4 = fcmp ule double %fabs.arg, 0xFFF0000000000000"
                "  %A5 = fcmp oge double %fabs.arg, 0xFFF0000000000000"
                "  %A6 = fcmp ult double %fabs.arg, 0xFFF0000000000000"
                "  ret i1 %A\n"
                "}\n");

  Value *ArgVal = F->getArg(0);

  auto [OltVal, OltClass] = fcmpToClassTest(
      CmpInst::FCMP_OLT, *A->getFunction(), A->getOperand(0), A->getOperand(1));
  EXPECT_EQ(ArgVal, OltVal);
  EXPECT_EQ(fcNone, OltClass);

  auto [UgeVal, UgeClass] =
      fcmpToClassTest(CmpInst::FCMP_UGE, *A2->getFunction(), A2->getOperand(0),
                      A2->getOperand(1));
  EXPECT_EQ(ArgVal, UgeVal);
  EXPECT_EQ(fcAllFlags, UgeClass);

  auto [OgtVal, OgtClass] =
      fcmpToClassTest(CmpInst::FCMP_OGT, *A3->getFunction(), A3->getOperand(0),
                      A3->getOperand(1));
  EXPECT_EQ(ArgVal, OgtVal);
  EXPECT_EQ(~fcNan, OgtClass);

  auto [UleVal, UleClass] =
      fcmpToClassTest(CmpInst::FCMP_ULE, *A4->getFunction(), A4->getOperand(0),
                      A4->getOperand(1));
  EXPECT_EQ(ArgVal, UleVal);
  EXPECT_EQ(fcNan, UleClass);

  auto [OgeVal, OgeClass] =
      fcmpToClassTest(CmpInst::FCMP_OGE, *A5->getFunction(), A5->getOperand(0),
                      A5->getOperand(1));
  EXPECT_EQ(ArgVal, OgeVal);
  EXPECT_EQ(~fcNan, OgeClass);

  auto [UltVal, UltClass] =
      fcmpToClassTest(CmpInst::FCMP_ULT, *A6->getFunction(), A6->getOperand(0),
                      A6->getOperand(1));
  EXPECT_EQ(ArgVal, UltVal);
  EXPECT_EQ(fcNan, UltClass);
}

TEST_F(ComputeKnownFPClassTest, FCmpToClassTest_PInf) {
  parseAssembly("define i1 @test(double %arg) {\n"
                "  %A = fcmp ogt double %arg, 0x7FF0000000000000"
                "  %A2 = fcmp ule double %arg, 0x7FF0000000000000"
                "  %A3 = fcmp ole double %arg, 0x7FF0000000000000"
                "  %A4 = fcmp ugt double %arg, 0x7FF0000000000000"
                "  ret i1 %A\n"
                "}\n");

  auto [OgtVal, OgtClass] = fcmpToClassTest(
      CmpInst::FCMP_OGT, *A->getFunction(), A->getOperand(0), A->getOperand(1));
  EXPECT_EQ(A->getOperand(0), OgtVal);
  EXPECT_EQ(fcNone, OgtClass);

  auto [UleVal, UleClass] =
      fcmpToClassTest(CmpInst::FCMP_ULE, *A2->getFunction(), A2->getOperand(0),
                      A2->getOperand(1));
  EXPECT_EQ(A2->getOperand(0), UleVal);
  EXPECT_EQ(fcAllFlags, UleClass);

  auto [OleVal, OleClass] =
      fcmpToClassTest(CmpInst::FCMP_OLE, *A3->getFunction(), A3->getOperand(0),
                      A3->getOperand(1));
  EXPECT_EQ(A->getOperand(0), OleVal);
  EXPECT_EQ(~fcNan, OleClass);

  auto [UgtVal, UgtClass] =
      fcmpToClassTest(CmpInst::FCMP_UGT, *A4->getFunction(), A4->getOperand(0),
                      A4->getOperand(1));
  EXPECT_EQ(A4->getOperand(0), UgtVal);
  EXPECT_EQ(fcNan, UgtClass);
}

TEST_F(ComputeKnownFPClassTest, SqrtNszSignBit) {
  parseAssembly(
      "declare float @llvm.sqrt.f32(float)\n"
      "define float @test(float %arg, float nofpclass(nan) %arg.nnan) {\n"
      "  %A = call float @llvm.sqrt.f32(float %arg)\n"
      "  %A2 = call nsz float @llvm.sqrt.f32(float %arg)\n"
      "  %A3 = call float @llvm.sqrt.f32(float %arg.nnan)\n"
      "  %A4 = call nsz float @llvm.sqrt.f32(float %arg.nnan)\n"
      "  ret float %A\n"
      "}\n");

  const FPClassTest SqrtMask = fcPositive | fcNegZero | fcNan;
  const FPClassTest NszSqrtMask = fcPositive | fcNan;

  {
    KnownFPClass UseInstrInfo =
        computeKnownFPClass(A, M->getDataLayout(), fcAllFlags, 0, nullptr,
                            nullptr, nullptr, nullptr, /*UseInstrInfo=*/true);
    EXPECT_EQ(SqrtMask, UseInstrInfo.KnownFPClasses);
    EXPECT_EQ(std::nullopt, UseInstrInfo.SignBit);

    KnownFPClass NoUseInstrInfo =
        computeKnownFPClass(A, M->getDataLayout(), fcAllFlags, 0, nullptr,
                            nullptr, nullptr, nullptr, /*UseInstrInfo=*/false);
    EXPECT_EQ(SqrtMask, NoUseInstrInfo.KnownFPClasses);
    EXPECT_EQ(std::nullopt, NoUseInstrInfo.SignBit);
  }

  {
    KnownFPClass UseInstrInfoNSZ =
        computeKnownFPClass(A2, M->getDataLayout(), fcAllFlags, 0, nullptr,
                            nullptr, nullptr, nullptr, /*UseInstrInfo=*/true);
    EXPECT_EQ(NszSqrtMask, UseInstrInfoNSZ.KnownFPClasses);
    EXPECT_EQ(std::nullopt, UseInstrInfoNSZ.SignBit);

    KnownFPClass NoUseInstrInfoNSZ =
        computeKnownFPClass(A2, M->getDataLayout(), fcAllFlags, 0, nullptr,
                            nullptr, nullptr, nullptr, /*UseInstrInfo=*/false);
    EXPECT_EQ(SqrtMask, NoUseInstrInfoNSZ.KnownFPClasses);
    EXPECT_EQ(std::nullopt, NoUseInstrInfoNSZ.SignBit);
  }

  {
    KnownFPClass UseInstrInfoNoNan =
        computeKnownFPClass(A3, M->getDataLayout(), fcAllFlags, 0, nullptr,
                            nullptr, nullptr, nullptr, /*UseInstrInfo=*/true);
    EXPECT_EQ(fcPositive | fcNegZero | fcQNan,
              UseInstrInfoNoNan.KnownFPClasses);
    EXPECT_EQ(std::nullopt, UseInstrInfoNoNan.SignBit);

    KnownFPClass NoUseInstrInfoNoNan =
        computeKnownFPClass(A3, M->getDataLayout(), fcAllFlags, 0, nullptr,
                            nullptr, nullptr, nullptr, /*UseInstrInfo=*/false);
    EXPECT_EQ(fcPositive | fcNegZero | fcQNan,
              NoUseInstrInfoNoNan.KnownFPClasses);
    EXPECT_EQ(std::nullopt, NoUseInstrInfoNoNan.SignBit);
  }

  {
    KnownFPClass UseInstrInfoNSZNoNan =
        computeKnownFPClass(A4, M->getDataLayout(), fcAllFlags, 0, nullptr,
                            nullptr, nullptr, nullptr, /*UseInstrInfo=*/true);
    EXPECT_EQ(fcPositive | fcQNan, UseInstrInfoNSZNoNan.KnownFPClasses);
    EXPECT_EQ(std::nullopt, UseInstrInfoNSZNoNan.SignBit);

    KnownFPClass NoUseInstrInfoNSZNoNan =
        computeKnownFPClass(A4, M->getDataLayout(), fcAllFlags, 0, nullptr,
                            nullptr, nullptr, nullptr, /*UseInstrInfo=*/false);
    EXPECT_EQ(fcPositive | fcNegZero | fcQNan,
              NoUseInstrInfoNSZNoNan.KnownFPClasses);
    EXPECT_EQ(std::nullopt, NoUseInstrInfoNSZNoNan.SignBit);
  }
}

TEST_F(ComputeKnownFPClassTest, Constants) {
  parseAssembly("declare float @func()\n"
                "define float @test() {\n"
                "  %A = call float @func()\n"
                "  ret float %A\n"
                "}\n");

  Type *F32 = Type::getFloatTy(Context);
  Type *V4F32 = FixedVectorType::get(F32, 4);

  {
    KnownFPClass ConstAggZero = computeKnownFPClass(
        ConstantAggregateZero::get(V4F32), M->getDataLayout(), fcAllFlags, 0,
        nullptr, nullptr, nullptr, nullptr);

    EXPECT_EQ(fcPosZero, ConstAggZero.KnownFPClasses);
    ASSERT_TRUE(ConstAggZero.SignBit);
    EXPECT_FALSE(*ConstAggZero.SignBit);
  }

  {
    KnownFPClass Undef =
        computeKnownFPClass(UndefValue::get(F32), M->getDataLayout(),
                            fcAllFlags, 0, nullptr, nullptr, nullptr, nullptr);
    EXPECT_EQ(fcAllFlags, Undef.KnownFPClasses);
    EXPECT_FALSE(Undef.SignBit);
  }

  {
    KnownFPClass Poison =
        computeKnownFPClass(PoisonValue::get(F32), M->getDataLayout(),
                            fcAllFlags, 0, nullptr, nullptr, nullptr, nullptr);
    EXPECT_EQ(fcNone, Poison.KnownFPClasses);
    ASSERT_TRUE(Poison.SignBit);
    EXPECT_FALSE(*Poison.SignBit);
  }

  {
    // Assume the poison element should be 0.
    Constant *ZeroF32 = ConstantFP::getZero(F32);
    Constant *PoisonF32 = PoisonValue::get(F32);

    KnownFPClass PartiallyPoison = computeKnownFPClass(
        ConstantVector::get({ZeroF32, PoisonF32}), M->getDataLayout(),
        fcAllFlags, 0, nullptr, nullptr, nullptr, nullptr);
    EXPECT_EQ(fcPosZero, PartiallyPoison.KnownFPClasses);
    ASSERT_TRUE(PartiallyPoison.SignBit);
    EXPECT_FALSE(*PartiallyPoison.SignBit);
  }

  {
    // Assume the poison element should be 1.
    Constant *NegZeroF32 = ConstantFP::getZero(F32, true);
    Constant *PoisonF32 = PoisonValue::get(F32);

    KnownFPClass PartiallyPoison = computeKnownFPClass(
        ConstantVector::get({NegZeroF32, PoisonF32}), M->getDataLayout(),
        fcAllFlags, 0, nullptr, nullptr, nullptr, nullptr);
    EXPECT_EQ(fcNegZero, PartiallyPoison.KnownFPClasses);
    ASSERT_TRUE(PartiallyPoison.SignBit);
    EXPECT_TRUE(*PartiallyPoison.SignBit);
  }

  {
    // Assume the poison element should be 1.
    Constant *NegZeroF32 = ConstantFP::getZero(F32, true);
    Constant *PoisonF32 = PoisonValue::get(F32);

    KnownFPClass PartiallyPoison = computeKnownFPClass(
        ConstantVector::get({PoisonF32, NegZeroF32}), M->getDataLayout(),
        fcAllFlags, 0, nullptr, nullptr, nullptr, nullptr);
    EXPECT_EQ(fcNegZero, PartiallyPoison.KnownFPClasses);
    EXPECT_TRUE(PartiallyPoison.SignBit);
  }
}

TEST_F(ValueTrackingTest, isNonZeroRecurrence) {
  parseAssembly(R"(
    define i1 @test(i8 %n, i8 %r) {
    entry:
      br label %loop
    loop:
      %p = phi i8 [ -1, %entry ], [ %next, %loop ]
      %next = add nsw i8 %p, -1
      %cmp1 = icmp eq i8 %p, %n
      br i1 %cmp1, label %exit, label %loop
    exit:
      %A = or i8 %p, %r
      %CxtI = icmp eq i8 %A, 0
      ret i1 %CxtI
    }
  )");
  const DataLayout &DL = M->getDataLayout();
  AssumptionCache AC(*F);
  EXPECT_TRUE(isKnownNonZero(A, SimplifyQuery(DL, /*DT=*/nullptr, &AC, CxtI)));
}

TEST_F(ValueTrackingTest, KnownNonZeroFromDomCond) {
  parseAssembly(R"(
    declare ptr @f_i8()
    define void @test(i1 %c) {
      %A = call ptr @f_i8()
      %B = call ptr @f_i8()
      %c1 = icmp ne ptr %A, null
      %cond = and i1 %c1, %c
      br i1 %cond, label %T, label %Q
    T:
      %CxtI = add i32 0, 0
      ret void
    Q:
      %CxtI2 = add i32 0, 0
      ret void
    }
  )");
  AssumptionCache AC(*F);
  DominatorTree DT(*F);
  const DataLayout &DL = M->getDataLayout();
  const SimplifyQuery SQ(DL, &DT, &AC);
  EXPECT_EQ(isKnownNonZero(A, SQ.getWithInstruction(CxtI)), true);
  EXPECT_EQ(isKnownNonZero(A, SQ.getWithInstruction(CxtI2)), false);
}

TEST_F(ValueTrackingTest, KnownNonZeroFromDomCond2) {
  parseAssembly(R"(
    declare ptr @f_i8()
    define void @test(i1 %c) {
      %A = call ptr @f_i8()
      %B = call ptr @f_i8()
      %c1 = icmp ne ptr %A, null
      %cond = select i1 %c, i1 %c1, i1 false
      br i1 %cond, label %T, label %Q
    T:
      %CxtI = add i32 0, 0
      ret void
    Q:
      %CxtI2 = add i32 0, 0
      ret void
    }
  )");
  AssumptionCache AC(*F);
  DominatorTree DT(*F);
  const DataLayout &DL = M->getDataLayout();
  const SimplifyQuery SQ(DL, &DT, &AC);
  EXPECT_EQ(isKnownNonZero(A, SQ.getWithInstruction(CxtI)), true);
  EXPECT_EQ(isKnownNonZero(A, SQ.getWithInstruction(CxtI2)), false);
}

TEST_F(ValueTrackingTest, IsImpliedConditionAnd) {
  parseAssembly(R"(
    define void @test(i32 %x, i32 %y) {
      %c1 = icmp ult i32 %x, 10
      %c2 = icmp ult i32 %y, 15
      %A = and i1 %c1, %c2
      ; x < 10 /\ y < 15
      %A2 = icmp ult i32 %x, 20
      %A3 = icmp uge i32 %y, 20
      %A4 = icmp ult i32 %x, 5
      ret void
    }
  )");
  const DataLayout &DL = M->getDataLayout();
  EXPECT_EQ(isImpliedCondition(A, A2, DL), true);
  EXPECT_EQ(isImpliedCondition(A, A3, DL), false);
  EXPECT_EQ(isImpliedCondition(A, A4, DL), std::nullopt);
}

TEST_F(ValueTrackingTest, IsImpliedConditionAnd2) {
  parseAssembly(R"(
    define void @test(i32 %x, i32 %y) {
      %c1 = icmp ult i32 %x, 10
      %c2 = icmp ult i32 %y, 15
      %A = select i1 %c1, i1 %c2, i1 false
      ; x < 10 /\ y < 15
      %A2 = icmp ult i32 %x, 20
      %A3 = icmp uge i32 %y, 20
      %A4 = icmp ult i32 %x, 5
      ret void
    }
  )");
  const DataLayout &DL = M->getDataLayout();
  EXPECT_EQ(isImpliedCondition(A, A2, DL), true);
  EXPECT_EQ(isImpliedCondition(A, A3, DL), false);
  EXPECT_EQ(isImpliedCondition(A, A4, DL), std::nullopt);
}

TEST_F(ValueTrackingTest, IsImpliedConditionAndVec) {
  parseAssembly(R"(
    define void @test(<2 x i8> %x, <2 x i8> %y) {
      %A = icmp ult <2 x i8> %x, %y
      %A2 = icmp ule <2 x i8> %x, %y
      ret void
    }
  )");
  const DataLayout &DL = M->getDataLayout();
  EXPECT_EQ(isImpliedCondition(A, A2, DL), true);
}

TEST_F(ValueTrackingTest, IsImpliedConditionOr) {
  parseAssembly(R"(
    define void @test(i32 %x, i32 %y) {
      %c1 = icmp ult i32 %x, 10
      %c2 = icmp ult i32 %y, 15
      %A = or i1 %c1, %c2 ; negated
      ; x >= 10 /\ y >= 15
      %A2 = icmp ult i32 %x, 5
      %A3 = icmp uge i32 %y, 10
      %A4 = icmp ult i32 %x, 15
      ret void
    }
  )");
  const DataLayout &DL = M->getDataLayout();
  EXPECT_EQ(isImpliedCondition(A, A2, DL, false), false);
  EXPECT_EQ(isImpliedCondition(A, A3, DL, false), true);
  EXPECT_EQ(isImpliedCondition(A, A4, DL, false), std::nullopt);
}

TEST_F(ValueTrackingTest, IsImpliedConditionOr2) {
  parseAssembly(R"(
    define void @test(i32 %x, i32 %y) {
      %c1 = icmp ult i32 %x, 10
      %c2 = icmp ult i32 %y, 15
      %A = select i1 %c1, i1 true, i1 %c2 ; negated
      ; x >= 10 /\ y >= 15
      %A2 = icmp ult i32 %x, 5
      %A3 = icmp uge i32 %y, 10
      %A4 = icmp ult i32 %x, 15
      ret void
    }
  )");
  const DataLayout &DL = M->getDataLayout();
  EXPECT_EQ(isImpliedCondition(A, A2, DL, false), false);
  EXPECT_EQ(isImpliedCondition(A, A3, DL, false), true);
  EXPECT_EQ(isImpliedCondition(A, A4, DL, false), std::nullopt);
}

TEST_F(ComputeKnownBitsTest, KnownNonZeroShift) {
  // %q is known nonzero without known bits.
  // Because %q is nonzero, %A[0] is known to be zero.
  parseAssembly(
      "define i8 @test(i8 %p, ptr %pq) {\n"
      "  %q = load i8, ptr %pq, !range !0\n"
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
  // ptrtoint truncates the pointer type. Make sure we don't crash.
  parseAssembly(
      "define void @test(ptr %p) {\n"
      "  %A = load ptr, ptr %p\n"
      "  %i = ptrtoint ptr %A to i32\n"
      "  %m = and i32 %i, 31\n"
      "  %c = icmp eq i32 %m, 0\n"
      "  call void @llvm.assume(i1 %c)\n"
      "  ret void\n"
      "}\n"
      "declare void @llvm.assume(i1)\n");
  AssumptionCache AC(*F);
  KnownBits Known = computeKnownBits(
      A, M->getDataLayout(), /* Depth */ 0, &AC, F->front().getTerminator());
  EXPECT_TRUE(Known.isUnknown());
}

TEST_F(ComputeKnownBitsTest, ComputeKnownBitsPtrToIntZext) {
  // ptrtoint zero extends the pointer type. Make sure we don't crash.
  parseAssembly(
      "define void @test(ptr %p) {\n"
      "  %A = load ptr, ptr %p\n"
      "  %i = ptrtoint ptr %A to i128\n"
      "  %m = and i128 %i, 31\n"
      "  %c = icmp eq i128 %m, 0\n"
      "  call void @llvm.assume(i1 %c)\n"
      "  ret void\n"
      "}\n"
      "declare void @llvm.assume(i1)\n");
  AssumptionCache AC(*F);
  KnownBits Known = computeKnownBits(
      A, M->getDataLayout(), /* Depth */ 0, &AC, F->front().getTerminator());
  EXPECT_TRUE(Known.isUnknown());
}

TEST_F(ComputeKnownBitsTest, ComputeKnownBitsFreeze) {
  parseAssembly("define void @test() {\n"
                "  %m = call i32 @any_num()\n"
                "  %A = freeze i32 %m\n"
                "  %n = and i32 %m, 31\n"
                "  %c = icmp eq i32 %n, 0\n"
                "  call void @llvm.assume(i1 %c)\n"
                "  ret void\n"
                "}\n"
                "declare void @llvm.assume(i1)\n"
                "declare i32 @any_num()\n");
  AssumptionCache AC(*F);
  KnownBits Known = computeKnownBits(A, M->getDataLayout(), /* Depth */ 0, &AC,
                                     F->front().getTerminator());
  EXPECT_EQ(Known.Zero.getZExtValue(), 31u);
  EXPECT_EQ(Known.One.getZExtValue(), 0u);
}

TEST_F(ComputeKnownBitsTest, ComputeKnownBitsReturnedRangeConflict) {
  parseAssembly(
      "declare i16 @foo(i16 returned)\n"
      "\n"
      "define i16 @test() {\n"
      "  %A = call i16 @foo(i16 4095), !range !{i16 32, i16 33}\n"
      "  ret i16 %A\n"
      "}\n");
  // The call returns 32 according to range metadata, but 4095 according to the
  // returned arg operand. Given the conflicting information we expect that the
  // known bits information simply is cleared.
  expectKnownBits(/*zero*/ 0u, /*one*/ 0u);
}

TEST_F(ComputeKnownBitsTest, ComputeKnownBitsAddWithRange) {
  parseAssembly("define void @test(ptr %p) {\n"
                "  %A = load i64, ptr %p, !range !{i64 64, i64 65536}\n"
                "  %APlus512 = add i64 %A, 512\n"
                "  %c = icmp ugt i64 %APlus512, 523\n"
                "  call void @llvm.assume(i1 %c)\n"
                "  ret void\n"
                "}\n"
                "declare void @llvm.assume(i1)\n");
  AssumptionCache AC(*F);
  KnownBits Known = computeKnownBits(A, M->getDataLayout(), /* Depth */ 0, &AC,
                                     F->front().getTerminator());
  EXPECT_EQ(Known.Zero.getZExtValue(), ~(65536llu - 1));
  EXPECT_EQ(Known.One.getZExtValue(), 0u);
  Instruction &APlus512 = findInstructionByName(F, "APlus512");
  Known = computeKnownBits(&APlus512, M->getDataLayout(), /* Depth */ 0, &AC,
                           F->front().getTerminator());
  // We know of one less zero because 512 may have produced a 1 that
  // got carried all the way to the first trailing zero.
  EXPECT_EQ(Known.Zero.getZExtValue(), (~(65536llu - 1)) << 1);
  EXPECT_EQ(Known.One.getZExtValue(), 0u);
  // The known range is not precise given computeKnownBits works
  // with the masks of zeros and ones, not the ranges.
  EXPECT_EQ(Known.getMinValue(), 0u);
  EXPECT_EQ(Known.getMaxValue(), 131071);
}

TEST_F(ComputeKnownBitsTest, ComputeKnownBitsUnknownVScale) {
  Module M("", Context);
  IRBuilder<> Builder(Context);
  Function *TheFn =
      Intrinsic::getDeclaration(&M, Intrinsic::vscale, {Builder.getInt32Ty()});
  CallInst *CI = Builder.CreateCall(TheFn, {}, {}, "");

  KnownBits Known = computeKnownBits(CI, M.getDataLayout(), /* Depth */ 0);
  // There is no parent function so we cannot look up the vscale_range
  // attribute to determine the number of bits.
  EXPECT_EQ(Known.One.getZExtValue(), 0u);
  EXPECT_EQ(Known.Zero.getZExtValue(), 0u);

  BasicBlock *BB = BasicBlock::Create(Context);
  CI->insertInto(BB, BB->end());
  Known = computeKnownBits(CI, M.getDataLayout(), /* Depth */ 0);
  // There is no parent function so we cannot look up the vscale_range
  // attribute to determine the number of bits.
  EXPECT_EQ(Known.One.getZExtValue(), 0u);
  EXPECT_EQ(Known.Zero.getZExtValue(), 0u);

  CI->removeFromParent();
  delete CI;
  delete BB;
}

// 512 + [32, 64) doesn't produce overlapping bits.
// Make sure we get all the individual bits properly.
TEST_F(ComputeKnownBitsTest, ComputeKnownBitsAddWithRangeNoOverlap) {
  parseAssembly("define void @test(ptr %p) {\n"
                "  %A = load i64, ptr %p, !range !{i64 32, i64 64}\n"
                "  %APlus512 = add i64 %A, 512\n"
                "  %c = icmp ugt i64 %APlus512, 523\n"
                "  call void @llvm.assume(i1 %c)\n"
                "  ret void\n"
                "}\n"
                "declare void @llvm.assume(i1)\n");
  AssumptionCache AC(*F);
  KnownBits Known = computeKnownBits(A, M->getDataLayout(), /* Depth */ 0, &AC,
                                     F->front().getTerminator());
  EXPECT_EQ(Known.Zero.getZExtValue(), ~(64llu - 1));
  EXPECT_EQ(Known.One.getZExtValue(), 32u);
  Instruction &APlus512 = findInstructionByName(F, "APlus512");
  Known = computeKnownBits(&APlus512, M->getDataLayout(), /* Depth */ 0, &AC,
                           F->front().getTerminator());
  EXPECT_EQ(Known.Zero.getZExtValue(), ~512llu & ~(64llu - 1));
  EXPECT_EQ(Known.One.getZExtValue(), 512u | 32u);
  // The known range is not precise given computeKnownBits works
  // with the masks of zeros and ones, not the ranges.
  EXPECT_EQ(Known.getMinValue(), 544);
  EXPECT_EQ(Known.getMaxValue(), 575);
}

TEST_F(ComputeKnownBitsTest, ComputeKnownBitsGEPWithRange) {
  parseAssembly(
      "define void @test(ptr %p) {\n"
      "  %A = load i64, ptr %p, !range !{i64 64, i64 65536}\n"
      "  %APtr = inttoptr i64 %A to float*"
      "  %APtrPlus512 = getelementptr float, float* %APtr, i32 128\n"
      "  %c = icmp ugt float* %APtrPlus512, inttoptr (i32 523 to float*)\n"
      "  call void @llvm.assume(i1 %c)\n"
      "  ret void\n"
      "}\n"
      "declare void @llvm.assume(i1)\n");
  AssumptionCache AC(*F);
  KnownBits Known = computeKnownBits(A, M->getDataLayout(), /* Depth */ 0, &AC,
                                     F->front().getTerminator());
  EXPECT_EQ(Known.Zero.getZExtValue(), ~(65536llu - 1));
  EXPECT_EQ(Known.One.getZExtValue(), 0u);
  Instruction &APtrPlus512 = findInstructionByName(F, "APtrPlus512");
  Known = computeKnownBits(&APtrPlus512, M->getDataLayout(), /* Depth */ 0, &AC,
                           F->front().getTerminator());
  // We know of one less zero because 512 may have produced a 1 that
  // got carried all the way to the first trailing zero.
  EXPECT_EQ(Known.Zero.getZExtValue(), ~(65536llu - 1) << 1);
  EXPECT_EQ(Known.One.getZExtValue(), 0u);
  // The known range is not precise given computeKnownBits works
  // with the masks of zeros and ones, not the ranges.
  EXPECT_EQ(Known.getMinValue(), 0u);
  EXPECT_EQ(Known.getMaxValue(), 131071);
}

// 4*128 + [32, 64) doesn't produce overlapping bits.
// Make sure we get all the individual bits properly.
// This test is useful to check that we account for the scaling factor
// in the gep. Indeed, gep float, [32,64), 128 is not 128 + [32,64).
TEST_F(ComputeKnownBitsTest, ComputeKnownBitsGEPWithRangeNoOverlap) {
  parseAssembly(
      "define void @test(ptr %p) {\n"
      "  %A = load i64, ptr %p, !range !{i64 32, i64 64}\n"
      "  %APtr = inttoptr i64 %A to float*"
      "  %APtrPlus512 = getelementptr float, float* %APtr, i32 128\n"
      "  %c = icmp ugt float* %APtrPlus512, inttoptr (i32 523 to float*)\n"
      "  call void @llvm.assume(i1 %c)\n"
      "  ret void\n"
      "}\n"
      "declare void @llvm.assume(i1)\n");
  AssumptionCache AC(*F);
  KnownBits Known = computeKnownBits(A, M->getDataLayout(), /* Depth */ 0, &AC,
                                     F->front().getTerminator());
  EXPECT_EQ(Known.Zero.getZExtValue(), ~(64llu - 1));
  EXPECT_EQ(Known.One.getZExtValue(), 32u);
  Instruction &APtrPlus512 = findInstructionByName(F, "APtrPlus512");
  Known = computeKnownBits(&APtrPlus512, M->getDataLayout(), /* Depth */ 0, &AC,
                           F->front().getTerminator());
  EXPECT_EQ(Known.Zero.getZExtValue(), ~512llu & ~(64llu - 1));
  EXPECT_EQ(Known.One.getZExtValue(), 512u | 32u);
  // The known range is not precise given computeKnownBits works
  // with the masks of zeros and ones, not the ranges.
  EXPECT_EQ(Known.getMinValue(), 544);
  EXPECT_EQ(Known.getMaxValue(), 575);
}

TEST_F(ComputeKnownBitsTest, ComputeKnownBitsAbsoluteSymbol) {
  auto M = parseModule(R"(
    @absolute_0_255 = external global [128 x i32], align 1, !absolute_symbol !0
    @absolute_0_256 = external global [128 x i32], align 1, !absolute_symbol !1
    @absolute_256_512 = external global [128 x i32], align 1, !absolute_symbol !2
    @absolute_0_neg1 = external global [128 x i32], align 1, !absolute_symbol !3
    @absolute_neg32_32 = external global [128 x i32], align 1, !absolute_symbol !4
    @absolute_neg32_33 = external global [128 x i32], align 1, !absolute_symbol !5
    @absolute_neg64_neg32 = external global [128 x i32], align 1, !absolute_symbol !6
    @absolute_0_256_align8 = external global [128 x i32], align 8, !absolute_symbol !1

    !0 = !{i64 0, i64 255}
    !1 = !{i64 0, i64 256}
    !2 = !{i64 256, i64 512}
    !3 = !{i64 0, i64 -1}
    !4 = !{i64 -32, i64 32}
    !5 = !{i64 -32, i64 33}
    !6 = !{i64 -64, i64 -32}
  )");

  GlobalValue *Absolute_0_255 = M->getNamedValue("absolute_0_255");
  GlobalValue *Absolute_0_256 = M->getNamedValue("absolute_0_256");
  GlobalValue *Absolute_256_512 = M->getNamedValue("absolute_256_512");
  GlobalValue *Absolute_0_Neg1 = M->getNamedValue("absolute_0_neg1");
  GlobalValue *Absolute_Neg32_32 = M->getNamedValue("absolute_neg32_32");
  GlobalValue *Absolute_Neg32_33 = M->getNamedValue("absolute_neg32_33");
  GlobalValue *Absolute_Neg64_Neg32 = M->getNamedValue("absolute_neg64_neg32");
  GlobalValue *Absolute_0_256_Align8 =
      M->getNamedValue("absolute_0_256_align8");

  KnownBits Known_0_255 = computeKnownBits(Absolute_0_255, M->getDataLayout());
  EXPECT_EQ(64u - 8u, Known_0_255.countMinLeadingZeros());
  EXPECT_EQ(0u, Known_0_255.countMinTrailingZeros());
  EXPECT_EQ(0u, Known_0_255.countMinLeadingOnes());
  EXPECT_EQ(0u, Known_0_255.countMinTrailingOnes());

  KnownBits Known_0_256 = computeKnownBits(Absolute_0_256, M->getDataLayout());
  EXPECT_EQ(64u - 8u, Known_0_256.countMinLeadingZeros());
  EXPECT_EQ(0u, Known_0_256.countMinTrailingZeros());
  EXPECT_EQ(0u, Known_0_256.countMinLeadingOnes());
  EXPECT_EQ(0u, Known_0_256.countMinTrailingOnes());

  KnownBits Known_256_512 =
      computeKnownBits(Absolute_256_512, M->getDataLayout());
  EXPECT_EQ(64u - 8u, Known_0_255.countMinLeadingZeros());
  EXPECT_EQ(0u, Known_0_255.countMinTrailingZeros());
  EXPECT_EQ(0u, Known_0_255.countMinLeadingOnes());
  EXPECT_EQ(0u, Known_0_255.countMinTrailingOnes());

  KnownBits Known_0_Neg1 =
      computeKnownBits(Absolute_0_Neg1, M->getDataLayout());
  EXPECT_EQ(0u, Known_0_Neg1.countMinLeadingZeros());
  EXPECT_EQ(0u, Known_0_Neg1.countMinTrailingZeros());
  EXPECT_EQ(0u, Known_0_Neg1.countMinLeadingOnes());
  EXPECT_EQ(0u, Known_0_Neg1.countMinTrailingOnes());

  KnownBits Known_Neg32_32 =
      computeKnownBits(Absolute_Neg32_32, M->getDataLayout());
  EXPECT_EQ(0u, Known_Neg32_32.countMinLeadingZeros());
  EXPECT_EQ(0u, Known_Neg32_32.countMinTrailingZeros());
  EXPECT_EQ(0u, Known_Neg32_32.countMinLeadingOnes());
  EXPECT_EQ(0u, Known_Neg32_32.countMinTrailingOnes());
  EXPECT_EQ(1u, Known_Neg32_32.countMinSignBits());

  KnownBits Known_Neg32_33 =
      computeKnownBits(Absolute_Neg32_33, M->getDataLayout());
  EXPECT_EQ(0u, Known_Neg32_33.countMinLeadingZeros());
  EXPECT_EQ(0u, Known_Neg32_33.countMinTrailingZeros());
  EXPECT_EQ(0u, Known_Neg32_33.countMinLeadingOnes());
  EXPECT_EQ(0u, Known_Neg32_33.countMinTrailingOnes());
  EXPECT_EQ(1u, Known_Neg32_33.countMinSignBits());

  KnownBits Known_Neg32_Neg32 =
      computeKnownBits(Absolute_Neg64_Neg32, M->getDataLayout());
  EXPECT_EQ(0u, Known_Neg32_Neg32.countMinLeadingZeros());
  EXPECT_EQ(0u, Known_Neg32_Neg32.countMinTrailingZeros());
  EXPECT_EQ(58u, Known_Neg32_Neg32.countMinLeadingOnes());
  EXPECT_EQ(0u, Known_Neg32_Neg32.countMinTrailingOnes());
  EXPECT_EQ(58u, Known_Neg32_Neg32.countMinSignBits());

  KnownBits Known_0_256_Align8 =
      computeKnownBits(Absolute_0_256_Align8, M->getDataLayout());
  EXPECT_EQ(64u - 8u, Known_0_256_Align8.countMinLeadingZeros());
  EXPECT_EQ(3u, Known_0_256_Align8.countMinTrailingZeros());
  EXPECT_EQ(0u, Known_0_256_Align8.countMinLeadingOnes());
  EXPECT_EQ(0u, Known_0_256_Align8.countMinTrailingOnes());
}

TEST_F(ValueTrackingTest, HaveNoCommonBitsSet) {
  {
    // Check for an inverted mask: (X & ~M) op (Y & M).
    auto M = parseModule(R"(
  define i32 @test(i32 %X, i32 %Y, i32 noundef %M) {
    %1 = xor i32 %M, -1
    %LHS = and i32 %1, %X
    %RHS = and i32 %Y, %M
    %Ret = add i32 %LHS, %RHS
    ret i32 %Ret
  })");

    auto *F = M->getFunction("test");
    auto *LHS = findInstructionByNameOrNull(F, "LHS");
    auto *RHS = findInstructionByNameOrNull(F, "RHS");

    const DataLayout &DL = M->getDataLayout();
    EXPECT_TRUE(haveNoCommonBitsSet(LHS, RHS, DL));
    EXPECT_TRUE(haveNoCommonBitsSet(RHS, LHS, DL));
  }
  {
    // Check for (A & B) and ~(A | B)
    auto M = parseModule(R"(
  define void @test(i32 noundef %A, i32 noundef %B) {
    %LHS = and i32 %A, %B
    %or = or i32 %A, %B
    %RHS = xor i32 %or, -1

    %LHS2 = and i32 %B, %A
    %or2 = or i32 %A, %B
    %RHS2 = xor i32 %or2, -1

    ret void
  })");

    auto *F = M->getFunction("test");
    const DataLayout &DL = M->getDataLayout();

    auto *LHS = findInstructionByNameOrNull(F, "LHS");
    auto *RHS = findInstructionByNameOrNull(F, "RHS");
    EXPECT_TRUE(haveNoCommonBitsSet(LHS, RHS, DL));
    EXPECT_TRUE(haveNoCommonBitsSet(RHS, LHS, DL));

    auto *LHS2 = findInstructionByNameOrNull(F, "LHS2");
    auto *RHS2 = findInstructionByNameOrNull(F, "RHS2");
    EXPECT_TRUE(haveNoCommonBitsSet(LHS2, RHS2, DL));
    EXPECT_TRUE(haveNoCommonBitsSet(RHS2, LHS2, DL));
  }
  {
    // Check for (A & B) and ~(A | B) in vector version
    auto M = parseModule(R"(
  define void @test(<2 x i32> noundef %A, <2 x i32> noundef %B) {
    %LHS = and <2 x i32> %A, %B
    %or = or <2 x i32> %A, %B
    %RHS = xor <2 x i32> %or, <i32 -1, i32 -1>

    %LHS2 = and <2 x i32> %B, %A
    %or2 = or <2 x i32> %A, %B
    %RHS2 = xor <2 x i32> %or2, <i32 -1, i32 -1>

    ret void
  })");

    auto *F = M->getFunction("test");
    const DataLayout &DL = M->getDataLayout();

    auto *LHS = findInstructionByNameOrNull(F, "LHS");
    auto *RHS = findInstructionByNameOrNull(F, "RHS");
    EXPECT_TRUE(haveNoCommonBitsSet(LHS, RHS, DL));
    EXPECT_TRUE(haveNoCommonBitsSet(RHS, LHS, DL));

    auto *LHS2 = findInstructionByNameOrNull(F, "LHS2");
    auto *RHS2 = findInstructionByNameOrNull(F, "RHS2");
    EXPECT_TRUE(haveNoCommonBitsSet(LHS2, RHS2, DL));
    EXPECT_TRUE(haveNoCommonBitsSet(RHS2, LHS2, DL));
  }
}

class IsBytewiseValueTest : public ValueTrackingTest,
                            public ::testing::WithParamInterface<
                                std::pair<const char *, const char *>> {
protected:
};

const std::pair<const char *, const char *> IsBytewiseValueTests[] = {
    {
        "i8 0",
        "ptr null",
    },
    {
        "i8 undef",
        "ptr undef",
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
        "i8 0",
        "ptr inttoptr (i64 0 to ptr)",
    },
    {
        "i8 -1",
        "ptr inttoptr (i64 -1 to ptr)",
    },
    {
        "i8 -86",
        "ptr inttoptr (i64 -6148914691236517206 to ptr)",
    },
    {
        "",
        "ptr inttoptr (i48 -1 to ptr)",
    },
    {
        "i8 -1",
        "ptr inttoptr (i96 -1 to ptr)",
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
        "{i8, i64, ptr} zeroinitializer",
    },
    {
        "i8 undef",
        "{i8, i64, ptr} undef",
    },
    {
        "i8 -86",
        "{i8, i64, ptr} {i8 -86, i64 -6148914691236517206, ptr undef}",
    },
    {
        "",
        "{i8, i64, ptr} {i8 86, i64 -6148914691236517206, ptr undef}",
    },
};

INSTANTIATE_TEST_SUITE_P(IsBytewiseValueParamTests, IsBytewiseValueTest,
                         ::testing::ValuesIn(IsBytewiseValueTests));

TEST_P(IsBytewiseValueTest, IsBytewiseValue) {
  auto M = parseModule(std::string("@test = global ") + GetParam().second);
  GlobalVariable *GV = dyn_cast<GlobalVariable>(M->getNamedValue("test"));
  Value *Actual = isBytewiseValue(GV->getInitializer(), M->getDataLayout());
  std::string Buff;
  raw_string_ostream S(Buff);
  if (Actual)
    S << *Actual;
  EXPECT_EQ(GetParam().first, Buff);
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
    ConstantRange CR1 = computeConstantRange(Stride, false, true, &AC, nullptr);
    EXPECT_TRUE(CR1.isFullSet());

    Instruction *I = &findInstructionByName(F, "stride.plus.one");
    ConstantRange CR2 = computeConstantRange(Stride, false, true, &AC, I);
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
    ConstantRange CR = computeConstantRange(Stride, false, true, &AC, I);
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
    ConstantRange CR = computeConstantRange(Stride, false, true, &AC, GT2);
    EXPECT_EQ(5, CR.getLower());
    EXPECT_EQ(0, CR.getUpper());

    Instruction *I = &findInstructionByName(F, "stride.plus.one");
    ConstantRange CR2 = computeConstantRange(Stride, false, true, &AC, I);
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
    ConstantRange CR = computeConstantRange(Stride, false, true, &AC, I);
    EXPECT_TRUE(CR.isEmptySet());
  }

  {
    // Assumptions:
    //  * x.1 >= 5
    //  * x.2 < x.1
    //
    // stride = [0, -1)
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
    Value *X1 = &*(F->arg_begin());
    Value *X2 = &*std::next(F->arg_begin());

    Instruction *I = &findInstructionByName(F, "stride.plus.one");
    ConstantRange CR1 = computeConstantRange(X1, false, true, &AC, I);
    ConstantRange CR2 = computeConstantRange(X2, false, true, &AC, I);

    EXPECT_EQ(5, CR1.getLower());
    EXPECT_EQ(0, CR1.getUpper());

    EXPECT_EQ(0, CR2.getLower());
    EXPECT_EQ(0xffffffff, CR2.getUpper());

    // Check the depth cutoff results in a conservative result (full set) by
    // passing Depth == MaxDepth == 6.
    ConstantRange CR3 = computeConstantRange(X2, false, true, &AC, I, nullptr, 6);
    EXPECT_TRUE(CR3.isFullSet());
  }
  {
    // Assumptions:
    //  * x.2 <= x.1
    auto M = parseModule(R"(
  declare void @llvm.assume(i1)

  define i32 @test(i32 %x.1, i32 %x.2) {
    %lt = icmp ule i32 %x.2, %x.1
    call void @llvm.assume(i1 %lt)
    %stride.plus.one = add nsw nuw i32 %x.1, 1
    ret i32 %stride.plus.one
  })");
    Function *F = M->getFunction("test");

    AssumptionCache AC(*F);
    Value *X2 = &*std::next(F->arg_begin());

    Instruction *I = &findInstructionByName(F, "stride.plus.one");
    ConstantRange CR1 = computeConstantRange(X2, false, true, &AC, I);
    // If we don't know the value of x.2, we don't know the value of x.1.
    EXPECT_TRUE(CR1.isFullSet());
  }
}

struct FindAllocaForValueTestParams {
  const char *IR;
  bool AnyOffsetResult;
  bool ZeroOffsetResult;
};

class FindAllocaForValueTest
    : public ValueTrackingTest,
      public ::testing::WithParamInterface<FindAllocaForValueTestParams> {
protected:
};

const FindAllocaForValueTestParams FindAllocaForValueTests[] = {
    {R"(
      define void @test() {
        %a = alloca i64
        %r = bitcast ptr %a to ptr
        ret void
      })",
     true, true},

    {R"(
      define void @test() {
        %a = alloca i32
        %r = getelementptr i32, ptr %a, i32 1
        ret void
      })",
     true, false},

    {R"(
      define void @test() {
        %a = alloca i32
        %r = getelementptr i32, ptr %a, i32 0
        ret void
      })",
     true, true},

    {R"(
      define void @test(i1 %cond) {
      entry:
        %a = alloca i32
        br label %bb1

      bb1:
        %r = phi ptr [ %a, %entry ], [ %r, %bb1 ]
        br i1 %cond, label %bb1, label %exit

      exit:
        ret void
      })",
     true, true},

    {R"(
      define void @test(i1 %cond) {
        %a = alloca i32
        %r = select i1 %cond, ptr %a, ptr %a
        ret void
      })",
     true, true},

    {R"(
      define void @test(i1 %cond) {
        %a = alloca i32
        %b = alloca i32
        %r = select i1 %cond, ptr %a, ptr %b
        ret void
      })",
     false, false},

    {R"(
      define void @test(i1 %cond) {
      entry:
        %a = alloca i64
        %a32 = bitcast ptr %a to ptr
        br label %bb1

      bb1:
        %x = phi ptr [ %a32, %entry ], [ %x, %bb1 ]
        %r = getelementptr i32, ptr %x, i32 1
        br i1 %cond, label %bb1, label %exit

      exit:
        ret void
      })",
     true, false},

    {R"(
      define void @test(i1 %cond) {
      entry:
        %a = alloca i64
        %a32 = bitcast ptr %a to ptr
        br label %bb1

      bb1:
        %x = phi ptr [ %a32, %entry ], [ %r, %bb1 ]
        %r = getelementptr i32, ptr %x, i32 1
        br i1 %cond, label %bb1, label %exit

      exit:
        ret void
      })",
     true, false},

    {R"(
      define void @test(i1 %cond, ptr %a) {
      entry:
        %r = bitcast ptr %a to ptr
        ret void
      })",
     false, false},

    {R"(
      define void @test(i1 %cond) {
      entry:
        %a = alloca i32
        %b = alloca i32
        br label %bb1

      bb1:
        %r = phi ptr [ %a, %entry ], [ %b, %bb1 ]
        br i1 %cond, label %bb1, label %exit

      exit:
        ret void
      })",
     false, false},
    {R"(
      declare ptr @retptr(ptr returned)
      define void @test(i1 %cond) {
        %a = alloca i32
        %r = call ptr @retptr(ptr %a)
        ret void
      })",
     true, true},
    {R"(
      declare ptr @fun(ptr)
      define void @test(i1 %cond) {
        %a = alloca i32
        %r = call ptr @fun(ptr %a)
        ret void
      })",
     false, false},
};

TEST_P(FindAllocaForValueTest, findAllocaForValue) {
  auto M = parseModule(GetParam().IR);
  Function *F = M->getFunction("test");
  Instruction *I = &findInstructionByName(F, "r");
  const AllocaInst *AI = findAllocaForValue(I);
  EXPECT_EQ(!!AI, GetParam().AnyOffsetResult);
}

TEST_P(FindAllocaForValueTest, findAllocaForValueZeroOffset) {
  auto M = parseModule(GetParam().IR);
  Function *F = M->getFunction("test");
  Instruction *I = &findInstructionByName(F, "r");
  const AllocaInst *AI = findAllocaForValue(I, true);
  EXPECT_EQ(!!AI, GetParam().ZeroOffsetResult);
}

INSTANTIATE_TEST_SUITE_P(FindAllocaForValueTest, FindAllocaForValueTest,
                         ::testing::ValuesIn(FindAllocaForValueTests));
