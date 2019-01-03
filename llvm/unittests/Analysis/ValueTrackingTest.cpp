//===- ValueTrackingTest.cpp - ValueTracking tests ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ValueTracking.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/KnownBits.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class ValueTrackingTest : public testing::Test {
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

TEST(ValueTracking, GuaranteedToTransferExecutionToSuccessor) {
  StringRef Assembly =
      "declare void @nounwind_readonly(i32*) nounwind readonly "
      "declare void @nounwind_argmemonly(i32*) nounwind argmemonly "
      "declare void @throws_but_readonly(i32*) readonly "
      "declare void @throws_but_argmemonly(i32*) argmemonly "
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
