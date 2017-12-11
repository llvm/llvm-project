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

class MatchSelectPatternTest : public testing::Test {
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

  void expectPattern(const SelectPatternResult &P) {
    Value *LHS, *RHS;
    Instruction::CastOps CastOp;
    SelectPatternResult R = matchSelectPattern(A, LHS, RHS, &CastOp);
    EXPECT_EQ(P.Flavor, R.Flavor);
    EXPECT_EQ(P.NaNBehavior, R.NaNBehavior);
    EXPECT_EQ(P.Ordered, R.Ordered);
  }

  LLVMContext Context;
  std::unique_ptr<Module> M;
  Instruction *A, *B;
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

TEST(ValueTracking, ComputeNumSignBits_PR32045) {
  StringRef Assembly = "define i32 @f(i32 %a) { "
                       "  %val = ashr i32 %a, -1 "
                       "  ret i32 %val "
                       "} ";

  LLVMContext Context;
  SMDiagnostic Error;
  auto M = parseAssemblyString(Assembly, Error, Context);
  assert(M && "Bad assembly?");

  auto *F = M->getFunction("f");
  assert(F && "Bad assembly?");

  auto *RVal =
      cast<ReturnInst>(F->getEntryBlock().getTerminator())->getOperand(0);
  EXPECT_EQ(ComputeNumSignBits(RVal, M->getDataLayout()), 1u);
}

TEST(ValueTracking, ComputeKnownBits) {
  StringRef Assembly = "define i32 @f(i32 %a, i32 %b) { "
                       "  %ash = mul i32 %a, 8 "
                       "  %aad = add i32 %ash, 7 "
                       "  %aan = and i32 %aad, 4095 "
                       "  %bsh = shl i32 %b, 4 "
                       "  %bad = or i32 %bsh, 6 "
                       "  %ban = and i32 %bad, 4095 "
                       "  %mul = mul i32 %aan, %ban "
                       "  ret i32 %mul "
                       "} ";

  LLVMContext Context;
  SMDiagnostic Error;
  auto M = parseAssemblyString(Assembly, Error, Context);
  assert(M && "Bad assembly?");

  auto *F = M->getFunction("f");
  assert(F && "Bad assembly?");

  auto *RVal =
      cast<ReturnInst>(F->getEntryBlock().getTerminator())->getOperand(0);
  auto Known = computeKnownBits(RVal, M->getDataLayout());
  ASSERT_FALSE(Known.hasConflict());
  EXPECT_EQ(Known.One.getZExtValue(), 10u);
  EXPECT_EQ(Known.Zero.getZExtValue(), 4278190085u);
}

TEST(ValueTracking, ComputeKnownMulBits) {
  StringRef Assembly = "define i32 @f(i32 %a, i32 %b) { "
                       "  %aa = shl i32 %a, 5 "
                       "  %bb = shl i32 %b, 5 "
                       "  %aaa = or i32 %aa, 24 "
                       "  %bbb = or i32 %bb, 28 "
                       "  %mul = mul i32 %aaa, %bbb "
                       "  ret i32 %mul "
                       "} ";

  LLVMContext Context;
  SMDiagnostic Error;
  auto M = parseAssemblyString(Assembly, Error, Context);
  assert(M && "Bad assembly?");

  auto *F = M->getFunction("f");
  assert(F && "Bad assembly?");

  auto *RVal =
      cast<ReturnInst>(F->getEntryBlock().getTerminator())->getOperand(0);
  auto Known = computeKnownBits(RVal, M->getDataLayout());
  ASSERT_FALSE(Known.hasConflict());
  EXPECT_EQ(Known.One.getZExtValue(), 32u);
  EXPECT_EQ(Known.Zero.getZExtValue(), 95u);
}
