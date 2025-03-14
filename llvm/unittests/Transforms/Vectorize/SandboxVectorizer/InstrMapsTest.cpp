//===- InstrMapsTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/InstrMaps.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/SandboxIR/Function.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/Support/SourceMgr.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;

struct InstrMapsTest : public testing::Test {
  LLVMContext C;
  std::unique_ptr<Module> M;

  void parseIR(LLVMContext &C, const char *IR) {
    SMDiagnostic Err;
    M = parseAssemblyString(IR, Err, C);
    if (!M)
      Err.print("InstrMapsTest", errs());
  }
};

TEST_F(InstrMapsTest, Basic) {
  parseIR(C, R"IR(
define void @foo(i8 %v0, i8 %v1, i8 %v2, i8 %v3, <2 x i8> %vec) {
  %add0 = add i8 %v0, %v0
  %add1 = add i8 %v1, %v1
  %add2 = add i8 %v2, %v2
  %add3 = add i8 %v3, %v3
  %vadd0 = add <2 x i8> %vec, %vec
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();

  auto *Add0 = cast<sandboxir::BinaryOperator>(&*It++);
  auto *Add1 = cast<sandboxir::BinaryOperator>(&*It++);
  auto *Add2 = cast<sandboxir::BinaryOperator>(&*It++);
  auto *Add3 = cast<sandboxir::BinaryOperator>(&*It++);
  auto *VAdd0 = cast<sandboxir::BinaryOperator>(&*It++);
  [[maybe_unused]] auto *Ret = cast<sandboxir::ReturnInst>(&*It++);

  sandboxir::InstrMaps IMaps;
  {
    // Check with empty IMaps.
    sandboxir::Action A(nullptr, {Add0}, {}, 0);
    EXPECT_EQ(IMaps.getVectorForOrig(Add0), nullptr);
    EXPECT_EQ(IMaps.getVectorForOrig(Add1), nullptr);
    EXPECT_FALSE(IMaps.getOrigLane(&A, Add0));
  }
  {
    // Check with 1 match.
    sandboxir::Action A(nullptr, {Add0, Add1}, {}, 0);
    sandboxir::Action OtherA(nullptr, {}, {}, 0);
    IMaps.registerVector({Add0, Add1}, &A);
    EXPECT_EQ(IMaps.getVectorForOrig(Add0), &A);
    EXPECT_EQ(IMaps.getVectorForOrig(Add1), &A);
    EXPECT_FALSE(IMaps.getOrigLane(&A, VAdd0));     // Bad Orig value
    EXPECT_FALSE(IMaps.getOrigLane(&OtherA, Add0)); // Bad Vector value
    EXPECT_EQ(*IMaps.getOrigLane(&A, Add0), 0U);
    EXPECT_EQ(*IMaps.getOrigLane(&A, Add1), 1U);
  }
  {
    // Check when the same vector maps to different original values (which is
    // common for vector constants).
    sandboxir::Action A(nullptr, {Add2, Add3}, {}, 0);
    IMaps.registerVector({Add2, Add3}, &A);
    EXPECT_EQ(*IMaps.getOrigLane(&A, Add2), 0U);
    EXPECT_EQ(*IMaps.getOrigLane(&A, Add3), 1U);
  }
  {
    // Check when we register for a second time.
    sandboxir::Action A(nullptr, {Add2, Add3}, {}, 0);
#ifndef NDEBUG
    EXPECT_DEATH(IMaps.registerVector({Add1, Add0}, &A), ".*exists.*");
#endif // NDEBUG
  }
}

TEST_F(InstrMapsTest, VectorLanes) {
  parseIR(C, R"IR(
define void @foo(<2 x i8> %v0, <2 x i8> %v1, <4 x i8> %v2, <4 x i8> %v3) {
  %vadd0 = add <2 x i8> %v0, %v1
  %vadd1 = add <2 x i8> %v0, %v1
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();

  auto *VAdd0 = cast<sandboxir::BinaryOperator>(&*It++);
  auto *VAdd1 = cast<sandboxir::BinaryOperator>(&*It++);

  sandboxir::InstrMaps IMaps;

  {
    // Check that the vector lanes are calculated correctly.
    sandboxir::Action A(nullptr, {VAdd0, VAdd1}, {}, 0);
    IMaps.registerVector({VAdd0, VAdd1}, &A);
    EXPECT_EQ(*IMaps.getOrigLane(&A, VAdd0), 0U);
    EXPECT_EQ(*IMaps.getOrigLane(&A, VAdd1), 2U);
  }
}
