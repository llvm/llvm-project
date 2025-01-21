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

  sandboxir::InstrMaps IMaps(Ctx);
  // Check with empty IMaps.
  EXPECT_EQ(IMaps.getVectorForOrig(Add0), nullptr);
  EXPECT_EQ(IMaps.getVectorForOrig(Add1), nullptr);
  EXPECT_FALSE(IMaps.getOrigLane(Add0, Add0));
  // Check with 1 match.
  IMaps.registerVector({Add0, Add1}, VAdd0);
  EXPECT_EQ(IMaps.getVectorForOrig(Add0), VAdd0);
  EXPECT_EQ(IMaps.getVectorForOrig(Add1), VAdd0);
  EXPECT_FALSE(IMaps.getOrigLane(VAdd0, VAdd0)); // Bad Orig value
  EXPECT_FALSE(IMaps.getOrigLane(Add0, Add0));   // Bad Vector value
  EXPECT_EQ(*IMaps.getOrigLane(VAdd0, Add0), 0U);
  EXPECT_EQ(*IMaps.getOrigLane(VAdd0, Add1), 1U);
  // Check when the same vector maps to different original values (which is
  // common for vector constants).
  IMaps.registerVector({Add2, Add3}, VAdd0);
  EXPECT_EQ(*IMaps.getOrigLane(VAdd0, Add2), 0U);
  EXPECT_EQ(*IMaps.getOrigLane(VAdd0, Add3), 1U);
  // Check when we register for a second time.
#ifndef NDEBUG
  EXPECT_DEATH(IMaps.registerVector({Add1, Add0}, VAdd0), ".*exists.*");
#endif // NDEBUG
  // Check callbacks: erase original instr.
  Add0->eraseFromParent();
  EXPECT_FALSE(IMaps.getOrigLane(VAdd0, Add0));
  EXPECT_EQ(*IMaps.getOrigLane(VAdd0, Add1), 1U);
  EXPECT_EQ(IMaps.getVectorForOrig(Add0), nullptr);
  // Check callbacks: erase vector instr.
  VAdd0->eraseFromParent();
  EXPECT_FALSE(IMaps.getOrigLane(VAdd0, Add1));
  EXPECT_EQ(IMaps.getVectorForOrig(Add1), nullptr);
}
