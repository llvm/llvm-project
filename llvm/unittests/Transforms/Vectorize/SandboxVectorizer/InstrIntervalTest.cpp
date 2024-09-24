//===- InstrIntervalTest.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/InstrInterval.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/SandboxIR/SandboxIR.h"
#include "llvm/Support/SourceMgr.h"
#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"

using namespace llvm;

struct InstrIntervalTest : public testing::Test {
  LLVMContext C;
  std::unique_ptr<Module> M;

  void parseIR(LLVMContext &C, const char *IR) {
    SMDiagnostic Err;
    M = parseAssemblyString(IR, Err, C);
    if (!M)
      Err.print("InstrIntervalTest", errs());
  }
};

TEST_F(InstrIntervalTest, Basic) {
  parseIR(C, R"IR(
define void @foo(i8 %v0) {
  %add0 = add i8 %v0, %v0
  %add1 = add i8 %v0, %v0
  %add2 = add i8 %v0, %v0
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto &F = *Ctx.createFunction(&LLVMF);
  auto *BB = &*F.begin();
  auto It = BB->begin();
  auto *I0 = &*It++;
  auto *I1 = &*It++;
  auto *I2 = &*It++;
  auto *Ret = &*It++;

  sandboxir::InstrInterval Interval(I0, Ret);
#ifndef NDEBUG
  EXPECT_DEATH(sandboxir::InstrInterval(I1, I0), ".*before.*");
#endif // NDEBUG
  // Check InstrInterval(ArrayRef), from(), to().
  {
    sandboxir::InstrInterval Interval(
        SmallVector<sandboxir::Instruction *>({I0, Ret}));
    EXPECT_EQ(Interval.top(), I0);
    EXPECT_EQ(Interval.bottom(), Ret);
  }
  {
    sandboxir::InstrInterval Interval(
        SmallVector<sandboxir::Instruction *>({Ret, I0}));
    EXPECT_EQ(Interval.top(), I0);
    EXPECT_EQ(Interval.bottom(), Ret);
  }
  {
    sandboxir::InstrInterval Interval(
        SmallVector<sandboxir::Instruction *>({I0, I0}));
    EXPECT_EQ(Interval.top(), I0);
    EXPECT_EQ(Interval.bottom(), I0);
  }

  // Check empty().
  EXPECT_FALSE(Interval.empty());
  sandboxir::InstrInterval Empty;
  EXPECT_TRUE(Empty.empty());
  sandboxir::InstrInterval One(I0, I0);
  EXPECT_FALSE(One.empty());
  // Check contains().
  for (auto &I : *BB) {
    EXPECT_TRUE(Interval.contains(&I));
    EXPECT_FALSE(Empty.contains(&I));
  }
  EXPECT_FALSE(One.contains(I1));
  EXPECT_FALSE(One.contains(I2));
  EXPECT_FALSE(One.contains(Ret));
  // Check iterator.
  auto BBIt = BB->begin();
  for (auto &I : Interval)
    EXPECT_EQ(&I, &*BBIt++);
}
