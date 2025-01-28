//===- IntervalTest.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Interval.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/SandboxIR/Context.h"
#include "llvm/SandboxIR/Function.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/Support/SourceMgr.h"
#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"

using namespace llvm;

struct IntervalTest : public testing::Test {
  LLVMContext C;
  std::unique_ptr<Module> M;

  void parseIR(LLVMContext &C, const char *IR) {
    SMDiagnostic Err;
    M = parseAssemblyString(IR, Err, C);
    if (!M)
      Err.print("InstrIntervalTest", errs());
  }
};

TEST_F(IntervalTest, Basic) {
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

  sandboxir::Interval<sandboxir::Instruction> Intvl(I0, Ret);
#ifndef NDEBUG
  EXPECT_DEATH(sandboxir::Interval<sandboxir::Instruction>(I1, I0),
               ".*before.*");
#endif // NDEBUG
  // Check Interval<sandboxir::Instruction>(ArrayRef), from(), to().
  {
    sandboxir::Interval<sandboxir::Instruction> Intvl(
        SmallVector<sandboxir::Instruction *>({I0, Ret}));
    EXPECT_EQ(Intvl.top(), I0);
    EXPECT_EQ(Intvl.bottom(), Ret);
  }
  {
    sandboxir::Interval<sandboxir::Instruction> Intvl(
        SmallVector<sandboxir::Instruction *>({Ret, I0}));
    EXPECT_EQ(Intvl.top(), I0);
    EXPECT_EQ(Intvl.bottom(), Ret);
  }
  {
    sandboxir::Interval<sandboxir::Instruction> Intvl(
        SmallVector<sandboxir::Instruction *>({I0, I0}));
    EXPECT_EQ(Intvl.top(), I0);
    EXPECT_EQ(Intvl.bottom(), I0);
  }

  // Check empty().
  EXPECT_FALSE(Intvl.empty());
  sandboxir::Interval<sandboxir::Instruction> Empty;
  EXPECT_TRUE(Empty.empty());
  sandboxir::Interval<sandboxir::Instruction> One(I0, I0);
  EXPECT_FALSE(One.empty());
  // Check contains().
  for (auto &I : *BB) {
    EXPECT_TRUE(Intvl.contains(&I));
    EXPECT_FALSE(Empty.contains(&I));
  }
  EXPECT_FALSE(One.contains(I1));
  EXPECT_FALSE(One.contains(I2));
  EXPECT_FALSE(One.contains(Ret));
  // Check iterator.
  auto BBIt = BB->begin();
  for (auto &I : Intvl)
    EXPECT_EQ(&I, &*BBIt++);
  {
    // Check equality.
    EXPECT_TRUE(Empty == Empty);
    EXPECT_FALSE(Empty == One);
    EXPECT_TRUE(One == One);
    sandboxir::Interval<sandboxir::Instruction> Intvl1(I0, I2);
    sandboxir::Interval<sandboxir::Instruction> Intvl2(I0, I2);
    EXPECT_TRUE(Intvl1 == Intvl1);
    EXPECT_TRUE(Intvl1 == Intvl2);
  }
  {
    // Check inequality.
    EXPECT_FALSE(Empty != Empty);
    EXPECT_TRUE(Empty != One);
    EXPECT_FALSE(One != One);
    sandboxir::Interval<sandboxir::Instruction> Intvl1(I0, I2);
    sandboxir::Interval<sandboxir::Instruction> Intvl2(I0, I2);
    EXPECT_FALSE(Intvl1 != Intvl1);
    EXPECT_FALSE(Intvl1 != Intvl2);
  }
  {
    // Check disjoint().
    EXPECT_TRUE(Empty.disjoint(Empty));
    EXPECT_TRUE(One.disjoint(Empty));
    EXPECT_TRUE(Empty.disjoint(One));
    sandboxir::Interval<sandboxir::Instruction> Intvl1(I0, I2);
    sandboxir::Interval<sandboxir::Instruction> Intvl2(I1, Ret);
    EXPECT_FALSE(Intvl1.disjoint(Intvl2));
    sandboxir::Interval<sandboxir::Instruction> Intvl3(I2, I2);
    EXPECT_FALSE(Intvl1.disjoint(Intvl3));
    EXPECT_TRUE(Intvl1.disjoint(Empty));
  }
  {
    // Check comesBefore().
    sandboxir::Interval<sandboxir::Instruction> Intvl1(I0, I0);
    sandboxir::Interval<sandboxir::Instruction> Intvl2(I2, I2);
    EXPECT_TRUE(Intvl1.comesBefore(Intvl2));
    EXPECT_FALSE(Intvl2.comesBefore(Intvl1));

    sandboxir::Interval<sandboxir::Instruction> Intvl12(I1, I2);
    EXPECT_TRUE(Intvl1.comesBefore(Intvl12));
    EXPECT_FALSE(Intvl12.comesBefore(Intvl1));
    {
#ifndef NDEBUG
      // Check comesBefore() with non-disjoint intervals.
      sandboxir::Interval<sandboxir::Instruction> Intvl1(I0, I2);
      sandboxir::Interval<sandboxir::Instruction> Intvl2(I2, I2);
      EXPECT_DEATH(Intvl1.comesBefore(Intvl2), ".*disjoint.*");
#endif // NDEBUG
    }
  }
}

// Helper function for returning a vector of instruction pointers from a range
// of references.
template <typename RangeT>
static SmallVector<sandboxir::Instruction *> getPtrVec(RangeT Range) {
  SmallVector<sandboxir::Instruction *> PtrVec;
  for (sandboxir::Instruction &I : Range)
    PtrVec.push_back(&I);
  return PtrVec;
}

TEST_F(IntervalTest, Difference) {
  parseIR(C, R"IR(
define void @foo(i8 %v0) {
  %I0 = add i8 %v0, %v0
  %I1 = add i8 %v0, %v0
  %I2 = add i8 %v0, %v0
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

  {
    // Check [I0,Ret] - []
    sandboxir::Interval<sandboxir::Instruction> I0Ret(I0, Ret);
    sandboxir::Interval<sandboxir::Instruction> Empty;
    auto Diffs = I0Ret - Empty;
    EXPECT_EQ(Diffs.size(), 1u);
    const sandboxir::Interval<sandboxir::Instruction> &Diff = Diffs[0];
    EXPECT_THAT(getPtrVec(Diff), testing::ElementsAre(I0, I1, I2, Ret));

    // Check getSingleDiff().
    EXPECT_EQ(I0Ret.getSingleDiff(Empty), Diff);
  }
  {
    // Check [] - [I0,Ret]
    sandboxir::Interval<sandboxir::Instruction> Empty;
    sandboxir::Interval<sandboxir::Instruction> I0Ret(I0, Ret);
    auto Diffs = Empty - I0Ret;
    EXPECT_EQ(Diffs.size(), 1u);
    const sandboxir::Interval<sandboxir::Instruction> &Diff = Diffs[0];
    EXPECT_TRUE(Diff.empty());

    // Check getSingleDiff().
    EXPECT_EQ(Empty.getSingleDiff(I0Ret), Diff);
  }
  {
    // Check [I0,Ret] - [I0].
    sandboxir::Interval<sandboxir::Instruction> I0Ret(I0, Ret);
    sandboxir::Interval<sandboxir::Instruction> I0I0(I0, I0);
    auto Diffs = I0Ret - I0I0;
    EXPECT_EQ(Diffs.size(), 1u);
    const sandboxir::Interval<sandboxir::Instruction> &Diff = Diffs[0];
    EXPECT_THAT(getPtrVec(Diff), testing::ElementsAre(I1, I2, Ret));

    // Check getSingleDiff().
    EXPECT_EQ(I0Ret.getSingleDiff(I0I0), Diff);
  }
  {
    // Check [I0,Ret] - [I1].
    sandboxir::Interval<sandboxir::Instruction> I0Ret(I0, Ret);
    sandboxir::Interval<sandboxir::Instruction> I1I1(I1, I1);
    auto Diffs = I0Ret - I1I1;
    EXPECT_EQ(Diffs.size(), 2u);
    const sandboxir::Interval<sandboxir::Instruction> &Diff0 = Diffs[0];
    EXPECT_THAT(getPtrVec(Diff0), testing::ElementsAre(I0));
    const sandboxir::Interval<sandboxir::Instruction> &Diff1 = Diffs[1];
    EXPECT_THAT(getPtrVec(Diff1), testing::ElementsAre(I2, Ret));

#ifndef NDEBUG
    // Check getSingleDiff().
    EXPECT_DEATH(I0Ret.getSingleDiff(I1I1), ".*single.*");
#endif // NDEBUG
  }
}

TEST_F(IntervalTest, Intersection) {
  parseIR(C, R"IR(
define void @foo(i8 %v0) {
  %I0 = add i8 %v0, %v0
  %I1 = add i8 %v0, %v0
  %I2 = add i8 %v0, %v0
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
  [[maybe_unused]] auto *I2 = &*It++;
  auto *Ret = &*It++;

  {
    // Check [I0,Ret] ^ []
    sandboxir::Interval<sandboxir::Instruction> I0Ret(I0, Ret);
    sandboxir::Interval<sandboxir::Instruction> Empty;
    auto Intersection = I0Ret.intersection(Empty);
    EXPECT_TRUE(Intersection.empty());
  }
  {
    // Check [] ^ [I0,Ret]
    sandboxir::Interval<sandboxir::Instruction> Empty;
    sandboxir::Interval<sandboxir::Instruction> I0Ret(I0, Ret);
    auto Intersection = Empty.intersection(I0Ret);
    EXPECT_TRUE(Intersection.empty());
  }
  {
    // Check [I0,Ret] ^ [I0]
    sandboxir::Interval<sandboxir::Instruction> I0Ret(I0, Ret);
    sandboxir::Interval<sandboxir::Instruction> I0I0(I0, I0);
    auto Intersection = I0Ret.intersection(I0I0);
    EXPECT_THAT(getPtrVec(Intersection), testing::ElementsAre(I0));
  }
  {
    // Check [I0] ^ [I0,Ret]
    sandboxir::Interval<sandboxir::Instruction> I0I0(I0, I0);
    sandboxir::Interval<sandboxir::Instruction> I0Ret(I0, Ret);
    auto Intersection = I0I0.intersection(I0Ret);
    EXPECT_THAT(getPtrVec(Intersection), testing::ElementsAre(I0));
  }
  {
    // Check [I0,Ret] ^ [I1].
    sandboxir::Interval<sandboxir::Instruction> I0Ret(I0, Ret);
    sandboxir::Interval<sandboxir::Instruction> I1I1(I1, I1);
    auto Intersection = I0Ret.intersection(I1I1);
    EXPECT_THAT(getPtrVec(Intersection), testing::ElementsAre(I1));
  }
}

TEST_F(IntervalTest, UnionInterval) {
  parseIR(C, R"IR(
define void @foo(i8 %v0) {
  %I0 = add i8 %v0, %v0
  %I1 = add i8 %v0, %v0
  %I2 = add i8 %v0, %v0
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
  [[maybe_unused]] auto *I2 = &*It++;
  auto *Ret = &*It++;

  {
    // Check [I0] unionInterval [I2].
    sandboxir::Interval<sandboxir::Instruction> I0I0(I0, I0);
    sandboxir::Interval<sandboxir::Instruction> I2I2(I2, I2);
    auto SingleUnion = I0I0.getUnionInterval(I2I2);
    EXPECT_THAT(getPtrVec(SingleUnion), testing::ElementsAre(I0, I1, I2));
  }
  {
    // Check [I0] unionInterval Empty.
    sandboxir::Interval<sandboxir::Instruction> I0I0(I0, I0);
    sandboxir::Interval<sandboxir::Instruction> Empty;
    auto SingleUnion = I0I0.getUnionInterval(Empty);
    EXPECT_THAT(getPtrVec(SingleUnion), testing::ElementsAre(I0));
  }
  {
    // Check [I0,I1] unionInterval [I1].
    sandboxir::Interval<sandboxir::Instruction> I0I1(I0, I1);
    sandboxir::Interval<sandboxir::Instruction> I1I1(I1, I1);
    auto SingleUnion = I0I1.getUnionInterval(I1I1);
    EXPECT_THAT(getPtrVec(SingleUnion), testing::ElementsAre(I0, I1));
  }
  {
    // Check [I2,Ret] unionInterval [I0].
    sandboxir::Interval<sandboxir::Instruction> I2Ret(I2, Ret);
    sandboxir::Interval<sandboxir::Instruction> I0I0(I0, I0);
    auto SingleUnion = I2Ret.getUnionInterval(I0I0);
    EXPECT_THAT(getPtrVec(SingleUnion), testing::ElementsAre(I0, I1, I2, Ret));
  }
}

TEST_F(IntervalTest, NotifyMoveInstr) {
  parseIR(C, R"IR(
define void @foo(i8 %v0) {
  %I0 = add i8 %v0, %v0
  %I1 = add i8 %v0, %v0
  %I2 = add i8 %v0, %v0
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
  {
    // Assert that we don't try to move external instr to the interval.
    sandboxir::Interval<sandboxir::Instruction> I2Ret(I2, Ret);
#ifndef NDEBUG
    EXPECT_DEATH(I2Ret.notifyMoveInstr(I0, Ret->getIterator()), ".*interval.*");
#endif // NDEBUG
  }
  {
    // Assert that we don't move before self.
    sandboxir::Interval<sandboxir::Instruction> I2Ret(I2, Ret);
#ifndef NDEBUG
    EXPECT_DEATH(I2Ret.notifyMoveInstr(Ret, Ret->getIterator()), ".*self.*");
#endif // NDEBUG
  }
  {
    // Single-element interval.
    sandboxir::Interval<sandboxir::Instruction> I2I2(I2, I2);
    I2I2.notifyMoveInstr(I2, Ret->getIterator());
    EXPECT_EQ(I2I2.top(), I2);
    EXPECT_EQ(I2I2.bottom(), I2);
  }
  {
    // Two-element interval swap.
    sandboxir::Interval<sandboxir::Instruction> I1I2(I1, I2);
    I1I2.notifyMoveInstr(I2, I1->getIterator());
    I2->moveBefore(I1);
    EXPECT_EQ(I1I2.top(), I2);
    EXPECT_EQ(I1I2.bottom(), I1);

    I2->moveAfter(I1);
  }
  {
    // Move to same position.
    sandboxir::Interval<sandboxir::Instruction> I0Ret(I0, Ret);
    I0Ret.notifyMoveInstr(I0, I1->getIterator());
    I0->moveBefore(I1);
    EXPECT_EQ(I0Ret.top(), I0);
    EXPECT_EQ(I0Ret.bottom(), Ret);
  }
  {
    // Move internal to internal.
    sandboxir::Interval<sandboxir::Instruction> I0Ret(I0, Ret);
    I0Ret.notifyMoveInstr(I2, I1->getIterator());
    I2->moveBefore(I1);
    EXPECT_EQ(I0Ret.top(), I0);
    EXPECT_EQ(I0Ret.bottom(), Ret);

    I2->moveAfter(I1);
  }
  {
    // Move internal before top.
    sandboxir::Interval<sandboxir::Instruction> I0Ret(I0, Ret);
    I0Ret.notifyMoveInstr(I2, I0->getIterator());
    I2->moveBefore(I0);
    EXPECT_EQ(I0Ret.top(), I2);
    EXPECT_EQ(I0Ret.bottom(), Ret);

    I2->moveAfter(I1);
  }
  {
    // Move internal to bottom.
    sandboxir::Interval<sandboxir::Instruction> I0Ret(I0, Ret);
    I0Ret.notifyMoveInstr(I2, BB->end());
    I2->moveAfter(Ret);
    EXPECT_EQ(I0Ret.top(), I0);
    EXPECT_EQ(I0Ret.bottom(), I2);

    I2->moveAfter(I1);
  }
  {
    // Move bottom before internal.
    sandboxir::Interval<sandboxir::Instruction> I0Ret(I0, Ret);
    I0Ret.notifyMoveInstr(Ret, I2->getIterator());
    Ret->moveBefore(I2);
    EXPECT_EQ(I0Ret.top(), I0);
    EXPECT_EQ(I0Ret.bottom(), I2);

    Ret->moveAfter(I2);
  }
  {
    // Move bottom before top.
    sandboxir::Interval<sandboxir::Instruction> I0Ret(I0, Ret);
    I0Ret.notifyMoveInstr(Ret, I0->getIterator());
    Ret->moveBefore(I0);
    EXPECT_EQ(I0Ret.top(), Ret);
    EXPECT_EQ(I0Ret.bottom(), I2);

    Ret->moveAfter(I2);
  }
}
