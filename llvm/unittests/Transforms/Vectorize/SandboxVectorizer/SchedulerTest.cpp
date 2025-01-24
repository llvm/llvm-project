//===- SchedulerTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Scheduler.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Dominators.h"
#include "llvm/SandboxIR/Context.h"
#include "llvm/SandboxIR/Function.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/Support/SourceMgr.h"
#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"

using namespace llvm;

struct SchedulerTest : public testing::Test {
  LLVMContext C;
  std::unique_ptr<Module> M;
  std::unique_ptr<AssumptionCache> AC;
  std::unique_ptr<DominatorTree> DT;
  std::unique_ptr<BasicAAResult> BAA;
  std::unique_ptr<AAResults> AA;

  void parseIR(LLVMContext &C, const char *IR) {
    SMDiagnostic Err;
    M = parseAssemblyString(IR, Err, C);
    if (!M)
      Err.print("SchedulerTest", errs());
  }

  AAResults &getAA(llvm::Function &LLVMF) {
    TargetLibraryInfoImpl TLII;
    TargetLibraryInfo TLI(TLII);
    AA = std::make_unique<AAResults>(TLI);
    AC = std::make_unique<AssumptionCache>(LLVMF);
    DT = std::make_unique<DominatorTree>(LLVMF);
    BAA = std::make_unique<BasicAAResult>(M->getDataLayout(), LLVMF, TLI, *AC,
                                          DT.get());
    AA->addAAResult(*BAA);
    return *AA;
  }
};

TEST_F(SchedulerTest, SchedBundle) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %v0, i8 %v1) {
  store i8 %v0, ptr %ptr
  %other = add i8 %v0, %v1
  store i8 %v1, ptr %ptr
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *S0 = cast<sandboxir::StoreInst>(&*It++);
  auto *Other = &*It++;
  auto *S1 = cast<sandboxir::StoreInst>(&*It++);
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);

  sandboxir::DependencyGraph DAG(getAA(*LLVMF), Ctx);
  DAG.extend({&*BB->begin(), BB->getTerminator()});
  auto *SN0 = DAG.getNode(S0);
  auto *SN1 = DAG.getNode(S1);
  sandboxir::SchedBundle Bndl({SN0, SN1});

  // Check getTop().
  EXPECT_EQ(Bndl.getTop(), SN0);
  // Check getBot().
  EXPECT_EQ(Bndl.getBot(), SN1);
  // Check cluster().
  Bndl.cluster(S1->getIterator());
  {
    auto It = BB->begin();
    EXPECT_EQ(&*It++, Other);
    EXPECT_EQ(&*It++, S0);
    EXPECT_EQ(&*It++, S1);
    EXPECT_EQ(&*It++, Ret);
    S0->moveBefore(Other);
  }

  Bndl.cluster(S0->getIterator());
  {
    auto It = BB->begin();
    EXPECT_EQ(&*It++, S0);
    EXPECT_EQ(&*It++, S1);
    EXPECT_EQ(&*It++, Other);
    EXPECT_EQ(&*It++, Ret);
    S1->moveAfter(Other);
  }

  Bndl.cluster(Other->getIterator());
  {
    auto It = BB->begin();
    EXPECT_EQ(&*It++, S0);
    EXPECT_EQ(&*It++, S1);
    EXPECT_EQ(&*It++, Other);
    EXPECT_EQ(&*It++, Ret);
    S1->moveAfter(Other);
  }

  Bndl.cluster(Ret->getIterator());
  {
    auto It = BB->begin();
    EXPECT_EQ(&*It++, Other);
    EXPECT_EQ(&*It++, S0);
    EXPECT_EQ(&*It++, S1);
    EXPECT_EQ(&*It++, Ret);
    Other->moveBefore(S1);
  }

  Bndl.cluster(BB->end());
  {
    auto It = BB->begin();
    EXPECT_EQ(&*It++, Other);
    EXPECT_EQ(&*It++, Ret);
    EXPECT_EQ(&*It++, S0);
    EXPECT_EQ(&*It++, S1);
    Ret->moveAfter(S1);
    Other->moveAfter(S0);
  }
  // Check iterators.
  EXPECT_THAT(Bndl, testing::ElementsAre(SN0, SN1));
  EXPECT_THAT((const sandboxir::SchedBundle &)Bndl,
              testing::ElementsAre(SN0, SN1));
}

TEST_F(SchedulerTest, Basic) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %v0, i8 %v1) {
  store i8 %v0, ptr %ptr
  store i8 %v1, ptr %ptr
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *S0 = cast<sandboxir::StoreInst>(&*It++);
  auto *S1 = cast<sandboxir::StoreInst>(&*It++);
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);

  {
    // Schedule all instructions in sequence.
    sandboxir::Scheduler Sched(getAA(*LLVMF), Ctx);
    EXPECT_TRUE(Sched.trySchedule({Ret}));
    EXPECT_TRUE(Sched.trySchedule({S1}));
    EXPECT_TRUE(Sched.trySchedule({S0}));
  }
  {
    // Skip instructions.
    sandboxir::Scheduler Sched(getAA(*LLVMF), Ctx);
    EXPECT_TRUE(Sched.trySchedule({Ret}));
    EXPECT_TRUE(Sched.trySchedule({S0}));
  }
  {
    // Try invalid scheduling. Dependency S0->S1.
    sandboxir::Scheduler Sched(getAA(*LLVMF), Ctx);
    EXPECT_TRUE(Sched.trySchedule({Ret}));
    EXPECT_FALSE(Sched.trySchedule({S0, S1}));
  }
}

TEST_F(SchedulerTest, Bundles) {
  parseIR(C, R"IR(
define void @foo(ptr noalias %ptr0, ptr noalias %ptr1) {
  %ld0 = load i8, ptr %ptr0
  %ld1 = load i8, ptr %ptr1
  store i8 %ld0, ptr %ptr0
  store i8 %ld1, ptr %ptr1
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *L0 = cast<sandboxir::LoadInst>(&*It++);
  auto *L1 = cast<sandboxir::LoadInst>(&*It++);
  auto *S0 = cast<sandboxir::StoreInst>(&*It++);
  auto *S1 = cast<sandboxir::StoreInst>(&*It++);
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);

  sandboxir::Scheduler Sched(getAA(*LLVMF), Ctx);
  EXPECT_TRUE(Sched.trySchedule({Ret}));
  EXPECT_TRUE(Sched.trySchedule({S0, S1}));
  EXPECT_TRUE(Sched.trySchedule({L0, L1}));
}

TEST_F(SchedulerTest, RescheduleAlreadyScheduled) {
  parseIR(C, R"IR(
define void @foo(ptr noalias %ptr0, ptr noalias %ptr1) {
  %ld0 = load i8, ptr %ptr0
  %ld1 = load i8, ptr %ptr1
  %add0 = add i8 %ld0, %ld0
  %add1 = add i8 %ld1, %ld1
  store i8 %add0, ptr %ptr0
  store i8 %add1, ptr %ptr1
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *L0 = cast<sandboxir::LoadInst>(&*It++);
  auto *L1 = cast<sandboxir::LoadInst>(&*It++);
  auto *Add0 = cast<sandboxir::BinaryOperator>(&*It++);
  auto *Add1 = cast<sandboxir::BinaryOperator>(&*It++);
  auto *S0 = cast<sandboxir::StoreInst>(&*It++);
  auto *S1 = cast<sandboxir::StoreInst>(&*It++);
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);

  sandboxir::Scheduler Sched(getAA(*LLVMF), Ctx);
  EXPECT_TRUE(Sched.trySchedule({Ret}));
  EXPECT_TRUE(Sched.trySchedule({S0, S1}));
  EXPECT_TRUE(Sched.trySchedule({L0, L1}));
  // At this point Add0 and Add1 should have been individually scheduled
  // as single bundles.
  // Check if rescheduling works.
  EXPECT_TRUE(Sched.trySchedule({Add0, Add1}));
  EXPECT_TRUE(Sched.trySchedule({L0, L1}));
}
