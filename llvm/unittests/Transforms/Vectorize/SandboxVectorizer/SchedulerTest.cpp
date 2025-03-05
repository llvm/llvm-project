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

static sandboxir::BasicBlock *getBasicBlockByName(sandboxir::Function *F,
                                                  StringRef Name) {
  for (sandboxir::BasicBlock &BB : *F)
    if (BB.getName() == Name)
      return &BB;
  llvm_unreachable("Expected to find basic block!");
}

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

// Check that when we erase a DAG node its SchedBundle gets updated.
TEST_F(SchedulerTest, SchedBundleEraseDGNode) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %v0, i8 %v1, i8 %v2, i8 %v3) {
  store i8 %v0, ptr %ptr
  store i8 %v1, ptr %ptr
  store i8 %v2, ptr %ptr
  store i8 %v3, ptr %ptr
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
  auto *S2 = cast<sandboxir::StoreInst>(&*It++);
  auto *S3 = cast<sandboxir::StoreInst>(&*It++);

  sandboxir::DependencyGraph DAG(getAA(*LLVMF), Ctx);
  DAG.extend({&*BB->begin(), BB->getTerminator()});
  auto *SN0 = DAG.getNode(S0);
  auto *SN1 = DAG.getNode(S1);
  auto *SN2 = DAG.getNode(S2);
  auto *SN3 = DAG.getNode(S3);
  {
    // Check the common case, when the bundle contains unique nodes.
    sandboxir::SchedBundle Bndl({SN0, SN1});
    S0->eraseFromParent();
    EXPECT_THAT(Bndl, testing::ElementsAre(SN1));
  }
  {
    // Check corner case when the node appears more than once.
    sandboxir::SchedBundle Bndl({SN2, SN3, SN2});
    S2->eraseFromParent();
    EXPECT_THAT(Bndl, testing::ElementsAre(SN3));
  }
}

// Check that assigning a bundle to a DAG Node that is already assigned to a
// bundle, removes the node from the old bundle.
TEST_F(SchedulerTest, SchedBundleReassign) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %v0, i8 %v1, i8 %v2) {
  store i8 %v0, ptr %ptr
  store i8 %v1, ptr %ptr
  store i8 %v2, ptr %ptr
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
  auto *S2 = cast<sandboxir::StoreInst>(&*It++);

  sandboxir::DependencyGraph DAG(getAA(*LLVMF), Ctx);
  DAG.extend({&*BB->begin(), BB->getTerminator()});
  auto *SN0 = DAG.getNode(S0);
  auto *SN1 = DAG.getNode(S1);
  auto *SN2 = DAG.getNode(S2);
  sandboxir::SchedBundle BndlOld({SN0, SN1});
  sandboxir::SchedBundle BndlNew({SN0, SN2});
  EXPECT_THAT(BndlOld, testing::ElementsAre(SN1));
  EXPECT_THAT(BndlNew, testing::ElementsAre(SN0, SN2));
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

TEST_F(SchedulerTest, TrimSchedule) {
  parseIR(C, R"IR(
define void @foo(ptr noalias %ptr0, ptr noalias %ptr1, i8 %arg) {
  %zext = zext i8 0 to i32
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
  auto *Z = cast<sandboxir::CastInst>(&*It++);
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
  // as singleton bundles, but {S0,S1} and {L0,L1} as vector bundles.
  // Check if rescheduling works.
  EXPECT_TRUE(Sched.trySchedule({Add0, Add1}));
  // These should fail because {L0,L1} is a vector bundle.
  EXPECT_FALSE(Sched.trySchedule({L0, Z}));
  EXPECT_FALSE(Sched.trySchedule({L1, Z}));
  // This should succeed because it matches the original vec bundle.
  EXPECT_TRUE(Sched.trySchedule({L0, L1}));
}

// Test that an instruction can't belong in two bundles!
TEST_F(SchedulerTest, CheckBundles) {
  parseIR(C, R"IR(
define void @foo(ptr noalias %ptr0, ptr noalias %ptr1, ptr noalias %ptr2) {
  %L0 = load i8, ptr %ptr0
  %L1 = load i8, ptr %ptr1 ; This belongs in 2 bundles!
  %L2 = load i8, ptr %ptr2
  %add0 = add i8 %L0, %L1
  %add1 = add i8 %L1, %L2
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
  auto *L2 = cast<sandboxir::LoadInst>(&*It++);
  auto *Add0 = cast<sandboxir::BinaryOperator>(&*It++);
  auto *Add1 = cast<sandboxir::BinaryOperator>(&*It++);
  auto *S0 = cast<sandboxir::StoreInst>(&*It++);
  auto *S1 = cast<sandboxir::StoreInst>(&*It++);

  sandboxir::Scheduler Sched(getAA(*LLVMF), Ctx);
  EXPECT_TRUE(Sched.trySchedule({S0, S1}));
  EXPECT_TRUE(Sched.trySchedule({Add0, Add1}));
  EXPECT_TRUE(Sched.trySchedule({L0, L1}));
  // This should fail because L1 is already part of {L0,L1}
  EXPECT_FALSE(Sched.trySchedule({L1, L2}));
  EXPECT_FALSE(Sched.trySchedule({L2, L1}));
}

// Try schedule a bundle {L1,L2} where L1 is already scheduled in {L0,L1}
// but L2 is not in the DAG at all
TEST_F(SchedulerTest, CheckBundles2) {
  parseIR(C, R"IR(
define void @foo(ptr noalias %ptr0, ptr noalias %ptr1, ptr noalias %ptr2) {
  %L2 = load i8, ptr %ptr2 ; This is not in the DAG
  %L1 = load i8, ptr %ptr1 ; This belongs in 2 bundles!
  %L0 = load i8, ptr %ptr0
  %add1 = add i8 %L1, %L2
  %add0 = add i8 %L0, %L1
  store i8 %add1, ptr %ptr1
  store i8 %add0, ptr %ptr0
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *L2 = cast<sandboxir::LoadInst>(&*It++);
  auto *L1 = cast<sandboxir::LoadInst>(&*It++);
  auto *L0 = cast<sandboxir::LoadInst>(&*It++);
  auto *Add1 = cast<sandboxir::BinaryOperator>(&*It++);
  auto *Add0 = cast<sandboxir::BinaryOperator>(&*It++);
  auto *S1 = cast<sandboxir::StoreInst>(&*It++);
  auto *S0 = cast<sandboxir::StoreInst>(&*It++);

  sandboxir::Scheduler Sched(getAA(*LLVMF), Ctx);
  EXPECT_TRUE(Sched.trySchedule({S0, S1}));
  EXPECT_TRUE(Sched.trySchedule({Add0, Add1}));
  EXPECT_TRUE(Sched.trySchedule({L0, L1}));
  // This should fail because L1 is already part of {L0,L1}.
  EXPECT_FALSE(Sched.trySchedule({L1, L2}));
  EXPECT_FALSE(Sched.trySchedule({L2, L1}));
}

// Try schedule a bundle {L1,L2} where L1 is already scheduled in {L0,L1}
// but L2 is in the DAG but isn't scheduled.
TEST_F(SchedulerTest, CheckBundles3) {
  parseIR(C, R"IR(
define void @foo(ptr noalias %ptr0, ptr noalias %ptr1, ptr noalias %ptr2) {
  %L2 = load i8, ptr %ptr2 ; This is not in the DAG
  %L1 = load i8, ptr %ptr1 ; This belongs in 2 bundles!
  %L0 = load i8, ptr %ptr0
  %add1 = add i8 %L1, %L2
  %add0 = add i8 %L0, %L1
  store i8 %add1, ptr %ptr1
  store i8 %add0, ptr %ptr0
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *L2 = cast<sandboxir::LoadInst>(&*It++);
  auto *L1 = cast<sandboxir::LoadInst>(&*It++);
  auto *L0 = cast<sandboxir::LoadInst>(&*It++);
  auto *Add1 = cast<sandboxir::BinaryOperator>(&*It++);
  auto *Add0 = cast<sandboxir::BinaryOperator>(&*It++);
  auto *S1 = cast<sandboxir::StoreInst>(&*It++);
  auto *S0 = cast<sandboxir::StoreInst>(&*It++);

  sandboxir::Scheduler Sched(getAA(*LLVMF), Ctx);
  EXPECT_TRUE(Sched.trySchedule({S0, S1}));
  EXPECT_TRUE(Sched.trySchedule({Add0, Add1}));
  EXPECT_TRUE(Sched.trySchedule({L0, L1}));
  // Add L2 to the DAG, but don't schedule it.
  auto &DAG = sandboxir::SchedulerInternalsAttorney::getDAG(Sched);
  DAG.extend(L2);
  // This should fail because L1 is already part of {L0,L1}.
  EXPECT_FALSE(Sched.trySchedule({L1, L2}));
  EXPECT_FALSE(Sched.trySchedule({L2, L1}));
}

// Check that Scheduler::getBndlSchedState() works correctly.
TEST_F(SchedulerTest, GetBndlSchedState) {
  parseIR(C, R"IR(
define void @foo(ptr noalias %ptr0, ptr noalias %ptr1, ptr noalias %ptr2) {
  %L2 = load i8, ptr %ptr2 ; This is not in the DAG
  %L1 = load i8, ptr %ptr1 ; This belongs in 2 bundles!
  %L0 = load i8, ptr %ptr0
  %add1 = add i8 %L1, %L2
  %add0 = add i8 %L0, %L1
  store i8 %add1, ptr %ptr1
  store i8 %add0, ptr %ptr0
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *L2 = cast<sandboxir::LoadInst>(&*It++);
  auto *L1 = cast<sandboxir::LoadInst>(&*It++);
  auto *L0 = cast<sandboxir::LoadInst>(&*It++);
  auto *Add1 = cast<sandboxir::BinaryOperator>(&*It++);
  auto *Add0 = cast<sandboxir::BinaryOperator>(&*It++);
  auto *S1 = cast<sandboxir::StoreInst>(&*It++);
  auto *S0 = cast<sandboxir::StoreInst>(&*It++);

  sandboxir::Scheduler Sched(getAA(*LLVMF), Ctx);
  auto &DAG = sandboxir::SchedulerInternalsAttorney::getDAG(Sched);
  auto GetBndlSchedState = [&Sched](ArrayRef<sandboxir::Instruction *> Instrs) {
    return sandboxir::SchedulerInternalsAttorney::getBndlSchedState(Sched,
                                                                    Instrs);
  };
  using BndlSchedState = sandboxir::SchedulerInternalsAttorney::BndlSchedState;
  // Check when instructions are not in the DAG.
  EXPECT_EQ(GetBndlSchedState({S0}), BndlSchedState::NoneScheduled);
  EXPECT_EQ(GetBndlSchedState({S0, S1}), BndlSchedState::NoneScheduled);
  EXPECT_EQ(GetBndlSchedState({S0, S1}), BndlSchedState::NoneScheduled);
  // Check when instructions are in the DAG.
  DAG.extend({S0, S1});
  EXPECT_EQ(GetBndlSchedState({S0}), BndlSchedState::NoneScheduled);
  EXPECT_EQ(GetBndlSchedState({S0, S1}), BndlSchedState::NoneScheduled);
  EXPECT_EQ(GetBndlSchedState({S0, S1}), BndlSchedState::NoneScheduled);
  // One instruction in the DAG and the other not in the DAG.
  EXPECT_EQ(GetBndlSchedState({S0, Add0}), BndlSchedState::NoneScheduled);

  // Check with scheduled instructions.
  Sched.clear(); // Manually extending the DAG messes with the scheduler.
  EXPECT_TRUE(Sched.trySchedule({S0, S1}));
  // Check fully scheduled.
  EXPECT_EQ(GetBndlSchedState({S0, S1}), BndlSchedState::FullyScheduled);
  // Check scheduled + not in DAG.
  EXPECT_EQ(GetBndlSchedState({S0, Add0}), BndlSchedState::AlreadyScheduled);
  EXPECT_EQ(GetBndlSchedState({Add0, S0}), BndlSchedState::AlreadyScheduled);
  EXPECT_EQ(GetBndlSchedState({Add0, S1}), BndlSchedState::AlreadyScheduled);
  EXPECT_EQ(GetBndlSchedState({Add0, Add1}), BndlSchedState::NoneScheduled);
  // Extend DAG such that Add0 and Add1 are in the DAG but are not scheduled.
  DAG.extend({Add0, Add1});
  // Check both in DAG but not scheduled.
  EXPECT_EQ(GetBndlSchedState({Add0, Add1}), BndlSchedState::NoneScheduled);
  // Check scheduled + in DAG but not scheduled.
  EXPECT_EQ(GetBndlSchedState({S0, Add0}), BndlSchedState::AlreadyScheduled);
  EXPECT_EQ(GetBndlSchedState({Add0, S0}), BndlSchedState::AlreadyScheduled);
  EXPECT_EQ(GetBndlSchedState({Add0, S1}), BndlSchedState::AlreadyScheduled);

  Sched.clear(); // Manually extending the DAG messes with the scheduler.
  // Schedule instructions towards the top so that intermediate instructions
  // (namely Add0, Add1) get temporarily scheduled in singleton bundles.
  EXPECT_TRUE(Sched.trySchedule({S0, S1}));
  EXPECT_TRUE(Sched.trySchedule({L0, L1}));
  // Check fully scheduled.
  EXPECT_EQ(GetBndlSchedState({L0, L1}), BndlSchedState::FullyScheduled);
  // Check both singletons.
  EXPECT_EQ(GetBndlSchedState({Add0, Add1}),
            BndlSchedState::TemporarilyScheduled);
  // Check single singleton.
  EXPECT_EQ(GetBndlSchedState({Add0}), BndlSchedState::TemporarilyScheduled);
  EXPECT_EQ(GetBndlSchedState({Add1}), BndlSchedState::TemporarilyScheduled);
  // Check singleton + scheduled.
  EXPECT_EQ(GetBndlSchedState({L0, S1}), BndlSchedState::AlreadyScheduled);
  EXPECT_EQ(GetBndlSchedState({S1, L0}), BndlSchedState::AlreadyScheduled);
  EXPECT_EQ(GetBndlSchedState({L0, Add1}), BndlSchedState::AlreadyScheduled);
  EXPECT_EQ(GetBndlSchedState({Add1, L0}), BndlSchedState::AlreadyScheduled);
  // Check singleton + not in DAG.
  EXPECT_EQ(GetBndlSchedState({Add1, L2}),
            BndlSchedState::TemporarilyScheduled);
  EXPECT_EQ(GetBndlSchedState({L2, Add0}),
            BndlSchedState::TemporarilyScheduled);

  // Check duplicates.
  // TODO: Should duplicates be allowed?
  EXPECT_EQ(GetBndlSchedState({L2, L2}), BndlSchedState::NoneScheduled);
  EXPECT_EQ(GetBndlSchedState({S0, S0}), BndlSchedState::FullyScheduled);
  EXPECT_EQ(GetBndlSchedState({Add0, Add1}),
            BndlSchedState::TemporarilyScheduled);
}

// Check scheduling in the following order: {A0,A1},{B0,B1},{C0,C1},{D0,D1}
// assuming program order: B0,B1,C0,C1,D0,D1,E0,D1.
// This will effectively schedule nodes below already scheduled nodes, which
// can expose issues in the code that adds nodes to the ready list.
// For example, we schedule {D0,D1} while {C0,C1} are scheduled and there is
// a dependency D0->C0 and D1->C1.
//
//                   {A0,A1}  {B0,B1}  {C0,C1}  {D0,D1}
//   B0,B1                    | S
//   |\                       |
//   | C0,C1                  |        | S      | S
//   |  | \                   |                 |
//   |  |  D0,D1              |                 | S
//   | /                      |
//   A0,A1             | S    | S
//                 +------------------------+
//                 | Legend   |: DAG        |
//                 |          S: Scheduled  |
TEST_F(SchedulerTest, ScheduledPredecessors) {
  parseIR(C, R"IR(
define void @foo(ptr noalias %ptrA0, ptr noalias %ptrA1,
                 ptr noalias %ptrB0, ptr noalias %ptrB1,
                 ptr noalias %ptrD0, ptr noalias %ptrD1) {
  %B0 = load i8, ptr %ptrB0
  %B1 = load i8, ptr %ptrB1
  %C0 = add i8 %B0, 0
  %C1 = add i8 %B1, 1
  store i8 %C0, ptr %ptrD0
  store i8 %C1, ptr %ptrD1
  store i8 %B0, ptr %ptrA0
  store i8 %B1, ptr %ptrA1
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *B1 = cast<sandboxir::LoadInst>(&*It++);
  auto *B0 = cast<sandboxir::LoadInst>(&*It++);
  auto *C1 = cast<sandboxir::BinaryOperator>(&*It++);
  auto *C0 = cast<sandboxir::BinaryOperator>(&*It++);
  auto *D1 = cast<sandboxir::StoreInst>(&*It++);
  auto *D0 = cast<sandboxir::StoreInst>(&*It++);
  auto *A1 = cast<sandboxir::StoreInst>(&*It++);
  auto *A0 = cast<sandboxir::StoreInst>(&*It++);
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);
  (void)Ret;

  sandboxir::Scheduler Sched(getAA(*LLVMF), Ctx);
  EXPECT_TRUE(Sched.trySchedule({A0, A1}));
  // NOTE: We schedule the intermediate nodes between {A0,A1} and {B0,B1} by
  // hand one by one to make sure they are scheduled in that order because
  // the scheduler may reorder them a bit if we let it do it.
  EXPECT_TRUE(Sched.trySchedule(D0));
  EXPECT_TRUE(Sched.trySchedule(D1));
  EXPECT_TRUE(Sched.trySchedule(C0));
  EXPECT_TRUE(Sched.trySchedule(C1));
  EXPECT_TRUE(Sched.trySchedule({B0, B1}));
  // At this point all nodes must have been scheduled from B0,B1 to A0,A1.
  // The ones in between are scheduled as single-instruction nodes.
  // So when we attempt to schedule {C0,C1} we will need to reschedule.
  // At this point we will trim the schedule from {C0,C1} upwards.
  EXPECT_TRUE(Sched.trySchedule({C0, C1}));
  // Now the schedule should only contain {C0,C1} which should be marked as
  // "scheduled".
  // {D0,D1} are below {C0,C1}, so we grow the DAG downwards, while
  // {C0,C1} are marked as "scheduled" above them.
  EXPECT_TRUE(Sched.trySchedule({D0, D1}));
}

TEST_F(SchedulerTest, DontCrossBBs) {
  parseIR(C, R"IR(
define void @foo(ptr noalias %ptr0, ptr noalias %ptr1, i8 %v0, i8 %v1) {
bb0:
  %add0 = add i8 %v0, 0
  %add1 = add i8 %v1, 1
  br label %bb1

bb1:
  store i8 %add0, ptr %ptr0
  store i8 %add1, ptr %ptr1
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB0 = getBasicBlockByName(F, "bb0");
  auto *BB1 = getBasicBlockByName(F, "bb1");
  auto It = BB0->begin();
  auto *Add0 = &*It++;
  auto *Add1 = &*It++;

  It = BB1->begin();
  auto *S0 = cast<sandboxir::StoreInst>(&*It++);
  auto *S1 = cast<sandboxir::StoreInst>(&*It++);
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);

  {
    // Schedule bottom-up
    sandboxir::Scheduler Sched(getAA(*LLVMF), Ctx);
    EXPECT_TRUE(Sched.trySchedule({Ret}));
    EXPECT_TRUE(Sched.trySchedule({S0, S1}));
    // Scheduling across blocks should fail.
    EXPECT_FALSE(Sched.trySchedule({Add0, Add1}));
  }
  {
    // Schedule top-down
    sandboxir::Scheduler Sched(getAA(*LLVMF), Ctx);
    EXPECT_TRUE(Sched.trySchedule({Add0, Add1}));
    // Scheduling across blocks should fail.
    EXPECT_FALSE(Sched.trySchedule({S0, S1}));
  }
}

TEST_F(SchedulerTest, NotifyCreateInst) {
  parseIR(C, R"IR(
define void @foo(ptr noalias %ptr, ptr noalias %ptr1, ptr noalias %ptr2) {
  %ld0 = load i8, ptr %ptr
  store i8 %ld0, ptr %ptr
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *L0 = cast<sandboxir::LoadInst>(&*It++);
  auto *S0 = cast<sandboxir::StoreInst>(&*It++);
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);
  auto *Ptr1 = F->getArg(1);
  auto *Ptr2 = F->getArg(2);

  sandboxir::Scheduler Sched(getAA(*LLVMF), Ctx);
  // Schedule Ret and S0. The top of schedule should be at S0.
  EXPECT_TRUE(Sched.trySchedule({Ret}));
  EXPECT_TRUE(Sched.trySchedule({S0}));
  auto &DAG = sandboxir::SchedulerInternalsAttorney::getDAG(Sched);
  DAG.extend({L0});
  auto *L0N = DAG.getNode(L0);
  EXPECT_EQ(L0N->getNumUnscheduledSuccs(), 0u);
  // We should have DAG nodes for all instructions at this point

  // Now create a new instruction below S0.
  sandboxir::StoreInst *NewS1 =
      sandboxir::StoreInst::create(L0, Ptr1, Align(8), Ret->getIterator(),
                                   /*IsVolatile=*/false, Ctx);
  // Check that it is marked as "scheduled".
  auto *NewS1N = DAG.getNode(NewS1);
  EXPECT_TRUE(NewS1N->scheduled());
  // Check that L0's UnscheduledSuccs are still == 0 since NewS1 is "scheduled".
  EXPECT_EQ(L0N->getNumUnscheduledSuccs(), 0u);

  // Now create a new instruction above S0.
  sandboxir::StoreInst *NewS2 =
      sandboxir::StoreInst::create(L0, Ptr2, Align(8), S0->getIterator(),
                                   /*IsVolatile=*/false, Ctx);
  // Check that it is not marked as "scheduled".
  auto *NewS2N = DAG.getNode(NewS2);
  EXPECT_FALSE(NewS2N->scheduled());
  // Check that L0's UnscheduledSuccs got updated because of NewS2.
  EXPECT_EQ(L0N->getNumUnscheduledSuccs(), 1u);

  sandboxir::ReadyListContainer ReadyList;
  // Check empty().
  EXPECT_TRUE(ReadyList.empty());
}

TEST_F(SchedulerTest, ReadyList) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %ld0 = load i8, ptr %ptr
  store i8 %ld0, ptr %ptr
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *L0 = cast<sandboxir::LoadInst>(&*It++);
  auto *S0 = cast<sandboxir::StoreInst>(&*It++);
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);

  sandboxir::DependencyGraph DAG(getAA(*LLVMF), Ctx);
  DAG.extend({&*BB->begin(), BB->getTerminator()});
  auto *L0N = DAG.getNode(L0);
  auto *S0N = DAG.getNode(S0);
  auto *RetN = DAG.getNode(Ret);

  sandboxir::ReadyListContainer ReadyList;
  // Check empty().
  EXPECT_TRUE(ReadyList.empty());
  // Check insert(), pop().
  ReadyList.insert(L0N);
  EXPECT_FALSE(ReadyList.empty());
  EXPECT_EQ(ReadyList.pop(), L0N);
  // Check clear().
  ReadyList.insert(L0N);
  EXPECT_FALSE(ReadyList.empty());
  ReadyList.clear();
  EXPECT_TRUE(ReadyList.empty());
  // Check remove().
  EXPECT_TRUE(ReadyList.empty());
  ReadyList.remove(L0N); // Removing a non-existing node should be valid.
  ReadyList.insert(L0N);
  ReadyList.insert(S0N);
  ReadyList.insert(RetN);
  ReadyList.remove(S0N);
  DenseSet<sandboxir::DGNode *> Nodes;
  Nodes.insert(ReadyList.pop());
  Nodes.insert(ReadyList.pop());
  EXPECT_TRUE(ReadyList.empty());
  EXPECT_THAT(Nodes, testing::UnorderedElementsAre(L0N, RetN));
}

TEST_F(SchedulerTest, ReadyListPriorities) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr) {
bb0:
  br label %bb1

bb1:
  %phi0 = phi i8 [0, %bb0], [1, %bb1]
  %phi1 = phi i8 [0, %bb0], [1, %bb1]
  %ld0 = load i8, ptr %ptr
  store i8 %ld0, ptr %ptr
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB1 = getBasicBlockByName(F, "bb1");
  auto It = BB1->begin();
  auto *Phi0 = cast<sandboxir::PHINode>(&*It++);
  auto *Phi1 = cast<sandboxir::PHINode>(&*It++);
  auto *L0 = cast<sandboxir::LoadInst>(&*It++);
  auto *S0 = cast<sandboxir::StoreInst>(&*It++);
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);

  sandboxir::DependencyGraph DAG(getAA(*LLVMF), Ctx);
  DAG.extend({&*BB1->begin(), BB1->getTerminator()});
  auto *Phi0N = DAG.getNode(Phi0);
  auto *Phi1N = DAG.getNode(Phi1);
  auto *L0N = DAG.getNode(L0);
  auto *S0N = DAG.getNode(S0);
  auto *RetN = DAG.getNode(Ret);

  sandboxir::ReadyListContainer ReadyList;
  // Check PHI vs non-PHI.
  ReadyList.insert(S0N);
  ReadyList.insert(Phi0N);
  EXPECT_EQ(ReadyList.pop(), Phi0N);
  EXPECT_EQ(ReadyList.pop(), S0N);
  ReadyList.insert(Phi0N);
  ReadyList.insert(S0N);
  EXPECT_EQ(ReadyList.pop(), Phi0N);
  EXPECT_EQ(ReadyList.pop(), S0N);
  // Check PHI vs terminator.
  ReadyList.insert(RetN);
  ReadyList.insert(Phi1N);
  EXPECT_EQ(ReadyList.pop(), Phi1N);
  EXPECT_EQ(ReadyList.pop(), RetN);
  ReadyList.insert(Phi1N);
  ReadyList.insert(RetN);
  EXPECT_EQ(ReadyList.pop(), Phi1N);
  EXPECT_EQ(ReadyList.pop(), RetN);
  // Check terminator vs non-terminator.
  ReadyList.insert(RetN);
  ReadyList.insert(L0N);
  EXPECT_EQ(ReadyList.pop(), L0N);
  EXPECT_EQ(ReadyList.pop(), RetN);
  ReadyList.insert(L0N);
  ReadyList.insert(RetN);
  EXPECT_EQ(ReadyList.pop(), L0N);
  EXPECT_EQ(ReadyList.pop(), RetN);
  // Check all, program order.
  ReadyList.insert(RetN);
  ReadyList.insert(L0N);
  ReadyList.insert(Phi1N);
  ReadyList.insert(S0N);
  ReadyList.insert(Phi0N);
  EXPECT_EQ(ReadyList.pop(), Phi0N);
  EXPECT_EQ(ReadyList.pop(), Phi1N);
  EXPECT_EQ(ReadyList.pop(), L0N);
  EXPECT_EQ(ReadyList.pop(), S0N);
  EXPECT_EQ(ReadyList.pop(), RetN);
}
