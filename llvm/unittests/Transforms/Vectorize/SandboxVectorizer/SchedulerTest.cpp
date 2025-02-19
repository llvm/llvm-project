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
