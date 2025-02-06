//===- DependencyGraphTest.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/DependencyGraph.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/SandboxIR/Context.h"
#include "llvm/SandboxIR/Function.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/Support/SourceMgr.h"
#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"

using namespace llvm;

struct DependencyGraphTest : public testing::Test {
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
      Err.print("DependencyGraphTest", errs());
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
  /// \Returns true if there is a dependency: SrcN->DstN.
  bool memDependency(sandboxir::DGNode *SrcN, sandboxir::DGNode *DstN) {
    if (auto *MemDstN = dyn_cast<sandboxir::MemDGNode>(DstN))
      return MemDstN->hasMemPred(SrcN);
    return false;
  }
};

TEST_F(DependencyGraphTest, isStackSaveOrRestoreIntrinsic) {
  parseIR(C, R"IR(
declare void @llvm.sideeffect()
define void @foo(i8 %v1, ptr %ptr) {
  %add = add i8 %v1, %v1
  %stacksave = call ptr @llvm.stacksave()
  call void @llvm.stackrestore(ptr %stacksave)
  call void @llvm.sideeffect()
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  sandboxir::Function *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *Add = cast<sandboxir::BinaryOperator>(&*It++);
  auto *StackSave = cast<sandboxir::CallInst>(&*It++);
  auto *StackRestore = cast<sandboxir::CallInst>(&*It++);
  auto *Other = cast<sandboxir::CallInst>(&*It++);
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);

  using DGNode = sandboxir::DGNode;
  EXPECT_FALSE(DGNode::isStackSaveOrRestoreIntrinsic(Add));
  EXPECT_TRUE(DGNode::isStackSaveOrRestoreIntrinsic(StackSave));
  EXPECT_TRUE(DGNode::isStackSaveOrRestoreIntrinsic(StackRestore));
  EXPECT_FALSE(DGNode::isStackSaveOrRestoreIntrinsic(Other));
  EXPECT_FALSE(DGNode::isStackSaveOrRestoreIntrinsic(Ret));
}

TEST_F(DependencyGraphTest, Instruction_isMemDepCandidate) {
  parseIR(C, R"IR(
declare void @llvm.fake.use(...)
declare void @llvm.sideeffect()
declare void @llvm.pseudoprobe(i64, i64, i32, i64)
declare void @bar()
define void @foo(i8 %v1, ptr %ptr) {
  %add0 = add i8 %v1, %v1
  %ld0 = load i8, ptr %ptr
  store i8 %v1, ptr %ptr
  call void @llvm.sideeffect()
  call void @llvm.pseudoprobe(i64 42, i64 1, i32 0, i64 -1)
  call void @llvm.fake.use(ptr %ptr)
  call void @bar()
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  sandboxir::Function *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *Add0 = cast<sandboxir::BinaryOperator>(&*It++);
  auto *Ld0 = cast<sandboxir::LoadInst>(&*It++);
  auto *St0 = cast<sandboxir::StoreInst>(&*It++);
  auto *SideEffect0 = cast<sandboxir::CallInst>(&*It++);
  auto *PseudoProbe0 = cast<sandboxir::CallInst>(&*It++);
  auto *OtherIntrinsic0 = cast<sandboxir::CallInst>(&*It++);
  auto *CallBar = cast<sandboxir::CallInst>(&*It++);
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);

  using DGNode = sandboxir::DGNode;

  EXPECT_FALSE(DGNode::isMemDepCandidate(Add0));
  EXPECT_TRUE(DGNode::isMemDepCandidate(Ld0));
  EXPECT_TRUE(DGNode::isMemDepCandidate(St0));
  EXPECT_FALSE(DGNode::isMemDepCandidate(SideEffect0));
  EXPECT_FALSE(DGNode::isMemDepCandidate(PseudoProbe0));
  EXPECT_TRUE(DGNode::isMemDepCandidate(OtherIntrinsic0));
  EXPECT_TRUE(DGNode::isMemDepCandidate(CallBar));
  EXPECT_FALSE(DGNode::isMemDepCandidate(Ret));
}

TEST_F(DependencyGraphTest, Instruction_isMemIntrinsic) {
  parseIR(C, R"IR(
declare void @llvm.sideeffect()
declare void @llvm.pseudoprobe(i64)
declare void @llvm.assume(i1)

define void @foo(ptr %ptr, i1 %cond) {
  call void @llvm.sideeffect()
  call void @llvm.pseudoprobe(i64 42)
  call void @llvm.assume(i1 %cond)
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  sandboxir::Function *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *SideEffect = cast<sandboxir::IntrinsicInst>(&*It++);
  auto *PseudoProbe = cast<sandboxir::IntrinsicInst>(&*It++);
  auto *OtherIntrinsic = cast<sandboxir::IntrinsicInst>(&*It++);

  using DGNode = sandboxir::DGNode;
  EXPECT_FALSE(DGNode::isMemIntrinsic(SideEffect));
  EXPECT_FALSE(DGNode::isMemIntrinsic(PseudoProbe));
  EXPECT_TRUE(DGNode::isMemIntrinsic(OtherIntrinsic));
}

TEST_F(DependencyGraphTest, MemDGNode) {
  parseIR(C, R"IR(
declare void @llvm.sideeffect()
declare void @llvm.pseudoprobe(i64, i64, i32, i64)
declare void @llvm.fake.use(...)
declare void @bar()
define void @foo(i8 %v1, ptr %ptr) {
  store i8 %v1, ptr %ptr
  %ld0 = load i8, ptr %ptr
  %add = add i8 %v1, %v1
  %stacksave = call ptr @llvm.stacksave()
  call void @llvm.stackrestore(ptr %stacksave)
  call void @llvm.sideeffect()
  call void @llvm.pseudoprobe(i64 42, i64 1, i32 0, i64 -1)
  call void @llvm.fake.use(ptr %ptr)
  call void @bar()
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *Store = cast<sandboxir::StoreInst>(&*It++);
  auto *Load = cast<sandboxir::LoadInst>(&*It++);
  auto *Add = cast<sandboxir::BinaryOperator>(&*It++);
  auto *StackSave = cast<sandboxir::CallInst>(&*It++);
  auto *StackRestore = cast<sandboxir::CallInst>(&*It++);
  auto *SideEffect = cast<sandboxir::CallInst>(&*It++);
  auto *PseudoProbe = cast<sandboxir::CallInst>(&*It++);
  auto *FakeUse = cast<sandboxir::CallInst>(&*It++);
  auto *Call = cast<sandboxir::CallInst>(&*It++);
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);

  sandboxir::DependencyGraph DAG(getAA(*LLVMF), Ctx);
  DAG.extend({&*BB->begin(), BB->getTerminator()});
  EXPECT_TRUE(isa<llvm::sandboxir::MemDGNode>(DAG.getNode(Store)));
  EXPECT_TRUE(isa<llvm::sandboxir::MemDGNode>(DAG.getNode(Load)));
  EXPECT_FALSE(isa<llvm::sandboxir::MemDGNode>(DAG.getNode(Add)));
  EXPECT_TRUE(isa<llvm::sandboxir::MemDGNode>(DAG.getNode(StackSave)));
  EXPECT_TRUE(isa<llvm::sandboxir::MemDGNode>(DAG.getNode(StackRestore)));
  EXPECT_FALSE(isa<llvm::sandboxir::MemDGNode>(DAG.getNode(SideEffect)));
  EXPECT_FALSE(isa<llvm::sandboxir::MemDGNode>(DAG.getNode(PseudoProbe)));
  EXPECT_TRUE(isa<llvm::sandboxir::MemDGNode>(DAG.getNode(FakeUse)));
  EXPECT_TRUE(isa<llvm::sandboxir::MemDGNode>(DAG.getNode(Call)));
  EXPECT_FALSE(isa<llvm::sandboxir::MemDGNode>(DAG.getNode(Ret)));
}

TEST_F(DependencyGraphTest, Basic) {
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
  sandboxir::DependencyGraph DAG(getAA(*LLVMF), Ctx);
  auto Span = DAG.extend({&*BB->begin(), BB->getTerminator()});
  // Check extend().
  EXPECT_EQ(Span.top(), &*BB->begin());
  EXPECT_EQ(Span.bottom(), BB->getTerminator());

  auto *N0 = cast<sandboxir::MemDGNode>(DAG.getNode(S0));
  auto *N1 = cast<sandboxir::MemDGNode>(DAG.getNode(S1));
  auto *N2 = DAG.getNode(Ret);

  // Check getInstruction().
  EXPECT_EQ(N0->getInstruction(), S0);
  EXPECT_EQ(N1->getInstruction(), S1);
  // Check hasMemPred()
  EXPECT_TRUE(N1->hasMemPred(N0));
  EXPECT_FALSE(N0->hasMemPred(N1));

  // Check preds().
  EXPECT_TRUE(N0->preds(DAG).empty());
  EXPECT_THAT(N1->preds(DAG), testing::ElementsAre(N0));

  // Check memPreds().
  EXPECT_TRUE(N0->memPreds().empty());
  EXPECT_THAT(N1->memPreds(), testing::ElementsAre(N0));
  EXPECT_TRUE(N2->preds(DAG).empty());

  // Check UnscheduledSuccs.
  EXPECT_EQ(N0->getNumUnscheduledSuccs(), 1u); // N1
  EXPECT_EQ(N1->getNumUnscheduledSuccs(), 0u);
  EXPECT_EQ(N2->getNumUnscheduledSuccs(), 0u);

  // Check decrUnscheduledSuccs.
  N0->decrUnscheduledSuccs();
  EXPECT_EQ(N0->getNumUnscheduledSuccs(), 0u);
#ifndef NDEBUG
  EXPECT_DEATH(N0->decrUnscheduledSuccs(), ".*Counting.*");
#endif // NDEBUG

  // Check scheduled(), setScheduled().
  EXPECT_FALSE(N0->scheduled());
  N0->setScheduled(true);
  EXPECT_TRUE(N0->scheduled());
}

TEST_F(DependencyGraphTest, Preds) {
  parseIR(C, R"IR(
declare ptr @bar(i8)
define i8 @foo(i8 %v0, i8 %v1) {
  %add0 = add i8 %v0, %v0
  %add1 = add i8 %v1, %v1
  %add2 = add i8 %add0, %add1
  %ptr = call ptr @bar(i8 %add1)
  store i8 %add2, ptr %ptr
  ret i8 %add2
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  sandboxir::DependencyGraph DAG(getAA(*LLVMF), Ctx);
  DAG.extend({&*BB->begin(), BB->getTerminator()});

  auto *AddN0 = DAG.getNode(cast<sandboxir::BinaryOperator>(&*It++));
  auto *AddN1 = DAG.getNode(cast<sandboxir::BinaryOperator>(&*It++));
  auto *AddN2 = DAG.getNode(cast<sandboxir::BinaryOperator>(&*It++));
  auto *CallN = DAG.getNode(cast<sandboxir::CallInst>(&*It++));
  auto *StN = DAG.getNode(cast<sandboxir::StoreInst>(&*It++));
  auto *RetN = DAG.getNode(cast<sandboxir::ReturnInst>(&*It++));

  // Check preds().
  EXPECT_THAT(AddN0->preds(DAG), testing::ElementsAre());
  EXPECT_THAT(AddN1->preds(DAG), testing::ElementsAre());
  EXPECT_THAT(AddN2->preds(DAG), testing::ElementsAre(AddN0, AddN1));
  EXPECT_THAT(CallN->preds(DAG), testing::ElementsAre(AddN1));
  EXPECT_THAT(StN->preds(DAG),
              testing::UnorderedElementsAre(CallN, CallN, AddN2));
  EXPECT_THAT(RetN->preds(DAG), testing::ElementsAre(AddN2));

  // Check UnscheduledSuccs.
  EXPECT_EQ(AddN0->getNumUnscheduledSuccs(), 1u); // AddN2
  EXPECT_EQ(AddN1->getNumUnscheduledSuccs(), 2u); // AddN2, CallN
  EXPECT_EQ(AddN2->getNumUnscheduledSuccs(), 2u); // StN, RetN
  EXPECT_EQ(CallN->getNumUnscheduledSuccs(), 2u); // StN, StN
  EXPECT_EQ(StN->getNumUnscheduledSuccs(), 0u);
  EXPECT_EQ(RetN->getNumUnscheduledSuccs(), 0u);
}

// Make sure we don't get null predecessors even if they are outside the DAG.
TEST_F(DependencyGraphTest, NonNullPreds) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %val) {
  %gep = getelementptr i8, ptr %ptr, i32 0
  store i8 %val, ptr %gep
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  [[maybe_unused]] auto *GEP = cast<sandboxir::GetElementPtrInst>(&*It++);
  auto *S0 = cast<sandboxir::StoreInst>(&*It++);
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);

  sandboxir::DependencyGraph DAG(getAA(*LLVMF), Ctx);
  // The DAG doesn't include GEP.
  DAG.extend({S0, Ret});

  auto *S0N = DAG.getNode(S0);
  // S0 has one operand (the GEP) that is outside the DAG and no memory
  // predecessors. So pred_begin() should be == pred_end().
  auto PredIt = S0N->preds_begin(DAG);
  auto PredItE = S0N->preds_end(DAG);
  EXPECT_EQ(PredIt, PredItE);
  // Check preds().
  for (auto *PredN : S0N->preds(DAG))
    EXPECT_NE(PredN, nullptr);
}

TEST_F(DependencyGraphTest, MemDGNode_getPrevNode_getNextNode) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %v0, i8 %v1) {
  store i8 %v0, ptr %ptr
  add i8 %v0, %v0
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
  [[maybe_unused]] auto *Add = cast<sandboxir::BinaryOperator>(&*It++);
  auto *S1 = cast<sandboxir::StoreInst>(&*It++);
  [[maybe_unused]] auto *Ret = cast<sandboxir::ReturnInst>(&*It++);

  sandboxir::DependencyGraph DAG(getAA(*LLVMF), Ctx);
  DAG.extend({&*BB->begin(), BB->getTerminator()});

  auto *S0N = cast<sandboxir::MemDGNode>(DAG.getNode(S0));
  auto *S1N = cast<sandboxir::MemDGNode>(DAG.getNode(S1));

  EXPECT_EQ(S0N->getPrevNode(), nullptr);
  EXPECT_EQ(S0N->getNextNode(), S1N);

  EXPECT_EQ(S1N->getPrevNode(), S0N);
  EXPECT_EQ(S1N->getNextNode(), nullptr);
}

TEST_F(DependencyGraphTest, DGNodeRange) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %v0, i8 %v1) {
  add i8 %v0, %v0
  store i8 %v0, ptr %ptr
  add i8 %v0, %v0
  store i8 %v1, ptr %ptr
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *Add0 = cast<sandboxir::BinaryOperator>(&*It++);
  auto *S0 = cast<sandboxir::StoreInst>(&*It++);
  auto *Add1 = cast<sandboxir::BinaryOperator>(&*It++);
  auto *S1 = cast<sandboxir::StoreInst>(&*It++);
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);

  sandboxir::DependencyGraph DAG(getAA(*LLVMF), Ctx);
  DAG.extend({&*BB->begin(), BB->getTerminator()});

  auto *S0N = cast<sandboxir::MemDGNode>(DAG.getNode(S0));
  auto *S1N = cast<sandboxir::MemDGNode>(DAG.getNode(S1));

  // Check getTopMemDGNode().
  using B = sandboxir::MemDGNodeIntervalBuilder;
  using InstrInterval = sandboxir::Interval<sandboxir::Instruction>;
  EXPECT_EQ(B::getTopMemDGNode(InstrInterval(S0, S0), DAG), S0N);
  EXPECT_EQ(B::getTopMemDGNode(InstrInterval(S0, Ret), DAG), S0N);
  EXPECT_EQ(B::getTopMemDGNode(InstrInterval(Add0, Add1), DAG), S0N);
  EXPECT_EQ(B::getTopMemDGNode(InstrInterval(Add0, Add0), DAG), nullptr);

  // Check getBotMemDGNode().
  EXPECT_EQ(B::getBotMemDGNode(InstrInterval(S1, S1), DAG), S1N);
  EXPECT_EQ(B::getBotMemDGNode(InstrInterval(Add0, S1), DAG), S1N);
  EXPECT_EQ(B::getBotMemDGNode(InstrInterval(Add0, Ret), DAG), S1N);
  EXPECT_EQ(B::getBotMemDGNode(InstrInterval(Ret, Ret), DAG), nullptr);

  // Check empty range.
  EXPECT_THAT(sandboxir::MemDGNodeIntervalBuilder::makeEmpty(),
              testing::ElementsAre());

  // Returns the pointers in Range.
  auto getPtrVec = [](const auto &Range) {
    SmallVector<const sandboxir::DGNode *> Vec;
    for (const sandboxir::DGNode &N : Range)
      Vec.push_back(&N);
    return Vec;
  };
  // Both TopN and BotN are memory.
  EXPECT_THAT(
      getPtrVec(sandboxir::MemDGNodeIntervalBuilder::make({S0, S1}, DAG)),
      testing::ElementsAre(S0N, S1N));
  // Only TopN is memory.
  EXPECT_THAT(
      getPtrVec(sandboxir::MemDGNodeIntervalBuilder::make({S0, Ret}, DAG)),
      testing::ElementsAre(S0N, S1N));
  EXPECT_THAT(
      getPtrVec(sandboxir::MemDGNodeIntervalBuilder::make({S0, Add1}, DAG)),
      testing::ElementsAre(S0N));
  // Only BotN is memory.
  EXPECT_THAT(
      getPtrVec(sandboxir::MemDGNodeIntervalBuilder::make({Add0, S1}, DAG)),
      testing::ElementsAre(S0N, S1N));
  EXPECT_THAT(
      getPtrVec(sandboxir::MemDGNodeIntervalBuilder::make({Add0, S0}, DAG)),
      testing::ElementsAre(S0N));
  // Neither TopN or BotN is memory.
  EXPECT_THAT(
      getPtrVec(sandboxir::MemDGNodeIntervalBuilder::make({Add0, Ret}, DAG)),
      testing::ElementsAre(S0N, S1N));
  EXPECT_THAT(
      getPtrVec(sandboxir::MemDGNodeIntervalBuilder::make({Add0, Add0}, DAG)),
      testing::ElementsAre());
}

TEST_F(DependencyGraphTest, AliasingStores) {
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
  sandboxir::DependencyGraph DAG(getAA(*LLVMF), Ctx);
  DAG.extend({&*BB->begin(), BB->getTerminator()});
  auto It = BB->begin();
  auto *Store0N = cast<sandboxir::MemDGNode>(
      DAG.getNode(cast<sandboxir::StoreInst>(&*It++)));
  auto *Store1N = cast<sandboxir::MemDGNode>(
      DAG.getNode(cast<sandboxir::StoreInst>(&*It++)));
  auto *RetN = DAG.getNode(cast<sandboxir::ReturnInst>(&*It++));
  EXPECT_TRUE(Store0N->memPreds().empty());
  EXPECT_THAT(Store1N->memPreds(), testing::ElementsAre(Store0N));
  EXPECT_TRUE(RetN->preds(DAG).empty());
}

TEST_F(DependencyGraphTest, NonAliasingStores) {
  parseIR(C, R"IR(
define void @foo(ptr noalias %ptr0, ptr noalias %ptr1, i8 %v0, i8 %v1) {
  store i8 %v0, ptr %ptr0
  store i8 %v1, ptr %ptr1
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  sandboxir::DependencyGraph DAG(getAA(*LLVMF), Ctx);
  DAG.extend({&*BB->begin(), BB->getTerminator()});
  auto It = BB->begin();
  auto *Store0N = cast<sandboxir::MemDGNode>(
      DAG.getNode(cast<sandboxir::StoreInst>(&*It++)));
  auto *Store1N = cast<sandboxir::MemDGNode>(
      DAG.getNode(cast<sandboxir::StoreInst>(&*It++)));
  auto *RetN = DAG.getNode(cast<sandboxir::ReturnInst>(&*It++));
  // We expect no dependencies because the stores don't alias.
  EXPECT_TRUE(Store0N->memPreds().empty());
  EXPECT_TRUE(Store1N->memPreds().empty());
  EXPECT_TRUE(RetN->preds(DAG).empty());
}

TEST_F(DependencyGraphTest, VolatileLoads) {
  parseIR(C, R"IR(
define void @foo(ptr noalias %ptr0, ptr noalias %ptr1) {
  %ld0 = load volatile i8, ptr %ptr0
  %ld1 = load volatile i8, ptr %ptr1
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  sandboxir::DependencyGraph DAG(getAA(*LLVMF), Ctx);
  DAG.extend({&*BB->begin(), BB->getTerminator()});
  auto It = BB->begin();
  auto *Ld0N = cast<sandboxir::MemDGNode>(
      DAG.getNode(cast<sandboxir::LoadInst>(&*It++)));
  auto *Ld1N = cast<sandboxir::MemDGNode>(
      DAG.getNode(cast<sandboxir::LoadInst>(&*It++)));
  auto *RetN = DAG.getNode(cast<sandboxir::ReturnInst>(&*It++));
  EXPECT_TRUE(Ld0N->memPreds().empty());
  EXPECT_THAT(Ld1N->memPreds(), testing::ElementsAre(Ld0N));
  EXPECT_TRUE(RetN->preds(DAG).empty());
}

TEST_F(DependencyGraphTest, VolatileSotres) {
  parseIR(C, R"IR(
define void @foo(ptr noalias %ptr0, ptr noalias %ptr1, i8 %v) {
  store volatile i8 %v, ptr %ptr0
  store volatile i8 %v, ptr %ptr1
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  sandboxir::DependencyGraph DAG(getAA(*LLVMF), Ctx);
  DAG.extend({&*BB->begin(), BB->getTerminator()});
  auto It = BB->begin();
  auto *Store0N = cast<sandboxir::MemDGNode>(
      DAG.getNode(cast<sandboxir::StoreInst>(&*It++)));
  auto *Store1N = cast<sandboxir::MemDGNode>(
      DAG.getNode(cast<sandboxir::StoreInst>(&*It++)));
  auto *RetN = DAG.getNode(cast<sandboxir::ReturnInst>(&*It++));
  EXPECT_TRUE(Store0N->memPreds().empty());
  EXPECT_THAT(Store1N->memPreds(), testing::ElementsAre(Store0N));
  EXPECT_TRUE(RetN->preds(DAG).empty());
}

TEST_F(DependencyGraphTest, Call) {
  parseIR(C, R"IR(
declare void @bar1()
declare void @bar2()
define void @foo(float %v1, float %v2) {
  call void @bar1()
  %add = fadd float %v1, %v2
  call void @bar2()
  ret void
}
)IR");
  Function *LLVMF = M->getFunction("foo");

  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();

  sandboxir::DependencyGraph DAG(getAA(*LLVMF), Ctx);
  DAG.extend({&*BB->begin(), BB->getTerminator()->getPrevNode()});

  auto It = BB->begin();
  auto *Call1N = cast<sandboxir::MemDGNode>(DAG.getNode(&*It++));
  auto *AddN = DAG.getNode(&*It++);
  auto *Call2N = cast<sandboxir::MemDGNode>(DAG.getNode(&*It++));

  EXPECT_THAT(Call1N->memPreds(), testing::ElementsAre());
  EXPECT_THAT(AddN->preds(DAG), testing::ElementsAre());
  EXPECT_THAT(Call2N->memPreds(), testing::ElementsAre(Call1N));
}

// Check that there is a dependency: stacksave -> alloca -> stackrestore.
TEST_F(DependencyGraphTest, StackSaveRestoreInAlloca) {
  parseIR(C, R"IR(
declare ptr @llvm.stacksave()
declare void @llvm.stackrestore(ptr %ptr)

define void @foo() {
  %stack0 = call ptr @llvm.stacksave()        ; Should depend on store
  %alloca0 = alloca inalloca i8               ; Should depend on stacksave
  call void @llvm.stackrestore(ptr %stack0)   ; Should depend transiently on %alloca0
  ret void
}
)IR");
  Function *LLVMF = M->getFunction("foo");

  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();

  sandboxir::DependencyGraph DAG(getAA(*LLVMF), Ctx);
  DAG.extend({&*BB->begin(), BB->getTerminator()->getPrevNode()});

  auto It = BB->begin();
  auto *StackSaveN = DAG.getNode(&*It++);
  auto *AllocaN = DAG.getNode(&*It++);
  auto *StackRestoreN = DAG.getNode(&*It++);

  EXPECT_TRUE(memDependency(AllocaN, StackRestoreN));
  EXPECT_TRUE(memDependency(StackSaveN, AllocaN));
}

// Checks that stacksave and stackrestore depend on other mem instrs.
TEST_F(DependencyGraphTest, StackSaveRestoreDependOnOtherMem) {
  parseIR(C, R"IR(
declare ptr @llvm.stacksave()
declare void @llvm.stackrestore(ptr %ptr)

define void @foo(i8 %v0, i8 %v1, ptr %ptr) {
  store volatile i8 %v0, ptr %ptr, align 4
  %stack0 = call ptr @llvm.stacksave()       ; Should depend on store
  call void @llvm.stackrestore(ptr %stack0)  ; Should depend on stacksave
  store volatile i8 %v1, ptr %ptr, align 4   ; Should depend on stackrestore
  ret void
}
)IR");
  Function *LLVMF = M->getFunction("foo");

  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();

  sandboxir::DependencyGraph DAG(getAA(*LLVMF), Ctx);
  DAG.extend({&*BB->begin(), BB->getTerminator()->getPrevNode()});

  auto It = BB->begin();
  auto *Store0N = DAG.getNode(&*It++);
  auto *StackSaveN = DAG.getNode(&*It++);
  auto *StackRestoreN = DAG.getNode(&*It++);
  auto *Store1N = DAG.getNode(&*It++);

  EXPECT_TRUE(memDependency(Store0N, StackSaveN));
  EXPECT_TRUE(memDependency(StackSaveN, StackRestoreN));
  EXPECT_TRUE(memDependency(StackRestoreN, Store1N));
}

// Make sure there is a dependency between a stackrestore and an alloca.
TEST_F(DependencyGraphTest, StackRestoreAndInAlloca) {
  parseIR(C, R"IR(
declare void @llvm.stackrestore(ptr %ptr)

define void @foo(ptr %ptr) {
  call void @llvm.stackrestore(ptr %ptr)
  %alloca0 = alloca inalloca i8              ; Should depend on stackrestore
  ret void
}
)IR");
  Function *LLVMF = M->getFunction("foo");

  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();

  sandboxir::DependencyGraph DAG(getAA(*LLVMF), Ctx);
  DAG.extend({&*BB->begin(), BB->getTerminator()->getPrevNode()});

  auto It = BB->begin();
  auto *StackRestoreN = DAG.getNode(&*It++);
  auto *AllocaN = DAG.getNode(&*It++);

  EXPECT_TRUE(memDependency(StackRestoreN, AllocaN));
}

// Make sure there is a dependency between the alloca and stacksave
TEST_F(DependencyGraphTest, StackSaveAndInAlloca) {
  parseIR(C, R"IR(
declare ptr @llvm.stacksave()

define void @foo(ptr %ptr) {
  %alloca0 = alloca inalloca i8              ; Should depend on stackrestore
  %stack0 = call ptr @llvm.stacksave()       ; Should depend on alloca0
  ret void
}
)IR");
  Function *LLVMF = M->getFunction("foo");

  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();

  sandboxir::DependencyGraph DAG(getAA(*LLVMF), Ctx);
  DAG.extend({&*BB->begin(), BB->getTerminator()->getPrevNode()});

  auto It = BB->begin();
  auto *AllocaN = DAG.getNode(&*It++);
  auto *StackSaveN = DAG.getNode(&*It++);

  EXPECT_TRUE(memDependency(AllocaN, StackSaveN));
}

// A non-InAlloca in a stacksave-stackrestore region does not need extra
// dependencies.
TEST_F(DependencyGraphTest, StackSaveRestoreNoInAlloca) {
  parseIR(C, R"IR(
declare ptr @llvm.stacksave()
declare void @llvm.stackrestore(ptr %ptr)
declare void @use(ptr %ptr)

define void @foo() {
  %stack = call ptr @llvm.stacksave()
  %alloca1 = alloca i8                         ; No dependency
  call void @llvm.stackrestore(ptr %stack)
  ret void
}
)IR");
  Function *LLVMF = M->getFunction("foo");

  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();

  sandboxir::DependencyGraph DAG(getAA(*LLVMF), Ctx);
  DAG.extend({&*BB->begin(), BB->getTerminator()->getPrevNode()});

  auto It = BB->begin();
  auto *StackSaveN = DAG.getNode(&*It++);
  auto *AllocaN = DAG.getNode(&*It++);
  auto *StackRestoreN = DAG.getNode(&*It++);

  EXPECT_FALSE(memDependency(StackSaveN, AllocaN));
  EXPECT_FALSE(memDependency(AllocaN, StackRestoreN));
}

TEST_F(DependencyGraphTest, Extend) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %v1, i8 %v2, i8 %v3, i8 %v4, i8 %v5) {
  store i8 %v1, ptr %ptr
  store i8 %v2, ptr %ptr
  store i8 %v3, ptr %ptr
  store i8 %v4, ptr %ptr
  store i8 %v5, ptr %ptr
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *S1 = cast<sandboxir::StoreInst>(&*It++);
  auto *S2 = cast<sandboxir::StoreInst>(&*It++);
  auto *S3 = cast<sandboxir::StoreInst>(&*It++);
  auto *S4 = cast<sandboxir::StoreInst>(&*It++);
  auto *S5 = cast<sandboxir::StoreInst>(&*It++);
  sandboxir::DependencyGraph DAG(getAA(*LLVMF), Ctx);
  {
    // Scenario 1: Build new DAG
    auto NewIntvl = DAG.extend({S3, S3});
    EXPECT_EQ(NewIntvl, sandboxir::Interval<sandboxir::Instruction>(S3, S3));
    EXPECT_EQ(DAG.getInterval().top(), S3);
    EXPECT_EQ(DAG.getInterval().bottom(), S3);
    [[maybe_unused]] auto *S3N = cast<sandboxir::MemDGNode>(DAG.getNode(S3));
    // Check UnscheduledSuccs.
    EXPECT_EQ(S3N->getNumUnscheduledSuccs(), 0u);
  }
  {
    // Scenario 2: Extend below
    auto NewIntvl = DAG.extend({S5, S5});
    EXPECT_EQ(NewIntvl, sandboxir::Interval<sandboxir::Instruction>(S4, S5));
    auto *S3N = cast<sandboxir::MemDGNode>(DAG.getNode(S3));
    auto *S4N = cast<sandboxir::MemDGNode>(DAG.getNode(S4));
    auto *S5N = cast<sandboxir::MemDGNode>(DAG.getNode(S5));
    EXPECT_TRUE(S4N->hasMemPred(S3N));
    EXPECT_TRUE(S5N->hasMemPred(S4N));
    EXPECT_TRUE(S5N->hasMemPred(S3N));
    // Check UnscheduledSuccs.
    EXPECT_EQ(S3N->getNumUnscheduledSuccs(), 2u); // S4N, S5N
    EXPECT_EQ(S4N->getNumUnscheduledSuccs(), 1u); // S5N
    EXPECT_EQ(S5N->getNumUnscheduledSuccs(), 0u);
  }
  {
    // Scenario 3: Extend above
    auto NewIntvl = DAG.extend({S1, S2});
    EXPECT_EQ(NewIntvl, sandboxir::Interval<sandboxir::Instruction>(S1, S2));
    auto *S1N = cast<sandboxir::MemDGNode>(DAG.getNode(S1));
    auto *S2N = cast<sandboxir::MemDGNode>(DAG.getNode(S2));
    auto *S3N = cast<sandboxir::MemDGNode>(DAG.getNode(S3));
    auto *S4N = cast<sandboxir::MemDGNode>(DAG.getNode(S4));
    auto *S5N = cast<sandboxir::MemDGNode>(DAG.getNode(S5));

    EXPECT_TRUE(S2N->hasMemPred(S1N));

    EXPECT_TRUE(S3N->hasMemPred(S2N));
    EXPECT_TRUE(S3N->hasMemPred(S1N));

    EXPECT_TRUE(S4N->hasMemPred(S3N));
    EXPECT_TRUE(S4N->hasMemPred(S2N));
    EXPECT_TRUE(S4N->hasMemPred(S1N));

    EXPECT_TRUE(S5N->hasMemPred(S4N));
    EXPECT_TRUE(S5N->hasMemPred(S3N));
    EXPECT_TRUE(S5N->hasMemPred(S2N));
    EXPECT_TRUE(S5N->hasMemPred(S1N));

    // Check UnscheduledSuccs.
    EXPECT_EQ(S1N->getNumUnscheduledSuccs(), 4u); // S2N, S3N, S4N, S5N
    EXPECT_EQ(S2N->getNumUnscheduledSuccs(), 3u); // S3N, S4N, S5N
    EXPECT_EQ(S3N->getNumUnscheduledSuccs(), 2u); // S4N, S5N
    EXPECT_EQ(S4N->getNumUnscheduledSuccs(), 1u); // S5N
    EXPECT_EQ(S5N->getNumUnscheduledSuccs(), 0u);
  }

  {
    // Check UnscheduledSuccs when a node is scheduled
    sandboxir::DependencyGraph DAG(getAA(*LLVMF), Ctx);
    DAG.extend({S2, S2});
    auto *S2N = cast<sandboxir::MemDGNode>(DAG.getNode(S2));
    S2N->setScheduled(true);

    DAG.extend({S1, S1});
    auto *S1N = cast<sandboxir::MemDGNode>(DAG.getNode(S1));
    EXPECT_EQ(S1N->getNumUnscheduledSuccs(), 0u); // S1 is scheduled
  }
}

TEST_F(DependencyGraphTest, CreateInstrCallback) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, ptr noalias %ptr2, i8 %v1, i8 %v2, i8 %v3, i8 %arg) {
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
  auto *S1 = cast<sandboxir::StoreInst>(&*It++);
  auto *S2 = cast<sandboxir::StoreInst>(&*It++);
  auto *S3 = cast<sandboxir::StoreInst>(&*It++);
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);

  // Check new instruction callback.
  sandboxir::DependencyGraph DAG(getAA(*LLVMF), Ctx);
  DAG.extend({S1, Ret});
  auto *Arg = F->getArg(3);
  auto *Ptr = S1->getPointerOperand();
  {
    sandboxir::StoreInst *NewS =
        sandboxir::StoreInst::create(Arg, Ptr, Align(8), S3->getIterator(),
                                     /*IsVolatile=*/true, Ctx);
    auto *NewSN = DAG.getNode(NewS);
    EXPECT_TRUE(NewSN != nullptr);

    // Check the MemDGNode chain.
    auto *S2MemN = cast<sandboxir::MemDGNode>(DAG.getNode(S2));
    auto *NewMemSN = cast<sandboxir::MemDGNode>(NewSN);
    auto *S3MemN = cast<sandboxir::MemDGNode>(DAG.getNode(S3));
    EXPECT_EQ(S2MemN->getNextNode(), NewMemSN);
    EXPECT_EQ(NewMemSN->getPrevNode(), S2MemN);
    EXPECT_EQ(NewMemSN->getNextNode(), S3MemN);
    EXPECT_EQ(S3MemN->getPrevNode(), NewMemSN);
  }

  {
    // Also check if new node is at the end of the BB, after Ret.
    sandboxir::StoreInst *NewS =
        sandboxir::StoreInst::create(Arg, Ptr, Align(8), BB->end(),
                                     /*IsVolatile=*/true, Ctx);
    // Check the MemDGNode chain.
    auto *S3MemN = cast<sandboxir::MemDGNode>(DAG.getNode(S3));
    auto *NewMemSN = cast<sandboxir::MemDGNode>(DAG.getNode(NewS));
    EXPECT_EQ(S3MemN->getNextNode(), NewMemSN);
    EXPECT_EQ(NewMemSN->getPrevNode(), S3MemN);
    EXPECT_EQ(NewMemSN->getNextNode(), nullptr);
  }

  // TODO: Check the dependencies to/from NewSN after they land.
}

TEST_F(DependencyGraphTest, EraseInstrCallback) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %v1, i8 %v2, i8 %v3, i8 %arg) {
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
  auto *S1 = cast<sandboxir::StoreInst>(&*It++);
  auto *S2 = cast<sandboxir::StoreInst>(&*It++);
  auto *S3 = cast<sandboxir::StoreInst>(&*It++);

  // Check erase instruction callback.
  sandboxir::DependencyGraph DAG(getAA(*LLVMF), Ctx);
  DAG.extend({S1, S3});
  S2->eraseFromParent();
  auto *DeletedN = DAG.getNodeOrNull(S2);
  EXPECT_TRUE(DeletedN == nullptr);

  // Check the MemDGNode chain.
  auto *S1MemN = cast<sandboxir::MemDGNode>(DAG.getNode(S1));
  auto *S3MemN = cast<sandboxir::MemDGNode>(DAG.getNode(S3));
  EXPECT_EQ(S1MemN->getNextNode(), S3MemN);
  EXPECT_EQ(S3MemN->getPrevNode(), S1MemN);

  // Check the chain when we erase the top node.
  S1->eraseFromParent();
  EXPECT_EQ(S3MemN->getPrevNode(), nullptr);

  // TODO: Check the dependencies to/from NewSN after they land.
}

TEST_F(DependencyGraphTest, MoveInstrCallback) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, ptr %ptr2, i8 %v1, i8 %v2, i8 %v3, i8 %arg) {
  %ld0 = load i8, ptr %ptr2
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
  auto *Ld = cast<sandboxir::LoadInst>(&*It++);
  auto *S1 = cast<sandboxir::StoreInst>(&*It++);
  auto *S2 = cast<sandboxir::StoreInst>(&*It++);
  auto *S3 = cast<sandboxir::StoreInst>(&*It++);

  sandboxir::DependencyGraph DAG(getAA(*LLVMF), Ctx);
  DAG.extend({Ld, S3});
  auto *LdN = cast<sandboxir::MemDGNode>(DAG.getNode(Ld));
  auto *S1N = cast<sandboxir::MemDGNode>(DAG.getNode(S1));
  auto *S2N = cast<sandboxir::MemDGNode>(DAG.getNode(S2));
  EXPECT_EQ(S1N->getPrevNode(), LdN);
  S1->moveBefore(Ld);
  EXPECT_EQ(S1N->getPrevNode(), nullptr);
  EXPECT_EQ(S1N->getNextNode(), LdN);
  EXPECT_EQ(LdN->getPrevNode(), S1N);
  EXPECT_EQ(LdN->getNextNode(), S2N);
}

// Check that the mem chain is maintained correctly when the move destination is
// not a mem node.
TEST_F(DependencyGraphTest, MoveInstrCallbackWithNonMemInstrs) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %v1, i8 %v2, i8 %arg) {
  %ld = load i8, ptr %ptr
  %zext1 = zext i8 %arg to i32
  %zext2 = zext i8 %arg to i32
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
  auto *Ld = cast<sandboxir::LoadInst>(&*It++);
  [[maybe_unused]] auto *Zext1 = cast<sandboxir::CastInst>(&*It++);
  auto *Zext2 = cast<sandboxir::CastInst>(&*It++);
  auto *S1 = cast<sandboxir::StoreInst>(&*It++);
  auto *S2 = cast<sandboxir::StoreInst>(&*It++);
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);

  sandboxir::DependencyGraph DAG(getAA(*LLVMF), Ctx);
  DAG.extend({Ld, S2});
  auto *LdN = cast<sandboxir::MemDGNode>(DAG.getNode(Ld));
  auto *S1N = cast<sandboxir::MemDGNode>(DAG.getNode(S1));
  auto *S2N = cast<sandboxir::MemDGNode>(DAG.getNode(S2));
  EXPECT_EQ(LdN->getNextNode(), S1N);
  EXPECT_EQ(S1N->getNextNode(), S2N);

  S1->moveBefore(Zext2);
  EXPECT_EQ(LdN->getNextNode(), S1N);
  EXPECT_EQ(S1N->getNextNode(), S2N);

  // Try move right after the end of the DAGInterval.
  S1->moveBefore(Ret);
  EXPECT_EQ(S2N->getNextNode(), S1N);
  EXPECT_EQ(S1N->getNextNode(), nullptr);
}
