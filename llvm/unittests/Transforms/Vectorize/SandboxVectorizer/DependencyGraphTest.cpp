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
  bool dependency(sandboxir::DGNode *SrcN, sandboxir::DGNode *DstN) {
    const auto &Preds = DstN->memPreds();
    auto It = find(Preds, SrcN);
    return It != Preds.end();
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

  sandboxir::DependencyGraph DAG(getAA(*LLVMF));
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
  sandboxir::DependencyGraph DAG(getAA(*LLVMF));
  auto Span = DAG.extend({&*BB->begin(), BB->getTerminator()});
  // Check extend().
  EXPECT_EQ(Span.top(), &*BB->begin());
  EXPECT_EQ(Span.bottom(), BB->getTerminator());

  sandboxir::DGNode *N0 = DAG.getNode(S0);
  sandboxir::DGNode *N1 = DAG.getNode(S1);
  sandboxir::DGNode *N2 = DAG.getNode(Ret);
  // Check getInstruction().
  EXPECT_EQ(N0->getInstruction(), S0);
  EXPECT_EQ(N1->getInstruction(), S1);
  // Check hasMemPred()
  EXPECT_TRUE(N1->hasMemPred(N0));
  EXPECT_FALSE(N0->hasMemPred(N1));

  // Check memPreds().
  EXPECT_TRUE(N0->memPreds().empty());
  EXPECT_THAT(N1->memPreds(), testing::ElementsAre(N0));
  EXPECT_TRUE(N2->memPreds().empty());
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

  sandboxir::DependencyGraph DAG(getAA(*LLVMF));
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

  sandboxir::DependencyGraph DAG(getAA(*LLVMF));
  DAG.extend({&*BB->begin(), BB->getTerminator()});

  auto *S0N = cast<sandboxir::MemDGNode>(DAG.getNode(S0));
  auto *S1N = cast<sandboxir::MemDGNode>(DAG.getNode(S1));

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
  sandboxir::DependencyGraph DAG(getAA(*LLVMF));
  DAG.extend({&*BB->begin(), BB->getTerminator()});
  auto It = BB->begin();
  auto *Store0N = DAG.getNode(cast<sandboxir::StoreInst>(&*It++));
  auto *Store1N = DAG.getNode(cast<sandboxir::StoreInst>(&*It++));
  auto *RetN = DAG.getNode(cast<sandboxir::ReturnInst>(&*It++));
  EXPECT_TRUE(Store0N->memPreds().empty());
  EXPECT_THAT(Store1N->memPreds(), testing::ElementsAre(Store0N));
  EXPECT_TRUE(RetN->memPreds().empty());
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
  sandboxir::DependencyGraph DAG(getAA(*LLVMF));
  DAG.extend({&*BB->begin(), BB->getTerminator()});
  auto It = BB->begin();
  auto *Store0N = DAG.getNode(cast<sandboxir::StoreInst>(&*It++));
  auto *Store1N = DAG.getNode(cast<sandboxir::StoreInst>(&*It++));
  auto *RetN = DAG.getNode(cast<sandboxir::ReturnInst>(&*It++));
  // We expect no dependencies because the stores don't alias.
  EXPECT_TRUE(Store0N->memPreds().empty());
  EXPECT_TRUE(Store1N->memPreds().empty());
  EXPECT_TRUE(RetN->memPreds().empty());
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
  sandboxir::DependencyGraph DAG(getAA(*LLVMF));
  DAG.extend({&*BB->begin(), BB->getTerminator()});
  auto It = BB->begin();
  auto *Ld0N = DAG.getNode(cast<sandboxir::LoadInst>(&*It++));
  auto *Ld1N = DAG.getNode(cast<sandboxir::LoadInst>(&*It++));
  auto *RetN = DAG.getNode(cast<sandboxir::ReturnInst>(&*It++));
  EXPECT_TRUE(Ld0N->memPreds().empty());
  EXPECT_THAT(Ld1N->memPreds(), testing::ElementsAre(Ld0N));
  EXPECT_TRUE(RetN->memPreds().empty());
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
  sandboxir::DependencyGraph DAG(getAA(*LLVMF));
  DAG.extend({&*BB->begin(), BB->getTerminator()});
  auto It = BB->begin();
  auto *Store0N = DAG.getNode(cast<sandboxir::StoreInst>(&*It++));
  auto *Store1N = DAG.getNode(cast<sandboxir::StoreInst>(&*It++));
  auto *RetN = DAG.getNode(cast<sandboxir::ReturnInst>(&*It++));
  EXPECT_TRUE(Store0N->memPreds().empty());
  EXPECT_THAT(Store1N->memPreds(), testing::ElementsAre(Store0N));
  EXPECT_TRUE(RetN->memPreds().empty());
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

  sandboxir::DependencyGraph DAG(getAA(*LLVMF));
  DAG.extend({&*BB->begin(), BB->getTerminator()->getPrevNode()});

  auto It = BB->begin();
  auto *Call1N = DAG.getNode(&*It++);
  auto *AddN = DAG.getNode(&*It++);
  auto *Call2N = DAG.getNode(&*It++);

  EXPECT_THAT(Call1N->memPreds(), testing::ElementsAre());
  EXPECT_THAT(AddN->memPreds(), testing::ElementsAre());
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

  sandboxir::DependencyGraph DAG(getAA(*LLVMF));
  DAG.extend({&*BB->begin(), BB->getTerminator()->getPrevNode()});

  auto It = BB->begin();
  auto *StackSaveN = DAG.getNode(&*It++);
  auto *AllocaN = DAG.getNode(&*It++);
  auto *StackRestoreN = DAG.getNode(&*It++);

  EXPECT_TRUE(dependency(AllocaN, StackRestoreN));
  EXPECT_TRUE(dependency(StackSaveN, AllocaN));
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

  sandboxir::DependencyGraph DAG(getAA(*LLVMF));
  DAG.extend({&*BB->begin(), BB->getTerminator()->getPrevNode()});

  auto It = BB->begin();
  auto *Store0N = DAG.getNode(&*It++);
  auto *StackSaveN = DAG.getNode(&*It++);
  auto *StackRestoreN = DAG.getNode(&*It++);
  auto *Store1N = DAG.getNode(&*It++);

  EXPECT_TRUE(dependency(Store0N, StackSaveN));
  EXPECT_TRUE(dependency(StackSaveN, StackRestoreN));
  EXPECT_TRUE(dependency(StackRestoreN, Store1N));
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

  sandboxir::DependencyGraph DAG(getAA(*LLVMF));
  DAG.extend({&*BB->begin(), BB->getTerminator()->getPrevNode()});

  auto It = BB->begin();
  auto *StackRestoreN = DAG.getNode(&*It++);
  auto *AllocaN = DAG.getNode(&*It++);

  EXPECT_TRUE(dependency(StackRestoreN, AllocaN));
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

  sandboxir::DependencyGraph DAG(getAA(*LLVMF));
  DAG.extend({&*BB->begin(), BB->getTerminator()->getPrevNode()});

  auto It = BB->begin();
  auto *AllocaN = DAG.getNode(&*It++);
  auto *StackSaveN = DAG.getNode(&*It++);

  EXPECT_TRUE(dependency(AllocaN, StackSaveN));
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

  sandboxir::DependencyGraph DAG(getAA(*LLVMF));
  DAG.extend({&*BB->begin(), BB->getTerminator()->getPrevNode()});

  auto It = BB->begin();
  auto *StackSaveN = DAG.getNode(&*It++);
  auto *AllocaN = DAG.getNode(&*It++);
  auto *StackRestoreN = DAG.getNode(&*It++);

  EXPECT_FALSE(dependency(StackSaveN, AllocaN));
  EXPECT_FALSE(dependency(AllocaN, StackRestoreN));
}
