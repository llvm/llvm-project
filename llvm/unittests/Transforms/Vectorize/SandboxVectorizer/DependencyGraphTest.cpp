//===- DependencyGraphTest.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/DependencyGraph.h"
#include "llvm/AsmParser/Parser.h"
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

  void parseIR(LLVMContext &C, const char *IR) {
    SMDiagnostic Err;
    M = parseAssemblyString(IR, Err, C);
    if (!M)
      Err.print("DependencyGraphTest", errs());
  }
};

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

  sandboxir::DependencyGraph DAG;
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
  sandboxir::DependencyGraph DAG;
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
  EXPECT_THAT(N2->memPreds(), testing::ElementsAre(N1));
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

  sandboxir::DependencyGraph DAG;
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

  sandboxir::DependencyGraph DAG;
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
