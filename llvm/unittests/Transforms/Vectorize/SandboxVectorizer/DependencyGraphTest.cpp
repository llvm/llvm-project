//===- DependencyGraphTest.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/DependencyGraph.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/SandboxIR/SandboxIR.h"
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
