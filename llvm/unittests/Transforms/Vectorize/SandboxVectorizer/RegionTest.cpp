//===- RegionTest.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Region.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/SandboxIR/SandboxIR.h"
#include "llvm/Support/SourceMgr.h"
#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"

using namespace llvm;

struct RegionTest : public testing::Test {
  LLVMContext C;
  std::unique_ptr<Module> M;

  void parseIR(LLVMContext &C, const char *IR) {
    SMDiagnostic Err;
    M = parseAssemblyString(IR, Err, C);
    if (!M)
      Err.print("RegionTest", errs());
  }
};

TEST_F(RegionTest, Basic) {
  parseIR(C, R"IR(
define i8 @foo(i8 %v0, i8 %v1) {
  %t0 = add i8 %v0, 1
  %t1 = add i8 %t0, %v1
  ret i8 %t1
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *T0 = cast<sandboxir::Instruction>(&*It++);
  auto *T1 = cast<sandboxir::Instruction>(&*It++);
  auto *Ret = cast<sandboxir::Instruction>(&*It++);
  sandboxir::Region Rgn(Ctx);

  // Check getters
  EXPECT_EQ(&Ctx, &Rgn.getContext());
  EXPECT_EQ(0U, Rgn.getID());

  // Check add / remove / empty.
  EXPECT_TRUE(Rgn.empty());
  Rgn.add(T0);
  EXPECT_FALSE(Rgn.empty());
  Rgn.remove(T0);
  EXPECT_TRUE(Rgn.empty());

  // Check iteration.
  Rgn.add(T0);
  Rgn.add(T1);
  Rgn.add(Ret);
  // Use an ordered matcher because we're supposed to preserve the insertion
  // order for determinism.
  EXPECT_THAT(Rgn.insts(), testing::ElementsAre(T0, T1, Ret));

  // Check contains
  EXPECT_TRUE(Rgn.contains(T0));
  Rgn.remove(T0);
  EXPECT_FALSE(Rgn.contains(T0));

#ifndef NDEBUG
  // Check equality comparison. Insert in reverse order into `Other` to check
  // that comparison is order-independent.
  sandboxir::Region Other(Ctx);
  Other.add(Ret);
  EXPECT_NE(Rgn, Other);
  Other.add(T1);
  EXPECT_EQ(Rgn, Other);
#endif
}
