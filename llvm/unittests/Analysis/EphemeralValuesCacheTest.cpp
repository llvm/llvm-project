//===--- EphemeralValuesCache.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/EphemeralValuesCache.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

struct EphemeralValuesCacheTest : public testing::Test {
  LLVMContext Ctx;
  std::unique_ptr<Module> M;

  void parseIR(const char *Assembly) {
    SMDiagnostic Error;
    M = parseAssemblyString(Assembly, Error, Ctx);
    if (!M)
      Error.print("EphemeralValuesCacheTest", errs());
  }
};

TEST_F(EphemeralValuesCacheTest, basic) {
  parseIR(R"IR(
declare void @llvm.assume(i1)

define void @foo(i8 %arg0, i8 %arg1) {
  %c0 = icmp eq i8 %arg0, 0
  call void @llvm.assume(i1 %c0)
  call void @foo(i8 %arg0, i8 %arg1)
  %c1 = icmp eq i8 %arg1, 0
  call void @llvm.assume(i1 %c1)
  ret void
}
)IR");
  Function *F = M->getFunction("foo");
  auto *BB = &*F->begin();
  AssumptionCache AC(*F);
  EphemeralValuesCache EVC(*F, AC);
  auto It = BB->begin();
  auto *C0 = &*It++;
  auto *Assume0 = &*It++;
  [[maybe_unused]] auto *NotEph = &*It++;
  auto *C1 = &*It++;
  auto *Assume1 = &*It++;
  [[maybe_unused]] auto *Ret = &*It++;
  // Check emphemeral values.
  EXPECT_THAT(EVC.ephValues(),
              testing::UnorderedElementsAre(C0, Assume0, C1, Assume1));
  // Clear the cache and try again.
  EVC.clear();
  EXPECT_THAT(EVC.ephValues(),
              testing::UnorderedElementsAre(C0, Assume0, C1, Assume1));
  // Modify the IR, clear cache and recompute.
  Assume1->eraseFromParent();
  C1->eraseFromParent();
  EXPECT_THAT(EVC.ephValues(),
              testing::UnorderedElementsAre(C0, Assume0, C1, Assume1));
  EVC.clear();
  EXPECT_THAT(EVC.ephValues(), testing::UnorderedElementsAre(C0, Assume0));
}

} // namespace
