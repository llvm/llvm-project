//===- AttributorTest.cpp - Attributor unit tests ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/Attributor.h"
#include "AttributorTestBase.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Transforms/Utils/CallGraphUpdater.h"
#include "gtest/gtest.h"
#include <memory>

namespace llvm {

TEST_F(AttributorTestBase, IRPPositionCallBaseContext) {
  const char *ModuleString = R"(
    define i32 @foo(i32 %a) {
    entry:
      ret i32 %a
    }
  )";

  parseModule(ModuleString);

  Function *F = M->getFunction("foo");
  IRPosition Pos =
      IRPosition::function(*F, (const llvm::CallBase *)(uintptr_t)0xDEADBEEF);
  EXPECT_TRUE(Pos.hasCallBaseContext());
  EXPECT_FALSE(Pos.stripCallBaseContext().hasCallBaseContext());
}

TEST_F(AttributorTestBase, TestCast) {
  const char *ModuleString = R"(
    define i32 @foo(i32 %a, i32 %b) {
    entry:
      %c = add i32 %a, %b
      ret i32 %c
    }
  )";

  Module &M = parseModule(ModuleString);

  SetVector<Function *> Functions;
  AnalysisGetter AG;
  for (Function &F : M)
    Functions.insert(&F);

  CallGraphUpdater CGUpdater;
  BumpPtrAllocator Allocator;
  InformationCache InfoCache(M, AG, Allocator, nullptr);
  Attributor A(Functions, InfoCache, CGUpdater);

  Function *F = M.getFunction("foo");

  const AbstractAttribute *AA =
      &A.getOrCreateAAFor<AAIsDead>(IRPosition::function(*F));

  EXPECT_TRUE(AA);

  const auto *SFail = dyn_cast<AAAlign>(AA);
  const auto *SSucc = dyn_cast<AAIsDead>(AA);

  ASSERT_EQ(SFail, nullptr);
  ASSERT_TRUE(SSucc);
}

TEST_F(AttributorTestBase, AAReachabilityTest) {
  const char *ModuleString = R"(
    @x = global i32 0
    define void @func4() {
      store i32 0, i32* @x
      ret void
    }

    define void @func3() {
      store i32 0, i32* @x
      ret void
    }

    define void @func2() {
    entry:
      call void @func3()
      ret void
    }

    define void @func1() {
    entry:
      call void @func2()
      ret void
    }

    define void @func5(void ()* %unknown) {
    entry:
      call void %unknown()
      ret void
    }

    define void @func6() {
    entry:
      call void @func5(void ()* @func3)
      ret void
    }

    define void @func7() {
    entry:
      call void @func2()
      call void @func4()
      ret void
    }
  )";

  Module &M = parseModule(ModuleString);

  SetVector<Function *> Functions;
  AnalysisGetter AG;
  for (Function &F : M)
    Functions.insert(&F);

  CallGraphUpdater CGUpdater;
  BumpPtrAllocator Allocator;
  InformationCache InfoCache(M, AG, Allocator, nullptr);
  Attributor A(Functions, InfoCache, CGUpdater);

  Function *F1 = M.getFunction("func1");
  Function *F3 = M.getFunction("func3");
  Function *F4 = M.getFunction("func4");
  Function *F6 = M.getFunction("func6");
  Function *F7 = M.getFunction("func7");

  // call void @func2()
  CallBase &F7FirstCB =
      *static_cast<CallBase *>(F7->getEntryBlock().getFirstNonPHI());

  const AAFunctionReachability &F1AA =
      A.getOrCreateAAFor<AAFunctionReachability>(IRPosition::function(*F1));

  const AAFunctionReachability &F6AA =
      A.getOrCreateAAFor<AAFunctionReachability>(IRPosition::function(*F6));

  const AAFunctionReachability &F7AA =
      A.getOrCreateAAFor<AAFunctionReachability>(IRPosition::function(*F7));

  F1AA.canReach(A, F3);
  F1AA.canReach(A, F4);
  F6AA.canReach(A, F4);
  F7AA.canReach(A, F7FirstCB, F3);
  F7AA.canReach(A, F7FirstCB, F4);

  A.run();

  ASSERT_TRUE(F1AA.canReach(A, F3));
  ASSERT_FALSE(F1AA.canReach(A, F4));

  ASSERT_TRUE(F7AA.canReach(A, F7FirstCB, F3));
  ASSERT_FALSE(F7AA.canReach(A, F7FirstCB, F4));

  // Assumed to be reacahable, since F6 can reach a function with
  // a unknown callee.
  ASSERT_TRUE(F6AA.canReach(A, F4));
}

} // namespace llvm
