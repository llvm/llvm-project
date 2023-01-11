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
  AttributorConfig AC(CGUpdater);
  Attributor A(Functions, InfoCache, AC);

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
    @x = external global i32
    define internal void @func4() {
      store i32 0, i32* @x
      ret void
    }

    define internal void @func3() {
      store i32 0, i32* @x
      ret void
    }

    define internal void @func8() {
      store i32 0, i32* @x
      ret void
    }

    define internal void @func2() {
    entry:
      call void @func3()
      ret void
    }

    define internal void @func1() {
    entry:
      call void @func2()
      ret void
    }

    declare void @unknown()
    define internal void @func5(void ()* %ptr) {
    entry:
      call void %ptr()
      call void @unknown()
      ret void
    }

    define internal void @func6() {
    entry:
      call void @func5(void ()* @func3)
      ret void
    }

    define internal void @func7() {
    entry:
      call void @func2()
      call void @func4()
      ret void
    }

    define internal void @func9() {
    entry:
      call void @func2()
      call void @func8()
      ret void
    }

    define void @func10() {
    entry:
      call void @func9()
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
  AttributorConfig AC(CGUpdater);
  AC.DeleteFns = false;
  Attributor A(Functions, InfoCache, AC);

  Function &F1 = *M.getFunction("func1");
  Function &F3 = *M.getFunction("func3");
  Function &F4 = *M.getFunction("func4");
  Function &F6 = *M.getFunction("func6");
  Function &F7 = *M.getFunction("func7");
  Function &F9 = *M.getFunction("func9");

  // call void @func2()
  CallBase &F7FirstCB = static_cast<CallBase &>(*F7.getEntryBlock().begin());
  // call void @func2()
  Instruction &F9FirstInst = *F9.getEntryBlock().begin();
  // call void @func8
  Instruction &F9SecondInst = *++(F9.getEntryBlock().begin());

  const AAInterFnReachability &F1AA =
      A.getOrCreateAAFor<AAInterFnReachability>(IRPosition::function(F1));

  const AAInterFnReachability &F6AA =
      A.getOrCreateAAFor<AAInterFnReachability>(IRPosition::function(F6));

  const AAInterFnReachability &F7AA =
      A.getOrCreateAAFor<AAInterFnReachability>(IRPosition::function(F7));

  const AAInterFnReachability &F9AA =
      A.getOrCreateAAFor<AAInterFnReachability>(IRPosition::function(F9));

  F1AA.canReach(A, F3);
  F1AA.canReach(A, F4);
  F6AA.canReach(A, F4);
  F7AA.instructionCanReach(A, F7FirstCB, F3);
  F7AA.instructionCanReach(A, F7FirstCB, F4);
  F9AA.instructionCanReach(A, F9FirstInst, F3);
  F9AA.instructionCanReach(A, F9FirstInst, F4);

  A.run();

  ASSERT_TRUE(F1AA.canReach(A, F3));
  ASSERT_FALSE(F1AA.canReach(A, F4));

  ASSERT_TRUE(F7AA.instructionCanReach(A, F7FirstCB, F3));
  ASSERT_TRUE(F7AA.instructionCanReach(A, F7FirstCB, F4));

  // Assumed to be reacahable, since F6 can reach a function with
  // a unknown callee.
  ASSERT_TRUE(F6AA.canReach(A, F4));

  // The second instruction of F9 can't reach the first call.
  ASSERT_FALSE(F9AA.instructionCanReach(A, F9SecondInst, F3));

  // The first instruction of F9 can reach the first call.
  ASSERT_TRUE(F9AA.instructionCanReach(A, F9FirstInst, F3));
  // Because func10 calls the func4 after the call to func9 it is reachable but
  // as it requires backwards logic we would need AA::isPotentiallyReachable.
  ASSERT_FALSE(F9AA.instructionCanReach(A, F9FirstInst, F4));
}

} // namespace llvm
