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

  AbstractAttribute *AA = (AbstractAttribute *)&(
      A.getOrCreateAAFor<AAIsDead>(IRPosition::function(*F)));

  EXPECT_TRUE(AA);

  const auto *SFail = dyn_cast<AAAlign>(AA);
  const auto *SSucc = dyn_cast<AAIsDead>(AA);

  ASSERT_EQ(SFail, nullptr);
  ASSERT_TRUE(SSucc);
}

} // namespace llvm