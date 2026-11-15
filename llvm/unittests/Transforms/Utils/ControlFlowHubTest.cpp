//===- ControlFlowHubTest.cpp - ControlFlowHub tests ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Utils/ControlFlowUtils.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("ControlFlowHubTest", errs());
  return Mod;
}

TEST(ControlFlowHub, IntegerGuardKeepsBranchWeights) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"(
define void @f(i1 %cond) !prof !1 {
entry:
  br i1 %cond, label %left, label %right, !prof !0
left:
  ret void
right:
  ret void
}
!0 = !{!"branch_weights", i32 9, i32 1}
!1 = !{!"function_entry_count", i64 100}
)");
  ASSERT_TRUE(M);
  Function *F = M->getFunction("f");
  BasicBlock *Entry = &F->getEntryBlock();
  BasicBlock *Left = Entry->getTerminator()->getSuccessor(0);
  BasicBlock *Right = Entry->getTerminator()->getSuccessor(1);

  // No boolean guards, so this branch becomes an integer select.
  ControlFlowHub Hub;
  Hub.addBranch(Entry, Left, Right);
  SmallVector<BasicBlock *> Guards;
  Hub.finalize(nullptr, Guards, "irr", /*MaxControlFlowBooleans=*/0);

  SelectInst *GuardSelect = nullptr;
  for (Instruction &I : instructions(*F))
    if (auto *Select = dyn_cast<SelectInst>(&I))
      GuardSelect = Select;
  ASSERT_TRUE(GuardSelect);
  uint64_t TrueWeight = 0, FalseWeight = 0;
  ASSERT_TRUE(extractBranchWeights(*GuardSelect, TrueWeight, FalseWeight));
  EXPECT_EQ(TrueWeight, 9u);
  EXPECT_EQ(FalseWeight, 1u);
}
