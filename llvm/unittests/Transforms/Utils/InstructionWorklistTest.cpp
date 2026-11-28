//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "instruction-worklist-test"
#include "llvm/Transforms/Utils/InstructionWorklist.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C) {
  const char *IR = R"(
    define void @f() {
      %a = add i32 0, 1
      %b = add i32 0, 2
      ret void
    }
  )";
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(IR, Err, C);
  if (!M)
    Err.print("InstructionWorklistTest", errs());
  return M;
}

TEST(InstructionWorklistTest, PushRemove) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C);
  Instruction *A = &M->getFunction("f")->front().front();
  Instruction *B = A->getNextNode();

  InstructionWorklist WL;
  WL.push(A);
  WL.push(B);
  WL.push(A);
  WL.remove(B);
  EXPECT_EQ(WL.removeOne(), nullptr);
  EXPECT_EQ(WL.removeOne(), A);
  EXPECT_TRUE(WL.isEmpty());
  WL.zap();
}

#if defined(GTEST_HAS_DEATH_TEST) && !defined(NDEBUG)
TEST(InstructionWorklistTest, Renumbered) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C);
  Function *F = M->getFunction("f");
  Instruction *A = &F->front().front();
  Instruction *B = A->getNextNode();

  InstructionWorklist WL;
  WL.push(A);
  A->removeFromParent();
  A->insertBefore(B->getIterator());
  EXPECT_DEATH(WL.removeOne(), "renumbered while in the worklist");
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  F->renumberInstructions();
  EXPECT_DEATH(WL.push(B), "renumbered while in the worklist");
#endif
}
#endif
