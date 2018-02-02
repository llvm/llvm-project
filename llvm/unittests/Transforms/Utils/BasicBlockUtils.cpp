//===- BasicBlockUtils.cpp - Unit tests for BasicBlockUtils ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("BasicBlockUtilsTests", errs());
  return Mod;
}

TEST(BasicBlockUtils, SplitBlockPredecessors) {
  LLVMContext C;

  std::unique_ptr<Module> M = parseIR(
    C,
    "define i32 @basic_func(i1 %cond) {\n"
    "entry:\n"
    "  br i1 %cond, label %bb0, label %bb1\n"
    "bb0:\n"
    "  br label %bb1\n"
    "bb1:\n"
    "  %phi = phi i32 [ 0, %entry ], [ 1, %bb0 ]"
    "  ret i32 %phi\n"
    "}\n"
    "\n"
    );

  auto *F = M->getFunction("basic_func");
  DominatorTree DT(*F);

  // Make sure the dominator tree is properly updated if calling this on the
  // entry block.
  SplitBlockPredecessors(&F->getEntryBlock(), {}, "split.entry", &DT);
  EXPECT_TRUE(DT.verify());
}
