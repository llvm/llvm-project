//===- UniformityAnalysisCallbackVHTest.cpp - Deletion callback tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests that UniformityAnalysis uses CallbackVH to keep UniformValues in sync
// when values are deleted.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUTargetMachine.h"
#include "AMDGPUUnitTests.h"
#include "llvm/ADT/GenericUniformityImpl.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/CycleInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

static UniformityInfo computeUniformity(TargetTransformInfo *TTI, Function *F) {
  DominatorTree DT(*F);
  CycleInfo CI;
  CI.compute(*F);
  UniformityInfo UI{DT, CI, TTI};
  if (TTI->hasBranchDivergence(F))
    UI.compute();
  return UI;
}

TEST(UniformityAnalysisCallbackVH, DeletionMakesNewInstDivergent) {
  // Delete a uniform instruction. CallbackVH::deleted() removes it from
  // UniformValues during eraseFromParent(). A newly created instruction is
  // not in UniformValues and must be conservatively reported as divergent.
  StringRef ModuleString = R"(
  target triple = "amdgcn-unknown-amdhsa"
  define amdgpu_kernel void @test(i32 inreg %a, i32 inreg %b) {
    %add = add i32 %a, %b
    ret void
  }
  )";
  LLVMContext Context;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(ModuleString, Err, Context);
  ASSERT_TRUE(M) << Err.getMessage();

  Function *F = M->getFunction("test");
  ASSERT_TRUE(F);

  auto TM =
      createAMDGPUTargetMachine("amdgcn-amd-", "gfx1010", "+wavefrontsize32");
  ASSERT_TRUE(TM);
  TargetTransformInfo TTI = TM->getTargetTransformInfo(*F);

  UniformityInfo UI = computeUniformity(&TTI, F);

  Instruction *AddInst = &*F->getEntryBlock().begin();
  ASSERT_TRUE(isa<BinaryOperator>(AddInst));
  EXPECT_FALSE(UI.isDivergent(AddInst)) << "%add should be uniform";

  // Delete %add. CallbackVH::deleted() removes it from UniformValues.
  AddInst->eraseFromParent();

  // New instruction was not present during analysis, so it is not in
  // UniformValues. isDivergent must return true (conservative).
  IRBuilder<> Builder(&F->getEntryBlock(), F->getEntryBlock().begin());
  Value *NewInst = Builder.CreateAdd(F->getArg(0), F->getArg(1), "new_add");

  EXPECT_TRUE(UI.isDivergent(NewInst))
      << "New instruction after deletion must be reported divergent";
}

} // namespace
