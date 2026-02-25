//===- UniformityAnalysisCallbackVHTest.cpp - Deletion/RAUW callback tests
//-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests that UniformityAnalysis uses CallbackVH to keep UniformValues in sync
// when values are deleted or RAUW'd. After finalization, newly created
// instructions that are not in UniformValues are conservatively reported as
// divergent.
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

TEST(UniformityAnalysisCallbackVH, RAUWRemovesOldValueFromUniformValues) {
  // After analysis, RAUW a uniform instruction with another uniform
  // instruction. The old value is removed from UniformValues by
  // CallbackVH::allUsesReplacedWith(). A new instruction at a possibly
  // reused address must be reported divergent.
  StringRef ModuleString = R"(
  target triple = "amdgcn-unknown-amdhsa"
  define amdgpu_kernel void @test(i32 inreg %a, i32 inreg %b) {
    %add = add i32 %a, %b
    %sub = sub i32 %a, %b
    %mul = mul i32 %add, 2
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

  // Find %add and %sub by opcode.
  Instruction *AddInst = nullptr;
  Instruction *SubInst = nullptr;
  for (auto &I : F->getEntryBlock()) {
    if (I.getOpcode() == Instruction::Add)
      AddInst = &I;
    else if (I.getOpcode() == Instruction::Sub)
      SubInst = &I;
  }
  ASSERT_TRUE(AddInst && SubInst);
  EXPECT_FALSE(UI.isDivergent(AddInst)) << "%add should be uniform";
  EXPECT_FALSE(UI.isDivergent(SubInst)) << "%sub should be uniform";

  // RAUW %add -> %sub. CallbackVH::allUsesReplacedWith() removes %add from
  // UniformValues. After this, %mul becomes: %mul = mul i32 %sub, 2
  AddInst->replaceAllUsesWith(SubInst);

  // %add is still alive (not yet erased), but the RAUW callback should have
  // already removed it from UniformValues. Verify this: isDivergent must now
  // return true for %add because it is no longer in the uniform set.
  EXPECT_TRUE(UI.isDivergent(AddInst))
      << "%add must be removed from UniformValues after RAUW";

  // %add is now unused. Delete it.
  AddInst->eraseFromParent();

  // Create a new instruction. It is not in UniformValues and must be divergent.
  IRBuilder<> Builder(&F->getEntryBlock(), F->getEntryBlock().begin());
  Value *NewInst = Builder.CreateAdd(F->getArg(0), F->getArg(1), "new_add");

  EXPECT_TRUE(UI.isDivergent(NewInst))
      << "New instruction after RAUW+deletion must be reported divergent";
}

} // namespace
