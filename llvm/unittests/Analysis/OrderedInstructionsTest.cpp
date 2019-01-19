//===- OrderedInstructions.cpp - Unit tests for OrderedInstructions  ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/OrderedInstructions.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "gtest/gtest.h"

using namespace llvm;

/// Check intra-basicblock and inter-basicblock dominance using
/// OrderedInstruction.
TEST(OrderedInstructionsTest, DominanceTest) {
  LLVMContext Ctx;
  Module M("test", Ctx);
  IRBuilder<> B(Ctx);
  FunctionType *FTy =
      FunctionType::get(Type::getVoidTy(Ctx), {B.getInt8PtrTy()}, false);
  Function *F = cast<Function>(M.getOrInsertFunction("f", FTy));

  // Create the function as follow and check for dominance relation.
  //
  // test():
  //  bbx:
  //    loadx;
  //    loady;
  //  bby:
  //    loadz;
  //    return;
  //
  // More specifically, check for loadx -> (dominates) loady,
  // loady -> loadx and loady -> loadz.
  //
  // Create BBX with 2 loads.
  BasicBlock *BBX = BasicBlock::Create(Ctx, "bbx", F);
  B.SetInsertPoint(BBX);
  Argument *PointerArg = &*F->arg_begin();
  LoadInst *LoadInstX = B.CreateLoad(PointerArg);
  LoadInst *LoadInstY = B.CreateLoad(PointerArg);

  // Create BBY with 1 load.
  BasicBlock *BBY = BasicBlock::Create(Ctx, "bby", F);
  B.SetInsertPoint(BBY);
  LoadInst *LoadInstZ = B.CreateLoad(PointerArg);
  B.CreateRet(LoadInstZ);
  std::unique_ptr<DominatorTree> DT(new DominatorTree(*F));
  OrderedInstructions OI(&*DT);

  // Intra-BB dominance test.
  EXPECT_TRUE(OI.dominates(LoadInstX, LoadInstY));
  EXPECT_FALSE(OI.dominates(LoadInstY, LoadInstX));

  // Inter-BB dominance test.
  EXPECT_TRUE(OI.dominates(LoadInstY, LoadInstZ));
}
