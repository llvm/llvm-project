//===- TrackingSSAUpdaterTest.cpp - Unit tests for TrackingSSAUpdater
//------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include "gtest/gtest.h"

using namespace llvm;

// Test that TrackingSSAUpdater follows RAUW and returns the replacement value.
TEST(TrackingSSAUpdater, RAUWUpdatesAvailableValue) {
  LLVMContext C;
  Module M("TrackingSSAUpdaterTest", C);
  IRBuilder<> B(C);
  Type *I32Ty = B.getInt32Ty();

  // Create: define void @f(i32 %arg) { entry: %v = add i32 %arg, 1; ret void }
  auto *F = Function::Create(FunctionType::get(B.getVoidTy(), {I32Ty}, false),
                             GlobalValue::ExternalLinkage, "f", &M);
  Argument *Arg = &*F->arg_begin();
  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  B.SetInsertPoint(Entry);
  Value *V = B.CreateAdd(Arg, ConstantInt::get(I32Ty, 1), "v");
  B.CreateRetVoid();

  // Register %v as the available value for Entry.
  TrackingSSAUpdater Updater;
  Updater.Initialize(I32Ty, "test");
  Updater.AddAvailableValue(Entry, V);
  EXPECT_EQ(Updater.FindValueForBlock(Entry), V);
  EXPECT_TRUE(Updater.HasValueForBlock(Entry));

  // RAUW %v with a constant.
  Value *Replacement = ConstantInt::get(I32Ty, 42);
  V->replaceAllUsesWith(Replacement);

  // TrackingSSAUpdater should now return the replacement.
  EXPECT_EQ(Updater.FindValueForBlock(Entry), Replacement);
}

// Test that AddAvailableValue overwrites a previously tracked value.
TEST(TrackingSSAUpdater, OverwriteAvailableValue) {
  LLVMContext C;
  Module M("TrackingSSAUpdaterTest", C);
  IRBuilder<> B(C);
  Type *I32Ty = B.getInt32Ty();

  auto *F = Function::Create(FunctionType::get(B.getVoidTy(), false),
                             GlobalValue::ExternalLinkage, "f", &M);
  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  B.SetInsertPoint(Entry);
  B.CreateRetVoid();

  Value *C1 = ConstantInt::get(I32Ty, 1);
  Value *C2 = ConstantInt::get(I32Ty, 2);

  TrackingSSAUpdater Updater;
  Updater.Initialize(I32Ty, "test");
  Updater.AddAvailableValue(Entry, C1);
  EXPECT_EQ(Updater.FindValueForBlock(Entry), C1);

  Updater.AddAvailableValue(Entry, C2);
  EXPECT_EQ(Updater.FindValueForBlock(Entry), C2);
}

// Test that GetValueInMiddleOfBlock returns the RAUW'd value after syncAll.
TEST(TrackingSSAUpdater, GetValueInMiddleOfBlockAfterRAUW) {
  LLVMContext C;
  Module M("TrackingSSAUpdaterTest", C);
  IRBuilder<> B(C);
  Type *I32Ty = B.getInt32Ty();

  // Create: define void @f(i32 %arg) { entry: %v = add %arg, 1; br bb2;
  //                                     bb2: ret void }
  auto *F = Function::Create(FunctionType::get(B.getVoidTy(), {I32Ty}, false),
                             GlobalValue::ExternalLinkage, "f", &M);
  Argument *Arg = &*F->arg_begin();
  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  BasicBlock *BB2 = BasicBlock::Create(C, "bb2", F);

  B.SetInsertPoint(Entry);
  Value *V = B.CreateAdd(Arg, ConstantInt::get(I32Ty, 1), "v");
  B.CreateBr(BB2);

  B.SetInsertPoint(BB2);
  B.CreateRetVoid();

  // Register %v for Entry.
  TrackingSSAUpdater Updater;
  Updater.Initialize(I32Ty, "test");
  Updater.AddAvailableValue(Entry, V);

  // RAUW %v -> constant.
  Value *Replacement = ConstantInt::get(I32Ty, 42);
  V->replaceAllUsesWith(Replacement);

  // GetValueInMiddleOfBlock for BB2 (single predecessor: Entry) should
  // return the replacement, not the stale pointer.
  Value *Result = Updater.GetValueInMiddleOfBlock(BB2);
  EXPECT_EQ(Result, Replacement);
}

// Test that FindValueForBlock returns nullptr for an unregistered block.
TEST(TrackingSSAUpdater, FindValueForUnknownBlock) {
  LLVMContext C;
  Module M("TrackingSSAUpdaterTest", C);
  IRBuilder<> B(C);
  Type *I32Ty = B.getInt32Ty();

  auto *F = Function::Create(FunctionType::get(B.getVoidTy(), false),
                             GlobalValue::ExternalLinkage, "f", &M);
  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  BasicBlock *Other = BasicBlock::Create(C, "other", F);
  B.SetInsertPoint(Entry);
  B.CreateBr(Other);
  B.SetInsertPoint(Other);
  B.CreateRetVoid();

  TrackingSSAUpdater Updater;
  Updater.Initialize(I32Ty, "test");
  Updater.AddAvailableValue(Entry, ConstantInt::get(I32Ty, 1));

  EXPECT_EQ(Updater.FindValueForBlock(Other), nullptr);
  EXPECT_FALSE(Updater.HasValueForBlock(Other));
}

// Test Initialize clears previous state.
TEST(TrackingSSAUpdater, InitializeClearsPreviousState) {
  LLVMContext C;
  Module M("TrackingSSAUpdaterTest", C);
  IRBuilder<> B(C);
  Type *I32Ty = B.getInt32Ty();

  auto *F = Function::Create(FunctionType::get(B.getVoidTy(), false),
                             GlobalValue::ExternalLinkage, "f", &M);
  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  B.SetInsertPoint(Entry);
  B.CreateRetVoid();

  TrackingSSAUpdater Updater;
  Updater.Initialize(I32Ty, "first");
  Updater.AddAvailableValue(Entry, ConstantInt::get(I32Ty, 1));
  EXPECT_TRUE(Updater.HasValueForBlock(Entry));

  // Re-initialize should clear tracked values.
  Updater.Initialize(I32Ty, "second");
  EXPECT_FALSE(Updater.HasValueForBlock(Entry));
}
