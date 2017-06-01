//===- FunctionTest.cpp - Function unit tests -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Function.h"
#include "gtest/gtest.h"
using namespace llvm;

namespace {

TEST(FunctionTest, hasLazyArguments) {
  LLVMContext C;

  Type *ArgTypes[] = {Type::getInt8Ty(C), Type::getInt32Ty(C)};
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(C), ArgTypes, false);

  // Functions start out with lazy arguments.
  std::unique_ptr<Function> F(
      Function::Create(FTy, GlobalValue::ExternalLinkage, "F"));
  EXPECT_TRUE(F->hasLazyArguments());

  // Checking for empty or size shouldn't force arguments to be instantiated.
  EXPECT_FALSE(F->arg_empty());
  EXPECT_TRUE(F->hasLazyArguments());
  EXPECT_EQ(2u, F->arg_size());
  EXPECT_TRUE(F->hasLazyArguments());

  // The argument list should be populated at first access.
  (void)F->arg_begin();
  EXPECT_FALSE(F->hasLazyArguments());
}

TEST(FunctionTest, stealArgumentListFrom) {
  LLVMContext C;

  Type *ArgTypes[] = {Type::getInt8Ty(C), Type::getInt32Ty(C)};
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(C), ArgTypes, false);
  std::unique_ptr<Function> F1(
      Function::Create(FTy, GlobalValue::ExternalLinkage, "F1"));
  std::unique_ptr<Function> F2(
      Function::Create(FTy, GlobalValue::ExternalLinkage, "F1"));
  EXPECT_TRUE(F1->hasLazyArguments());
  EXPECT_TRUE(F2->hasLazyArguments());

  // Steal arguments before they've been accessed.  Nothing should change; both
  // functions should still have lazy arguments.
  //
  //   steal(empty); drop (empty)
  F1->stealArgumentListFrom(*F2);
  EXPECT_TRUE(F1->hasLazyArguments());
  EXPECT_TRUE(F2->hasLazyArguments());

  // Save arguments from F1 for later assertions.  F1 won't have lazy arguments
  // anymore.
  SmallVector<Argument *, 4> Args;
  for (Argument &A : F1->args())
    Args.push_back(&A);
  EXPECT_EQ(2u, Args.size());
  EXPECT_FALSE(F1->hasLazyArguments());

  // Steal arguments from F1 to F2.  F1's arguments should be lazy again.
  //
  //   steal(real); drop (empty)
  F2->stealArgumentListFrom(*F1);
  EXPECT_TRUE(F1->hasLazyArguments());
  EXPECT_FALSE(F2->hasLazyArguments());
  unsigned I = 0;
  for (Argument &A : F2->args()) {
    EXPECT_EQ(Args[I], &A);
    I++;
  }
  EXPECT_EQ(2u, I);

  // Check that arguments in F1 don't have pointer equality with the saved ones.
  // This also instantiates F1's arguments.
  I = 0;
  for (Argument &A : F1->args()) {
    EXPECT_NE(Args[I], &A);
    I++;
  }
  EXPECT_EQ(2u, I);
  EXPECT_FALSE(F1->hasLazyArguments());
  EXPECT_FALSE(F2->hasLazyArguments());

  // Steal back from F2.  F2's arguments should be lazy again.
  //
  //   steal(real); drop (real)
  F1->stealArgumentListFrom(*F2);
  EXPECT_FALSE(F1->hasLazyArguments());
  EXPECT_TRUE(F2->hasLazyArguments());
  I = 0;
  for (Argument &A : F1->args()) {
    EXPECT_EQ(Args[I], &A);
    I++;
  }
  EXPECT_EQ(2u, I);

  // Steal from F2 a second time.  Now both functions should have lazy
  // arguments.
  //
  //   steal(empty); drop (real)
  F1->stealArgumentListFrom(*F2);
  EXPECT_TRUE(F1->hasLazyArguments());
  EXPECT_TRUE(F2->hasLazyArguments());
}

} // end namespace
