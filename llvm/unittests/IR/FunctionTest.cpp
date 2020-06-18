//===- FunctionTest.cpp - Function unit tests -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
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

  // Checking that getArg gets the arguments from F1 in the correct order.
  unsigned i = 0;
  for (Argument &A : F->args()) {
    EXPECT_EQ(&A, F->getArg(i));
    ++i;
  }
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

// Test setting and removing section information
TEST(FunctionTest, setSection) {
  LLVMContext C;
  Module M("test", C);

  llvm::Function *F =
      Function::Create(llvm::FunctionType::get(llvm::Type::getVoidTy(C), false),
                       llvm::GlobalValue::ExternalLinkage, "F", &M);

  F->setSection(".text.test");
  EXPECT_TRUE(F->getSection() == ".text.test");
  EXPECT_TRUE(F->hasSection());
  F->setSection("");
  EXPECT_FALSE(F->hasSection());
  F->setSection(".text.test");
  F->setSection(".text.test2");
  EXPECT_TRUE(F->getSection() == ".text.test2");
  EXPECT_TRUE(F->hasSection());
}

TEST(FunctionTest, GetPointerAlignment) {
  LLVMContext Context;
  Type *VoidType(Type::getVoidTy(Context));
  FunctionType *FuncType(FunctionType::get(VoidType, false));
  std::unique_ptr<Function> Func(Function::Create(
      FuncType, GlobalValue::ExternalLinkage));
  EXPECT_EQ(Align(1), Func->getPointerAlignment(DataLayout("")));
  EXPECT_EQ(Align(1), Func->getPointerAlignment(DataLayout("Fi8")));
  EXPECT_EQ(Align(1), Func->getPointerAlignment(DataLayout("Fn8")));
  EXPECT_EQ(Align(2), Func->getPointerAlignment(DataLayout("Fi16")));
  EXPECT_EQ(Align(2), Func->getPointerAlignment(DataLayout("Fn16")));
  EXPECT_EQ(Align(4), Func->getPointerAlignment(DataLayout("Fi32")));
  EXPECT_EQ(Align(4), Func->getPointerAlignment(DataLayout("Fn32")));

  Func->setAlignment(Align(4));

  EXPECT_EQ(Align(1), Func->getPointerAlignment(DataLayout("")));
  EXPECT_EQ(Align(1), Func->getPointerAlignment(DataLayout("Fi8")));
  EXPECT_EQ(Align(4), Func->getPointerAlignment(DataLayout("Fn8")));
  EXPECT_EQ(Align(2), Func->getPointerAlignment(DataLayout("Fi16")));
  EXPECT_EQ(Align(4), Func->getPointerAlignment(DataLayout("Fn16")));
  EXPECT_EQ(Align(4), Func->getPointerAlignment(DataLayout("Fi32")));
  EXPECT_EQ(Align(4), Func->getPointerAlignment(DataLayout("Fn32")));
}

} // end namespace
