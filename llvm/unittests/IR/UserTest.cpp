//===- llvm/unittest/IR/UserTest.cpp - User unit tests --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/User.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"
using namespace llvm;

namespace {

TEST(UserTest, ValueOpIteration) {
  LLVMContext C;

  const char *ModuleString = "define void @f(i32 %x, i32 %y) {\n"
                             "entry:\n"
                             "  switch i32 undef, label %s0\n"
                             "      [ i32 1, label %s1\n"
                             "        i32 2, label %s2\n"
                             "        i32 3, label %s3\n"
                             "        i32 4, label %s4\n"
                             "        i32 5, label %s5\n"
                             "        i32 6, label %s6\n"
                             "        i32 7, label %s7\n"
                             "        i32 8, label %s8\n"
                             "        i32 9, label %s9 ]\n"
                             "\n"
                             "s0:\n"
                             "  br label %exit\n"
                             "s1:\n"
                             "  br label %exit\n"
                             "s2:\n"
                             "  br label %exit\n"
                             "s3:\n"
                             "  br label %exit\n"
                             "s4:\n"
                             "  br label %exit\n"
                             "s5:\n"
                             "  br label %exit\n"
                             "s6:\n"
                             "  br label %exit\n"
                             "s7:\n"
                             "  br label %exit\n"
                             "s8:\n"
                             "  br label %exit\n"
                             "s9:\n"
                             "  br label %exit\n"
                             "\n"
                             "exit:\n"
                             "  %phi = phi i32 [ 0, %s0 ], [ 1, %s1 ],\n"
                             "                 [ 2, %s2 ], [ 3, %s3 ],\n"
                             "                 [ 4, %s4 ], [ 5, %s5 ],\n"
                             "                 [ 6, %s6 ], [ 7, %s7 ],\n"
                             "                 [ 8, %s8 ], [ 9, %s9 ]\n"
                             "  ret void\n"
                             "}\n";
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(ModuleString, Err, C);

  Function *F = M->getFunction("f");
  BasicBlock &ExitBB = F->back();
  PHINode &P = cast<PHINode>(ExitBB.front());
  EXPECT_TRUE(P.value_op_begin() == P.value_op_begin());
  EXPECT_FALSE(P.value_op_begin() == P.value_op_end());
  EXPECT_TRUE(P.value_op_begin() != P.value_op_end());
  EXPECT_FALSE(P.value_op_end() != P.value_op_end());
  EXPECT_TRUE(P.value_op_begin() < P.value_op_end());
  EXPECT_FALSE(P.value_op_begin() < P.value_op_begin());
  EXPECT_TRUE(P.value_op_end() > P.value_op_begin());
  EXPECT_FALSE(P.value_op_begin() > P.value_op_begin());
  EXPECT_TRUE(P.value_op_begin() <= P.value_op_begin());
  EXPECT_FALSE(P.value_op_end() <= P.value_op_begin());
  EXPECT_TRUE(P.value_op_begin() >= P.value_op_begin());
  EXPECT_FALSE(P.value_op_begin() >= P.value_op_end());
  EXPECT_EQ(10, std::distance(P.value_op_begin(), P.value_op_end()));

  // const value op iteration
  const PHINode *IP = &P;
  EXPECT_TRUE(IP->value_op_begin() == IP->value_op_begin());
  EXPECT_FALSE(IP->value_op_begin() == IP->value_op_end());
  EXPECT_TRUE(IP->value_op_begin() != IP->value_op_end());
  EXPECT_FALSE(IP->value_op_end() != IP->value_op_end());
  EXPECT_TRUE(IP->value_op_begin() < IP->value_op_end());
  EXPECT_FALSE(IP->value_op_begin() < IP->value_op_begin());
  EXPECT_TRUE(IP->value_op_end() > IP->value_op_begin());
  EXPECT_FALSE(IP->value_op_begin() > IP->value_op_begin());
  EXPECT_TRUE(IP->value_op_begin() <= IP->value_op_begin());
  EXPECT_FALSE(IP->value_op_end() <= IP->value_op_begin());
  EXPECT_TRUE(IP->value_op_begin() >= IP->value_op_begin());
  EXPECT_FALSE(IP->value_op_begin() >= IP->value_op_end());
  EXPECT_EQ(10, std::distance(IP->value_op_begin(), IP->value_op_end()));

  User::value_op_iterator I = P.value_op_begin();
  I += 3;
  EXPECT_EQ(std::next(P.value_op_begin(), 3), I);
  EXPECT_EQ(P.getOperand(3), *I);
  I++;
  EXPECT_EQ(P.getOperand(6), I[2]);
  EXPECT_EQ(P.value_op_end(), (I - 2) + 8);

  // const value op
  User::const_value_op_iterator CI = IP->value_op_begin();
  CI += 3;
  EXPECT_EQ(std::next(IP->value_op_begin(), 3), CI);
  EXPECT_EQ(IP->getOperand(3), *CI);
  CI++;
  EXPECT_EQ(IP->getOperand(6), CI[2]);
  EXPECT_EQ(IP->value_op_end(), (CI - 2) + 8);
}

TEST(UserTest, replaceUseOfWith) {
  LLVMContext C;

  const char *ModuleString = "define void @f(i32 %x) {\n"
                             "entry:\n"
                             "  %v0 = add i32 1, 1\n"
                             "  %v1 = add i32 %x, 2\n"
                             "  ret void\n"
                             "}\n";
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(ModuleString, Err, C);
  Function *F = M->getFunction("f");
  EXPECT_TRUE(F);
  EXPECT_TRUE(F->arg_begin() != F->arg_end());
  BasicBlock& entryBB = F->front();
  Instruction& I0 = *(entryBB.begin());
  Instruction& I1 = *(++(entryBB.begin()));

  Argument &X = *F->arg_begin();
  EXPECT_EQ("x", X.getName());
  EXPECT_NE(X.user_begin() ,X.user_end());
  EXPECT_EQ(I0.user_begin() ,I0.user_end());


  auto XUser = find(X.users(), &(I1));
  EXPECT_NE(XUser, X.user_end());
 
  XUser->replaceUsesOfWith(&X, &I0);
  EXPECT_EQ(X.user_begin() ,X.user_end());
  EXPECT_NE(I0.user_begin() ,I0.user_end());
}

TEST(UserTest, PersonalityUser) {
  LLVMContext Context;
  Module M("", Context);
  FunctionType *RetVoidTy = FunctionType::get(Type::getVoidTy(Context), false);
  Function *PersonalityF = Function::Create(
      RetVoidTy, GlobalValue::ExternalLinkage, "PersonalityFn", &M);
  Function *TestF =
      Function::Create(RetVoidTy, GlobalValue::ExternalLinkage, "TestFn", &M);

  // Set up the personality function
  TestF->setPersonalityFn(PersonalityF);
  auto PersonalityUsers = PersonalityF->user_begin();

  // One user and that user is the Test function
  EXPECT_EQ(*PersonalityUsers, TestF);
  EXPECT_EQ(++PersonalityUsers, PersonalityF->user_end());

  // Reset the personality function
  TestF->setPersonalityFn(nullptr);

  // No users should remain
  EXPECT_TRUE(TestF->user_empty());
}

} // end anonymous namespace
