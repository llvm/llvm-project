//===- llvm/unittest/IR/WaymarkTest.cpp - getUser() unit tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// we perform white-box tests
//
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "gtest/gtest.h"
#include <algorithm>

namespace llvm {
namespace {

TEST(WaymarkTest, NativeArray) {
  LLVMContext Context;
  static uint8_t tail[22] = "s02s33s30y2y0s1x0syxS";
  Value * values[22];
  std::transform(tail, tail + 22, values, [&](char c) {
    return ConstantInt::get(Type::getInt8Ty(Context), c);
  });
  FunctionType *FT = FunctionType::get(Type::getVoidTy(Context), true);
  std::unique_ptr<Function> F(
      Function::Create(FT, GlobalValue::ExternalLinkage));
  const CallInst *A = CallInst::Create(F.get(), makeArrayRef(values));
  ASSERT_NE(A, (const CallInst*)nullptr);
  ASSERT_EQ(1U + 22, A->getNumOperands());
  const Use *U = &A->getOperandUse(0);
  const Use *Ue = &A->getOperandUse(22);
  for (; U != Ue; ++U)
  {
    EXPECT_EQ(A, U->getUser());
  }
  delete A;
}

TEST(WaymarkTest, TwoBit) {
  Use* many = (Use*)calloc(sizeof(Use), 8212 + 1);
  ASSERT_TRUE(many);
  Use::initTags(many, many + 8212);
  for (Use *U = many, *Ue = many + 8212 - 1; U != Ue; ++U)
  {
    EXPECT_EQ(reinterpret_cast<User *>(Ue + 1), U->getUser());
  }
  free(many);
}

}  // end anonymous namespace
}  // end namespace llvm
