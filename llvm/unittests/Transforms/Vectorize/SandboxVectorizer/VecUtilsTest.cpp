//===- VecUtilsTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/VecUtils.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/SandboxIR/Context.h"
#include "llvm/SandboxIR/Type.h"
#include "gtest/gtest.h"

using namespace llvm;

struct VecUtilsTest : public testing::Test {
  LLVMContext C;
};

TEST_F(VecUtilsTest, GetNumElements) {
  sandboxir::Context Ctx(C);
  auto *ElemTy = sandboxir::Type::getInt32Ty(Ctx);
  EXPECT_EQ(sandboxir::VecUtils::getNumElements(ElemTy), 1);
  auto *VTy = sandboxir::FixedVectorType::get(ElemTy, 2);
  EXPECT_EQ(sandboxir::VecUtils::getNumElements(VTy), 2);
  auto *VTy1 = sandboxir::FixedVectorType::get(ElemTy, 1);
  EXPECT_EQ(sandboxir::VecUtils::getNumElements(VTy1), 1);
}

TEST_F(VecUtilsTest, GetElementType) {
  sandboxir::Context Ctx(C);
  auto *ElemTy = sandboxir::Type::getInt32Ty(Ctx);
  EXPECT_EQ(sandboxir::VecUtils::getElementType(ElemTy), ElemTy);
  auto *VTy = sandboxir::FixedVectorType::get(ElemTy, 2);
  EXPECT_EQ(sandboxir::VecUtils::getElementType(VTy), ElemTy);
}
