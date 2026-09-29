//===- TensorSpecTest.cpp - test for TensorSpec ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/TensorSpec.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

using namespace llvm;

extern const char *TestMainArgv0;

TEST(TensorSpecTest, JSONParsing) {
  auto Value = json::parse(
      R"({"name": "tensor_name", 
        "port": 2, 
        "type": "int32_t", 
        "shape":[1,4]
        })");
  EXPECT_TRUE(!!Value);
  LLVMContext Ctx;
  std::optional<TensorSpec> Spec = getTensorSpecFromJSON(Ctx, *Value);
  EXPECT_TRUE(Spec);
  EXPECT_EQ(*Spec, TensorSpec::createSpec<int32_t>("tensor_name", {1, 4}, 2));
}

TEST(TensorSpecTest, JSONParsingInvalidTensorType) {
  auto Value = json::parse(
      R"(
        {"name": "tensor_name", 
        "port": 2, 
        "type": "no such type", 
        "shape":[1,4]
        }
      )");
  EXPECT_TRUE(!!Value);
  LLVMContext Ctx;
  auto Spec = getTensorSpecFromJSON(Ctx, *Value);
  EXPECT_FALSE(Spec);
}

TEST(TensorSpecTest, TensorSpecSizesAndTypes) {
  auto Spec1D = TensorSpec::createSpec<int16_t>("Hi1", {1});
  auto Spec2D = TensorSpec::createSpec<int16_t>("Hi2", {1, 1});
  auto Spec1DLarge = TensorSpec::createSpec<float>("Hi3", {10});
  auto Spec3DLarge = TensorSpec::createSpec<float>("Hi3", {2, 4, 10});
  EXPECT_TRUE(Spec1D.isElementType<int16_t>());
  EXPECT_FALSE(Spec3DLarge.isElementType<double>());
  EXPECT_EQ(Spec1D.getElementCount(), 1U);
  EXPECT_EQ(Spec2D.getElementCount(), 1U);
  EXPECT_EQ(Spec1DLarge.getElementCount(), 10U);
  EXPECT_EQ(Spec3DLarge.getElementCount(), 80U);
  EXPECT_EQ(Spec3DLarge.getElementByteSize(), sizeof(float));
  EXPECT_EQ(Spec1D.getElementByteSize(), sizeof(int16_t));
}

TEST(TensorSpecTest, PrintValueForDebug) {
  std::vector<int32_t> Values{1, 3};
  EXPECT_EQ(tensorValueToString(reinterpret_cast<const char *>(Values.data()),
                                TensorSpec::createSpec<int32_t>("name", {2})),
            "1,3");
}
