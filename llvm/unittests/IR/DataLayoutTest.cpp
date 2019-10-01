//===- ConstantRangeTest.cpp - ConstantRange tests ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DataLayout.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(DataLayoutTest, FunctionPtrAlign) {
  EXPECT_EQ(0U, DataLayout("").getFunctionPtrAlign());
  EXPECT_EQ(1U, DataLayout("Fi8").getFunctionPtrAlign());
  EXPECT_EQ(2U, DataLayout("Fi16").getFunctionPtrAlign());
  EXPECT_EQ(4U, DataLayout("Fi32").getFunctionPtrAlign());
  EXPECT_EQ(8U, DataLayout("Fi64").getFunctionPtrAlign());
  EXPECT_EQ(1U, DataLayout("Fn8").getFunctionPtrAlign());
  EXPECT_EQ(2U, DataLayout("Fn16").getFunctionPtrAlign());
  EXPECT_EQ(4U, DataLayout("Fn32").getFunctionPtrAlign());
  EXPECT_EQ(8U, DataLayout("Fn64").getFunctionPtrAlign());
  EXPECT_EQ(DataLayout::FunctionPtrAlignType::Independent, \
      DataLayout("").getFunctionPtrAlignType());
  EXPECT_EQ(DataLayout::FunctionPtrAlignType::Independent, \
      DataLayout("Fi8").getFunctionPtrAlignType());
  EXPECT_EQ(DataLayout::FunctionPtrAlignType::MultipleOfFunctionAlign, \
      DataLayout("Fn8").getFunctionPtrAlignType());
  EXPECT_EQ(DataLayout("Fi8"), DataLayout("Fi8"));
  EXPECT_NE(DataLayout("Fi8"), DataLayout("Fi16"));
  EXPECT_NE(DataLayout("Fi8"), DataLayout("Fn8"));

  DataLayout a(""), b("Fi8"), c("Fn8");
  EXPECT_NE(a, b);
  EXPECT_NE(a, c);
  EXPECT_NE(b, c);

  a = b;
  EXPECT_EQ(a, b);
  a = c;
  EXPECT_EQ(a, c);
}

}  // anonymous namespace
