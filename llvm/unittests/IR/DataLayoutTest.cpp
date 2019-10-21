//===- ConstantRangeTest.cpp - ConstantRange tests ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(DataLayoutTest, FunctionPtrAlign) {
  EXPECT_EQ(MaybeAlign(0), DataLayout("").getFunctionPtrAlign());
  EXPECT_EQ(MaybeAlign(1), DataLayout("Fi8").getFunctionPtrAlign());
  EXPECT_EQ(MaybeAlign(2), DataLayout("Fi16").getFunctionPtrAlign());
  EXPECT_EQ(MaybeAlign(4), DataLayout("Fi32").getFunctionPtrAlign());
  EXPECT_EQ(MaybeAlign(8), DataLayout("Fi64").getFunctionPtrAlign());
  EXPECT_EQ(MaybeAlign(1), DataLayout("Fn8").getFunctionPtrAlign());
  EXPECT_EQ(MaybeAlign(2), DataLayout("Fn16").getFunctionPtrAlign());
  EXPECT_EQ(MaybeAlign(4), DataLayout("Fn32").getFunctionPtrAlign());
  EXPECT_EQ(MaybeAlign(8), DataLayout("Fn64").getFunctionPtrAlign());
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

TEST(DataLayoutTest, ValueOrABITypeAlignment) {
  const DataLayout DL("Fi8");
  LLVMContext Context;
  Type *const FourByteAlignType = Type::getInt32Ty(Context);
  EXPECT_EQ(Align(16),
            DL.getValueOrABITypeAlignment(MaybeAlign(16), FourByteAlignType));
  EXPECT_EQ(Align(4),
            DL.getValueOrABITypeAlignment(MaybeAlign(), FourByteAlignType));
}

}  // anonymous namespace
