//===- llvm/unittest/IR/AttributesTest.cpp - Attributes unit tests --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Attributes.h"
#include "llvm/IR/LLVMContext.h"
#include "gtest/gtest.h"
using namespace llvm;

namespace {

TEST(Attributes, Uniquing) {
  LLVMContext C;

  Attribute AttrA = Attribute::get(C, Attribute::AlwaysInline);
  Attribute AttrB = Attribute::get(C, Attribute::AlwaysInline);
  EXPECT_EQ(AttrA, AttrB);

  AttributeList ASs[] = {AttributeList::get(C, 1, Attribute::ZExt),
                         AttributeList::get(C, 2, Attribute::SExt)};

  AttributeList SetA = AttributeList::get(C, ASs);
  AttributeList SetB = AttributeList::get(C, ASs);
  EXPECT_EQ(SetA, SetB);
}

TEST(Attributes, Ordering) {
  LLVMContext C;

  Attribute Align4 = Attribute::get(C, Attribute::Alignment, 4);
  Attribute Align5 = Attribute::get(C, Attribute::Alignment, 5);
  Attribute Deref4 = Attribute::get(C, Attribute::Dereferenceable, 4);
  Attribute Deref5 = Attribute::get(C, Attribute::Dereferenceable, 5);
  EXPECT_TRUE(Align4 < Align5);
  EXPECT_TRUE(Align4 < Deref4);
  EXPECT_TRUE(Align4 < Deref5);
  EXPECT_TRUE(Align5 < Deref4);

  AttributeList ASs[] = {AttributeList::get(C, 2, Attribute::ZExt),
                         AttributeList::get(C, 1, Attribute::SExt)};

  AttributeList SetA = AttributeList::get(C, ASs);
  AttributeList SetB = SetA.removeAttributes(C, 1, ASs[1].getAttributes(1));
  EXPECT_NE(SetA, SetB);
}

TEST(Attributes, AddAttributes) {
  LLVMContext C;
  AttributeList AL;
  AttrBuilder B;
  B.addAttribute(Attribute::NoReturn);
  AL = AL.addAttributes(C, AttributeList::FunctionIndex, AttributeSet::get(C, B));
  EXPECT_TRUE(AL.hasFnAttribute(Attribute::NoReturn));
  B.clear();
  B.addAttribute(Attribute::SExt);
  AL = AL.addAttributes(C, AttributeList::ReturnIndex, B);
  EXPECT_TRUE(AL.hasAttribute(AttributeList::ReturnIndex, Attribute::SExt));
  EXPECT_TRUE(AL.hasFnAttribute(Attribute::NoReturn));
}

TEST(Attributes, AddMatchingAlignAttr) {
  LLVMContext C;
  AttributeList AL;
  AL = AL.addAttribute(C, AttributeList::FirstArgIndex,
                       Attribute::getWithAlignment(C, 8));
  AL = AL.addAttribute(C, AttributeList::FirstArgIndex + 1,
                       Attribute::getWithAlignment(C, 32));
  EXPECT_EQ(8U, AL.getParamAlignment(0));
  EXPECT_EQ(32U, AL.getParamAlignment(1));

  AttrBuilder B;
  B.addAttribute(Attribute::NonNull);
  B.addAlignmentAttr(8);
  AL = AL.addAttributes(C, AttributeList::FirstArgIndex, B);
  EXPECT_EQ(8U, AL.getParamAlignment(0));
  EXPECT_EQ(32U, AL.getParamAlignment(1));
  EXPECT_TRUE(AL.hasParamAttribute(0, Attribute::NonNull));
}

} // end anonymous namespace
