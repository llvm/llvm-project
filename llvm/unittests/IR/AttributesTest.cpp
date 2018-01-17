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

TEST(Attributes, RemoveAlign) {
  LLVMContext C;

  Attribute AlignAttr = Attribute::getWithAlignment(C, 8);
  Attribute StackAlignAttr = Attribute::getWithStackAlignment(C, 32);
  AttrBuilder B_align_readonly;
  B_align_readonly.addAttribute(AlignAttr);
  B_align_readonly.addAttribute(Attribute::ReadOnly);
  AttrBuilder B_align;
  B_align.addAttribute(AlignAttr);
  AttrBuilder B_stackalign_optnone;
  B_stackalign_optnone.addAttribute(StackAlignAttr);
  B_stackalign_optnone.addAttribute(Attribute::OptimizeNone);
  AttrBuilder B_stackalign;
  B_stackalign.addAttribute(StackAlignAttr);

  AttributeSet AS = AttributeSet::get(C, B_align_readonly);
  EXPECT_TRUE(AS.getAlignment() == 8);
  EXPECT_TRUE(AS.hasAttribute(Attribute::ReadOnly));
  AS = AS.removeAttribute(C, Attribute::Alignment);
  EXPECT_FALSE(AS.hasAttribute(Attribute::Alignment));
  EXPECT_TRUE(AS.hasAttribute(Attribute::ReadOnly));
  AS = AttributeSet::get(C, B_align_readonly);
  AS = AS.removeAttributes(C, B_align);
  EXPECT_TRUE(AS.getAlignment() == 0);
  EXPECT_TRUE(AS.hasAttribute(Attribute::ReadOnly));

  AttributeList AL;
  AL = AL.addParamAttributes(C, 0, B_align_readonly);
  AL = AL.addAttributes(C, 0, B_stackalign_optnone);
  EXPECT_TRUE(AL.hasAttributes(0));
  EXPECT_TRUE(AL.hasAttribute(0, Attribute::StackAlignment));
  EXPECT_TRUE(AL.hasAttribute(0, Attribute::OptimizeNone));
  EXPECT_TRUE(AL.getStackAlignment(0) == 32);
  EXPECT_TRUE(AL.hasParamAttrs(0));
  EXPECT_TRUE(AL.hasParamAttr(0, Attribute::Alignment));
  EXPECT_TRUE(AL.hasParamAttr(0, Attribute::ReadOnly));
  EXPECT_TRUE(AL.getParamAlignment(0) == 8);

  AL = AL.removeParamAttribute(C, 0, Attribute::Alignment);
  EXPECT_FALSE(AL.hasParamAttr(0, Attribute::Alignment));
  EXPECT_TRUE(AL.hasParamAttr(0, Attribute::ReadOnly));
  EXPECT_TRUE(AL.hasAttribute(0, Attribute::StackAlignment));
  EXPECT_TRUE(AL.hasAttribute(0, Attribute::OptimizeNone));
  EXPECT_TRUE(AL.getStackAlignment(0) == 32);

  AL = AL.removeAttribute(C, 0, Attribute::StackAlignment);
  EXPECT_FALSE(AL.hasParamAttr(0, Attribute::Alignment));
  EXPECT_TRUE(AL.hasParamAttr(0, Attribute::ReadOnly));
  EXPECT_FALSE(AL.hasAttribute(0, Attribute::StackAlignment));
  EXPECT_TRUE(AL.hasAttribute(0, Attribute::OptimizeNone));

  AttributeList AL2;
  AL2 = AL2.addParamAttributes(C, 0, B_align_readonly);
  AL2 = AL2.addAttributes(C, 0, B_stackalign_optnone);

  AL2 = AL2.removeParamAttributes(C, 0, B_align);
  EXPECT_FALSE(AL2.hasParamAttr(0, Attribute::Alignment));
  EXPECT_TRUE(AL2.hasParamAttr(0, Attribute::ReadOnly));
  EXPECT_TRUE(AL2.hasAttribute(0, Attribute::StackAlignment));
  EXPECT_TRUE(AL2.hasAttribute(0, Attribute::OptimizeNone));
  EXPECT_TRUE(AL2.getStackAlignment(0) == 32);

  AL2 = AL2.removeAttributes(C, 0, B_stackalign);
  EXPECT_FALSE(AL2.hasParamAttr(0, Attribute::Alignment));
  EXPECT_TRUE(AL2.hasParamAttr(0, Attribute::ReadOnly));
  EXPECT_FALSE(AL2.hasAttribute(0, Attribute::StackAlignment));
  EXPECT_TRUE(AL2.hasAttribute(0, Attribute::OptimizeNone));
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

TEST(Attributes, EmptyGet) {
  LLVMContext C;
  AttributeList EmptyLists[] = {AttributeList(), AttributeList()};
  AttributeList AL = AttributeList::get(C, EmptyLists);
  EXPECT_TRUE(AL.isEmpty());
}

} // end anonymous namespace
