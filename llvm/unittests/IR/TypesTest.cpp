//===- llvm/unittest/IR/TypesTest.cpp - Type unit tests -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "gtest/gtest.h"
using namespace llvm;

namespace {

TEST(TypesTest, StructType) {
  LLVMContext C;

  // PR13522
  StructType *Struct = StructType::create(C, "FooBar");
  EXPECT_EQ("FooBar", Struct->getName());
  Struct->setName(Struct->getName().substr(0, 3));
  EXPECT_EQ("Foo", Struct->getName());
  Struct->setName("");
  EXPECT_TRUE(Struct->getName().empty());
  EXPECT_FALSE(Struct->hasName());
}

TEST(TypesTest, LayoutIdenticalEmptyStructs) {
  LLVMContext C;

  StructType *Foo = StructType::create(C, "Foo");
  StructType *Bar = StructType::create(C, "Bar");
  EXPECT_TRUE(Foo->isLayoutIdentical(Bar));
}

}  // end anonymous namespace
