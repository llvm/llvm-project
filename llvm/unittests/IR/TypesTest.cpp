//===- llvm/unittest/IR/TypesTest.cpp - Type unit tests -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

TEST(TypesTest, CopyPointerType) {
  LLVMContext COpaquePointers;
  COpaquePointers.enableOpaquePointers();

  PointerType *P1 = PointerType::get(COpaquePointers, 1);
  EXPECT_TRUE(P1->isOpaque());
  PointerType *P1C = PointerType::getWithSamePointeeType(P1, 1);
  EXPECT_EQ(P1, P1C);
  EXPECT_TRUE(P1C->isOpaque());
  PointerType *P1C0 = PointerType::getWithSamePointeeType(P1, 0);
  EXPECT_NE(P1, P1C0);
  EXPECT_TRUE(P1C0->isOpaque());

  LLVMContext CTypedPointers;
  Type *Int8 = Type::getInt8Ty(CTypedPointers);
  PointerType *P2 = PointerType::get(Int8, 1);
  EXPECT_FALSE(P2->isOpaque());
  PointerType *P2C = PointerType::getWithSamePointeeType(P2, 1);
  EXPECT_EQ(P2, P2C);
  EXPECT_FALSE(P2C->isOpaque());
  PointerType *P2C0 = PointerType::getWithSamePointeeType(P2, 0);
  EXPECT_NE(P2, P2C0);
  EXPECT_FALSE(P2C0->isOpaque());
}

}  // end anonymous namespace
