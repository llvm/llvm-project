//===- llvm/unittest/IR/TypesTest.cpp - Type unit tests -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/TypedPointerType.h"
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

TEST(TypesTest, TargetExtType) {
  LLVMContext Context;
  Type *A = TargetExtType::get(Context, "typea");
  Type *Aparam = TargetExtType::get(Context, "typea", {}, {0, 1});
  Type *Aparam2 = TargetExtType::get(Context, "typea", {}, {0, 1});
  // Opaque types with same parameters are identical...
  EXPECT_EQ(Aparam, Aparam2);
  // ... but just having the same name is not enough.
  EXPECT_NE(A, Aparam);
}

TEST(TypedPointerType, PrintTest) {
  std::string Buffer;
  LLVMContext Context;
  raw_string_ostream OS(Buffer);

  Type *I8Ptr = TypedPointerType::get(Type::getInt8Ty(Context), 0);
  I8Ptr->print(OS);
  EXPECT_EQ(StringRef(Buffer), ("typedptr(i8, 0)"));
}

}  // end anonymous namespace
