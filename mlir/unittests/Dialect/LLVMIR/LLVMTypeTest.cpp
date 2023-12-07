//===- LLVMTypeTest.cpp - Tests for LLVM types ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLVMTestBase.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

using namespace mlir;
using namespace mlir::LLVM;

TEST_F(LLVMIRTest, IsStructTypeMutable) {
  auto structTy = LLVMStructType::getIdentified(&context, "foo");
  ASSERT_TRUE(bool(structTy));
  ASSERT_TRUE(structTy.hasTrait<TypeTrait::IsMutable>());
}

TEST_F(LLVMIRTest, MutualReferencedSubElementTypes) {
  auto fooStructTy = LLVMStructType::getIdentified(&context, "foo");
  ASSERT_TRUE(bool(fooStructTy));
  auto barStructTy = LLVMStructType::getIdentified(&context, "bar");
  ASSERT_TRUE(bool(barStructTy));

  // Created two structs that are referencing each other.
  Type fooBody[] = {LLVMPointerType::get(barStructTy)};
  ASSERT_TRUE(succeeded(fooStructTy.setBody(fooBody, /*isPacked=*/false)));
  Type barBody[] = {LLVMPointerType::get(fooStructTy)};
  ASSERT_TRUE(succeeded(barStructTy.setBody(barBody, /*isPacked=*/false)));

  // Test if walkSubElements goes into infinite loops.
  SmallVector<Type, 4> subElementTypes;
  fooStructTy.walk([&](Type type) { subElementTypes.push_back(type); });
  ASSERT_EQ(subElementTypes.size(), 4U);

  // !llvm.ptr<struct<"foo",...>>
  ASSERT_TRUE(isa<LLVMPointerType>(subElementTypes[0]));

  // !llvm.struct<"bar",...>
  auto structType = dyn_cast<LLVMStructType>(subElementTypes[1]);
  ASSERT_TRUE(bool(structType));
  ASSERT_TRUE(structType.getName().equals("bar"));

  // !llvm.ptr<struct<"bar",...>>
  ASSERT_TRUE(isa<LLVMPointerType>(subElementTypes[2]));

  // !llvm.struct<"foo",...>
  structType = dyn_cast<LLVMStructType>(subElementTypes[3]);
  ASSERT_TRUE(bool(structType));
  ASSERT_TRUE(structType.getName().equals("foo"));
}
