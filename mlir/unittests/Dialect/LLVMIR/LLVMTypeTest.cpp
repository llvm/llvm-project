//===- LLVMTypeTest.cpp - Tests for LLVM types ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLVMTestBase.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/SubElementInterfaces.h"

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

  auto subElementInterface = fooStructTy.dyn_cast<SubElementTypeInterface>();
  ASSERT_TRUE(bool(subElementInterface));
  // Test if walkSubElements goes into infinite loops.
  SmallVector<Type, 4> subElementTypes;
  subElementInterface.walkSubElements(
      [](Attribute attr) {},
      [&](Type type) { subElementTypes.push_back(type); });
  // We don't record LLVMPointerType (because it's immutable), thus
  // !llvm.ptr<struct<"bar",...>> will be visited twice.
  ASSERT_EQ(subElementTypes.size(), 5U);

  // !llvm.ptr<struct<"bar",...>>
  ASSERT_TRUE(subElementTypes[0].isa<LLVMPointerType>());

  // !llvm.struct<"foo",...>
  auto structType = subElementTypes[1].dyn_cast<LLVMStructType>();
  ASSERT_TRUE(bool(structType));
  ASSERT_TRUE(structType.getName().equals("foo"));

  // !llvm.ptr<struct<"foo",...>>
  ASSERT_TRUE(subElementTypes[2].isa<LLVMPointerType>());

  // !llvm.struct<"bar",...>
  structType = subElementTypes[3].dyn_cast<LLVMStructType>();
  ASSERT_TRUE(bool(structType));
  ASSERT_TRUE(structType.getName().equals("bar"));

  // !llvm.ptr<struct<"bar",...>>
  ASSERT_TRUE(subElementTypes[4].isa<LLVMPointerType>());
}
