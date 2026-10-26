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

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG)
using LLVMIRDeathTest = LLVMIRTest;

TEST_F(LLVMIRDeathTest, StructTypeGetVerifiesElementTypes) {
  EXPECT_DEATH(
      LLVMStructType::getLiteral(&context, {LLVMVoidType::get(&context)}),
      "invalid LLVM structure element type");
}
#endif

TEST_F(LLVMIRTest, StructElementOffsets) {
  DataLayout dataLayout;
  Type i8 = IntegerType::get(&context, 8);
  Type i32 = IntegerType::get(&context, 32);
  Type f64 = Float64Type::get(&context);
  Type f80 = Float80Type::get(&context);

  // Check ABI alignment for unpacked structures.
  auto unpacked = LLVMStructType::getLiteral(&context, {i8, f64, i32});
  EXPECT_EQ(unpacked.getElementOffset(dataLayout, 0), 0u);
  EXPECT_EQ(unpacked.getElementOffset(dataLayout, 1), 8u);
  EXPECT_EQ(unpacked.getElementOffset(dataLayout, 2), 16u);
  EXPECT_EQ(dataLayout.getTypeSize(unpacked), 24u);

  // Check no alignment for packed structures.
  auto packed = LLVMStructType::getLiteral(&context, {i8, f64, i32},
                                           /*isPacked=*/true);
  EXPECT_EQ(packed.getElementOffset(dataLayout, 0), 0u);
  EXPECT_EQ(packed.getElementOffset(dataLayout, 1), 1u);
  EXPECT_EQ(packed.getElementOffset(dataLayout, 2), 9u);
  EXPECT_EQ(dataLayout.getTypeSize(packed), 13u);

  // Check that fields get their allocation size, even in packed
  // structures.
  auto alloc = LLVMStructType::getLiteral(&context, {f80, i8});
  EXPECT_EQ(alloc.getElementOffset(dataLayout, 1), 16u);
  EXPECT_EQ(dataLayout.getTypeSize(alloc), 32u);
  auto packedAlloc =
      LLVMStructType::getLiteral(&context, {f80, i8}, /*isPacked=*/true);
  EXPECT_EQ(packedAlloc.getElementOffset(dataLayout, 1), 16u);
  EXPECT_EQ(dataLayout.getTypeSize(packedAlloc), 17u);
}
