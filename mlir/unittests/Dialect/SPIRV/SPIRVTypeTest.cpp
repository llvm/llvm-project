//===- SPIRVTypeTest.cpp - SPIR-V Type Tests ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace mlir;

TEST(SPIRVTypeTest, ArraySizeInBytes) {
  MLIRContext context;
  context.loadDialect<spirv::SPIRVDialect>();
  Builder b(&context);
  Type f32 = b.getF32Type();

  // Tightly packed array: size is element size times element count.
  auto array = spirv::ArrayType::get(f32, 16);
  EXPECT_EQ(array.getSizeInBytes(), std::optional<int64_t>(64));

  // Explicitly strided array: stride is the per-element byte distance, so it
  // already accounts for the element size and must not be added to it.
  auto stridedArray = spirv::ArrayType::get(f32, 16, /*stride=*/4);
  EXPECT_EQ(stridedArray.getSizeInBytes(), std::optional<int64_t>(64));

  // Padded array: stride larger than the element size includes the padding.
  auto paddedArray = spirv::ArrayType::get(f32, 16, /*stride=*/8);
  EXPECT_EQ(paddedArray.getSizeInBytes(), std::optional<int64_t>(128));
}

TEST(SPIRVTypeTest, ArrayOfVectorSizeInBytes) {
  MLIRContext context;
  context.loadDialect<spirv::SPIRVDialect>();
  Builder b(&context);
  auto vec4f32 = VectorType::get({4}, b.getF32Type());

  // Tightly packed array of vectors: 4 * 4 bytes per element, 8 elements.
  auto array = spirv::ArrayType::get(vec4f32, 8);
  EXPECT_EQ(array.getSizeInBytes(), std::optional<int64_t>(128));

  // Strided array of vectors: stride already covers the whole vector element.
  auto stridedArray = spirv::ArrayType::get(vec4f32, 8, /*stride=*/16);
  EXPECT_EQ(stridedArray.getSizeInBytes(), std::optional<int64_t>(128));
}

TEST(SPIRVTypeTest, NestedArraySizeInBytes) {
  MLIRContext context;
  context.loadDialect<spirv::SPIRVDialect>();
  Builder b(&context);
  Type f32 = b.getF32Type();

  // Array of tightly packed arrays: inner is 4 * 4 = 16 bytes, outer has 3.
  auto inner = spirv::ArrayType::get(f32, 4);
  auto outer = spirv::ArrayType::get(inner, 3);
  EXPECT_EQ(outer.getSizeInBytes(), std::optional<int64_t>(48));

  // Outer stride dominates and includes any inner padding.
  auto stridedOuter = spirv::ArrayType::get(inner, 3, /*stride=*/32);
  EXPECT_EQ(stridedOuter.getSizeInBytes(), std::optional<int64_t>(96));
}
