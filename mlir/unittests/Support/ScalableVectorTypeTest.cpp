//===- ScalableVectorTypeTest.cpp - ScalableVectorType Tests --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/ScalableVectorType.h"
#include "mlir/IR/Dialect.h"
#include "gtest/gtest.h"

using namespace mlir;

TEST(ScalableVectorTypeTest, TestVectorDim) {
  auto fixedDim = VectorDim::getFixed(4);
  ASSERT_FALSE(fixedDim.isScalable());
  ASSERT_TRUE(fixedDim.isFixed());
  ASSERT_EQ(fixedDim.getFixedSize(), 4);

  auto scalableDim = VectorDim::getScalable(8);
  ASSERT_TRUE(scalableDim.isScalable());
  ASSERT_FALSE(scalableDim.isFixed());
  ASSERT_EQ(scalableDim.getMinSize(), 8);
}

TEST(ScalableVectorTypeTest, BasicFunctionality) {
  MLIRContext context;

  Type f32 = FloatType::getF32(&context);

  // Construct n-D scalable vector.
  VectorType scalableVector = ScalableVectorType::get(
      {VectorDim::getFixed(1), VectorDim::getFixed(2),
       VectorDim::getScalable(3), VectorDim::getFixed(4),
       VectorDim::getScalable(5)},
      f32);
  // Construct fixed vector.
  VectorType fixedVector = ScalableVectorType::get(VectorDim::getFixed(1), f32);

  // Check casts.
  ASSERT_TRUE(isa<ScalableVectorType>(scalableVector));
  ASSERT_FALSE(isa<ScalableVectorType>(fixedVector));
  ASSERT_FALSE(VectorDimList::from(fixedVector).hasScalableDims());

  // Check rank/size.
  auto vType = cast<ScalableVectorType>(scalableVector);
  ASSERT_EQ(vType.getDims().size(), unsigned(scalableVector.getRank()));
  ASSERT_TRUE(vType.getDims().hasScalableDims());

  // Check iterating over dimensions.
  std::array expectedDims{VectorDim::getFixed(1), VectorDim::getFixed(2),
                          VectorDim::getScalable(3), VectorDim::getFixed(4),
                          VectorDim::getScalable(5)};
  unsigned i = 0;
  for (VectorDim dim : vType.getDims()) {
    ASSERT_EQ(dim, expectedDims[i]);
    i++;
  }
}

TEST(ScalableVectorTypeTest, VectorDimListHelpers) {
  std::array<int64_t, 4> sizes{42, 10, 3, 1};
  std::array<bool, 4> scalableFlags{false, true, false, true};

  // Manually construct from sizes + flags.
  VectorDimList dimList(sizes, scalableFlags);

  ASSERT_EQ(dimList.size(), 4U);

  ASSERT_EQ(dimList.front(), VectorDim::getFixed(42));
  ASSERT_EQ(dimList.back(), VectorDim::getScalable(1));

  std::array innerDims{VectorDim::getScalable(10), VectorDim::getFixed(3)};
  ASSERT_EQ(dimList.slice(1, 2), innerDims);
}
