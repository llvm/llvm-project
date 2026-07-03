//===- ReshapeOpsUtilsTest.cpp - ReshapeOpsUtils unit tests ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"
#include <optional>

using namespace mlir;

/// Helper to make constructing
/// `std::optional<SmallVector<ReassociationIndices>>` more readable.
static std::optional<SmallVector<ReassociationIndices>>
makeOptionalIndices(std::initializer_list<ReassociationIndices> list) {
  return std::optional<SmallVector<ReassociationIndices>>(list);
}

TEST(ReassociationIndicesForCollapse, ScalarTest) {
  EXPECT_EQ(getReassociationIndicesForCollapse({1}, {}),
            makeOptionalIndices({}));
  EXPECT_EQ(getReassociationIndicesForCollapse({1, 1}, {}),
            makeOptionalIndices({}));
  EXPECT_EQ(getReassociationIndicesForCollapse({ShapedType::kDynamic}, {}),
            makeOptionalIndices({}));
  EXPECT_EQ(getReassociationIndicesForCollapse({1, ShapedType::kDynamic,
                                                ShapedType::kDynamic, 1,
                                                ShapedType::kDynamic},
                                               {}),
            makeOptionalIndices({}));
}

TEST(ReassociationIndicesForCollapse, ScalarTestFailure) {
  EXPECT_EQ(getReassociationIndicesForCollapse({}, {}), std::nullopt);
  EXPECT_EQ(getReassociationIndicesForCollapse({}, {1}), std::nullopt);
  EXPECT_EQ(getReassociationIndicesForCollapse({2}, {}), std::nullopt);
  EXPECT_EQ(
      getReassociationIndicesForCollapse({1, 2, ShapedType::kDynamic, 1}, {}),
      std::nullopt);
}

TEST(ReassociationIndicesForCollapse, StaticTest) {
  EXPECT_EQ(getReassociationIndicesForCollapse({10, 20}, {200}),
            makeOptionalIndices({{0, 1}}));
  EXPECT_EQ(getReassociationIndicesForCollapse({10, 20, 30}, {10, 600}),
            makeOptionalIndices({{0}, {1, 2}}));
  EXPECT_EQ(getReassociationIndicesForCollapse({10, 20, 30}, {200, 30}),
            makeOptionalIndices({{0, 1}, {2}}));
}

TEST(ReassociationIndicesForCollapse, StaticTestFailure) {
  // No-op reassociation
  EXPECT_EQ(getReassociationIndicesForCollapse({10, 20}, {10, 20}),
            std::nullopt);
  // Invalid static reassociations
  EXPECT_EQ(getReassociationIndicesForCollapse({10, 20}, {10}), std::nullopt);
  EXPECT_EQ(getReassociationIndicesForCollapse({10, 20, 30}, {200, 300}),
            std::nullopt);
  // Non-collapsing (expanding) reassociation
  EXPECT_EQ(getReassociationIndicesForCollapse({10, 20, 30}, {1, 10, 20, 30}),
            std::nullopt);
}

TEST(ReassociationIndicesForCollapse, StaticTestUnitDims) {
  EXPECT_EQ(getReassociationIndicesForCollapse({10, 1}, {10}),
            makeOptionalIndices({{0, 1}}));
  EXPECT_EQ(getReassociationIndicesForCollapse({1, 20, 30}, {600}),
            makeOptionalIndices({{0, 1, 2}}));
  EXPECT_EQ(getReassociationIndicesForCollapse({1, 1, 1}, {1}),
            makeOptionalIndices({{0, 1, 2}}));
  EXPECT_EQ(getReassociationIndicesForCollapse({1, 1, 1, 1}, {1, 1, 1}),
            makeOptionalIndices({{0}, {1}, {2, 3}}));
}

TEST(ReassociationIndicesForCollapse, DynamicTest) {
  EXPECT_EQ(getReassociationIndicesForCollapse({ShapedType::kDynamic, 1},
                                               {ShapedType::kDynamic}),
            makeOptionalIndices({{0, 1}}));
  EXPECT_EQ(getReassociationIndicesForCollapse({ShapedType::kDynamic, 1, 1},
                                               {ShapedType::kDynamic}),
            makeOptionalIndices({{0, 1, 2}}));
  EXPECT_EQ(getReassociationIndicesForCollapse(
                {1, ShapedType::kDynamic, 1, ShapedType::kDynamic, 1},
                {ShapedType::kDynamic, ShapedType::kDynamic}),
            makeOptionalIndices({{0, 1}, {2, 3, 4}}));
  EXPECT_EQ(
      getReassociationIndicesForCollapse(
          {ShapedType::kDynamic, ShapedType::kDynamic}, {ShapedType::kDynamic}),
      makeOptionalIndices({{0, 1}}));
  EXPECT_EQ(getReassociationIndicesForCollapse(
                {1, ShapedType::kDynamic, ShapedType::kDynamic},
                {1, ShapedType::kDynamic}),
            makeOptionalIndices({{0}, {1, 2}}));

  EXPECT_EQ(getReassociationIndicesForCollapse({ShapedType::kDynamic, 10},
                                               {ShapedType::kDynamic}),
            makeOptionalIndices({{0, 1}}));
  EXPECT_EQ(getReassociationIndicesForCollapse(
                {1, ShapedType::kDynamic, ShapedType::kDynamic},
                {ShapedType::kDynamic}),
            makeOptionalIndices({{0, 1, 2}}));
  EXPECT_EQ(getReassociationIndicesForCollapse({10, ShapedType::kDynamic},
                                               {ShapedType::kDynamic}),
            makeOptionalIndices({{0, 1}}));
  EXPECT_EQ(getReassociationIndicesForCollapse(
                {ShapedType::kDynamic, 1, 2, ShapedType::kDynamic, 10},
                {ShapedType::kDynamic, 10}),
            makeOptionalIndices({{0, 1, 2, 3}, {4}}));
  EXPECT_EQ(getReassociationIndicesForCollapse({ShapedType::kDynamic, 10, 20},
                                               {ShapedType::kDynamic, 20}),
            makeOptionalIndices({{0, 1}, {2}}));
  EXPECT_EQ(getReassociationIndicesForCollapse({10, ShapedType::kDynamic, 20},
                                               {ShapedType::kDynamic, 20}),
            makeOptionalIndices({{0, 1}, {2}}));
  EXPECT_EQ(getReassociationIndicesForCollapse(
                {ShapedType::kDynamic, 3, 2, 5, 2}, {ShapedType::kDynamic, 20}),
            makeOptionalIndices({{0, 1}, {2, 3, 4}}));
  EXPECT_EQ(getReassociationIndicesForCollapse(
                {10, ShapedType::kDynamic, 20, ShapedType::kDynamic, 1},
                {ShapedType::kDynamic, 20, ShapedType::kDynamic}),
            makeOptionalIndices({{0, 1}, {2}, {3, 4}}));
  EXPECT_EQ(getReassociationIndicesForCollapse({1, ShapedType::kDynamic, 1},
                                               {ShapedType::kDynamic}),
            makeOptionalIndices({{0, 1, 2}}));
  EXPECT_EQ(getReassociationIndicesForCollapse(
                {ShapedType::kDynamic, ShapedType::kDynamic, 1},
                {ShapedType::kDynamic, ShapedType::kDynamic}),
            makeOptionalIndices({{0}, {1, 2}}));
  EXPECT_EQ(getReassociationIndicesForCollapse(
                {1, ShapedType::kDynamic, ShapedType::kDynamic},
                {ShapedType::kDynamic, ShapedType::kDynamic}),
            makeOptionalIndices({{0, 1}, {2}}));
  EXPECT_EQ(getReassociationIndicesForCollapse(
                {ShapedType::kDynamic, 1, ShapedType::kDynamic},
                {ShapedType::kDynamic, ShapedType::kDynamic}),
            makeOptionalIndices({{0}, {1, 2}}));
}

TEST(ReassociationIndicesForCollapse, DynamicTestFailure) {
  EXPECT_EQ(getReassociationIndicesForCollapse({ShapedType::kDynamic, 10, 20},
                                               {ShapedType::kDynamic, 10}),
            std::nullopt);
  EXPECT_EQ(getReassociationIndicesForCollapse(
                {ShapedType::kDynamic, 10, ShapedType::kDynamic},
                {ShapedType::kDynamic, ShapedType::kDynamic}),
            std::nullopt);
  EXPECT_EQ(getReassociationIndicesForCollapse(
                {20, ShapedType::kDynamic, 10, ShapedType::kDynamic},
                {ShapedType::kDynamic, ShapedType::kDynamic}),
            std::nullopt);
  EXPECT_EQ(getReassociationIndicesForCollapse(
                {ShapedType::kDynamic, 5, 3, 2, 2}, {ShapedType::kDynamic, 20}),
            std::nullopt);
  EXPECT_EQ(
      getReassociationIndicesForCollapse(
          {ShapedType::kDynamic, ShapedType::kDynamic, ShapedType::kDynamic},
          {ShapedType::kDynamic, ShapedType::kDynamic}),
      std::nullopt);
  EXPECT_EQ(getReassociationIndicesForCollapse(
                {ShapedType::kDynamic, ShapedType::kDynamic, 10, 1,
                 ShapedType::kDynamic},
                {ShapedType::kDynamic, ShapedType::kDynamic}),
            std::nullopt);
  EXPECT_EQ(getReassociationIndicesForCollapse(
                {ShapedType::kDynamic, 10, 10, 10, ShapedType::kDynamic},
                {ShapedType::kDynamic, 10, ShapedType::kDynamic}),
            std::nullopt);
  EXPECT_EQ(getReassociationIndicesForCollapse(
                {ShapedType::kDynamic, 10, 10, 10, ShapedType::kDynamic},
                {ShapedType::kDynamic, 2, 2, ShapedType::kDynamic}),
            std::nullopt);
  EXPECT_EQ(getReassociationIndicesForCollapse(
                {ShapedType::kDynamic, 3, 4, 3, ShapedType::kDynamic},
                {ShapedType::kDynamic, 12, ShapedType::kDynamic}),
            std::nullopt);
  EXPECT_EQ(getReassociationIndicesForCollapse(
                {ShapedType::kDynamic, 8, 4, 2, 16, ShapedType::kDynamic},
                {ShapedType::kDynamic, 32, ShapedType::kDynamic}),
            std::nullopt);

  //===----------------------------------------------------------------------===//
  // TODO: Reassociation for the following examples can be computed, but isn't
  // supported by `getReassociationIndicesForCollapse`.
  //===----------------------------------------------------------------------===//

  // TODO: Fails because there's no backtracking when some source dimensions
  // remain unmatched at either edge.
  EXPECT_EQ(getReassociationIndicesForCollapse(
                {ShapedType::kDynamic, 10, ShapedType::kDynamic, 10},
                {ShapedType::kDynamic, 10}),
            std::nullopt);
  EXPECT_EQ(getReassociationIndicesForCollapse({1, ShapedType::kDynamic, 2, 2},
                                               {1, ShapedType::kDynamic, 2}),
            std::nullopt);
  EXPECT_EQ(getReassociationIndicesForCollapse({2, 2, ShapedType::kDynamic, 1},
                                               {2, ShapedType::kDynamic}),
            std::nullopt);
}
