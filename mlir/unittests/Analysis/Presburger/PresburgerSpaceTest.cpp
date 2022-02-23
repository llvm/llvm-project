//===- PresburgerSpaceTest.cpp - Tests for PresburgerSpace ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using IdKind = PresburgerSpace::IdKind;

TEST(PresburgerSpaceTest, insertId) {
  PresburgerSpace space = PresburgerSpace::getRelationSpace(2, 2, 1);

  // Try inserting 2 domain ids.
  space.insertId(PresburgerSpace::IdKind::Domain, 0, 2);
  EXPECT_EQ(space.getNumDomainIds(), 4u);

  // Try inserting 1 range ids.
  space.insertId(PresburgerSpace::IdKind::Range, 0, 1);
  EXPECT_EQ(space.getNumRangeIds(), 3u);
}

TEST(PresburgerSpaceTest, insertIdSet) {
  PresburgerSpace space = PresburgerSpace::getSetSpace(2, 1);

  // Try inserting 2 dimension ids. The space should have 4 range ids since
  // spaces which do not distinguish between domain, range are implemented like
  // this.
  space.insertId(PresburgerSpace::IdKind::SetDim, 0, 2);
  EXPECT_EQ(space.getNumRangeIds(), 4u);
}

TEST(PresburgerSpaceTest, removeIdRange) {
  PresburgerSpace space = PresburgerSpace::getRelationSpace(2, 1, 3);

  // Remove 1 domain identifier.
  space.removeIdRange(0, 1);
  EXPECT_EQ(space.getNumDomainIds(), 1u);

  // Remove 1 symbol and 1 range identifier.
  space.removeIdRange(1, 3);
  EXPECT_EQ(space.getNumDomainIds(), 1u);
  EXPECT_EQ(space.getNumRangeIds(), 0u);
  EXPECT_EQ(space.getNumSymbolIds(), 2u);
}
