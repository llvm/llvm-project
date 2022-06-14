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
using namespace presburger;

TEST(PresburgerSpaceTest, insertId) {
  PresburgerSpace space = PresburgerSpace::getRelationSpace(2, 2, 1);

  // Try inserting 2 domain ids.
  space.insertId(IdKind::Domain, 0, 2);
  EXPECT_EQ(space.getNumDomainIds(), 4u);

  // Try inserting 1 range ids.
  space.insertId(IdKind::Range, 0, 1);
  EXPECT_EQ(space.getNumRangeIds(), 3u);
}

TEST(PresburgerSpaceTest, insertIdSet) {
  PresburgerSpace space = PresburgerSpace::getSetSpace(2, 1);

  // Try inserting 2 dimension ids. The space should have 4 range ids since
  // spaces which do not distinguish between domain, range are implemented like
  // this.
  space.insertId(IdKind::SetDim, 0, 2);
  EXPECT_EQ(space.getNumRangeIds(), 4u);
}

TEST(PresburgerSpaceTest, removeIdRange) {
  PresburgerSpace space = PresburgerSpace::getRelationSpace(2, 1, 3);

  // Remove 1 domain identifier.
  space.removeIdRange(IdKind::Domain, 0, 1);
  EXPECT_EQ(space.getNumDomainIds(), 1u);

  // Remove 1 symbol and 1 range identifier.
  space.removeIdRange(IdKind::Symbol, 0, 1);
  space.removeIdRange(IdKind::Range, 0, 1);
  EXPECT_EQ(space.getNumDomainIds(), 1u);
  EXPECT_EQ(space.getNumRangeIds(), 0u);
  EXPECT_EQ(space.getNumSymbolIds(), 2u);
}

TEST(PresburgerSpaceTest, insertIdAttachement) {
  PresburgerSpace space = PresburgerSpace::getRelationSpace(2, 2, 1, 0);
  space.resetAttachements<int *>();

  // Attach attachement to domain ids.
  int attachements[2] = {0, 1};
  space.setAttachement<int *>(IdKind::Domain, 0, &attachements[0]);
  space.setAttachement<int *>(IdKind::Domain, 1, &attachements[1]);

  // Try inserting 2 domain ids.
  space.insertId(IdKind::Domain, 0, 2);
  EXPECT_EQ(space.getNumDomainIds(), 4u);

  // Try inserting 1 range ids.
  space.insertId(IdKind::Range, 0, 1);
  EXPECT_EQ(space.getNumRangeIds(), 3u);

  // Check if the attachements for the old ids are still attached properly.
  EXPECT_EQ(*space.getAttachement<int *>(IdKind::Domain, 2), attachements[0]);
  EXPECT_EQ(*space.getAttachement<int *>(IdKind::Domain, 3), attachements[1]);
}

TEST(PresburgerSpaceTest, removeIdRangeAttachement) {
  PresburgerSpace space = PresburgerSpace::getRelationSpace(2, 1, 3, 0);
  space.resetAttachements<int *>();

  int attachements[6] = {0, 1, 2, 3, 4, 5};

  // Attach attachements to domain identifiers.
  space.setAttachement<int *>(IdKind::Domain, 0, &attachements[0]);
  space.setAttachement<int *>(IdKind::Domain, 1, &attachements[1]);

  // Attach attachements to range identifiers.
  space.setAttachement<int *>(IdKind::Range, 0, &attachements[2]);

  // Attach attachements to symbol identifiers.
  space.setAttachement<int *>(IdKind::Symbol, 0, &attachements[3]);
  space.setAttachement<int *>(IdKind::Symbol, 1, &attachements[4]);
  space.setAttachement<int *>(IdKind::Symbol, 2, &attachements[5]);

  // Remove 1 domain identifier.
  space.removeIdRange(IdKind::Domain, 0, 1);
  EXPECT_EQ(space.getNumDomainIds(), 1u);

  // Remove 1 symbol and 1 range identifier.
  space.removeIdRange(IdKind::Symbol, 0, 1);
  space.removeIdRange(IdKind::Range, 0, 1);
  EXPECT_EQ(space.getNumDomainIds(), 1u);
  EXPECT_EQ(space.getNumRangeIds(), 0u);
  EXPECT_EQ(space.getNumSymbolIds(), 2u);

  // Check if domain attachements are attached properly.
  EXPECT_EQ(*space.getAttachement<int *>(IdKind::Domain, 0), attachements[1]);

  // Check if symbol attachements are attached properly.
  EXPECT_EQ(*space.getAttachement<int *>(IdKind::Range, 0), attachements[4]);
  EXPECT_EQ(*space.getAttachement<int *>(IdKind::Range, 1), attachements[5]);
}
