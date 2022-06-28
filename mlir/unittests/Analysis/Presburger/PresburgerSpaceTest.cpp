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
  space.insertVar(VarKind::Domain, 0, 2);
  EXPECT_EQ(space.getNumDomainVars(), 4u);

  // Try inserting 1 range ids.
  space.insertVar(VarKind::Range, 0, 1);
  EXPECT_EQ(space.getNumRangeVars(), 3u);
}

TEST(PresburgerSpaceTest, insertIdSet) {
  PresburgerSpace space = PresburgerSpace::getSetSpace(2, 1);

  // Try inserting 2 dimension ids. The space should have 4 range ids since
  // spaces which do not distinguish between domain, range are implemented like
  // this.
  space.insertVar(VarKind::SetDim, 0, 2);
  EXPECT_EQ(space.getNumRangeVars(), 4u);
}

TEST(PresburgerSpaceTest, removeIdRange) {
  PresburgerSpace space = PresburgerSpace::getRelationSpace(2, 1, 3);

  // Remove 1 domain identifier.
  space.removeVarRange(VarKind::Domain, 0, 1);
  EXPECT_EQ(space.getNumDomainVars(), 1u);

  // Remove 1 symbol and 1 range identifier.
  space.removeVarRange(VarKind::Symbol, 0, 1);
  space.removeVarRange(VarKind::Range, 0, 1);
  EXPECT_EQ(space.getNumDomainVars(), 1u);
  EXPECT_EQ(space.getNumRangeVars(), 0u);
  EXPECT_EQ(space.getNumSymbolVars(), 2u);
}

TEST(PresburgerSpaceTest, insertIdAttachment) {
  PresburgerSpace space = PresburgerSpace::getRelationSpace(2, 2, 1, 0);
  space.resetAttachments<int *>();

  // Attach attachment to domain ids.
  int attachments[2] = {0, 1};
  space.setAttachment<int *>(VarKind::Domain, 0, &attachments[0]);
  space.setAttachment<int *>(VarKind::Domain, 1, &attachments[1]);

  // Try inserting 2 domain ids.
  space.insertVar(VarKind::Domain, 0, 2);
  EXPECT_EQ(space.getNumDomainVars(), 4u);

  // Try inserting 1 range ids.
  space.insertVar(VarKind::Range, 0, 1);
  EXPECT_EQ(space.getNumRangeVars(), 3u);

  // Check if the attachments for the old ids are still attached properly.
  EXPECT_EQ(*space.getAttachment<int *>(VarKind::Domain, 2), attachments[0]);
  EXPECT_EQ(*space.getAttachment<int *>(VarKind::Domain, 3), attachments[1]);
}

TEST(PresburgerSpaceTest, removeIdRangeAttachment) {
  PresburgerSpace space = PresburgerSpace::getRelationSpace(2, 1, 3, 0);
  space.resetAttachments<int *>();

  int attachments[6] = {0, 1, 2, 3, 4, 5};

  // Attach attachments to domain identifiers.
  space.setAttachment<int *>(VarKind::Domain, 0, &attachments[0]);
  space.setAttachment<int *>(VarKind::Domain, 1, &attachments[1]);

  // Attach attachments to range identifiers.
  space.setAttachment<int *>(VarKind::Range, 0, &attachments[2]);

  // Attach attachments to symbol identifiers.
  space.setAttachment<int *>(VarKind::Symbol, 0, &attachments[3]);
  space.setAttachment<int *>(VarKind::Symbol, 1, &attachments[4]);
  space.setAttachment<int *>(VarKind::Symbol, 2, &attachments[5]);

  // Remove 1 domain identifier.
  space.removeVarRange(VarKind::Domain, 0, 1);
  EXPECT_EQ(space.getNumDomainVars(), 1u);

  // Remove 1 symbol and 1 range identifier.
  space.removeVarRange(VarKind::Symbol, 0, 1);
  space.removeVarRange(VarKind::Range, 0, 1);
  EXPECT_EQ(space.getNumDomainVars(), 1u);
  EXPECT_EQ(space.getNumRangeVars(), 0u);
  EXPECT_EQ(space.getNumSymbolVars(), 2u);

  // Check if domain attachments are attached properly.
  EXPECT_EQ(*space.getAttachment<int *>(VarKind::Domain, 0), attachments[1]);

  // Check if symbol attachments are attached properly.
  EXPECT_EQ(*space.getAttachment<int *>(VarKind::Range, 0), attachments[4]);
  EXPECT_EQ(*space.getAttachment<int *>(VarKind::Range, 1), attachments[5]);
}
