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

TEST(PresburgerSpaceTest, insertVarIdentifier) {
  PresburgerSpace space = PresburgerSpace::getRelationSpace(2, 2, 1, 0);
  space.resetIds();

  int identifiers[2] = {0, 1};

  // Attach identifiers to domain ids.
  space.getId(VarKind::Domain, 0) = Identifier(&identifiers[0]);
  space.getId(VarKind::Domain, 1) = Identifier(&identifiers[1]);

  // Try inserting 2 domain ids.
  space.insertVar(VarKind::Domain, 0, 2);
  EXPECT_EQ(space.getNumDomainVars(), 4u);

  // Try inserting 1 range ids.
  space.insertVar(VarKind::Range, 0, 1);
  EXPECT_EQ(space.getNumRangeVars(), 3u);

  // Check if the identifiers for the old ids are still attached properly.
  EXPECT_EQ(space.getId(VarKind::Domain, 2), Identifier(&identifiers[0]));
  EXPECT_EQ(space.getId(VarKind::Domain, 3), Identifier(&identifiers[1]));
}

TEST(PresburgerSpaceTest, removeVarRangeIdentifier) {
  PresburgerSpace space = PresburgerSpace::getRelationSpace(2, 1, 3, 0);
  space.resetIds();

  int identifiers[6] = {0, 1, 2, 3, 4, 5};

  // Attach identifiers to domain identifiers.
  space.getId(VarKind::Domain, 0) = Identifier(&identifiers[0]);
  space.getId(VarKind::Domain, 1) = Identifier(&identifiers[1]);

  // Attach identifiers to range identifiers.
  space.getId(VarKind::Range, 0) = Identifier(&identifiers[2]);

  // Attach identifiers to symbol identifiers.
  space.getId(VarKind::Symbol, 0) = Identifier(&identifiers[3]);
  space.getId(VarKind::Symbol, 1) = Identifier(&identifiers[4]);
  space.getId(VarKind::Symbol, 2) = Identifier(&identifiers[5]);

  // Remove 1 domain identifier.
  space.removeVarRange(VarKind::Domain, 0, 1);
  EXPECT_EQ(space.getNumDomainVars(), 1u);

  // Remove 1 symbol and 1 range identifier.
  space.removeVarRange(VarKind::Symbol, 0, 1);
  space.removeVarRange(VarKind::Range, 0, 1);
  EXPECT_EQ(space.getNumDomainVars(), 1u);
  EXPECT_EQ(space.getNumRangeVars(), 0u);
  EXPECT_EQ(space.getNumSymbolVars(), 2u);

  // Check if domain identifiers are attached properly.
  EXPECT_EQ(space.getId(VarKind::Domain, 0), Identifier(&identifiers[1]));

  // Check if symbol identifiers are attached properly.
  EXPECT_EQ(space.getId(VarKind::Range, 0), Identifier(&identifiers[4]));
  EXPECT_EQ(space.getId(VarKind::Range, 1), Identifier(&identifiers[5]));
}

TEST(PresburgerSpaceTest, IdentifierIsEqual) {
  PresburgerSpace space = PresburgerSpace::getRelationSpace(1, 2, 0, 0);
  space.resetIds();

  int identifiers[2] = {0, 1};
  space.getId(VarKind::Domain, 0) = Identifier(&identifiers[0]);
  space.getId(VarKind::Range, 0) = Identifier(&identifiers[0]);
  space.getId(VarKind::Range, 1) = Identifier(&identifiers[1]);

  EXPECT_EQ(space.getId(VarKind::Domain, 0), space.getId(VarKind::Range, 0));
  EXPECT_FALSE(
      space.getId(VarKind::Range, 0).isEqual(space.getId(VarKind::Range, 1)));
}

TEST(PresburgerSpaceTest, convertVarKind) {
  PresburgerSpace space = PresburgerSpace::getRelationSpace(2, 2, 0, 0);
  space.resetIds();

  // Attach identifiers.
  int identifiers[4] = {0, 1, 2, 3};
  space.getId(VarKind::Domain, 0) = Identifier(&identifiers[0]);
  space.getId(VarKind::Domain, 1) = Identifier(&identifiers[1]);
  space.getId(VarKind::Range, 0) = Identifier(&identifiers[2]);
  space.getId(VarKind::Range, 1) = Identifier(&identifiers[3]);

  // Convert Range variables to symbols.
  space.convertVarKind(VarKind::Range, 0, 2, VarKind::Symbol, 0);

  // Check if the identifiers are moved to symbols.
  EXPECT_EQ(space.getId(VarKind::Symbol, 0), Identifier(&identifiers[2]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 1), Identifier(&identifiers[3]));

  // Convert 1 symbol to range identifier.
  space.convertVarKind(VarKind::Symbol, 1, 1, VarKind::Range, 0);

  // Check if the identifier is moved to range.
  EXPECT_EQ(space.getId(VarKind::Range, 0), Identifier(&identifiers[3]));
}

TEST(PresburgerSpaceTest, convertVarKindLocals) {
  PresburgerSpace space = PresburgerSpace::getRelationSpace(2, 2, 0, 0);
  space.resetIds();

  // Attach identifiers to range variables.
  int identifiers[4] = {0, 1};
  space.getId(VarKind::Range, 0) = Identifier(&identifiers[0]);
  space.getId(VarKind::Range, 1) = Identifier(&identifiers[1]);

  // Convert Range variables to locals i.e. project them out.
  space.convertVarKind(VarKind::Range, 0, 2, VarKind::Local, 0);

  // Check if the variables were moved.
  EXPECT_EQ(space.getNumVarKind(VarKind::Range), 0u);
  EXPECT_EQ(space.getNumVarKind(VarKind::Local), 2u);

  // Convert the Local variables back to Range variables.
  space.convertVarKind(VarKind::Local, 0, 2, VarKind::Range, 0);

  // The identifier information should be lost.
  EXPECT_FALSE(space.getId(VarKind::Range, 0).hasValue());
  EXPECT_FALSE(space.getId(VarKind::Range, 1).hasValue());
}

TEST(PresburgerSpaceTest, convertVarKind2) {
  PresburgerSpace space = PresburgerSpace::getRelationSpace(0, 2, 2, 0);
  space.resetIds();

  // Attach identifiers.
  int identifiers[4] = {0, 1, 2, 3};
  space.getId(VarKind::Range, 0) = Identifier(&identifiers[0]);
  space.getId(VarKind::Range, 1) = Identifier(&identifiers[1]);
  space.getId(VarKind::Symbol, 0) = Identifier(&identifiers[2]);
  space.getId(VarKind::Symbol, 1) = Identifier(&identifiers[3]);

  // Convert Range variables to symbols.
  space.convertVarKind(VarKind::Range, 0, 2, VarKind::Symbol, 1);

  // Check if the identifiers are moved to symbols.
  EXPECT_EQ(space.getId(VarKind::Symbol, 0), Identifier(&identifiers[2]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 1), Identifier(&identifiers[0]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 2), Identifier(&identifiers[1]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 3), Identifier(&identifiers[3]));
}

TEST(PresburgerSpaceTest, mergeAndAlignSymbols) {
  PresburgerSpace space = PresburgerSpace::getRelationSpace(3, 3, 2, 0);
  space.resetIds();

  PresburgerSpace otherSpace = PresburgerSpace::getRelationSpace(3, 2, 3, 0);
  otherSpace.resetIds();

  // Attach identifiers.
  int identifiers[7] = {0, 1, 2, 3, 4, 5, 6};
  int otherIdentifiers[8] = {10, 11, 12, 13, 14, 15, 16, 17};

  space.getId(VarKind::Domain, 0) = Identifier(&identifiers[0]);
  space.getId(VarKind::Domain, 1) = Identifier(&identifiers[1]);
  // Note the common identifier.
  space.getId(VarKind::Domain, 2) = Identifier(&otherIdentifiers[2]);
  space.getId(VarKind::Range, 0) = Identifier(&identifiers[2]);
  space.getId(VarKind::Range, 1) = Identifier(&identifiers[3]);
  space.getId(VarKind::Range, 2) = Identifier(&identifiers[4]);
  space.getId(VarKind::Symbol, 0) = Identifier(&identifiers[5]);
  space.getId(VarKind::Symbol, 1) = Identifier(&identifiers[6]);

  otherSpace.getId(VarKind::Domain, 0) = Identifier(&otherIdentifiers[0]);
  otherSpace.getId(VarKind::Domain, 1) = Identifier(&otherIdentifiers[1]);
  otherSpace.getId(VarKind::Domain, 2) = Identifier(&otherIdentifiers[2]);
  otherSpace.getId(VarKind::Range, 0) = Identifier(&otherIdentifiers[3]);
  otherSpace.getId(VarKind::Range, 1) = Identifier(&otherIdentifiers[4]);
  // Note the common identifier.
  otherSpace.getId(VarKind::Symbol, 0) = Identifier(&identifiers[6]);
  otherSpace.getId(VarKind::Symbol, 1) = Identifier(&otherIdentifiers[5]);
  otherSpace.getId(VarKind::Symbol, 2) = Identifier(&otherIdentifiers[7]);

  space.mergeAndAlignSymbols(otherSpace);

  // Check if merge & align is successful.
  // Check symbol var identifiers.
  EXPECT_EQ(4u, space.getNumSymbolVars());
  EXPECT_EQ(4u, otherSpace.getNumSymbolVars());
  EXPECT_EQ(space.getId(VarKind::Symbol, 0), Identifier(&identifiers[5]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 1), Identifier(&identifiers[6]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 2), Identifier(&otherIdentifiers[5]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 3), Identifier(&otherIdentifiers[7]));
  EXPECT_EQ(otherSpace.getId(VarKind::Symbol, 0), Identifier(&identifiers[5]));
  EXPECT_EQ(otherSpace.getId(VarKind::Symbol, 1), Identifier(&identifiers[6]));
  EXPECT_EQ(otherSpace.getId(VarKind::Symbol, 2),
            Identifier(&otherIdentifiers[5]));
  EXPECT_EQ(otherSpace.getId(VarKind::Symbol, 3),
            Identifier(&otherIdentifiers[7]));
  // Check that domain and range var identifiers are not affected.
  EXPECT_EQ(3u, space.getNumDomainVars());
  EXPECT_EQ(3u, space.getNumRangeVars());
  EXPECT_EQ(space.getId(VarKind::Domain, 0), Identifier(&identifiers[0]));
  EXPECT_EQ(space.getId(VarKind::Domain, 1), Identifier(&identifiers[1]));
  EXPECT_EQ(space.getId(VarKind::Domain, 2), Identifier(&otherIdentifiers[2]));
  EXPECT_EQ(space.getId(VarKind::Range, 0), Identifier(&identifiers[2]));
  EXPECT_EQ(space.getId(VarKind::Range, 1), Identifier(&identifiers[3]));
  EXPECT_EQ(space.getId(VarKind::Range, 2), Identifier(&identifiers[4]));
  EXPECT_EQ(3u, otherSpace.getNumDomainVars());
  EXPECT_EQ(2u, otherSpace.getNumRangeVars());
  EXPECT_EQ(otherSpace.getId(VarKind::Domain, 0),
            Identifier(&otherIdentifiers[0]));
  EXPECT_EQ(otherSpace.getId(VarKind::Domain, 1),
            Identifier(&otherIdentifiers[1]));
  EXPECT_EQ(otherSpace.getId(VarKind::Domain, 2),
            Identifier(&otherIdentifiers[2]));
  EXPECT_EQ(otherSpace.getId(VarKind::Range, 0),
            Identifier(&otherIdentifiers[3]));
  EXPECT_EQ(otherSpace.getId(VarKind::Range, 1),
            Identifier(&otherIdentifiers[4]));
}
