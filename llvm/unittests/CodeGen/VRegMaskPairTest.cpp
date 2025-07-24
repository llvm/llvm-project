//===- VRegMaskPairTest.cpp - Unit tests for VRegMaskPairSet -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Register.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include "AMDGPUSSARAUtils.h"
#include "SIRegisterInfo.h"
#include "VRegMaskPair.h"

using namespace llvm;

namespace {

class VRegMaskPairTest : public ::testing::Test {
protected:
  Register R1 = Register::index2VirtReg(1);
  Register R2 = Register::index2VirtReg(2);
  Register R3 = Register::index2VirtReg(3);
  Register R4 = Register::index2VirtReg(4);

  LaneBitmask M0 = LaneBitmask::getLane(0); // sub0
  LaneBitmask M1 = LaneBitmask::getLane(1); // sub1
  LaneBitmask M2 = LaneBitmask::getLane(2); // sub2
  LaneBitmask M3 = LaneBitmask::getLane(3); // sub3
  LaneBitmask M01 = M0 | M1;
  LaneBitmask FULL = LaneBitmask::getAll();
};

TEST_F(VRegMaskPairTest, BasicInsertAndCoverage) {
  VRegMaskPairSet Set;
  EXPECT_TRUE(Set.insert({R1, M0}));
  EXPECT_TRUE(Set.insert({R1, M1}));
  EXPECT_FALSE(Set.insert({R1, M1})); // duplicate

  LaneCoverageResult Cov = Set.getCoverage({R1, M01});
  EXPECT_TRUE(Cov.isFullyCovered());
  EXPECT_EQ(Cov.getCovered(), M01);
  EXPECT_EQ(Cov.getNotCovered(), LaneBitmask::getNone());
}

TEST_F(VRegMaskPairTest, ExactContains) {
  VRegMaskPairSet Set;
  Set.insert({R2, M2});
  EXPECT_TRUE(Set.contains({R2, M2}));
  EXPECT_FALSE(Set.contains({R2, M3}));
}

TEST_F(VRegMaskPairTest, UnionAndJoinPreserveEntries) {
  VRegMaskPairSet A, B;
  A.insert({R1, M0});
  A.insert({R2, M1});

  B.insert({R1, M1});
  B.insert({R3, M0});

  VRegMaskPairSet U = A.set_join(B);
  EXPECT_TRUE(U.contains({R1, M0}));
  EXPECT_TRUE(U.contains({R1, M1}));
  EXPECT_TRUE(U.contains({R2, M1}));
  EXPECT_TRUE(U.contains({R3, M0}));
}

TEST_F(VRegMaskPairTest, IntersectionKeepsOnlyCoveredParts) {
  VRegMaskPairSet A, B;
  A.insert({R1, M0 | M1});
  A.insert({R2, M0});
  A.insert({R3, FULL});
  A.insert({R4, M1});

  B.insert({R1, M1});
  B.insert({R2, M1});
  B.insert({R4, FULL});

  VRegMaskPairSet I = A.set_intersection(B);
  EXPECT_TRUE(I.contains({R1, M1}));
  EXPECT_FALSE(I.contains({R2, M0}));
  EXPECT_FALSE(I.contains({R3, FULL}));
  EXPECT_TRUE(I.contains({R4, M1}));
}

TEST_F(VRegMaskPairTest, SubtractionRemovesCoveredParts) {
  VRegMaskPairSet A, B;
  A.insert({R1, M0 | M1});
  A.insert({R2, M1});
  A.insert({R3, M2});

  B.insert({R1, M1});
  B.insert({R3, M2});

  VRegMaskPairSet D = A.set_difference(B);
  EXPECT_TRUE(D.contains({R1, M0}));
  EXPECT_FALSE(D.contains({R1, M1}));
  EXPECT_TRUE(D.contains({R2, M1}));
  EXPECT_FALSE(D.contains({R3, M2}));
}

TEST_F(VRegMaskPairTest, SetOperations) {
  VRegMaskPairSet A, B;
  A.insert({R1, M0});
  A.insert({R2, M0});
  A.insert({R3, FULL});
  A.insert({R4, M1});

  B.insert({R1, M1});
  B.insert({R2, M1});
  B.insert({R4, FULL});

  VRegMaskPairSet I = A.set_intersection(B);
  EXPECT_FALSE(I.contains({R1, M1}));
  EXPECT_FALSE(I.contains({R2, M0}));
  EXPECT_FALSE(I.contains({R3, FULL}));
  EXPECT_TRUE(I.contains({R4, M1}));

  VRegMaskPairSet D = A.set_difference(B);
  EXPECT_TRUE(D.contains({R1, M0}));
  EXPECT_TRUE(D.contains({R2, M0}));
  EXPECT_TRUE(D.contains({R3, FULL}));
  EXPECT_FALSE(D.contains({R4, M1}));

  VRegMaskPairSet U = A.set_join(B);
  EXPECT_TRUE(U.contains({R1, M0}));
  EXPECT_TRUE(U.contains({R1, M1}));
  EXPECT_TRUE(U.contains({R2, M0}));
  EXPECT_TRUE(U.contains({R2, M1}));
  EXPECT_TRUE(U.contains({R3, FULL}));
  EXPECT_TRUE(U.contains({R4, M1}));
}

TEST_F(VRegMaskPairTest, InPlaceSetOperations) {
  VRegMaskPairSet A, B;
  A.insert({R1, M0});
  A.insert({R2, M1});

  B.insert({R1, M1});
  B.insert({R3, M0});

  VRegMaskPairSet AU = A;
  AU.set_union(B);
  EXPECT_TRUE(AU.contains({R1, M0}));
  EXPECT_TRUE(AU.contains({R1, M1}));
  EXPECT_TRUE(AU.contains({R2, M1}));
  EXPECT_TRUE(AU.contains({R3, M0}));

  VRegMaskPairSet AI = A;
  AI.set_intersect(B);
  EXPECT_FALSE(AI.contains({R1, M1}));
  EXPECT_FALSE(AI.contains({R1, M0}));
  EXPECT_FALSE(AI.contains({R2, M1}));

  VRegMaskPairSet AD = A;
  AD.set_subtract(B);
  EXPECT_TRUE(AD.contains({R1, M0}));
  EXPECT_TRUE(AD.contains({R2, M1}));
  EXPECT_FALSE(AD.contains({R1, M1}));
}

TEST_F(VRegMaskPairTest, RemoveAndPop) {
  VRegMaskPairSet Set;
  Set.insert({R1, M0});
  Set.insert({R2, M1});
  Set.insert({R3, M2});

  Set.remove({R2, M1});
  EXPECT_FALSE(Set.contains({R2, M1}));
  EXPECT_EQ(Set.size(), 2u);

  VRegMaskPair Last = Set.pop_back_val();
  EXPECT_FALSE(Set.contains(Last));
  EXPECT_EQ(Set.size(), 1u);
}

} // namespace
