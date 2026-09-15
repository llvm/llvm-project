//===- VersionTupleTests.cpp - Version Number Handling Tests --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/VersionTuple.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(VersionTuple, getAsString) {
  EXPECT_EQ("0", VersionTuple().getAsString());
  EXPECT_EQ("1", VersionTuple(1).getAsString());
  EXPECT_EQ("1.2", VersionTuple(1, 2).getAsString());
  EXPECT_EQ("1.2.3", VersionTuple(1, 2, 3).getAsString());
  EXPECT_EQ("1.2.3.4", VersionTuple(1, 2, 3, 4).getAsString());
  EXPECT_EQ("1.2.3.4.5", VersionTuple(1, 2, 3, 4, 5).getAsString());

  VersionTuple v(1, 2, 3, 4, 5);
  EXPECT_EQ(v.getMajor(), 1u);
  EXPECT_EQ(v.getMinor(), 2u);
  EXPECT_EQ(v.getSubminor(), 3u);
  EXPECT_EQ(v.getBuild(), 4u);
  EXPECT_EQ(v.getSubbuild(), 5u);
}

TEST(VersionTuple, tryParse) {
  VersionTuple VT;

  EXPECT_FALSE(VT.tryParse("1"));
  EXPECT_EQ("1", VT.getAsString());

  EXPECT_FALSE(VT.tryParse("1.2"));
  EXPECT_EQ("1.2", VT.getAsString());

  EXPECT_FALSE(VT.tryParse("1.2.3"));
  EXPECT_EQ("1.2.3", VT.getAsString());

  EXPECT_FALSE(VT.tryParse("1.2.3.4"));
  EXPECT_EQ("1.2.3.4", VT.getAsString());

  EXPECT_FALSE(VT.tryParse("1.2.3.4.5"));
  EXPECT_EQ("1.2.3.4.5", VT.getAsString());

  EXPECT_TRUE(VT.tryParse(""));
  EXPECT_TRUE(VT.tryParse("1."));
  EXPECT_TRUE(VT.tryParse("1.2."));
  EXPECT_TRUE(VT.tryParse("1.2.3."));
  EXPECT_TRUE(VT.tryParse("1.2.3.4."));
  EXPECT_TRUE(VT.tryParse("1.2.3.4.5."));
  EXPECT_TRUE(VT.tryParse("1.2.3.4.5.6"));
  EXPECT_TRUE(VT.tryParse("1-2"));
  EXPECT_TRUE(VT.tryParse("1+2"));
  EXPECT_TRUE(VT.tryParse(".1"));
  EXPECT_TRUE(VT.tryParse(" 1"));
  EXPECT_TRUE(VT.tryParse("1 "));
  EXPECT_TRUE(VT.tryParse("."));
  EXPECT_TRUE(VT.tryParse("1.2.3.1048576"));
  EXPECT_TRUE(VT.tryParse("1.2.3.4.1024"));
}

TEST(VersionTuple, withMajorReplaced) {
  VersionTuple VT(2);
  VersionTuple ReplacedVersion = VT.withMajorReplaced(7);
  EXPECT_FALSE(ReplacedVersion.getMinor().has_value());
  EXPECT_FALSE(ReplacedVersion.getSubminor().has_value());
  EXPECT_FALSE(ReplacedVersion.getBuild().has_value());
  EXPECT_EQ(VersionTuple(7), ReplacedVersion);

  VT = VersionTuple(100, 1);
  ReplacedVersion = VT.withMajorReplaced(7);
  EXPECT_TRUE(ReplacedVersion.getMinor().has_value());
  EXPECT_FALSE(ReplacedVersion.getSubminor().has_value());
  EXPECT_FALSE(ReplacedVersion.getBuild().has_value());
  EXPECT_EQ(VersionTuple(7, 1), ReplacedVersion);

  VT = VersionTuple(101, 11, 12);
  ReplacedVersion = VT.withMajorReplaced(7);
  EXPECT_TRUE(ReplacedVersion.getMinor().has_value());
  EXPECT_TRUE(ReplacedVersion.getSubminor().has_value());
  EXPECT_FALSE(ReplacedVersion.getBuild().has_value());
  EXPECT_EQ(VersionTuple(7, 11, 12), ReplacedVersion);

  VT = VersionTuple(101, 11, 12, 2);
  ReplacedVersion = VT.withMajorReplaced(7);
  EXPECT_TRUE(ReplacedVersion.getMinor().has_value());
  EXPECT_TRUE(ReplacedVersion.getSubminor().has_value());
  EXPECT_TRUE(ReplacedVersion.getBuild().has_value());
  EXPECT_EQ(VersionTuple(7, 11, 12, 2), ReplacedVersion);

  VT = VersionTuple(101, 11, 12, 2, 8);
  ReplacedVersion = VT.withMajorReplaced(7);
  EXPECT_TRUE(ReplacedVersion.getMinor().has_value());
  EXPECT_TRUE(ReplacedVersion.getSubminor().has_value());
  EXPECT_TRUE(ReplacedVersion.getBuild().has_value());
  EXPECT_TRUE(ReplacedVersion.getSubbuild().has_value());
  EXPECT_EQ(VersionTuple(7, 11, 12, 2, 8), ReplacedVersion);
}

TEST(VersionTuple, DenseMapInfo) {
  VersionTuple VT16(16);
  VersionTuple VT16_0(16, 0);

  VersionTuple VT17(17);
  VersionTuple VT17_0(17, 0);

  // In C++, if two objects are equal, their hashes should be equal.
  // DenseMapInfo relies on the same relation for comparing keys.
  // If isEqual returns true, getHashValue should return the same value.
  EXPECT_TRUE(DenseMapInfo<VersionTuple>::isEqual(VT16, VT16_0));
  EXPECT_EQ(DenseMapInfo<VersionTuple>::getHashValue(VT16),
            DenseMapInfo<VersionTuple>::getHashValue(VT16_0));

  EXPECT_TRUE(DenseMapInfo<VersionTuple>::isEqual(VT17, VT17_0));
  EXPECT_EQ(DenseMapInfo<VersionTuple>::getHashValue(VT17),
            DenseMapInfo<VersionTuple>::getHashValue(VT17_0));
}
