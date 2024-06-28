//===- ConstantRangeListTest.cpp - ConstantRangeList tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/ConstantRangeList.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

using ConstantRangeListTest = ::testing::Test;

TEST_F(ConstantRangeListTest, Basics) {
  ConstantRangeList CRL1a;
  CRL1a.insert(0, 12);
  EXPECT_FALSE(CRL1a.empty());

  ConstantRangeList CRL1b;
  CRL1b.insert(0, 4);
  CRL1b.insert(4, 8);
  CRL1b.insert(8, 12);
  EXPECT_TRUE(CRL1a == CRL1b);

  ConstantRangeList CRL1c;
  CRL1c.insert(0, 4);
  CRL1c.insert(8, 12);
  CRL1c.insert(4, 8);
  EXPECT_TRUE(CRL1a == CRL1c);

  ConstantRangeList CRL2;
  CRL2.insert(-4, 0);
  CRL2.insert(8, 12);
  EXPECT_TRUE(CRL1a != CRL2);
}

TEST_F(ConstantRangeListTest, getConstantRangeList) {
  SmallVector<ConstantRange, 2> Empty;
  EXPECT_TRUE(ConstantRangeList::getConstantRangeList(Empty).has_value());

  SmallVector<ConstantRange, 2> Valid;
  Valid.push_back(ConstantRange(APInt(64, 0, true), APInt(64, 4, true)));
  Valid.push_back(ConstantRange(APInt(64, 8, true), APInt(64, 12, true)));
  EXPECT_TRUE(ConstantRangeList::getConstantRangeList(Valid).has_value());

  SmallVector<ConstantRange, 2> Invalid1;
  Invalid1.push_back(ConstantRange(APInt(64, 4, true), APInt(64, 0, true)));
  EXPECT_EQ(ConstantRangeList::getConstantRangeList(Invalid1), std::nullopt);

  SmallVector<ConstantRange, 2> Invalid2;
  Invalid2.push_back(ConstantRange(APInt(64, 0, true), APInt(64, 4, true)));
  Invalid2.push_back(ConstantRange(APInt(64, 12, true), APInt(64, 8, true)));
  EXPECT_EQ(ConstantRangeList::getConstantRangeList(Invalid2), std::nullopt);

  SmallVector<ConstantRange, 2> Invalid3;
  Invalid3.push_back(ConstantRange(APInt(64, 0, true), APInt(64, 4, true)));
  Invalid3.push_back(ConstantRange(APInt(64, 4, true), APInt(64, 8, true)));
  EXPECT_EQ(ConstantRangeList::getConstantRangeList(Invalid3), std::nullopt);

  SmallVector<ConstantRange, 2> Invalid4;
  Invalid4.push_back(ConstantRange(APInt(64, 0, true), APInt(64, 12, true)));
  Invalid4.push_back(ConstantRange(APInt(64, 8, true), APInt(64, 16, true)));
  EXPECT_EQ(ConstantRangeList::getConstantRangeList(Invalid4), std::nullopt);
}

TEST_F(ConstantRangeListTest, Insert) {
  ConstantRangeList CRL;
  CRL.insert(0, 4);
  CRL.insert(8, 12);
  // No overlap, left
  CRL.insert(-8, -4);
  // No overlap, right
  CRL.insert(16, 20);
  // No overlap, middle
  CRL.insert(13, 15);
  // Overlap with left
  CRL.insert(-6, -2);
  // Overlap with right
  CRL.insert(5, 9);
  // Overlap with left and right
  CRL.insert(14, 18);
  // Overlap cross ranges
  CRL.insert(2, 14);
  // An existing range
  CRL.insert(0, 20);

  ConstantRangeList Expected;
  Expected.insert(-8, -2);
  Expected.insert(0, 20);
  EXPECT_TRUE(CRL == Expected);
}

ConstantRangeList GetCRL(ArrayRef<std::pair<APInt, APInt>> Pairs) {
  SmallVector<ConstantRange, 2> Ranges;
  for (auto &[Start, End] : Pairs)
    Ranges.push_back(ConstantRange(Start, End));
  return ConstantRangeList(Ranges);
}

TEST_F(ConstantRangeListTest, Union) {
  APInt APN4 = APInt(64, -4, /*isSigned=*/true);
  APInt APN2 = APInt(64, -2, /*isSigned=*/true);
  APInt AP0 = APInt(64, 0, /*isSigned=*/true);
  APInt AP2 = APInt(64, 2, /*isSigned=*/true);
  APInt AP4 = APInt(64, 4, /*isSigned=*/true);
  APInt AP6 = APInt(64, 6, /*isSigned=*/true);
  APInt AP7 = APInt(64, 7, /*isSigned=*/true);
  APInt AP8 = APInt(64, 8, /*isSigned=*/true);
  APInt AP10 = APInt(64, 10, /*isSigned=*/true);
  APInt AP11 = APInt(64, 11, /*isSigned=*/true);
  APInt AP12 = APInt(64, 12, /*isSigned=*/true);
  APInt AP16 = APInt(64, 16, /*isSigned=*/true);
  APInt AP18 = APInt(64, 18, /*isSigned=*/true);
  ConstantRangeList CRL = GetCRL({{AP0, AP4}, {AP8, AP12}});

  // Union with a subset.
  ConstantRangeList Empty;
  EXPECT_EQ(CRL.unionWith(Empty), CRL);
  EXPECT_EQ(Empty.unionWith(CRL), CRL);

  EXPECT_EQ(CRL.unionWith(GetCRL({{AP0, AP2}})), CRL);
  EXPECT_EQ(CRL.unionWith(GetCRL({{AP10, AP12}})), CRL);

  EXPECT_EQ(CRL.unionWith(GetCRL({{AP0, AP2}, {AP8, AP10}})), CRL);
  EXPECT_EQ(CRL.unionWith(GetCRL({{AP0, AP2}, {AP10, AP12}})), CRL);
  EXPECT_EQ(CRL.unionWith(GetCRL({{AP2, AP4}, {AP8, AP10}})), CRL);
  EXPECT_EQ(CRL.unionWith(GetCRL({{AP2, AP4}, {AP10, AP12}})), CRL);

  EXPECT_EQ(CRL.unionWith(GetCRL({{AP0, AP4}, {AP8, AP10}, {AP11, AP12}})),
            CRL);

  EXPECT_EQ(CRL.unionWith(CRL), CRL);

  // Union with new ranges.
  EXPECT_EQ(CRL.unionWith(GetCRL({{APN4, APN2}})),
            GetCRL({{APN4, APN2}, {AP0, AP4}, {AP8, AP12}}));
  EXPECT_EQ(CRL.unionWith(GetCRL({{AP6, AP7}})),
            GetCRL({{AP0, AP4}, {AP6, AP7}, {AP8, AP12}}));
  EXPECT_EQ(CRL.unionWith(GetCRL({{AP16, AP18}})),
            GetCRL({{AP0, AP4}, {AP8, AP12}, {AP16, AP18}}));

  EXPECT_EQ(CRL.unionWith(GetCRL({{APN2, AP2}})),
            GetCRL({{APN2, AP4}, {AP8, AP12}}));
  EXPECT_EQ(CRL.unionWith(GetCRL({{AP2, AP6}})),
            GetCRL({{AP0, AP6}, {AP8, AP12}}));
  EXPECT_EQ(CRL.unionWith(GetCRL({{AP10, AP16}})),
            GetCRL({{AP0, AP4}, {AP8, AP16}}));

  EXPECT_EQ(CRL.unionWith(GetCRL({{APN2, AP10}})), GetCRL({{APN2, AP12}}));
  EXPECT_EQ(CRL.unionWith(GetCRL({{AP2, AP10}})), GetCRL({{AP0, AP12}}));
  EXPECT_EQ(CRL.unionWith(GetCRL({{AP4, AP16}})), GetCRL({{AP0, AP16}}));
  EXPECT_EQ(CRL.unionWith(GetCRL({{APN2, AP16}})), GetCRL({{APN2, AP16}}));
}

TEST_F(ConstantRangeListTest, Intersect) {
  APInt APN2 = APInt(64, -2, /*isSigned=*/true);
  APInt AP0 = APInt(64, 0, /*isSigned=*/true);
  APInt AP2 = APInt(64, 2, /*isSigned=*/true);
  APInt AP4 = APInt(64, 4, /*isSigned=*/true);
  APInt AP6 = APInt(64, 6, /*isSigned=*/true);
  APInt AP7 = APInt(64, 7, /*isSigned=*/true);
  APInt AP8 = APInt(64, 8, /*isSigned=*/true);
  APInt AP10 = APInt(64, 10, /*isSigned=*/true);
  APInt AP11 = APInt(64, 11, /*isSigned=*/true);
  APInt AP12 = APInt(64, 12, /*isSigned=*/true);
  APInt AP16 = APInt(64, 16, /*isSigned=*/true);
  ConstantRangeList CRL = GetCRL({{AP0, AP4}, {AP8, AP12}});

  // No intersection.
  ConstantRangeList Empty;
  EXPECT_EQ(CRL.intersectWith(Empty), Empty);
  EXPECT_EQ(Empty.intersectWith(CRL), Empty);

  EXPECT_EQ(CRL.intersectWith(GetCRL({{APN2, AP0}})), Empty);
  EXPECT_EQ(CRL.intersectWith(GetCRL({{AP6, AP8}})), Empty);
  EXPECT_EQ(CRL.intersectWith(GetCRL({{AP12, AP16}})), Empty);

  // Single intersect range.
  EXPECT_EQ(CRL.intersectWith(GetCRL({{APN2, AP2}})), GetCRL({{AP0, AP2}}));
  EXPECT_EQ(CRL.intersectWith(GetCRL({{APN2, AP6}})), GetCRL({{AP0, AP4}}));
  EXPECT_EQ(CRL.intersectWith(GetCRL({{AP2, AP4}})), GetCRL({{AP2, AP4}}));
  EXPECT_EQ(CRL.intersectWith(GetCRL({{AP2, AP6}})), GetCRL({{AP2, AP4}}));
  EXPECT_EQ(CRL.intersectWith(GetCRL({{AP6, AP10}})), GetCRL({{AP8, AP10}}));
  EXPECT_EQ(CRL.intersectWith(GetCRL({{AP6, AP16}})), GetCRL({{AP8, AP12}}));
  EXPECT_EQ(CRL.intersectWith(GetCRL({{AP10, AP12}})), GetCRL({{AP10, AP12}}));
  EXPECT_EQ(CRL.intersectWith(GetCRL({{AP10, AP16}})), GetCRL({{AP10, AP12}}));

  // Multiple intersect ranges.
  EXPECT_EQ(CRL.intersectWith(GetCRL({{APN2, AP10}})),
            GetCRL({{AP0, AP4}, {AP8, AP10}}));
  EXPECT_EQ(CRL.intersectWith(GetCRL({{APN2, AP16}})), CRL);
  EXPECT_EQ(CRL.intersectWith(GetCRL({{AP2, AP10}})),
            GetCRL({{AP2, AP4}, {AP8, AP10}}));
  EXPECT_EQ(CRL.intersectWith(GetCRL({{AP2, AP16}})),
            GetCRL({{AP2, AP4}, {AP8, AP12}}));
  EXPECT_EQ(CRL.intersectWith(GetCRL({{APN2, AP2}, {AP6, AP10}})),
            GetCRL({{AP0, AP2}, {AP8, AP10}}));
  EXPECT_EQ(CRL.intersectWith(GetCRL({{AP2, AP6}, {AP10, AP16}})),
            GetCRL({{AP2, AP4}, {AP10, AP12}}));
  EXPECT_EQ(CRL.intersectWith(GetCRL({{APN2, AP2}, {AP7, AP10}, {AP11, AP16}})),
            GetCRL({{AP0, AP2}, {AP8, AP10}, {AP11, AP12}}));
  EXPECT_EQ(CRL.intersectWith(CRL), CRL);
}

} // anonymous namespace
