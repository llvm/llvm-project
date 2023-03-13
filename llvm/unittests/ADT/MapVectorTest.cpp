//===- unittest/ADT/MapVectorTest.cpp - MapVector unit tests ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/iterator_range.h"
#include "gtest/gtest.h"
#include <utility>

using namespace llvm;

TEST(MapVectorTest, swap) {
  MapVector<int, int> MV1, MV2;
  std::pair<MapVector<int, int>::iterator, bool> R;

  R = MV1.insert(std::make_pair(1, 2));
  ASSERT_EQ(R.first, MV1.begin());
  EXPECT_EQ(R.first->first, 1);
  EXPECT_EQ(R.first->second, 2);
  EXPECT_TRUE(R.second);

  EXPECT_FALSE(MV1.empty());
  EXPECT_TRUE(MV2.empty());
  MV2.swap(MV1);
  EXPECT_TRUE(MV1.empty());
  EXPECT_FALSE(MV2.empty());

  auto I = MV1.find(1);
  ASSERT_EQ(MV1.end(), I);

  I = MV2.find(1);
  ASSERT_EQ(I, MV2.begin());
  EXPECT_EQ(I->first, 1);
  EXPECT_EQ(I->second, 2);
}

TEST(MapVectorTest, insert_pop) {
  MapVector<int, int> MV;
  std::pair<MapVector<int, int>::iterator, bool> R;

  R = MV.insert(std::make_pair(1, 2));
  ASSERT_EQ(R.first, MV.begin());
  EXPECT_EQ(R.first->first, 1);
  EXPECT_EQ(R.first->second, 2);
  EXPECT_TRUE(R.second);

  R = MV.insert(std::make_pair(1, 3));
  ASSERT_EQ(R.first, MV.begin());
  EXPECT_EQ(R.first->first, 1);
  EXPECT_EQ(R.first->second, 2);
  EXPECT_FALSE(R.second);

  R = MV.insert(std::make_pair(4, 5));
  ASSERT_NE(R.first, MV.end());
  EXPECT_EQ(R.first->first, 4);
  EXPECT_EQ(R.first->second, 5);
  EXPECT_TRUE(R.second);

  EXPECT_EQ(MV.size(), 2u);
  EXPECT_EQ(MV[1], 2);
  EXPECT_EQ(MV[4], 5);

  MV.pop_back();
  EXPECT_EQ(MV.size(), 1u);
  EXPECT_EQ(MV[1], 2);

  R = MV.insert(std::make_pair(4, 7));
  ASSERT_NE(R.first, MV.end());
  EXPECT_EQ(R.first->first, 4);
  EXPECT_EQ(R.first->second, 7);
  EXPECT_TRUE(R.second);

  EXPECT_EQ(MV.size(), 2u);
  EXPECT_EQ(MV[1], 2);
  EXPECT_EQ(MV[4], 7);
}

TEST(MapVectorTest, erase) {
  MapVector<int, int> MV;

  MV.insert(std::make_pair(1, 2));
  MV.insert(std::make_pair(3, 4));
  MV.insert(std::make_pair(5, 6));
  ASSERT_EQ(MV.size(), 3u);

  ASSERT_TRUE(MV.contains(1));
  MV.erase(MV.find(1));
  ASSERT_EQ(MV.size(), 2u);
  ASSERT_FALSE(MV.contains(1));
  ASSERT_EQ(MV.find(1), MV.end());
  ASSERT_EQ(MV[3], 4);
  ASSERT_EQ(MV[5], 6);

  ASSERT_EQ(MV.erase(3), 1u);
  ASSERT_EQ(MV.size(), 1u);
  ASSERT_EQ(MV.find(3), MV.end());
  ASSERT_EQ(MV[5], 6);

  ASSERT_EQ(MV.erase(79), 0u);
  ASSERT_EQ(MV.size(), 1u);
}

TEST(MapVectorTest, remove_if) {
  MapVector<int, int> MV;

  MV.insert(std::make_pair(1, 11));
  MV.insert(std::make_pair(2, 12));
  MV.insert(std::make_pair(3, 13));
  MV.insert(std::make_pair(4, 14));
  MV.insert(std::make_pair(5, 15));
  MV.insert(std::make_pair(6, 16));
  ASSERT_EQ(MV.size(), 6u);

  MV.remove_if([](const std::pair<int, int> &Val) { return Val.second % 2; });
  ASSERT_EQ(MV.size(), 3u);
  ASSERT_EQ(MV.find(1), MV.end());
  ASSERT_EQ(MV.find(3), MV.end());
  ASSERT_EQ(MV.find(5), MV.end());
  ASSERT_EQ(MV[2], 12);
  ASSERT_EQ(MV[4], 14);
  ASSERT_EQ(MV[6], 16);
}

TEST(MapVectorTest, iteration_test) {
  MapVector<int, int> MV;

  MV.insert(std::make_pair(1, 11));
  MV.insert(std::make_pair(2, 12));
  MV.insert(std::make_pair(3, 13));
  MV.insert(std::make_pair(4, 14));
  MV.insert(std::make_pair(5, 15));
  MV.insert(std::make_pair(6, 16));
  ASSERT_EQ(MV.size(), 6u);

  int count = 1;
  for (auto P : make_range(MV.begin(), MV.end())) {
    ASSERT_EQ(P.first, count);
    count++;
  }

  count = 6;
  for (auto P : make_range(MV.rbegin(), MV.rend())) {
    ASSERT_EQ(P.first, count);
    count--;
  }
}

TEST(MapVectorTest, NonCopyable) {
  MapVector<int, std::unique_ptr<int>> MV;
  MV.insert(std::make_pair(1, std::make_unique<int>(1)));
  MV.insert(std::make_pair(2, std::make_unique<int>(2)));

  ASSERT_EQ(MV.count(1), 1u);
  ASSERT_EQ(*MV.find(2)->second, 2);
}

template <class IntType> struct MapVectorMappedTypeTest : ::testing::Test {
  using int_type = IntType;
};

using MapIntTypes = ::testing::Types<int, long, long long, unsigned,
                                     unsigned long, unsigned long long>;
TYPED_TEST_SUITE(MapVectorMappedTypeTest, MapIntTypes, );

TYPED_TEST(MapVectorMappedTypeTest, DifferentDenseMap) {
  // Test that using a map with a mapped type other than 'unsigned' compiles
  // and works.
  using IntType = typename TestFixture::int_type;
  using MapVectorType = MapVector<int, int, DenseMap<int, IntType>>;

  MapVectorType MV;
  std::pair<typename MapVectorType::iterator, bool> R;

  R = MV.insert(std::make_pair(1, 2));
  ASSERT_EQ(R.first, MV.begin());
  EXPECT_EQ(R.first->first, 1);
  EXPECT_EQ(R.first->second, 2);
  EXPECT_TRUE(R.second);

  const std::pair<int, int> Elem(1, 3);
  R = MV.insert(Elem);
  ASSERT_EQ(R.first, MV.begin());
  EXPECT_EQ(R.first->first, 1);
  EXPECT_EQ(R.first->second, 2);
  EXPECT_FALSE(R.second);

  int& value = MV[4];
  EXPECT_EQ(value, 0);
  value = 5;

  EXPECT_EQ(MV.size(), 2u);
  EXPECT_EQ(MV[1], 2);
  EXPECT_EQ(MV[4], 5);
}

TEST(SmallMapVectorSmallTest, insert_pop) {
  SmallMapVector<int, int, 32> MV;
  std::pair<SmallMapVector<int, int, 32>::iterator, bool> R;

  R = MV.insert(std::make_pair(1, 2));
  ASSERT_EQ(R.first, MV.begin());
  EXPECT_EQ(R.first->first, 1);
  EXPECT_EQ(R.first->second, 2);
  EXPECT_TRUE(R.second);

  R = MV.insert(std::make_pair(1, 3));
  ASSERT_EQ(R.first, MV.begin());
  EXPECT_EQ(R.first->first, 1);
  EXPECT_EQ(R.first->second, 2);
  EXPECT_FALSE(R.second);

  R = MV.insert(std::make_pair(4, 5));
  ASSERT_NE(R.first, MV.end());
  EXPECT_EQ(R.first->first, 4);
  EXPECT_EQ(R.first->second, 5);
  EXPECT_TRUE(R.second);

  EXPECT_EQ(MV.size(), 2u);
  EXPECT_EQ(MV[1], 2);
  EXPECT_EQ(MV[4], 5);

  MV.pop_back();
  EXPECT_EQ(MV.size(), 1u);
  EXPECT_EQ(MV[1], 2);

  R = MV.insert(std::make_pair(4, 7));
  ASSERT_NE(R.first, MV.end());
  EXPECT_EQ(R.first->first, 4);
  EXPECT_EQ(R.first->second, 7);
  EXPECT_TRUE(R.second);

  EXPECT_EQ(MV.size(), 2u);
  EXPECT_EQ(MV[1], 2);
  EXPECT_EQ(MV[4], 7);
}

TEST(SmallMapVectorSmallTest, erase) {
  SmallMapVector<int, int, 32> MV;

  MV.insert(std::make_pair(1, 2));
  MV.insert(std::make_pair(3, 4));
  MV.insert(std::make_pair(5, 6));
  ASSERT_EQ(MV.size(), 3u);

  MV.erase(MV.find(1));
  ASSERT_EQ(MV.size(), 2u);
  ASSERT_EQ(MV.find(1), MV.end());
  ASSERT_EQ(MV[3], 4);
  ASSERT_EQ(MV[5], 6);

  ASSERT_EQ(MV.erase(3), 1u);
  ASSERT_EQ(MV.size(), 1u);
  ASSERT_EQ(MV.find(3), MV.end());
  ASSERT_EQ(MV[5], 6);

  ASSERT_EQ(MV.erase(79), 0u);
  ASSERT_EQ(MV.size(), 1u);
}

TEST(SmallMapVectorSmallTest, remove_if) {
  SmallMapVector<int, int, 32> MV;

  MV.insert(std::make_pair(1, 11));
  MV.insert(std::make_pair(2, 12));
  MV.insert(std::make_pair(3, 13));
  MV.insert(std::make_pair(4, 14));
  MV.insert(std::make_pair(5, 15));
  MV.insert(std::make_pair(6, 16));
  ASSERT_EQ(MV.size(), 6u);

  MV.remove_if([](const std::pair<int, int> &Val) { return Val.second % 2; });
  ASSERT_EQ(MV.size(), 3u);
  ASSERT_EQ(MV.find(1), MV.end());
  ASSERT_EQ(MV.find(3), MV.end());
  ASSERT_EQ(MV.find(5), MV.end());
  ASSERT_EQ(MV[2], 12);
  ASSERT_EQ(MV[4], 14);
  ASSERT_EQ(MV[6], 16);
}

TEST(SmallMapVectorSmallTest, iteration_test) {
  SmallMapVector<int, int, 32> MV;

  MV.insert(std::make_pair(1, 11));
  MV.insert(std::make_pair(2, 12));
  MV.insert(std::make_pair(3, 13));
  MV.insert(std::make_pair(4, 14));
  MV.insert(std::make_pair(5, 15));
  MV.insert(std::make_pair(6, 16));
  ASSERT_EQ(MV.size(), 6u);

  int count = 1;
  for (auto P : make_range(MV.begin(), MV.end())) {
    ASSERT_EQ(P.first, count);
    count++;
  }

  count = 6;
  for (auto P : make_range(MV.rbegin(), MV.rend())) {
    ASSERT_EQ(P.first, count);
    count--;
  }
}

TEST(SmallMapVectorSmallTest, NonCopyable) {
  SmallMapVector<int, std::unique_ptr<int>, 8> MV;
  MV.insert(std::make_pair(1, std::make_unique<int>(1)));
  MV.insert(std::make_pair(2, std::make_unique<int>(2)));

  ASSERT_EQ(MV.count(1), 1u);
  ASSERT_EQ(*MV.find(2)->second, 2);
}

TEST(SmallMapVectorLargeTest, insert_pop) {
  SmallMapVector<int, int, 1> MV;
  std::pair<SmallMapVector<int, int, 1>::iterator, bool> R;

  R = MV.insert(std::make_pair(1, 2));
  ASSERT_EQ(R.first, MV.begin());
  EXPECT_EQ(R.first->first, 1);
  EXPECT_EQ(R.first->second, 2);
  EXPECT_TRUE(R.second);

  R = MV.insert(std::make_pair(1, 3));
  ASSERT_EQ(R.first, MV.begin());
  EXPECT_EQ(R.first->first, 1);
  EXPECT_EQ(R.first->second, 2);
  EXPECT_FALSE(R.second);

  R = MV.insert(std::make_pair(4, 5));
  ASSERT_NE(R.first, MV.end());
  EXPECT_EQ(R.first->first, 4);
  EXPECT_EQ(R.first->second, 5);
  EXPECT_TRUE(R.second);

  EXPECT_EQ(MV.size(), 2u);
  EXPECT_EQ(MV[1], 2);
  EXPECT_EQ(MV[4], 5);

  MV.pop_back();
  EXPECT_EQ(MV.size(), 1u);
  EXPECT_EQ(MV[1], 2);

  R = MV.insert(std::make_pair(4, 7));
  ASSERT_NE(R.first, MV.end());
  EXPECT_EQ(R.first->first, 4);
  EXPECT_EQ(R.first->second, 7);
  EXPECT_TRUE(R.second);

  EXPECT_EQ(MV.size(), 2u);
  EXPECT_EQ(MV[1], 2);
  EXPECT_EQ(MV[4], 7);
}

TEST(SmallMapVectorLargeTest, erase) {
  SmallMapVector<int, int, 1> MV;

  MV.insert(std::make_pair(1, 2));
  MV.insert(std::make_pair(3, 4));
  MV.insert(std::make_pair(5, 6));
  ASSERT_EQ(MV.size(), 3u);

  MV.erase(MV.find(1));
  ASSERT_EQ(MV.size(), 2u);
  ASSERT_EQ(MV.find(1), MV.end());
  ASSERT_EQ(MV[3], 4);
  ASSERT_EQ(MV[5], 6);

  ASSERT_EQ(MV.erase(3), 1u);
  ASSERT_EQ(MV.size(), 1u);
  ASSERT_EQ(MV.find(3), MV.end());
  ASSERT_EQ(MV[5], 6);

  ASSERT_EQ(MV.erase(79), 0u);
  ASSERT_EQ(MV.size(), 1u);
}

TEST(SmallMapVectorLargeTest, remove_if) {
  SmallMapVector<int, int, 1> MV;

  MV.insert(std::make_pair(1, 11));
  MV.insert(std::make_pair(2, 12));
  MV.insert(std::make_pair(3, 13));
  MV.insert(std::make_pair(4, 14));
  MV.insert(std::make_pair(5, 15));
  MV.insert(std::make_pair(6, 16));
  ASSERT_EQ(MV.size(), 6u);

  MV.remove_if([](const std::pair<int, int> &Val) { return Val.second % 2; });
  ASSERT_EQ(MV.size(), 3u);
  ASSERT_EQ(MV.find(1), MV.end());
  ASSERT_EQ(MV.find(3), MV.end());
  ASSERT_EQ(MV.find(5), MV.end());
  ASSERT_EQ(MV[2], 12);
  ASSERT_EQ(MV[4], 14);
  ASSERT_EQ(MV[6], 16);
}

TEST(SmallMapVectorLargeTest, iteration_test) {
  SmallMapVector<int, int, 1> MV;

  MV.insert(std::make_pair(1, 11));
  MV.insert(std::make_pair(2, 12));
  MV.insert(std::make_pair(3, 13));
  MV.insert(std::make_pair(4, 14));
  MV.insert(std::make_pair(5, 15));
  MV.insert(std::make_pair(6, 16));
  ASSERT_EQ(MV.size(), 6u);

  int count = 1;
  for (auto P : make_range(MV.begin(), MV.end())) {
    ASSERT_EQ(P.first, count);
    count++;
  }

  count = 6;
  for (auto P : make_range(MV.rbegin(), MV.rend())) {
    ASSERT_EQ(P.first, count);
    count--;
  }
}
