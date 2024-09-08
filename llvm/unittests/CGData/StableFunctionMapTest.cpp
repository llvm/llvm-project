//===- StableFunctionMapTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CGData/StableFunctionMap.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(StableFunctionMap, Name) {
  StableFunctionMap Map;
  EXPECT_TRUE(Map.empty());
  EXPECT_TRUE(Map.getNames().empty());
  unsigned ID1 = Map.getIdOrCreateForName("Func1");
  unsigned ID2 = Map.getIdOrCreateForName("Func2");
  unsigned ID3 = Map.getIdOrCreateForName("Func1");

  EXPECT_EQ(Map.getNames().size(), 2u);
  // The different names should return different IDs.
  EXPECT_NE(ID1, ID2);
  // The same name should return the same ID.
  EXPECT_EQ(ID1, ID3);
  // The IDs should be valid.
  EXPECT_EQ(*Map.getNameForId(ID1), "Func1");
  EXPECT_EQ(*Map.getNameForId(ID2), "Func2");
}

TEST(StableFunctionMap, Insert) {
  StableFunctionMap Map;

  StableFunction Func1{1, "Func1", "Mod1", 2, {{{0, 1}, 3}}};
  StableFunction Func2{1, "Func2", "Mod1", 2, {{{0, 1}, 2}}};
  Map.insert(Func1);
  Map.insert(Func2);
  // We only have a unique hash, 1
  EXPECT_EQ(Map.size(), 1u);
  // We have two functions with the same hash which are potentially mergeable.
  EXPECT_EQ(Map.size(StableFunctionMap::SizeType::TotalFunctionCount), 2u);
  EXPECT_EQ(Map.size(StableFunctionMap::SizeType::MergeableFunctionCount), 2u);
}

TEST(StableFunctionMap, InsertEntry) {
  StableFunctionMap Map;

  unsigned ID1 = Map.getIdOrCreateForName("Func1");
  unsigned ID2 = Map.getIdOrCreateForName("Mod1");
  unsigned ID3 = Map.getIdOrCreateForName("Func2");

  // Create a function entry and insert it into the map.
  auto IndexOperandHashMap1 = std::make_unique<IndexOperandHashMapType>();
  IndexOperandHashMap1->insert({{0, 1}, 3});
  auto FuncEntry1 = std::make_unique<StableFunctionEntry>(
      1, ID1, ID2, 2, std::move(IndexOperandHashMap1));
  Map.insert(std::move(FuncEntry1));

  // Create another function entry and insert it into the map.
  auto IndexOperandHashMap2 = std::make_unique<IndexOperandHashMapType>();
  IndexOperandHashMap2->insert({{0, 1}, 2});
  auto FuncEntry2 = std::make_unique<StableFunctionEntry>(
      1, ID3, ID2, 2, std::move(IndexOperandHashMap2));
  Map.insert(std::move(FuncEntry2));

  // We only have a unique hash, 1
  EXPECT_EQ(Map.size(), 1u);
  // We have two functions with the same hash which are potentially mergeable.
  EXPECT_EQ(Map.size(StableFunctionMap::SizeType::TotalFunctionCount), 2u);
}

TEST(OutlinedHashTreeTest, Merge) {
  StableFunctionMap Map1;
  StableFunction Func1{1, "Func1", "Mod1", 2, {{{0, 1}, 3}}};
  StableFunction Func2{1, "Func2", "Mod1", 2, {{{0, 1}, 2}}};
  StableFunction Func3{2, "Func3", "Mod1", 2, {{{1, 1}, 2}}};
  Map1.insert(Func1);
  Map1.insert(Func2);
  Map1.insert(Func3);

  StableFunctionMap Map2;
  StableFunction Func4{1, "Func4", "Mod2", 2, {{{0, 1}, 4}}};
  StableFunction Func5{2, "Func5", "Mod2", 2, {{{1, 1}, 5}}};
  StableFunction Func6{3, "Func6", "Mod2", 2, {{{1, 1}, 6}}};
  Map2.insert(Func4);
  Map2.insert(Func5);
  Map2.insert(Func6);

  // Merge two maps.
  Map1.merge(Map2);

  // We only have two unique hashes, 1, 2 and 3
  EXPECT_EQ(Map1.size(), 3u);
  // We have total 6 functions.
  EXPECT_EQ(Map1.size(StableFunctionMap::SizeType::TotalFunctionCount), 6u);
  // We have 5 mergeable functions. Func6 only has a unique hash, 3.
  EXPECT_EQ(Map1.size(StableFunctionMap::SizeType::MergeableFunctionCount), 5u);
}

} // end namespace
