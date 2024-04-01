//===- SampleProfileMatcherTests.cpp - SampleProfileMatcher Unit Tests -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/SampleProfileMatcher.h"
#include "gtest/gtest.h"

using namespace llvm;

MyersDiff Diff;

std::vector<Anchor>
createAnchorsFromStrings(const std::vector<std::string> &SV) {
  std::vector<Anchor> Anchors;
  for (uint64_t I = 0; I < SV.size(); I++) {
    Anchors.push_back(Anchor(LineLocation(I, 0), FunctionId(SV[I])));
  }
  return Anchors;
}

LocToLocMap
createEqualLocations(const std::vector<std::pair<uint32_t, uint32_t>> &V) {
  LocToLocMap LocMap;
  for (auto P : V) {
    LocMap.emplace(LineLocation(P.first, 0), LineLocation(P.second, 0));
  }
  return LocMap;
}

std::vector<LineLocation> createLocations(const std::vector<uint32_t> &V) {
  std::vector<LineLocation> Locations;
  for (auto I : V) {
    Locations.emplace_back(LineLocation(I, 0));
  }
  return Locations;
}

TEST(SampleProfileMatcherTests, MyersDiffTest1) {

  std::vector<Anchor> AnchorsA;
  std::vector<Anchor> AnchorsB;
  auto R = Diff.shortestEdit(AnchorsA, AnchorsB);
  EXPECT_TRUE(R.EqualLocations.empty());
  EXPECT_TRUE(R.Deletions.empty());
  EXPECT_TRUE(R.Insertions.empty());
}

TEST(SampleProfileMatcherTests, MyersDiffTest2) {
  std::vector<std::string> A({"a", "b", "c"});
  std::vector<Anchor> AnchorsA = createAnchorsFromStrings(A);
  std::vector<Anchor> AnchorsB;
  auto R = Diff.shortestEdit(AnchorsA, AnchorsB);
  EXPECT_TRUE(R.EqualLocations.empty());
  EXPECT_EQ(R.Insertions, createLocations(std::vector<uint32_t>({2, 1, 0})));
  EXPECT_TRUE(R.Deletions.empty());
}

TEST(SampleProfileMatcherTests, MyersDiffTest3) {

  std::vector<Anchor> AnchorsA;
  std::vector<std::string> B({"a", "b", "c"});
  std::vector<Anchor> AnchorsB = createAnchorsFromStrings(B);
  auto R = Diff.shortestEdit(AnchorsA, AnchorsB);
  EXPECT_TRUE(R.EqualLocations.empty());
  EXPECT_TRUE(R.Insertions.empty());
  EXPECT_EQ(R.Deletions, createLocations(std::vector<uint32_t>({2, 1, 0})));
}

TEST(SampleProfileMatcherTests, MyersDiffTest4) {
  std::vector<std::string> A({"a", "b", "c"});
  std::vector<std::string> B({"a", "b", "c"});
  std::vector<Anchor> AnchorsA = createAnchorsFromStrings(A);
  std::vector<Anchor> AnchorsB = createAnchorsFromStrings(B);
  LocToLocMap ExpectEqualLocations =
      createEqualLocations({{0, 0}, {1, 1}, {2, 2}});
  auto R = Diff.shortestEdit(AnchorsA, AnchorsB);
  EXPECT_EQ(R.EqualLocations, ExpectEqualLocations);
  EXPECT_TRUE(R.Insertions.empty());
  EXPECT_TRUE(R.Deletions.empty());
}

TEST(SampleProfileMatcherTests, MyersDiffTest5) {
  std::vector<std::string> A({"a", "b", "c"});
  std::vector<std::string> B({"b", "c", "d"});
  std::vector<Anchor> AnchorsA = createAnchorsFromStrings(A);
  std::vector<Anchor> AnchorsB = createAnchorsFromStrings(B);
  LocToLocMap ExpectEqualLocations = createEqualLocations({{1, 0}, {2, 1}});
  auto R = Diff.shortestEdit(AnchorsA, AnchorsB);
  EXPECT_EQ(R.EqualLocations, ExpectEqualLocations);
  EXPECT_EQ(R.Insertions, createLocations(std::vector<uint32_t>({0})));
  EXPECT_EQ(R.Deletions, createLocations(std::vector<uint32_t>({2})));
}

TEST(SampleProfileMatcherTests, MyersDiffTest6) {
  std::vector<std::string> A({"a", "b", "d"});
  std::vector<std::string> B({"a", "c", "d"});
  std::vector<Anchor> AnchorsA = createAnchorsFromStrings(A);
  std::vector<Anchor> AnchorsB = createAnchorsFromStrings(B);
  LocToLocMap ExpectEqualLocations = createEqualLocations({{0, 0}, {2, 2}});
  auto R = Diff.shortestEdit(AnchorsA, AnchorsB);
  EXPECT_EQ(R.EqualLocations, ExpectEqualLocations);
  EXPECT_EQ(R.Insertions, createLocations(std::vector<uint32_t>({1})));
  EXPECT_EQ(R.Deletions, createLocations(std::vector<uint32_t>({1})));
}

TEST(SampleProfileMatcherTests, MyersDiffTest7) {
  std::vector<std::string> A({"a", "b", "c", "a", "b", "b", "a"});
  std::vector<std::string> B({"c", "b", "a", "b", "a", "c"});
  std::vector<Anchor> AnchorsA = createAnchorsFromStrings(A);
  std::vector<Anchor> AnchorsB = createAnchorsFromStrings(B);
  LocToLocMap ExpectEqualLocations =
      createEqualLocations({{2, 0}, {3, 2}, {4, 3}, {6, 4}});
  auto R = Diff.shortestEdit(AnchorsA, AnchorsB);
  EXPECT_EQ(R.EqualLocations, ExpectEqualLocations);
  EXPECT_EQ(R.Insertions, createLocations(std::vector<uint32_t>({5, 1, 0})));
  EXPECT_EQ(R.Deletions, createLocations(std::vector<uint32_t>({5, 1})));
}

TEST(SampleProfileMatcherTests, MyersDiffTest8) {
  std::vector<std::string> A({"a", "c", "b", "c", "b", "d", "e"});
  std::vector<std::string> B({"a", "b", "c", "a", "a", "b", "c", "c", "d"});
  std::vector<Anchor> AnchorsA = createAnchorsFromStrings(A);
  std::vector<Anchor> AnchorsB = createAnchorsFromStrings(B);
  LocToLocMap ExpectEqualLocations =
      createEqualLocations({{0, 0}, {2, 1}, {3, 2}, {4, 5}, {5, 8}});
  auto R = Diff.shortestEdit(AnchorsA, AnchorsB);
  EXPECT_EQ(R.EqualLocations, ExpectEqualLocations);
  EXPECT_EQ(R.Insertions, createLocations(std::vector<uint32_t>({6, 1})));
  EXPECT_EQ(R.Deletions, createLocations(std::vector<uint32_t>({7, 6, 4, 3})));
}
