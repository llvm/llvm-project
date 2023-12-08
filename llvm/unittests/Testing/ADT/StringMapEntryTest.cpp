//===- StringMapEntryTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Testing/ADT/StringMapEntry.h"
#include "llvm/ADT/StringMap.h"

#include "gtest/gtest.h"
#include <sstream>

namespace llvm {
namespace {

using testing::Gt;
using testing::Matcher;
using testing::StrCaseEq;
using testing::StringMatchResultListener;
using testing::UnorderedElementsAre;

template <typename T> std::string Describe(const Matcher<T> &M, bool Match) {
  std::stringstream SS;
  if (Match) {
    M.DescribeTo(&SS);
  } else {
    M.DescribeNegationTo(&SS);
  }
  return SS.str();
}

template <typename T, typename V>
std::string ExplainMatch(const Matcher<T> &Matcher, const V &Value) {
  StringMatchResultListener Listener;
  Matcher.MatchAndExplain(Value, &Listener);
  return Listener.str();
}

TEST(IsStringMapEntryTest, InnerMatchersAreExactValues) {
  llvm::StringMap<int> Map = {{"A", 1}};
  EXPECT_THAT(*Map.find("A"), IsStringMapEntry("A", 1));
}

TEST(IsStringMapEntryTest, InnerMatchersAreOtherMatchers) {
  llvm::StringMap<int> Map = {{"A", 1}};
  EXPECT_THAT(*Map.find("A"), IsStringMapEntry(StrCaseEq("a"), Gt(0)));
}

TEST(IsStringMapEntryTest, UseAsInnerMatcher) {
  llvm::StringMap<int> Map = {{"A", 1}, {"B", 2}};
  EXPECT_THAT(Map, UnorderedElementsAre(IsStringMapEntry("A", 1),
                                        IsStringMapEntry("B", 2)));
}

TEST(IsStringMapEntryTest, DescribeSelf) {
  Matcher<llvm::StringMapEntry<int>> M = IsStringMapEntry("A", 1);
  EXPECT_EQ(
      R"(has a string key that is equal to "A", and has a value that is equal to 1)",
      Describe(M, true));
  EXPECT_EQ(
      R"(has a string key that isn't equal to "A", or has a value that isn't equal to 1)",
      Describe(M, false));
}

TEST(IsStringMapEntryTest, ExplainSelfMatchSuccess) {
  llvm::StringMap<int> Map = {{"A", 1}};
  Matcher<llvm::StringMapEntry<int>> M = IsStringMapEntry("A", 1);
  EXPECT_EQ(R"(which is a match)", ExplainMatch(M, *Map.find("A")));
}

TEST(IsStringMapEntryTest, ExplainSelfMatchFailsOnKey) {
  llvm::StringMap<int> Map = {{"B", 1}};
  Matcher<llvm::StringMapEntry<int>> M = IsStringMapEntry("A", 1);
  EXPECT_EQ(R"(which has a string key that doesn't match)",
            ExplainMatch(M, *Map.find("B")));
}

TEST(IsStringMapEntryTest, ExplainSelfMatchFailsOnValue) {
  llvm::StringMap<int> Map = {{"A", 2}};
  Matcher<llvm::StringMapEntry<int>> M = IsStringMapEntry("A", 1);
  EXPECT_EQ(R"(which has a value that doesn't match)",
            ExplainMatch(M, *Map.find("A")));
}

} // namespace
} // namespace llvm
