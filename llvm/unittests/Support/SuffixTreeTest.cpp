//===- unittests/Support/SuffixTreeTest.cpp - suffix tree tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/SuffixTree.h"
#include "gtest/gtest.h"
#include <vector>

using namespace llvm;

namespace {

// Each example vector has a unique element at the end to represent the end of
// the string

// Tests that The SuffixTree finds a simple repetition of the substring {1, 2}
// {1, 2} twice in the provided string.
TEST(SuffixTreeTest, TestSingleRepetition) {
  std::vector<unsigned> SimpleRepetitionData = {1, 2, 1, 2, 3};
  SuffixTree ST(SimpleRepetitionData);
  std::vector<SuffixTree::RepeatedSubstring> SubStrings;
  for (auto It = ST.begin(); It != ST.end(); It++)
    SubStrings.push_back(*It);
  ASSERT_EQ(SubStrings.size(), 1u);
  EXPECT_EQ(SubStrings[0].Length, 2u);
  EXPECT_EQ(SubStrings[0].StartIndices.size(), 2u);
  for (unsigned StartIdx : SubStrings[0].StartIndices) {
    EXPECT_EQ(SimpleRepetitionData[StartIdx], 1u);
    EXPECT_EQ(SimpleRepetitionData[StartIdx + 1], 2u);
  }
}

// Tests that the SuffixTree is able to find the substrings {1, 2, 3} at
// at indices 0 and 3 as well as the substrings {2, 3} at indices 1 and 4.
// This test also serves as a flag for improvements to the suffix tree.
//
// FIXME: Right now, the longest repeated substring from a specific index is
// returned, this could be improved to return the longest repeated substring, as
// well as those that are smaller.
TEST(SuffixTreeTest, TestLongerRepetition) {
  std::vector<unsigned> RepeatedRepetitionData = {1, 2, 3, 1, 2, 3, 4};
  SuffixTree ST(RepeatedRepetitionData);
  std::vector<SuffixTree::RepeatedSubstring> SubStrings;
  for (auto It = ST.begin(); It != ST.end(); It++)
    SubStrings.push_back(*It);
  EXPECT_EQ(SubStrings.size(), 2u);
  unsigned Len;
  for (SuffixTree::RepeatedSubstring &RS : SubStrings) {
    Len = RS.Length;
    bool IsExpectedLen = (Len == 3u || Len == 2u);
    bool IsExpectedIndex;
    ASSERT_TRUE(IsExpectedLen);

    if (Len == 3u) {
      for (unsigned StartIdx : RS.StartIndices) {
        IsExpectedIndex = (StartIdx == 0u || StartIdx == 3u);
        EXPECT_TRUE(IsExpectedIndex);
        EXPECT_EQ(RepeatedRepetitionData[StartIdx], 1u);
        EXPECT_EQ(RepeatedRepetitionData[StartIdx + 1], 2u);
        EXPECT_EQ(RepeatedRepetitionData[StartIdx + 2], 3u);
      }
    } else {
      for (unsigned StartIdx : RS.StartIndices) {
        IsExpectedIndex = (StartIdx == 1u || StartIdx == 4u);
        EXPECT_TRUE(IsExpectedIndex);
        EXPECT_EQ(RepeatedRepetitionData[StartIdx], 2u);
        EXPECT_EQ(RepeatedRepetitionData[StartIdx + 1], 3u);
      }
    }
  }
}

// Tests that the SuffixTree is able to find substring {1, 1, 1, 1, 1} at
// indices 0 and 1.
//
// FIXME: Add support for detecting {1, 1} and {1, 1, 1}
TEST(SuffixTreeTest, TestSingleCharacterRepeat) {
  std::vector<unsigned> RepeatedRepetitionData = {1, 1, 1, 1, 1, 1, 2};
  std::vector<unsigned>::iterator RRDIt, RRDIt2;
  SuffixTree ST(RepeatedRepetitionData);
  std::vector<SuffixTree::RepeatedSubstring> SubStrings;
  for (auto It = ST.begin(); It != ST.end(); It++)
    SubStrings.push_back(*It);
  EXPECT_EQ(SubStrings.size(), 1u);
  for (SuffixTree::RepeatedSubstring &RS : SubStrings) {
    EXPECT_EQ(RS.StartIndices.size(),
              RepeatedRepetitionData.size() - RS.Length);
    for (unsigned StartIdx : SubStrings[0].StartIndices) {
      RRDIt = RRDIt2 = RepeatedRepetitionData.begin();
      std::advance(RRDIt, StartIdx);
      std::advance(RRDIt2, StartIdx + SubStrings[0].Length);
      ASSERT_TRUE(
          all_of(make_range<std::vector<unsigned>::iterator>(RRDIt, RRDIt2),
                 [](unsigned Elt) { return Elt == 1; }));
    }
  }
}

// The suffix tree cannot currently find repeated substrings in strings of the
// form {1, 2, 3, 1, 2, 3}, because the two {1, 2, 3}s are adjacent ("tandem
// repeats")
//
// FIXME: Teach the SuffixTree to recognize these cases.
TEST(SuffixTreeTest, TestTandemRepeat) {
  std::vector<unsigned> RepeatedRepetitionData = {1, 2, 3, 1, 2, 3};
  SuffixTree ST(RepeatedRepetitionData);
  std::vector<SuffixTree::RepeatedSubstring> SubStrings;
  for (auto It = ST.begin(); It != ST.end(); It++)
    SubStrings.push_back(*It);
  EXPECT_EQ(SubStrings.size(), 0u);
}

// Tests that the SuffixTree does not erroneously include values that are not
// in repeated substrings.  That is, only finds {1, 1} at indices 0 and 3 and
// does not include 2 and 3.
TEST(SuffixTreeTest, TestExclusion) {
  std::vector<unsigned> RepeatedRepetitionData = {1, 1, 2, 1, 1, 3};
  std::vector<unsigned>::iterator RRDIt, RRDIt2;
  SuffixTree ST(RepeatedRepetitionData);
  std::vector<SuffixTree::RepeatedSubstring> SubStrings;
  for (auto It = ST.begin(); It != ST.end(); It++)
    SubStrings.push_back(*It);
  EXPECT_EQ(SubStrings.size(), 1u);
  bool IsExpectedIndex;
  for (SuffixTree::RepeatedSubstring &RS : SubStrings) {
    for (unsigned StartIdx : RS.StartIndices) {
      IsExpectedIndex = (StartIdx == 0u || StartIdx == 3u);
      EXPECT_TRUE(IsExpectedIndex);
      RRDIt = RRDIt2 = RepeatedRepetitionData.begin();
      std::advance(RRDIt, StartIdx);
      std::advance(RRDIt2, StartIdx + RS.Length);
      EXPECT_TRUE(
          all_of(make_range<std::vector<unsigned>::iterator>(RRDIt, RRDIt2),
                 [](unsigned Elt) { return Elt == 1; }));
    }
  }
}

} // namespace
