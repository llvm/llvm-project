//===- llvm/unittest/ADT/RadixTreeTest.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/RadixTree.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iterator>
#include <list>
#include <vector>

using namespace llvm;
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

// Test with StringRef.

TEST(RadixTreeTest, Empty) {
  RadixTree<StringRef, int> T;
  EXPECT_TRUE(T.empty());
  EXPECT_EQ(T.size(), 0u);

  EXPECT_TRUE(T.find_prefixes("").empty());
  EXPECT_TRUE(T.find_prefixes("A").empty());

  EXPECT_EQ(T.countNodes(), 1u);
}

TEST(RadixTreeTest, InsertEmpty) {
  RadixTree<StringRef, int> T;
  auto [It, IsNew] = T.emplace("", 4);
  EXPECT_TRUE(!T.empty());
  EXPECT_EQ(T.size(), 1u);
  EXPECT_TRUE(IsNew);
  const auto &[K, V] = *It;
  EXPECT_TRUE(K.empty());
  EXPECT_EQ(4, V);

  EXPECT_THAT(T, ElementsAre(Pair("", 4)));

  EXPECT_THAT(T.find_prefixes(""), ElementsAre(Pair("", 4)));

  EXPECT_THAT(T.find_prefixes("a"), ElementsAre(Pair("", 4)));

  EXPECT_EQ(T.countNodes(), 1u);
}

TEST(RadixTreeTest, Complex) {
  RadixTree<StringRef, int> T;
  T.emplace("abcd", 1);
  EXPECT_EQ(T.countNodes(), 2u);
  T.emplace("abklm", 2);
  EXPECT_EQ(T.countNodes(), 4u);
  T.emplace("123abklm", 3);
  EXPECT_EQ(T.countNodes(), 5u);
  T.emplace("123abklm", 4);
  EXPECT_EQ(T.countNodes(), 5u);
  T.emplace("ab", 5);
  EXPECT_EQ(T.countNodes(), 5u);
  T.emplace("1234567", 6);
  EXPECT_EQ(T.countNodes(), 7u);
  T.emplace("123456", 7);
  EXPECT_EQ(T.countNodes(), 8u);
  T.emplace("123456789", 8);
  EXPECT_EQ(T.countNodes(), 9u);

  EXPECT_THAT(T, UnorderedElementsAre(Pair("abcd", 1), Pair("abklm", 2),
                                      Pair("123abklm", 3), Pair("ab", 5),
                                      Pair("1234567", 6), Pair("123456", 7),
                                      Pair("123456789", 8)));

  EXPECT_THAT(T.find_prefixes("1234567890"),
              UnorderedElementsAre(Pair("1234567", 6), Pair("123456", 7),
                                   Pair("123456789", 8)));

  EXPECT_THAT(T.find_prefixes("123abklm"),
              UnorderedElementsAre(Pair("123abklm", 3)));

  EXPECT_THAT(T.find_prefixes("abcdefg"),
              UnorderedElementsAre(Pair("abcd", 1), Pair("ab", 5)));

  EXPECT_EQ(T.countNodes(), 9u);
}

TEST(RadixTreeTest, ValueWith2Parameters) {
  RadixTree<StringRef, std::pair<std::string, int>> T;
  T.emplace("abcd", "a", 3);

  EXPECT_THAT(T, UnorderedElementsAre(Pair("abcd", Pair("a", 3))));
}

// Test different types, less readable.

template <typename T> struct TestData {
  static const T Data1[];
  static const T Data2[];
};

template <> const char TestData<char>::Data1[] = "abcdedcba";
template <> const char TestData<char>::Data2[] = "abCDEDCba";

template <> const int TestData<int>::Data1[] = {1, 2, 3, 4, 5, 4, 3, 2, 1};
template <> const int TestData<int>::Data2[] = {1, 2, 4, 8, 16, 8, 4, 2, 1};

template <typename T> class RadixTreeTypeTest : public ::testing::Test {
public:
  using IteratorType = decltype(adl_begin(std::declval<const T &>()));
  using CharType = remove_cvref_t<decltype(*adl_begin(std::declval<T &>()))>;

  T make(const CharType *Data, size_t N) { return T(StringRef(Data, N)); }

  T make1(size_t N) { return make(TestData<CharType>::Data1, N); }
  T make2(size_t N) { return make(TestData<CharType>::Data2, N); }
};

template <>
iterator_range<StringRef::const_iterator>
RadixTreeTypeTest<iterator_range<StringRef::const_iterator>>::make(
    const char *Data, size_t N) {
  return StringRef(Data).take_front(N);
}

template <>
iterator_range<StringRef::const_reverse_iterator>
RadixTreeTypeTest<iterator_range<StringRef::const_reverse_iterator>>::make(
    const char *Data, size_t N) {
  return reverse(StringRef(Data).take_back(N));
}

template <>
ArrayRef<int> RadixTreeTypeTest<ArrayRef<int>>::make(const int *Data,
                                                     size_t N) {
  return ArrayRef<int>(Data, Data + N);
}

template <>
std::vector<int> RadixTreeTypeTest<std::vector<int>>::make(const int *Data,
                                                           size_t N) {
  return std::vector<int>(Data, Data + N);
}

template <>
std::list<int> RadixTreeTypeTest<std::list<int>>::make(const int *Data,
                                                       size_t N) {
  return std::list<int>(Data, Data + N);
}

class TypeNameGenerator {
public:
  template <typename T> static std::string GetName(int) {
    if (std::is_same_v<T, StringRef>)
      return "StringRef";
    if (std::is_same_v<T, std::string>)
      return "string";
    if (std::is_same_v<T, iterator_range<StringRef::const_iterator>>)
      return "iterator_range";
    if (std::is_same_v<T, iterator_range<StringRef::const_reverse_iterator>>)
      return "reverse_iterator_range";
    if (std::is_same_v<T, ArrayRef<int>>)
      return "ArrayRef";
    if (std::is_same_v<T, std::vector<int>>)
      return "vector";
    if (std::is_same_v<T, std::list<int>>)
      return "list";
    return "Unknown";
  }
};

using TestTypes =
    ::testing::Types<StringRef, std::string,
                     iterator_range<StringRef::const_iterator>,
                     iterator_range<StringRef::const_reverse_iterator>,
                     ArrayRef<int>, std::vector<int>, std::list<int>>;

TYPED_TEST_SUITE(RadixTreeTypeTest, TestTypes, TypeNameGenerator);

TYPED_TEST(RadixTreeTypeTest, Helpers) {
  for (size_t i = 0; i < 9; ++i) {
    auto R1 = this->make1(i);
    auto R2 = this->make2(i);
    EXPECT_EQ(llvm::range_size(R1), i);
    EXPECT_EQ(llvm::range_size(R2), i);
    auto [I1, I2] = llvm::mismatch(R1, R2);
    // Exactly 2 first elements of Data1 and Data2 must match.
    EXPECT_EQ(std::distance(R1.begin(), I1), std::min<int>(2, i));
  }
}

TYPED_TEST(RadixTreeTypeTest, Empty) {
  RadixTree<TypeParam, int> T;
  EXPECT_TRUE(T.empty());
  EXPECT_EQ(T.size(), 0u);

  EXPECT_TRUE(T.find_prefixes(this->make1(0)).empty());
  EXPECT_TRUE(T.find_prefixes(this->make2(1)).empty());

  EXPECT_EQ(T.countNodes(), 1u);
}

TYPED_TEST(RadixTreeTypeTest, InsertEmpty) {
  using TreeType = RadixTree<TypeParam, int>;
  TreeType T;
  auto [It, IsNew] = T.emplace(this->make1(0), 5);
  EXPECT_TRUE(!T.empty());
  EXPECT_EQ(T.size(), 1u);
  EXPECT_TRUE(IsNew);
  const auto &[K, V] = *It;
  EXPECT_TRUE(K.empty());
  EXPECT_EQ(5, V);

  EXPECT_THAT(T.find_prefixes(this->make1(0)),
              ElementsAre(Pair(ElementsAre(), 5)));

  EXPECT_THAT(T.find_prefixes(this->make2(1)),
              ElementsAre(Pair(ElementsAre(), 5)));

  EXPECT_THAT(T, ElementsAre(Pair(ElementsAre(), 5)));

  EXPECT_EQ(T.countNodes(), 1u);
}

TYPED_TEST(RadixTreeTypeTest, InsertEmptyTwice) {
  using TreeType = RadixTree<TypeParam, int>;
  TreeType T;
  T.emplace(this->make1(0), 5);
  auto [It, IsNew] = T.emplace(this->make1(0), 6);
  EXPECT_TRUE(!T.empty());
  EXPECT_EQ(T.size(), 1u);
  EXPECT_TRUE(!IsNew);
  const auto &[K, V] = *It;
  EXPECT_TRUE(K.empty());
  EXPECT_EQ(5, V);

  EXPECT_THAT(T.find_prefixes(this->make1(0)),
              ElementsAre(Pair(ElementsAre(), 5)));

  EXPECT_THAT(T.find_prefixes(this->make2(1)),
              ElementsAre(Pair(ElementsAre(), 5)));

  EXPECT_THAT(T, ElementsAre(Pair(ElementsAre(), 5)));

  EXPECT_EQ(T.countNodes(), 1u);
}

TYPED_TEST(RadixTreeTypeTest, InsertOne) {
  using TreeType = RadixTree<TypeParam, int>;
  TreeType T;
  auto [It, IsNew] = T.emplace(this->make1(1), 4);
  EXPECT_TRUE(!T.empty());
  EXPECT_EQ(T.size(), 1u);
  EXPECT_TRUE(IsNew);
  const auto &[K, V] = *It;
  EXPECT_THAT(K, ElementsAreArray(this->make1(1)));
  EXPECT_EQ(4, V);

  EXPECT_THAT(T, ElementsAre(Pair(ElementsAreArray(this->make1(1)), 4)));

  EXPECT_THAT(T.find_prefixes(this->make1(1)),
              ElementsAre(Pair(ElementsAreArray(this->make1(1)), 4)));

  EXPECT_THAT(T.find_prefixes(this->make1(2)),
              ElementsAre(Pair(ElementsAreArray(this->make1(1)), 4)));

  EXPECT_EQ(T.countNodes(), 2u);
}

TYPED_TEST(RadixTreeTypeTest, InsertOneTwice) {
  using TreeType = RadixTree<TypeParam, int>;
  TreeType T;
  T.emplace(this->make1(1), 4);
  auto [It, IsNew] = T.emplace(this->make1(1), 4);
  EXPECT_TRUE(!T.empty());
  EXPECT_EQ(T.size(), 1u);
  EXPECT_TRUE(!IsNew);

  EXPECT_THAT(T, ElementsAre(Pair(ElementsAreArray(this->make1(1)), 4)));
  EXPECT_EQ(T.countNodes(), 2u);
}

TYPED_TEST(RadixTreeTypeTest, InsertSuperStrings) {
  using TreeType = RadixTree<TypeParam, int>;
  TreeType T;

  for (size_t Len = 0; Len < 7; Len += 2) {
    auto [It, IsNew] = T.emplace(this->make1(Len), Len);
    EXPECT_TRUE(IsNew);
  }

  EXPECT_THAT(T,
              UnorderedElementsAre(Pair(ElementsAreArray(this->make1(0)), 0),
                                   Pair(ElementsAreArray(this->make1(2)), 2),
                                   Pair(ElementsAreArray(this->make1(4)), 4),
                                   Pair(ElementsAreArray(this->make1(6)), 6)));

  EXPECT_THAT(T.find_prefixes(this->make1(0)),
              UnorderedElementsAre(Pair(ElementsAreArray(this->make1(0)), 0)));

  EXPECT_THAT(T.find_prefixes(this->make1(3)),
              UnorderedElementsAre(Pair(ElementsAreArray(this->make1(0)), 0),
                                   Pair(ElementsAreArray(this->make1(2)), 2)));

  EXPECT_THAT(T.find_prefixes(this->make1(7)),
              UnorderedElementsAre(Pair(ElementsAreArray(this->make1(0)), 0),
                                   Pair(ElementsAreArray(this->make1(2)), 2),
                                   Pair(ElementsAreArray(this->make1(4)), 4),
                                   Pair(ElementsAreArray(this->make1(6)), 6)));

  EXPECT_EQ(T.countNodes(), 4u);
}

TYPED_TEST(RadixTreeTypeTest, InsertSubStrings) {
  using TreeType = RadixTree<TypeParam, int>;
  TreeType T;

  for (size_t Len = 0; Len < 7; Len += 2) {
    auto [It, IsNew] = T.emplace(this->make1(7 - Len), 7 - Len);
    EXPECT_TRUE(IsNew);
  }

  EXPECT_THAT(T,
              UnorderedElementsAre(Pair(ElementsAreArray(this->make1(1)), 1),
                                   Pair(ElementsAreArray(this->make1(3)), 3),
                                   Pair(ElementsAreArray(this->make1(5)), 5),
                                   Pair(ElementsAreArray(this->make1(7)), 7)));

  EXPECT_THAT(T.find_prefixes(this->make1(0)), UnorderedElementsAre());

  EXPECT_THAT(T.find_prefixes(this->make1(3)),
              UnorderedElementsAre(Pair(ElementsAreArray(this->make1(1)), 1),
                                   Pair(ElementsAreArray(this->make1(3)), 3)));

  EXPECT_THAT(T.find_prefixes(this->make1(6)),
              UnorderedElementsAre(Pair(ElementsAreArray(this->make1(1)), 1),
                                   Pair(ElementsAreArray(this->make1(3)), 3),
                                   Pair(ElementsAreArray(this->make1(5)), 5)));

  EXPECT_EQ(T.countNodes(), 5u);
}

TYPED_TEST(RadixTreeTypeTest, InsertVShape) {
  using TreeType = RadixTree<TypeParam, int>;
  TreeType T;

  EXPECT_EQ(T.countNodes(), 1u);
  T.emplace(this->make1(5), 15);
  EXPECT_EQ(T.countNodes(), 2u);
  T.emplace(this->make2(6), 26);
  EXPECT_EQ(T.countNodes(), 4u);
  T.emplace(this->make2(1), 21);
  EXPECT_EQ(T.countNodes(), 5u);

  EXPECT_THAT(T,
              UnorderedElementsAre(Pair(ElementsAreArray(this->make1(5)), 15),
                                   Pair(ElementsAreArray(this->make2(6)), 26),
                                   Pair(ElementsAreArray(this->make2(1)), 21)));

  EXPECT_THAT(T.find_prefixes(this->make1(7)),
              UnorderedElementsAre(Pair(ElementsAreArray(this->make2(1)), 21),
                                   Pair(ElementsAreArray(this->make1(5)), 15)));

  EXPECT_THAT(T.find_prefixes(this->make2(7)),
              UnorderedElementsAre(Pair(ElementsAreArray(this->make2(1)), 21),
                                   Pair(ElementsAreArray(this->make2(6)), 26)));

  EXPECT_EQ(T.countNodes(), 5u);
}

} // namespace
