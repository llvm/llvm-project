//===- llvm/unittest/ADT/SmallSetTest.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SmallSet unit tests.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/STLExtras.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <string>

using namespace llvm;

TEST(SmallSetTest, ConstructorIteratorPair) {
  std::initializer_list<int> L = {1, 2, 3, 4, 5};
  SmallSet<int, 4> S(std::begin(L), std::end(L));
  EXPECT_THAT(S, testing::UnorderedElementsAreArray(L));
}

TEST(SmallSet, ConstructorRange) {
  std::initializer_list<int> L = {1, 2, 3, 4, 5};

  SmallSet<int, 4> S(llvm::make_range(std::begin(L), std::end(L)));
  EXPECT_THAT(S, testing::UnorderedElementsAreArray(L));
}

TEST(SmallSet, ConstructorInitializerList) {
  std::initializer_list<int> L = {1, 2, 3, 4, 5};
  SmallSet<int, 4> S = {1, 2, 3, 4, 5};
  EXPECT_THAT(S, testing::UnorderedElementsAreArray(L));
}

TEST(SmallSet, CopyConstructor) {
  SmallSet<int, 4> S = {1, 2, 3};
  SmallSet<int, 4> T = S;

  EXPECT_THAT(S, testing::ContainerEq(T));
}

TEST(SmallSet, MoveConstructor) {
  std::initializer_list<int> L = {1, 2, 3};
  SmallSet<int, 4> S = L;
  SmallSet<int, 4> T = std::move(S);

  EXPECT_THAT(T, testing::UnorderedElementsAreArray(L));
}

TEST(SmallSet, CopyAssignment) {
  SmallSet<int, 4> S = {1, 2, 3};
  SmallSet<int, 4> T;
  T = S;

  EXPECT_THAT(S, testing::ContainerEq(T));
}

TEST(SmallSet, MoveAssignment) {
  std::initializer_list<int> L = {1, 2, 3};
  SmallSet<int, 4> S = L;
  SmallSet<int, 4> T;
  T = std::move(S);

  EXPECT_THAT(T, testing::UnorderedElementsAreArray(L));
}

TEST(SmallSetTest, Insert) {

  SmallSet<int, 4> s1;

  for (int i = 0; i < 4; i++) {
    auto InsertResult = s1.insert(i);
    EXPECT_EQ(*InsertResult.first, i);
    EXPECT_EQ(InsertResult.second, true);
  }

  for (int i = 0; i < 4; i++) {
    auto InsertResult = s1.insert(i);
    EXPECT_EQ(*InsertResult.first, i);
    EXPECT_EQ(InsertResult.second, false);
  }

  EXPECT_EQ(4u, s1.size());

  for (int i = 0; i < 4; i++)
    EXPECT_EQ(1u, s1.count(i));

  EXPECT_EQ(0u, s1.count(4));
}

TEST(SmallSetTest, InsertPerfectFwd) {
  struct Value {
    int Key;
    bool Moved;

    Value(int Key) : Key(Key), Moved(false) {}
    Value(const Value &) = default;
    Value(Value &&Other) : Key(Other.Key), Moved(false) { Other.Moved = true; }
    bool operator==(const Value &Other) const { return Key == Other.Key; }
    bool operator<(const Value &Other) const { return Key < Other.Key; }
  };

  {
    SmallSet<Value, 4> S;
    Value V1(1), V2(2);

    S.insert(V1);
    EXPECT_EQ(V1.Moved, false);

    S.insert(std::move(V2));
    EXPECT_EQ(V2.Moved, true);
  }
  {
    SmallSet<Value, 1> S;
    Value V1(1), V2(2);

    S.insert(V1);
    EXPECT_EQ(V1.Moved, false);

    S.insert(std::move(V2));
    EXPECT_EQ(V2.Moved, true);
  }
}

TEST(SmallSetTest, Grow) {
  SmallSet<int, 4> s1;

  for (int i = 0; i < 8; i++) {
    auto InsertResult = s1.insert(i);
    EXPECT_EQ(*InsertResult.first, i);
    EXPECT_EQ(InsertResult.second, true);
  }

  for (int i = 0; i < 8; i++) {
    auto InsertResult = s1.insert(i);
    EXPECT_EQ(*InsertResult.first, i);
    EXPECT_EQ(InsertResult.second, false);
  }

  EXPECT_EQ(8u, s1.size());

  for (int i = 0; i < 8; i++)
    EXPECT_EQ(1u, s1.count(i));

  EXPECT_EQ(0u, s1.count(8));
}

TEST(SmallSetTest, Erase) {
  SmallSet<int, 4> s1;

  for (int i = 0; i < 8; i++)
    s1.insert(i);

  EXPECT_EQ(8u, s1.size());

  // Remove elements one by one and check if all other elements are still there.
  for (int i = 0; i < 8; i++) {
    EXPECT_EQ(1u, s1.count(i));
    EXPECT_TRUE(s1.erase(i));
    EXPECT_EQ(0u, s1.count(i));
    EXPECT_EQ(8u - i - 1, s1.size());
    for (int j = i + 1; j < 8; j++)
      EXPECT_EQ(1u, s1.count(j));
  }

  EXPECT_EQ(0u, s1.count(8));
}

TEST(SmallSetTest, IteratorInt) {
  SmallSet<int, 4> s1;

  // Test the 'small' case.
  for (int i = 0; i < 3; i++)
    s1.insert(i);

  std::vector<int> V(s1.begin(), s1.end());
  // Make sure the elements are in the expected order.
  llvm::sort(V);
  for (int i = 0; i < 3; i++)
    EXPECT_EQ(i, V[i]);

  // Test the 'big' case by adding a few more elements to switch to std::set
  // internally.
  for (int i = 3; i < 6; i++)
    s1.insert(i);

  V.assign(s1.begin(), s1.end());
  // Make sure the elements are in the expected order.
  llvm::sort(V);
  for (int i = 0; i < 6; i++)
    EXPECT_EQ(i, V[i]);
}

TEST(SmallSetTest, IteratorString) {
  // Test SmallSetIterator for SmallSet with a type with non-trivial
  // ctors/dtors.
  SmallSet<std::string, 2> s1;

  s1.insert("str 1");
  s1.insert("str 2");
  s1.insert("str 1");

  std::vector<std::string> V(s1.begin(), s1.end());
  llvm::sort(V);
  EXPECT_EQ(2u, s1.size());
  EXPECT_EQ("str 1", V[0]);
  EXPECT_EQ("str 2", V[1]);

  s1.insert("str 4");
  s1.insert("str 0");
  s1.insert("str 4");

  V.assign(s1.begin(), s1.end());
  // Make sure the elements are in the expected order.
  llvm::sort(V);
  EXPECT_EQ(4u, s1.size());
  EXPECT_EQ("str 0", V[0]);
  EXPECT_EQ("str 1", V[1]);
  EXPECT_EQ("str 2", V[2]);
  EXPECT_EQ("str 4", V[3]);
}

TEST(SmallSetTest, IteratorIncMoveCopy) {
  // Test SmallSetIterator for SmallSet with a type with non-trivial
  // ctors/dtors.
  SmallSet<std::string, 2> s1;

  s1.insert("str 1");
  s1.insert("str 2");

  auto Iter = s1.begin();
  EXPECT_EQ("str 1", *Iter);
  ++Iter;
  EXPECT_EQ("str 2", *Iter);

  s1.insert("str 4");
  s1.insert("str 0");
  auto Iter2 = s1.begin();
  Iter = std::move(Iter2);
  EXPECT_EQ("str 0", *Iter);
}

TEST(SmallSetTest, EqualityComparisonTest) {
  SmallSet<int, 8> s1small;
  SmallSet<int, 10> s2small;
  SmallSet<int, 3> s3large;
  SmallSet<int, 8> s4large;

  for (int i = 1; i < 5; i++) {
    s1small.insert(i);
    s2small.insert(5 - i);
    s3large.insert(i);
  }
  for (int i = 1; i < 11; i++)
    s4large.insert(i);

  EXPECT_EQ(s1small, s1small);
  EXPECT_EQ(s3large, s3large);

  EXPECT_EQ(s1small, s2small);
  EXPECT_EQ(s1small, s3large);
  EXPECT_EQ(s2small, s3large);

  EXPECT_NE(s1small, s4large);
  EXPECT_NE(s4large, s3large);
}

TEST(SmallSetTest, Contains) {
  SmallSet<int, 2> Set;
  EXPECT_FALSE(Set.contains(0));
  EXPECT_FALSE(Set.contains(1));

  Set.insert(0);
  Set.insert(1);
  EXPECT_TRUE(Set.contains(0));
  EXPECT_TRUE(Set.contains(1));

  Set.insert(1);
  EXPECT_TRUE(Set.contains(0));
  EXPECT_TRUE(Set.contains(1));

  Set.erase(1);
  EXPECT_TRUE(Set.contains(0));
  EXPECT_FALSE(Set.contains(1));

  Set.insert(1);
  Set.insert(2);
  EXPECT_TRUE(Set.contains(0));
  EXPECT_TRUE(Set.contains(1));
  EXPECT_TRUE(Set.contains(2));
}
