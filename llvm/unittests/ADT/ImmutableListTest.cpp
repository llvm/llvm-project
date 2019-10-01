//===--------- ImmutableListTest.cpp - ImmutableList unit tests --*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ImmutableList.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

template <typename Fundamental> struct Wrapper : llvm::FoldingSetNode {
  Fundamental F;

  Wrapper(Fundamental F) : F(F) {}

  operator Fundamental() const { return F; }

  void Profile(FoldingSetNodeID &ID) const { ID.AddInteger(F); }
};

class ImmutableListTest : public testing::Test {};

void concat(const ImmutableList<Wrapper<char>> &L, char *Buffer) {
  int Index = 0;
  for (ImmutableList<Wrapper<char>>::iterator It = L.begin(), End = L.end();
       It != End; ++It) {
    Buffer[Index++] = *It;
  }
  Buffer[Index] = '\0';
}

TEST_F(ImmutableListTest, EmptyIntListTest) {
  ImmutableList<Wrapper<int>>::Factory f;

  EXPECT_TRUE(f.getEmptyList() == f.getEmptyList());
  EXPECT_TRUE(f.getEmptyList().isEqual(f.getEmptyList()));
  EXPECT_TRUE(f.getEmptyList().isEmpty());

  ImmutableList<Wrapper<int>> L = f.getEmptyList();
  EXPECT_EQ(nullptr, L.getTail().getInternalPointer());
  EXPECT_TRUE(L.getTail().isEmpty());
  EXPECT_TRUE(L.begin() == L.end());
}

TEST_F(ImmutableListTest, OneElemIntListTest) {
  ImmutableList<Wrapper<int>>::Factory f;
  ImmutableList<Wrapper<int>> L = f.getEmptyList();

  ImmutableList<Wrapper<int>> L2 = f.add(3, L);
  EXPECT_TRUE(L.isEmpty());
  EXPECT_FALSE(L2.isEmpty());
  EXPECT_TRUE(L2.getTail().isEmpty());

  EXPECT_FALSE(L == L2);
  EXPECT_TRUE(L == L2.getTail());
  EXPECT_FALSE(L.isEqual(L2));
  EXPECT_TRUE(L.isEqual(L2.getTail()));
  EXPECT_FALSE(L2.begin() == L2.end());
  EXPECT_TRUE(L2.begin() != L2.end());

  EXPECT_FALSE(L.contains(3));
  EXPECT_EQ(3, L2.getHead());
  EXPECT_TRUE(L2.contains(3));

  ImmutableList<Wrapper<int>> L3 = f.add(2, L);
  EXPECT_TRUE(L.isEmpty());
  EXPECT_FALSE(L3.isEmpty());
  EXPECT_FALSE(L == L3);
  EXPECT_FALSE(L.contains(2));
  EXPECT_TRUE(L3.contains(2));
  EXPECT_EQ(2, L3.getHead());

  EXPECT_FALSE(L2 == L3);
  EXPECT_FALSE(L2.contains(2));
}

// We'll store references to objects of this type.
struct Unmodifiable {
  Unmodifiable() = default;

  // We'll delete all of these special member functions to make sure no copy or
  // move happens during insertation.
  Unmodifiable(const Unmodifiable &) = delete;
  Unmodifiable(const Unmodifiable &&) = delete;
  Unmodifiable &operator=(const Unmodifiable &) = delete;
  Unmodifiable &operator=(const Unmodifiable &&) = delete;

  void doNothing() const {}

  void Profile(FoldingSetNodeID &ID) const { ID.AddPointer(this); }
};

// Mostly just a check whether ImmutableList::iterator can be instantiated
// with a reference type as a template argument.
TEST_F(ImmutableListTest, ReferenceStoringTest) {
  ImmutableList<const Unmodifiable &>::Factory f;

  Unmodifiable N;
  ImmutableList<const Unmodifiable &> L = f.create(N);
  for (ImmutableList<const Unmodifiable &>::iterator It = L.begin(),
                                                     E = L.end();
       It != E; ++It) {
    It->doNothing();
  }
}

TEST_F(ImmutableListTest, CreatingIntListTest) {
  ImmutableList<Wrapper<int>>::Factory f;

  ImmutableList<Wrapper<int>> L = f.getEmptyList();
  ImmutableList<Wrapper<int>> L2 = f.create(3);

  EXPECT_FALSE(L2.isEmpty());
  EXPECT_TRUE(L2.getTail().isEmpty());
  EXPECT_EQ(3, L2.getHead());
  EXPECT_TRUE(L.isEqual(L2.getTail()));
  EXPECT_TRUE(L2.getTail().isEqual(L));
}

TEST_F(ImmutableListTest, MultiElemIntListTest) {
  ImmutableList<Wrapper<int>>::Factory f;

  ImmutableList<Wrapper<int>> L = f.getEmptyList();
  ImmutableList<Wrapper<int>> L2 = f.add(5, f.add(4, f.add(3, L)));
  ImmutableList<Wrapper<int>> L3 = f.add(43, f.add(20, f.add(9, L2)));
  ImmutableList<Wrapper<int>> L4 = f.add(9, L2);
  ImmutableList<Wrapper<int>> L5 = f.add(9, L2);

  EXPECT_TRUE(L.isEmpty());
  EXPECT_FALSE(L2.isEmpty());
  EXPECT_FALSE(L3.isEmpty());
  EXPECT_FALSE(L4.isEmpty());

  EXPECT_FALSE(L.contains(3));
  EXPECT_FALSE(L.contains(9));

  EXPECT_TRUE(L2.contains(3));
  EXPECT_TRUE(L2.contains(4));
  EXPECT_TRUE(L2.contains(5));
  EXPECT_FALSE(L2.contains(9));
  EXPECT_FALSE(L2.contains(0));

  EXPECT_EQ(5, L2.getHead());
  EXPECT_EQ(4, L2.getTail().getHead());
  EXPECT_EQ(3, L2.getTail().getTail().getHead());

  EXPECT_TRUE(L3.contains(43));
  EXPECT_TRUE(L3.contains(20));
  EXPECT_TRUE(L3.contains(9));
  EXPECT_TRUE(L3.contains(3));
  EXPECT_TRUE(L3.contains(4));
  EXPECT_TRUE(L3.contains(5));
  EXPECT_FALSE(L3.contains(0));

  EXPECT_EQ(43, L3.getHead());
  EXPECT_EQ(20, L3.getTail().getHead());
  EXPECT_EQ(9, L3.getTail().getTail().getHead());

  EXPECT_TRUE(L3.getTail().getTail().getTail() == L2);
  EXPECT_TRUE(L2 == L3.getTail().getTail().getTail());
  EXPECT_TRUE(L3.getTail().getTail().getTail().isEqual(L2));
  EXPECT_TRUE(L2.isEqual(L3.getTail().getTail().getTail()));

  EXPECT_TRUE(L4.contains(9));
  EXPECT_TRUE(L4.contains(3));
  EXPECT_TRUE(L4.contains(4));
  EXPECT_TRUE(L4.contains(5));
  EXPECT_FALSE(L4.contains(20));
  EXPECT_FALSE(L4.contains(43));
  EXPECT_TRUE(L4.isEqual(L4));
  EXPECT_TRUE(L4.isEqual(L5));

  EXPECT_TRUE(L5.isEqual(L4));
  EXPECT_TRUE(L5.isEqual(L5));
}

template <typename Fundamental>
struct ExplicitCtorWrapper : public Wrapper<Fundamental> {
  explicit ExplicitCtorWrapper(Fundamental F) : Wrapper<Fundamental>(F) {}
  ExplicitCtorWrapper(const ExplicitCtorWrapper &) = delete;
  ExplicitCtorWrapper(ExplicitCtorWrapper &&) = default;
  ExplicitCtorWrapper &operator=(const ExplicitCtorWrapper &) = delete;
  ExplicitCtorWrapper &operator=(ExplicitCtorWrapper &&) = default;
};

TEST_F(ImmutableListTest, EmplaceIntListTest) {
  ImmutableList<ExplicitCtorWrapper<int>>::Factory f;

  ImmutableList<ExplicitCtorWrapper<int>> L = f.getEmptyList();
  ImmutableList<ExplicitCtorWrapper<int>> L2 = f.emplace(L, 3);

  ImmutableList<ExplicitCtorWrapper<int>> L3 =
      f.add(ExplicitCtorWrapper<int>(2), L2);

  ImmutableList<ExplicitCtorWrapper<int>> L4 =
      f.emplace(L3, ExplicitCtorWrapper<int>(1));

  ImmutableList<ExplicitCtorWrapper<int>> L5 =
      f.add(ExplicitCtorWrapper<int>(1), L3);

  EXPECT_FALSE(L2.isEmpty());
  EXPECT_TRUE(L2.getTail().isEmpty());
  EXPECT_EQ(3, L2.getHead());
  EXPECT_TRUE(L.isEqual(L2.getTail()));
  EXPECT_TRUE(L2.getTail().isEqual(L));

  EXPECT_FALSE(L3.isEmpty());
  EXPECT_FALSE(L2 == L3);
  EXPECT_EQ(2, L3.getHead());
  EXPECT_TRUE(L2 == L3.getTail());

  EXPECT_FALSE(L4.isEmpty());
  EXPECT_EQ(1, L4.getHead());
  EXPECT_TRUE(L3 == L4.getTail());

  EXPECT_TRUE(L4 == L5);
  EXPECT_TRUE(L3 == L5.getTail());
}

TEST_F(ImmutableListTest, CharListOrderingTest) {
  ImmutableList<Wrapper<char>>::Factory f;
  ImmutableList<Wrapper<char>> L = f.getEmptyList();

  ImmutableList<Wrapper<char>> L2 = f.add('i', f.add('e', f.add('a', L)));
  ImmutableList<Wrapper<char>> L3 = f.add('u', f.add('o', L2));

  char Buffer[10];
  concat(L3, Buffer);

  ASSERT_STREQ("uoiea", Buffer);
}

TEST_F(ImmutableListTest, LongListOrderingTest) {
  ImmutableList<Wrapper<long>>::Factory f;
  ImmutableList<Wrapper<long>> L = f.getEmptyList();

  ImmutableList<Wrapper<long>> L2 = f.add(3, f.add(4, f.add(5, L)));
  ImmutableList<Wrapper<long>> L3 = f.add(0, f.add(1, f.add(2, L2)));

  int i = 0;
  for (ImmutableList<Wrapper<long>>::iterator I = L.begin(), E = L.end();
       I != E; ++I) {
    ASSERT_EQ(i, *I);
    i++;
  }
  ASSERT_EQ(0, i);

  i = 3;
  for (ImmutableList<Wrapper<long>>::iterator I = L2.begin(), E = L2.end();
       I != E; ++I) {
    ASSERT_EQ(i, *I);
    i++;
  }
  ASSERT_EQ(6, i);

  i = 0;
  for (ImmutableList<Wrapper<long>>::iterator I = L3.begin(), E = L3.end();
       I != E; ++I) {
    ASSERT_EQ(i, *I);
    i++;
  }
  ASSERT_EQ(6, i);
}

static_assert(is_trivially_copyable<ImmutableList<Wrapper<long>>>::value,
              "trivially copyable");

} // namespace
