//===- StringPoolTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for orc-rt's StringPool.h APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/StringPool.h"
#include "gtest/gtest.h"

#include <unordered_set>

using namespace orc_rt;

TEST(StringPoolTest, EmptyByDefault) {
  StringPool SP;
  EXPECT_TRUE(SP.empty());
}

TEST(StringPoolTest, InternReturnsEqualContent) {
  StringPool SP;
  auto Foo = SP.intern("foo");
  EXPECT_TRUE(Foo);
  EXPECT_EQ(*Foo, "foo");
  EXPECT_FALSE(SP.empty());
}

TEST(StringPoolTest, RepeatedInternIsIdentical) {
  StringPool SP;
  auto Foo1 = SP.intern("foo");
  auto Foo2 = SP.intern("foo");
  EXPECT_EQ(Foo1, Foo2);
}

TEST(StringPoolTest, DifferentContentIsDistinct) {
  StringPool SP;
  auto Foo = SP.intern("foo");
  auto Bar = SP.intern("bar");
  EXPECT_NE(Foo, Bar);
}

TEST(StringPoolTest, DifferentPoolsAreDistinct) {
  StringPool SP1, SP2;
  auto Foo1 = SP1.intern("foo");
  auto Foo2 = SP2.intern("foo");
  EXPECT_EQ(*Foo1, *Foo2);
  EXPECT_NE(Foo1, Foo2);
}

TEST(StringPoolTest, DefaultConstructedIsNull) {
  PooledStringPtr Null;
  EXPECT_FALSE(Null);
  EXPECT_EQ(Null, PooledStringPtr(nullptr));
}

TEST(StringPoolTest, CopyKeepsEntryAlive) {
  StringPool SP;
  PooledStringPtr Copy;
  {
    auto Foo = SP.intern("foo");
    Copy = Foo;
  }
  // Foo has been destroyed. If copy-assignment above failed to incRef, the
  // entry's refcount would already be zero and clearDeadEntries() would
  // reclaim it.
  SP.clearDeadEntries();
  ASSERT_FALSE(SP.empty()) << "Copy should have kept the entry alive";
  EXPECT_EQ(*Copy, "foo");
}

TEST(StringPoolTest, ClearDeadEntriesReclaimsUnreferenced) {
  StringPool SP;
  {
    auto Foo = SP.intern("foo");
  }
  EXPECT_FALSE(SP.empty());
  SP.clearDeadEntries();
  EXPECT_TRUE(SP.empty());
}

TEST(StringPoolTest, ClearDeadEntriesKeepsReferenced) {
  StringPool SP;
  auto Foo = SP.intern("foo");
  {
    auto Bar = SP.intern("bar");
  }
  SP.clearDeadEntries();
  ASSERT_FALSE(SP.empty());
  EXPECT_EQ(*Foo, "foo");
}

TEST(StringPoolTest, MoveLeavesSourceNull) {
  StringPool SP;
  auto Foo = SP.intern("foo");
  auto Moved = std::move(Foo);
  EXPECT_FALSE(Foo);
  EXPECT_TRUE(Moved);
  EXPECT_EQ(*Moved, "foo");
}

TEST(StringPoolTest, NonOwningPtrComparesEqualToOwning) {
  StringPool SP;
  auto Foo = SP.intern("foo");
  NonOwningPooledStringPtr NonOwningFoo(Foo);
  EXPECT_EQ(Foo, NonOwningFoo);
  EXPECT_EQ(*NonOwningFoo, "foo");
}

TEST(StringPoolTest, NonOwningPtrDoesNotKeepEntryAlive) {
  StringPool SP;
  NonOwningPooledStringPtr NonOwningFoo;
  {
    auto Foo = SP.intern("foo");
    NonOwningFoo = NonOwningPooledStringPtr(Foo);
  }
  // Foo has been destroyed and was the only owner, so the entry should be
  // reclaimed even though NonOwningFoo still points at it.
  SP.clearDeadEntries();
  EXPECT_TRUE(SP.empty());
}

TEST(StringPoolTest, ConstructOwningFromNonOwningIncrementsRefcount) {
  StringPool SP;
  NonOwningPooledStringPtr NonOwningFoo;
  {
    auto Foo = SP.intern("foo");
    NonOwningFoo = NonOwningPooledStringPtr(Foo);
  }
  // The entry's refcount is now zero, but it has not yet been reclaimed by
  // clearDeadEntries(), so re-deriving an owning ptr from NonOwningFoo here
  // is well-defined and should keep the entry alive.
  PooledStringPtr Reowned(NonOwningFoo);
  SP.clearDeadEntries();
  ASSERT_FALSE(SP.empty());
  EXPECT_EQ(*Reowned, "foo");
}

TEST(StringPoolTest, UsableAsUnorderedSetKey) {
  StringPool SP;
  auto Foo1 = SP.intern("foo");
  auto Foo2 = SP.intern("foo");
  auto Bar = SP.intern("bar");

  std::unordered_set<PooledStringPtr> S;
  S.insert(Foo1);
  S.insert(Foo2);
  S.insert(Bar);

  EXPECT_EQ(S.size(), 2U);
  EXPECT_TRUE(S.count(Foo1));
  EXPECT_TRUE(S.count(Bar));
}

TEST(StringPoolTest, OwningAndNonOwningHashInterchangeably) {
  StringPool SP;
  auto Foo = SP.intern("foo");
  NonOwningPooledStringPtr NonOwningFoo(Foo);

  EXPECT_EQ(std::hash<PooledStringPtr>()(Foo),
            std::hash<NonOwningPooledStringPtr>()(NonOwningFoo));
}
