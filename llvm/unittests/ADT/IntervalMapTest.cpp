//===---- ADT/IntervalMapTest.cpp - IntervalMap unit tests ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/IntervalMap.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

typedef IntervalMap<unsigned, unsigned, 4> UUMap;
typedef IntervalMap<unsigned, unsigned, 4,
                    IntervalMapHalfOpenInfo<unsigned>> UUHalfOpenMap;

// Empty map tests
TEST(IntervalMapTest, EmptyMap) {
  UUMap::Allocator allocator;
  UUMap map(allocator);
  EXPECT_TRUE(map.empty());

  // Lookup on empty map.
  EXPECT_EQ(0u, map.lookup(0));
  EXPECT_EQ(7u, map.lookup(0, 7));
  EXPECT_EQ(0u, map.lookup(~0u-1));
  EXPECT_EQ(7u, map.lookup(~0u-1, 7));

  // Iterators.
  EXPECT_TRUE(map.begin() == map.begin());
  EXPECT_TRUE(map.begin() == map.end());
  EXPECT_TRUE(map.end() == map.end());
  EXPECT_FALSE(map.begin() != map.begin());
  EXPECT_FALSE(map.begin() != map.end());
  EXPECT_FALSE(map.end() != map.end());
  EXPECT_FALSE(map.begin().valid());
  EXPECT_FALSE(map.end().valid());
  UUMap::iterator I = map.begin();
  EXPECT_FALSE(I.valid());
  EXPECT_TRUE(I == map.end());

  // Default constructor and cross-constness compares.
  UUMap::const_iterator CI;
  CI = map.begin();
  EXPECT_TRUE(CI == I);
  UUMap::iterator I2;
  I2 = map.end();
  EXPECT_TRUE(I2 == CI);
}

// Single entry map tests
TEST(IntervalMapTest, SingleEntryMap) {
  UUMap::Allocator allocator;
  UUMap map(allocator);
  map.insert(100, 150, 1);
  EXPECT_FALSE(map.empty());

  // Lookup around interval.
  EXPECT_EQ(0u, map.lookup(0));
  EXPECT_EQ(0u, map.lookup(99));
  EXPECT_EQ(1u, map.lookup(100));
  EXPECT_EQ(1u, map.lookup(101));
  EXPECT_EQ(1u, map.lookup(125));
  EXPECT_EQ(1u, map.lookup(149));
  EXPECT_EQ(1u, map.lookup(150));
  EXPECT_EQ(0u, map.lookup(151));
  EXPECT_EQ(0u, map.lookup(200));
  EXPECT_EQ(0u, map.lookup(~0u-1));

  // Iterators.
  EXPECT_TRUE(map.begin() == map.begin());
  EXPECT_FALSE(map.begin() == map.end());
  EXPECT_TRUE(map.end() == map.end());
  EXPECT_TRUE(map.begin().valid());
  EXPECT_FALSE(map.end().valid());

  // Iter deref.
  UUMap::iterator I = map.begin();
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(100u, I.start());
  EXPECT_EQ(150u, I.stop());
  EXPECT_EQ(1u, I.value());

  // Preincrement.
  ++I;
  EXPECT_FALSE(I.valid());
  EXPECT_FALSE(I == map.begin());
  EXPECT_TRUE(I == map.end());

  // PreDecrement.
  --I;
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(100u, I.start());
  EXPECT_EQ(150u, I.stop());
  EXPECT_EQ(1u, I.value());
  EXPECT_TRUE(I == map.begin());
  EXPECT_FALSE(I == map.end());

  // Change the value.
  I.setValue(2);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(100u, I.start());
  EXPECT_EQ(150u, I.stop());
  EXPECT_EQ(2u, I.value());

  // Grow the bounds.
  I.setStart(0);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(0u, I.start());
  EXPECT_EQ(150u, I.stop());
  EXPECT_EQ(2u, I.value());

  I.setStop(200);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(0u, I.start());
  EXPECT_EQ(200u, I.stop());
  EXPECT_EQ(2u, I.value());

  // Shrink the bounds.
  I.setStart(150);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(150u, I.start());
  EXPECT_EQ(200u, I.stop());
  EXPECT_EQ(2u, I.value());

  // Shrink the interval to have a length of 1
  I.setStop(150);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(150u, I.start());
  EXPECT_EQ(150u, I.stop());
  EXPECT_EQ(2u, I.value());

  I.setStop(160);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(150u, I.start());
  EXPECT_EQ(160u, I.stop());
  EXPECT_EQ(2u, I.value());

  // Shrink the interval to have a length of 1
  I.setStart(160);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(160u, I.start());
  EXPECT_EQ(160u, I.stop());
  EXPECT_EQ(2u, I.value());

  // Erase last elem.
  I.erase();
  EXPECT_TRUE(map.empty());
  EXPECT_EQ(0, std::distance(map.begin(), map.end()));
}

// Single entry half-open map tests
TEST(IntervalMapTest, SingleEntryHalfOpenMap) {
  UUHalfOpenMap::Allocator allocator;
  UUHalfOpenMap map(allocator);
  map.insert(100, 150, 1);
  EXPECT_FALSE(map.empty());

  UUHalfOpenMap::iterator I = map.begin();
  ASSERT_TRUE(I.valid());

  // Shrink the interval to have a length of 1
  I.setStart(149);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(149u, I.start());
  EXPECT_EQ(150u, I.stop());
  EXPECT_EQ(1u, I.value());

  I.setStop(160);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(149u, I.start());
  EXPECT_EQ(160u, I.stop());
  EXPECT_EQ(1u, I.value());

  // Shrink the interval to have a length of 1
  I.setStop(150);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(149u, I.start());
  EXPECT_EQ(150u, I.stop());
  EXPECT_EQ(1u, I.value());
}

// Flat coalescing tests.
TEST(IntervalMapTest, RootCoalescing) {
  UUMap::Allocator allocator;
  UUMap map(allocator);
  map.insert(100, 150, 1);

  // Coalesce from the left.
  map.insert(90, 99, 1);
  EXPECT_EQ(1, std::distance(map.begin(), map.end()));
  EXPECT_EQ(90u, map.start());
  EXPECT_EQ(150u, map.stop());

  // Coalesce from the right.
  map.insert(151, 200, 1);
  EXPECT_EQ(1, std::distance(map.begin(), map.end()));
  EXPECT_EQ(90u, map.start());
  EXPECT_EQ(200u, map.stop());

  // Non-coalesce from the left.
  map.insert(60, 89, 2);
  EXPECT_EQ(2, std::distance(map.begin(), map.end()));
  EXPECT_EQ(60u, map.start());
  EXPECT_EQ(200u, map.stop());
  EXPECT_EQ(2u, map.lookup(89));
  EXPECT_EQ(1u, map.lookup(90));

  UUMap::iterator I = map.begin();
  EXPECT_EQ(60u, I.start());
  EXPECT_EQ(89u, I.stop());
  EXPECT_EQ(2u, I.value());
  ++I;
  EXPECT_EQ(90u, I.start());
  EXPECT_EQ(200u, I.stop());
  EXPECT_EQ(1u, I.value());
  ++I;
  EXPECT_FALSE(I.valid());

  // Non-coalesce from the right.
  map.insert(201, 210, 2);
  EXPECT_EQ(3, std::distance(map.begin(), map.end()));
  EXPECT_EQ(60u, map.start());
  EXPECT_EQ(210u, map.stop());
  EXPECT_EQ(2u, map.lookup(201));
  EXPECT_EQ(1u, map.lookup(200));

  // Erase from the left.
  map.begin().erase();
  EXPECT_EQ(2, std::distance(map.begin(), map.end()));
  EXPECT_EQ(90u, map.start());
  EXPECT_EQ(210u, map.stop());

  // Erase from the right.
  (--map.end()).erase();
  EXPECT_EQ(1, std::distance(map.begin(), map.end()));
  EXPECT_EQ(90u, map.start());
  EXPECT_EQ(200u, map.stop());

  // Add non-coalescing, then trigger coalescing with setValue.
  map.insert(80, 89, 2);
  map.insert(201, 210, 2);
  EXPECT_EQ(3, std::distance(map.begin(), map.end()));
  (++map.begin()).setValue(2);
  EXPECT_EQ(1, std::distance(map.begin(), map.end()));
  I = map.begin();
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(80u, I.start());
  EXPECT_EQ(210u, I.stop());
  EXPECT_EQ(2u, I.value());
}

// Flat multi-coalescing tests.
TEST(IntervalMapTest, RootMultiCoalescing) {
  UUMap::Allocator allocator;
  UUMap map(allocator);
  map.insert(140, 150, 1);
  map.insert(160, 170, 1);
  map.insert(100, 110, 1);
  map.insert(120, 130, 1);
  EXPECT_EQ(4, std::distance(map.begin(), map.end()));
  EXPECT_EQ(100u, map.start());
  EXPECT_EQ(170u, map.stop());

  // Verify inserts.
  UUMap::iterator I = map.begin();
  EXPECT_EQ(100u, I.start());
  EXPECT_EQ(110u, I.stop());
  ++I;
  EXPECT_EQ(120u, I.start());
  EXPECT_EQ(130u, I.stop());
  ++I;
  EXPECT_EQ(140u, I.start());
  EXPECT_EQ(150u, I.stop());
  ++I;
  EXPECT_EQ(160u, I.start());
  EXPECT_EQ(170u, I.stop());
  ++I;
  EXPECT_FALSE(I.valid());

  // Test advanceTo on flat tree.
  I = map.begin();
  I.advanceTo(135);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(140u, I.start());
  EXPECT_EQ(150u, I.stop());

  I.advanceTo(145);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(140u, I.start());
  EXPECT_EQ(150u, I.stop());

  I.advanceTo(200);
  EXPECT_FALSE(I.valid());

  I.advanceTo(300);
  EXPECT_FALSE(I.valid());

  // Coalesce left with followers.
  // [100;110] [120;130] [140;150] [160;170]
  map.insert(111, 115, 1);
  I = map.begin();
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(100u, I.start());
  EXPECT_EQ(115u, I.stop());
  ++I;
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(120u, I.start());
  EXPECT_EQ(130u, I.stop());
  ++I;
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(140u, I.start());
  EXPECT_EQ(150u, I.stop());
  ++I;
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(160u, I.start());
  EXPECT_EQ(170u, I.stop());
  ++I;
  EXPECT_FALSE(I.valid());

  // Coalesce right with followers.
  // [100;115] [120;130] [140;150] [160;170]
  map.insert(135, 139, 1);
  I = map.begin();
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(100u, I.start());
  EXPECT_EQ(115u, I.stop());
  ++I;
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(120u, I.start());
  EXPECT_EQ(130u, I.stop());
  ++I;
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(135u, I.start());
  EXPECT_EQ(150u, I.stop());
  ++I;
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(160u, I.start());
  EXPECT_EQ(170u, I.stop());
  ++I;
  EXPECT_FALSE(I.valid());

  // Coalesce left and right with followers.
  // [100;115] [120;130] [135;150] [160;170]
  map.insert(131, 134, 1);
  I = map.begin();
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(100u, I.start());
  EXPECT_EQ(115u, I.stop());
  ++I;
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(120u, I.start());
  EXPECT_EQ(150u, I.stop());
  ++I;
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(160u, I.start());
  EXPECT_EQ(170u, I.stop());
  ++I;
  EXPECT_FALSE(I.valid());

  // Test clear() on non-branched map.
  map.clear();
  EXPECT_TRUE(map.empty());
  EXPECT_TRUE(map.begin() == map.end());
}

// Branched, non-coalescing tests.
TEST(IntervalMapTest, Branched) {
  UUMap::Allocator allocator;
  UUMap map(allocator);

  // Insert enough intervals to force a branched tree.
  // This creates 9 leaf nodes with 11 elements each, tree height = 1.
  for (unsigned i = 1; i < 100; ++i) {
    map.insert(10*i, 10*i+5, i);
    EXPECT_EQ(10u, map.start());
    EXPECT_EQ(10*i+5, map.stop());
  }

  // Tree limits.
  EXPECT_FALSE(map.empty());
  EXPECT_EQ(10u, map.start());
  EXPECT_EQ(995u, map.stop());

  // Tree lookup.
  for (unsigned i = 1; i < 100; ++i) {
    EXPECT_EQ(0u, map.lookup(10*i-1));
    EXPECT_EQ(i, map.lookup(10*i));
    EXPECT_EQ(i, map.lookup(10*i+5));
    EXPECT_EQ(0u, map.lookup(10*i+6));
  }

  // Forward iteration.
  UUMap::iterator I = map.begin();
  for (unsigned i = 1; i < 100; ++i) {
    ASSERT_TRUE(I.valid());
    EXPECT_EQ(10*i, I.start());
    EXPECT_EQ(10*i+5, I.stop());
    EXPECT_EQ(i, *I);
    ++I;
  }
  EXPECT_FALSE(I.valid());
  EXPECT_TRUE(I == map.end());

  // Backwards iteration.
  for (unsigned i = 99; i; --i) {
    --I;
    ASSERT_TRUE(I.valid());
    EXPECT_EQ(10*i, I.start());
    EXPECT_EQ(10*i+5, I.stop());
    EXPECT_EQ(i, *I);
  }
  EXPECT_TRUE(I == map.begin());

  // Test advanceTo in same node.
  I.advanceTo(20);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(20u, I.start());
  EXPECT_EQ(25u, I.stop());

  // Change value, no coalescing.
  I.setValue(0);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(20u, I.start());
  EXPECT_EQ(25u, I.stop());
  EXPECT_EQ(0u, I.value());

  // Close the gap right, no coalescing.
  I.setStop(29);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(20u, I.start());
  EXPECT_EQ(29u, I.stop());
  EXPECT_EQ(0u, I.value());

  // Change value, no coalescing.
  I.setValue(2);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(20u, I.start());
  EXPECT_EQ(29u, I.stop());
  EXPECT_EQ(2u, I.value());

  // Change value, now coalescing.
  I.setValue(3);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(20u, I.start());
  EXPECT_EQ(35u, I.stop());
  EXPECT_EQ(3u, I.value());

  // Close the gap, now coalescing.
  I.setValue(4);
  ASSERT_TRUE(I.valid());
  I.setStop(39);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(20u, I.start());
  EXPECT_EQ(45u, I.stop());
  EXPECT_EQ(4u, I.value());

  // advanceTo another node.
  I.advanceTo(200);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(200u, I.start());
  EXPECT_EQ(205u, I.stop());

  // Close the gap left, no coalescing.
  I.setStart(196);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(196u, I.start());
  EXPECT_EQ(205u, I.stop());
  EXPECT_EQ(20u, I.value());

  // Change value, no coalescing.
  I.setValue(0);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(196u, I.start());
  EXPECT_EQ(205u, I.stop());
  EXPECT_EQ(0u, I.value());

  // Change value, now coalescing.
  I.setValue(19);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(190u, I.start());
  EXPECT_EQ(205u, I.stop());
  EXPECT_EQ(19u, I.value());

  // Close the gap, now coalescing.
  I.setValue(18);
  ASSERT_TRUE(I.valid());
  I.setStart(186);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(180u, I.start());
  EXPECT_EQ(205u, I.stop());
  EXPECT_EQ(18u, I.value());

  // Erase from the front.
  I = map.begin();
  for (unsigned i = 0; i != 20; ++i) {
    I.erase();
    EXPECT_TRUE(I == map.begin());
    EXPECT_FALSE(map.empty());
    EXPECT_EQ(I.start(), map.start());
    EXPECT_EQ(995u, map.stop());
  }

  // Test clear() on branched map.
  map.clear();
  EXPECT_TRUE(map.empty());
  EXPECT_TRUE(map.begin() == map.end());
}

// Branched, high, non-coalescing tests.
TEST(IntervalMapTest, Branched2) {
  UUMap::Allocator allocator;
  UUMap map(allocator);

  // Insert enough intervals to force a height >= 2 tree.
  for (unsigned i = 1; i < 1000; ++i)
    map.insert(10*i, 10*i+5, i);

  // Tree limits.
  EXPECT_FALSE(map.empty());
  EXPECT_EQ(10u, map.start());
  EXPECT_EQ(9995u, map.stop());

  // Tree lookup.
  for (unsigned i = 1; i < 1000; ++i) {
    EXPECT_EQ(0u, map.lookup(10*i-1));
    EXPECT_EQ(i, map.lookup(10*i));
    EXPECT_EQ(i, map.lookup(10*i+5));
    EXPECT_EQ(0u, map.lookup(10*i+6));
  }

  // Forward iteration.
  UUMap::iterator I = map.begin();
  for (unsigned i = 1; i < 1000; ++i) {
    ASSERT_TRUE(I.valid());
    EXPECT_EQ(10*i, I.start());
    EXPECT_EQ(10*i+5, I.stop());
    EXPECT_EQ(i, *I);
    ++I;
  }
  EXPECT_FALSE(I.valid());
  EXPECT_TRUE(I == map.end());

  // Backwards iteration.
  for (unsigned i = 999; i; --i) {
    --I;
    ASSERT_TRUE(I.valid());
    EXPECT_EQ(10*i, I.start());
    EXPECT_EQ(10*i+5, I.stop());
    EXPECT_EQ(i, *I);
  }
  EXPECT_TRUE(I == map.begin());

  // Test advanceTo in same node.
  I.advanceTo(20);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(20u, I.start());
  EXPECT_EQ(25u, I.stop());

  // advanceTo sibling leaf node.
  I.advanceTo(200);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(200u, I.start());
  EXPECT_EQ(205u, I.stop());

  // advanceTo further.
  I.advanceTo(2000);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(2000u, I.start());
  EXPECT_EQ(2005u, I.stop());

  // advanceTo beyond end()
  I.advanceTo(20000);
  EXPECT_FALSE(I.valid());

  // end().advanceTo() is valid as long as x > map.stop()
  I.advanceTo(30000);
  EXPECT_FALSE(I.valid());

  // Test clear() on branched map.
  map.clear();
  EXPECT_TRUE(map.empty());
  EXPECT_TRUE(map.begin() == map.end());
}

// Random insertions, coalescing to a single interval.
TEST(IntervalMapTest, RandomCoalescing) {
  UUMap::Allocator allocator;
  UUMap map(allocator);

  // This is a poor PRNG with maximal period:
  // x_n = 5 x_{n-1} + 1 mod 2^N

  unsigned x = 100;
  for (unsigned i = 0; i != 4096; ++i) {
    map.insert(10*x, 10*x+9, 1);
    EXPECT_GE(10*x, map.start());
    EXPECT_LE(10*x+9, map.stop());
    x = (5*x+1)%4096;
  }

  // Map should be fully coalesced after that exercise.
  EXPECT_FALSE(map.empty());
  EXPECT_EQ(0u, map.start());
  EXPECT_EQ(40959u, map.stop());
  EXPECT_EQ(1, std::distance(map.begin(), map.end()));

}

TEST(IntervalMapTest, Overlaps) {
  UUMap::Allocator allocator;
  UUMap map(allocator);
  map.insert(10, 20, 0);
  map.insert(30, 40, 0);
  map.insert(50, 60, 0);

  EXPECT_FALSE(map.overlaps(0, 9));
  EXPECT_TRUE(map.overlaps(0, 10));
  EXPECT_TRUE(map.overlaps(0, 15));
  EXPECT_TRUE(map.overlaps(0, 25));
  EXPECT_TRUE(map.overlaps(0, 45));
  EXPECT_TRUE(map.overlaps(10, 45));
  EXPECT_TRUE(map.overlaps(30, 45));
  EXPECT_TRUE(map.overlaps(35, 36));
  EXPECT_TRUE(map.overlaps(40, 45));
  EXPECT_FALSE(map.overlaps(45, 45));
  EXPECT_TRUE(map.overlaps(60, 60));
  EXPECT_TRUE(map.overlaps(60, 66));
  EXPECT_FALSE(map.overlaps(66, 66));
}

TEST(IntervalMapTest, OverlapsHalfOpen) {
  UUHalfOpenMap::Allocator allocator;
  UUHalfOpenMap map(allocator);
  map.insert(10, 20, 0);
  map.insert(30, 40, 0);
  map.insert(50, 60, 0);

  EXPECT_FALSE(map.overlaps(0, 9));
  EXPECT_FALSE(map.overlaps(0, 10));
  EXPECT_TRUE(map.overlaps(0, 15));
  EXPECT_TRUE(map.overlaps(0, 25));
  EXPECT_TRUE(map.overlaps(0, 45));
  EXPECT_TRUE(map.overlaps(10, 45));
  EXPECT_TRUE(map.overlaps(30, 45));
  EXPECT_TRUE(map.overlaps(35, 36));
  EXPECT_FALSE(map.overlaps(40, 45));
  EXPECT_FALSE(map.overlaps(45, 46));
  EXPECT_FALSE(map.overlaps(60, 61));
  EXPECT_FALSE(map.overlaps(60, 66));
  EXPECT_FALSE(map.overlaps(66, 67));
}

TEST(IntervalMapOverlapsTest, SmallMaps) {
  typedef IntervalMapOverlaps<UUMap,UUMap> UUOverlaps;
  UUMap::Allocator allocator;
  UUMap mapA(allocator);
  UUMap mapB(allocator);

  // empty, empty.
  EXPECT_FALSE(UUOverlaps(mapA, mapB).valid());

  mapA.insert(1, 2, 3);

  // full, empty
  EXPECT_FALSE(UUOverlaps(mapA, mapB).valid());
  // empty, full
  EXPECT_FALSE(UUOverlaps(mapB, mapA).valid());

  mapB.insert(3, 4, 5);

  // full, full, non-overlapping
  EXPECT_FALSE(UUOverlaps(mapA, mapB).valid());
  EXPECT_FALSE(UUOverlaps(mapB, mapA).valid());

  // Add an overlapping segment.
  mapA.insert(4, 5, 6);

  UUOverlaps AB(mapA, mapB);
  ASSERT_TRUE(AB.valid());
  EXPECT_EQ(4u, AB.a().start());
  EXPECT_EQ(3u, AB.b().start());
  ++AB;
  EXPECT_FALSE(AB.valid());

  UUOverlaps BA(mapB, mapA);
  ASSERT_TRUE(BA.valid());
  EXPECT_EQ(3u, BA.a().start());
  EXPECT_EQ(4u, BA.b().start());
  // advance past end.
  BA.advanceTo(6);
  EXPECT_FALSE(BA.valid());
  // advance an invalid iterator.
  BA.advanceTo(7);
  EXPECT_FALSE(BA.valid());
}

TEST(IntervalMapOverlapsTest, BigMaps) {
  typedef IntervalMapOverlaps<UUMap,UUMap> UUOverlaps;
  UUMap::Allocator allocator;
  UUMap mapA(allocator);
  UUMap mapB(allocator);

  // [0;4] [10;14] [20;24] ...
  for (unsigned n = 0; n != 100; ++n)
    mapA.insert(10*n, 10*n+4, n);

  // [5;6] [15;16] [25;26] ...
  for (unsigned n = 10; n != 20; ++n)
    mapB.insert(10*n+5, 10*n+6, n);

  // [208;209] [218;219] ...
  for (unsigned n = 20; n != 30; ++n)
    mapB.insert(10*n+8, 10*n+9, n);

  // insert some overlapping segments.
  mapB.insert(400, 400, 400);
  mapB.insert(401, 401, 401);
  mapB.insert(402, 500, 402);
  mapB.insert(600, 601, 402);

  UUOverlaps AB(mapA, mapB);
  ASSERT_TRUE(AB.valid());
  EXPECT_EQ(400u, AB.a().start());
  EXPECT_EQ(400u, AB.b().start());
  ++AB;
  ASSERT_TRUE(AB.valid());
  EXPECT_EQ(400u, AB.a().start());
  EXPECT_EQ(401u, AB.b().start());
  ++AB;
  ASSERT_TRUE(AB.valid());
  EXPECT_EQ(400u, AB.a().start());
  EXPECT_EQ(402u, AB.b().start());
  ++AB;
  ASSERT_TRUE(AB.valid());
  EXPECT_EQ(410u, AB.a().start());
  EXPECT_EQ(402u, AB.b().start());
  ++AB;
  ASSERT_TRUE(AB.valid());
  EXPECT_EQ(420u, AB.a().start());
  EXPECT_EQ(402u, AB.b().start());
  AB.skipB();
  ASSERT_TRUE(AB.valid());
  EXPECT_EQ(600u, AB.a().start());
  EXPECT_EQ(600u, AB.b().start());
  ++AB;
  EXPECT_FALSE(AB.valid());

  // Test advanceTo.
  UUOverlaps AB2(mapA, mapB);
  AB2.advanceTo(410);
  ASSERT_TRUE(AB2.valid());
  EXPECT_EQ(410u, AB2.a().start());
  EXPECT_EQ(402u, AB2.b().start());

  // It is valid to advanceTo with any monotonic sequence.
  AB2.advanceTo(411);
  ASSERT_TRUE(AB2.valid());
  EXPECT_EQ(410u, AB2.a().start());
  EXPECT_EQ(402u, AB2.b().start());

  // Check reversed maps.
  UUOverlaps BA(mapB, mapA);
  ASSERT_TRUE(BA.valid());
  EXPECT_EQ(400u, BA.b().start());
  EXPECT_EQ(400u, BA.a().start());
  ++BA;
  ASSERT_TRUE(BA.valid());
  EXPECT_EQ(400u, BA.b().start());
  EXPECT_EQ(401u, BA.a().start());
  ++BA;
  ASSERT_TRUE(BA.valid());
  EXPECT_EQ(400u, BA.b().start());
  EXPECT_EQ(402u, BA.a().start());
  ++BA;
  ASSERT_TRUE(BA.valid());
  EXPECT_EQ(410u, BA.b().start());
  EXPECT_EQ(402u, BA.a().start());
  ++BA;
  ASSERT_TRUE(BA.valid());
  EXPECT_EQ(420u, BA.b().start());
  EXPECT_EQ(402u, BA.a().start());
  BA.skipA();
  ASSERT_TRUE(BA.valid());
  EXPECT_EQ(600u, BA.b().start());
  EXPECT_EQ(600u, BA.a().start());
  ++BA;
  EXPECT_FALSE(BA.valid());

  // Test advanceTo.
  UUOverlaps BA2(mapB, mapA);
  BA2.advanceTo(410);
  ASSERT_TRUE(BA2.valid());
  EXPECT_EQ(410u, BA2.b().start());
  EXPECT_EQ(402u, BA2.a().start());

  BA2.advanceTo(411);
  ASSERT_TRUE(BA2.valid());
  EXPECT_EQ(410u, BA2.b().start());
  EXPECT_EQ(402u, BA2.a().start());
}

} // namespace
