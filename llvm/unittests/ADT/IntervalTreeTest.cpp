//===---- ADT/IntervalTreeTest.cpp - IntervalTree unit tests --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/IntervalTree.h"
#include "gtest/gtest.h"

// The test cases for the IntervalTree implementation, follow the below steps:
// a) Insert a series of intervals with their associated mapped value.
// b) Create the interval tree.
// c) Query for specific interval point, covering points inside and outside
//    of any given intervals.
// d) Traversal for specific interval point, using the iterators.
//
// When querying for a set of intervals containing a given value, the query is
// done three times, by calling:
// 1) Intervals = getContaining(...).
// 2) Intervals = getContaining(...).
//    sortIntervals(Intervals, Sorting=Ascending).
// 3) Intervals = getContaining(...).
//    sortIntervals(Intervals, Sorting=Ascending).
//
// The returned intervals are:
// 1) In their location order within the tree.
// 2) Smaller intervals first.
// 3) Bigger intervals first.

using namespace llvm;

namespace {

// Helper function to test a specific item or iterator.
template <typename TPoint, typename TItem, typename TValue>
void checkItem(TPoint Point, TItem Item, TPoint Left, TPoint Right,
               TValue Value) {
  EXPECT_TRUE(Item->contains(Point));
  EXPECT_EQ(Item->left(), Left);
  EXPECT_EQ(Item->right(), Right);
  EXPECT_EQ(Item->value(), Value);
}

// User class tree tests.
TEST(IntervalTreeTest, UserClass) {
  using UUPoint = unsigned;
  using UUValue = double;
  class MyData : public IntervalData<UUPoint, UUValue> {
    using UUData = IntervalData<UUPoint, UUValue>;

  public:
    // Inherit Base's constructors.
    using UUData::UUData;
    PointType left() const { return UUData::left(); }
    PointType right() const { return UUData::right(); }
    ValueType value() const { return UUData::value(); }

    bool left(const PointType &Point) const { return UUData::left(Point); }
    bool right(const PointType &Point) const { return UUData::right(Point); }
    bool contains(const PointType &Point) const {
      return UUData::contains(Point);
    }
  };

  using UUTree = IntervalTree<UUPoint, UUValue, MyData>;
  using UUReferences = UUTree::IntervalReferences;
  using UUData = UUTree::DataType;
  using UUAlloc = UUTree::Allocator;

  auto CheckData = [](UUPoint Point, const UUData *Data, UUPoint Left,
                      UUPoint Right, UUValue Value) {
    checkItem<UUPoint, const UUData *, UUValue>(Point, Data, Left, Right,
                                                Value);
  };

  UUAlloc Allocator;
  UUTree Tree(Allocator);
  UUReferences Intervals;
  UUPoint Point;

  EXPECT_TRUE(Tree.empty());
  Tree.clear();
  EXPECT_TRUE(Tree.empty());

  // [10, 20] <- (10.20)
  // [30, 40] <- (30.40)
  //
  //    [10...20]   [30...40]
  Tree.insert(10, 20, 10.20);
  Tree.insert(30, 40, 30.40);
  Tree.create();

  // Invalid interval values: x < [10
  Point = 5;
  Intervals = Tree.getContaining(Point);
  EXPECT_TRUE(Intervals.empty());

  // Valid interval values: [10...20]
  Point = 10;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  CheckData(Point, Intervals[0], 10, 20, 10.20);

  Point = 15;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  CheckData(Point, Intervals[0], 10, 20, 10.20);

  Point = 20;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  CheckData(Point, Intervals[0], 10, 20, 10.20);

  // Invalid interval values: 20] < x < [30
  Point = 25;
  Intervals = Tree.getContaining(Point);
  EXPECT_TRUE(Intervals.empty());

  // Valid interval values: [30...40]
  Point = 30;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  CheckData(Point, Intervals[0], 30, 40, 30.40);

  Point = 35;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  CheckData(Point, Intervals[0], 30, 40, 30.40);

  Point = 40;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  CheckData(Point, Intervals[0], 30, 40, 30.40);

  // Invalid interval values: 40] < x
  Point = 45;
  Intervals = Tree.getContaining(Point);
  EXPECT_TRUE(Intervals.empty());
}

using UUPoint = unsigned; // Interval endpoint type.
using UUValue = unsigned; // Mapped value type.

using UUTree = IntervalTree<UUPoint, UUValue>;
using UUReferences = UUTree::IntervalReferences;
using UUData = UUTree::DataType;
using UUSorting = UUTree::Sorting;
using UUPoint = UUTree::PointType;
using UUValue = UUTree::ValueType;
using UUIter = UUTree::find_iterator;
using UUAlloc = UUTree::Allocator;

void checkData(UUPoint Point, const UUData *Data, UUPoint Left, UUPoint Right,
               UUValue Value) {
  checkItem<UUPoint, const UUData *, UUValue>(Point, Data, Left, Right, Value);
}

void checkData(UUPoint Point, UUIter Iter, UUPoint Left, UUPoint Right,
               UUValue Value) {
  checkItem<UUPoint, UUIter, UUValue>(Point, Iter, Left, Right, Value);
}

// Empty tree tests.
TEST(IntervalTreeTest, NoIntervals) {
  UUAlloc Allocator;
  UUTree Tree(Allocator);
  EXPECT_TRUE(Tree.empty());
  Tree.clear();
  EXPECT_TRUE(Tree.empty());

  // Create the tree and switch to query mode.
  Tree.create();
  EXPECT_TRUE(Tree.empty());
  EXPECT_EQ(Tree.find(1), Tree.find_end());
}

// One item tree tests.
TEST(IntervalTreeTest, OneInterval) {
  UUAlloc Allocator;
  UUTree Tree(Allocator);
  UUReferences Intervals;
  UUPoint Point;

  // [10, 20] <- (1020)
  //
  //    [10...20]
  Tree.insert(10, 20, 1020);

  EXPECT_TRUE(Tree.empty());
  Tree.create();
  EXPECT_FALSE(Tree.empty());

  // Invalid interval values: x < [10.
  Point = 5;
  Intervals = Tree.getContaining(Point);
  EXPECT_TRUE(Intervals.empty());

  // Valid interval values: [10...20].
  Point = 10;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  checkData(Point, Intervals[0], 10, 20, 1020);

  Point = 15;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  checkData(Point, Intervals[0], 10, 20, 1020);

  Point = 20;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  checkData(Point, Intervals[0], 10, 20, 1020);

  // Invalid interval values: 20] < x
  Point = 25;
  Intervals = Tree.getContaining(Point);
  EXPECT_TRUE(Intervals.empty());
}

// Two items tree tests. No overlapping.
TEST(IntervalTreeTest, TwoIntervals) {
  UUAlloc Allocator;
  UUTree Tree(Allocator);
  UUReferences Intervals;
  UUPoint Point;

  // [10, 20] <- (1020)
  // [30, 40] <- (3040)
  //
  //    [10...20]   [30...40]
  Tree.insert(10, 20, 1020);
  Tree.insert(30, 40, 3040);
  Tree.create();

  // Invalid interval values: x < [10
  Point = 5;
  Intervals = Tree.getContaining(Point);
  EXPECT_TRUE(Intervals.empty());

  // Valid interval values: [10...20]
  Point = 10;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  checkData(Point, Intervals[0], 10, 20, 1020);

  Point = 15;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  checkData(Point, Intervals[0], 10, 20, 1020);

  Point = 20;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  checkData(Point, Intervals[0], 10, 20, 1020);

  // Invalid interval values: 20] < x < [30
  Point = 25;
  Intervals = Tree.getContaining(Point);
  EXPECT_TRUE(Intervals.empty());

  // Valid interval values: [30...40]
  Point = 30;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  checkData(Point, Intervals[0], 30, 40, 3040);

  Point = 35;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  checkData(Point, Intervals[0], 30, 40, 3040);

  Point = 40;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  checkData(Point, Intervals[0], 30, 40, 3040);

  // Invalid interval values: 40] < x
  Point = 45;
  Intervals = Tree.getContaining(Point);
  EXPECT_TRUE(Intervals.empty());
}

// Three items tree tests. No overlapping.
TEST(IntervalTreeTest, ThreeIntervals) {
  UUAlloc Allocator;
  UUTree Tree(Allocator);
  UUReferences Intervals;
  UUPoint Point;

  // [10, 20] <- (1020)
  // [30, 40] <- (3040)
  // [50, 60] <- (5060)
  //
  //    [10...20]   [30...40]   [50...60]
  Tree.insert(10, 20, 1020);
  Tree.insert(30, 40, 3040);
  Tree.insert(50, 60, 5060);
  Tree.create();

  // Invalid interval values: x < [10
  Point = 5;
  Intervals = Tree.getContaining(Point);
  EXPECT_TRUE(Intervals.empty());

  // Valid interval values: [10...20]
  Point = 10;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  checkData(Point, Intervals[0], 10, 20, 1020);

  Point = 15;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  checkData(Point, Intervals[0], 10, 20, 1020);

  Point = 20;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  checkData(Point, Intervals[0], 10, 20, 1020);

  // Invalid interval values: 20] < x < [30
  Point = 25;
  Intervals = Tree.getContaining(Point);
  EXPECT_TRUE(Intervals.empty());

  // Valid interval values: [30...40]
  Point = 30;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  checkData(Point, Intervals[0], 30, 40, 3040);

  Point = 35;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  checkData(Point, Intervals[0], 30, 40, 3040);

  Point = 40;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  checkData(Point, Intervals[0], 30, 40, 3040);

  // Invalid interval values: 40] < x < [50
  Point = 45;
  Intervals = Tree.getContaining(Point);
  EXPECT_TRUE(Intervals.empty());

  // Valid interval values: [50...60]
  Point = 50;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  checkData(Point, Intervals[0], 50, 60, 5060);

  Point = 55;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  checkData(Point, Intervals[0], 50, 60, 5060);

  Point = 60;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  checkData(Point, Intervals[0], 50, 60, 5060);

  // Invalid interval values: 60] < x
  Point = 65;
  Intervals = Tree.getContaining(Point);
  EXPECT_TRUE(Intervals.empty());
}

// One item tree tests.
TEST(IntervalTreeTest, EmptyIntervals) {
  UUAlloc Allocator;
  UUTree Tree(Allocator);
  UUReferences Intervals;
  UUPoint Point;

  // [40, 60] <- (4060)
  // [50, 50] <- (5050)
  // [10, 10] <- (1010)
  // [70, 70] <- (7070)
  //
  //                [40...............60]
  //                      [50...50]
  //    [10...10]
  //                                        [70...70]
  Tree.insert(40, 60, 4060);
  Tree.insert(50, 50, 5050);
  Tree.insert(10, 10, 1010);
  Tree.insert(70, 70, 7070);

  EXPECT_TRUE(Tree.empty());
  Tree.create();
  EXPECT_FALSE(Tree.empty());

  // Invalid interval values: x < [10.
  Point = 5;
  Intervals = Tree.getContaining(Point);
  EXPECT_TRUE(Intervals.empty());

  // Valid interval values: [10...10].
  Point = 10;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  checkData(Point, Intervals[0], 10, 10, 1010);

  // Invalid interval values: 10] < x
  Point = 15;
  Intervals = Tree.getContaining(Point);
  EXPECT_TRUE(Intervals.empty());

  // Invalid interval values: x < [50.
  Point = 45;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  checkData(Point, Intervals[0], 40, 60, 4060);

  // Valid interval values: [50...50].
  Point = 50;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 2u);
  checkData(Point, Intervals[0], 40, 60, 4060);
  checkData(Point, Intervals[1], 50, 50, 5050);

  // Invalid interval values: 50] < x
  Point = 55;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  checkData(Point, Intervals[0], 40, 60, 4060);

  // Invalid interval values: x < [70.
  Point = 65;
  Intervals = Tree.getContaining(Point);
  EXPECT_TRUE(Intervals.empty());

  // Valid interval values: [70...70].
  Point = 70;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  checkData(Point, Intervals[0], 70, 70, 7070);

  // Invalid interval values: 70] < x
  Point = 75;
  Intervals = Tree.getContaining(Point);
  EXPECT_TRUE(Intervals.empty());
}

// Simple overlapping tests.
TEST(IntervalTreeTest, SimpleIntervalsOverlapping) {
  UUAlloc Allocator;
  UUTree Tree(Allocator);
  UUReferences Intervals;
  UUPoint Point;

  // [40, 60] <- (4060)
  // [30, 70] <- (3070)
  // [20, 80] <- (2080)
  // [10, 90] <- (1090)
  //
  //                      [40...60]
  //                [30...............70]
  //          [20...........................80]
  //    [10.......................................90]
  Tree.insert(40, 60, 4060);
  Tree.insert(30, 70, 3070);
  Tree.insert(20, 80, 2080);
  Tree.insert(10, 90, 1090);
  Tree.create();

  // Invalid interval values: x < [10
  Point = 5;
  Intervals = Tree.getContaining(Point);
  EXPECT_TRUE(Intervals.empty());

  // Valid interval values:
  Point = 10;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  checkData(Point, Intervals[0], 10, 90, 1090);

  Point = 15;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  checkData(Point, Intervals[0], 10, 90, 1090);

  Point = 20;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 2u);
  checkData(Point, Intervals[0], 10, 90, 1090);
  checkData(Point, Intervals[1], 20, 80, 2080);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 2u);
  checkData(Point, Intervals[0], 20, 80, 2080);
  checkData(Point, Intervals[1], 10, 90, 1090);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 2u);
  checkData(Point, Intervals[0], 10, 90, 1090);
  checkData(Point, Intervals[1], 20, 80, 2080);

  Point = 25;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 2u);
  checkData(Point, Intervals[0], 10, 90, 1090);
  checkData(Point, Intervals[1], 20, 80, 2080);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 2u);
  checkData(Point, Intervals[0], 20, 80, 2080);
  checkData(Point, Intervals[1], 10, 90, 1090);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 2u);
  checkData(Point, Intervals[0], 10, 90, 1090);
  checkData(Point, Intervals[1], 20, 80, 2080);

  Point = 30;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 3u);
  checkData(Point, Intervals[0], 10, 90, 1090);
  checkData(Point, Intervals[1], 20, 80, 2080);
  checkData(Point, Intervals[2], 30, 70, 3070);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 3u);
  checkData(Point, Intervals[0], 30, 70, 3070);
  checkData(Point, Intervals[1], 20, 80, 2080);
  checkData(Point, Intervals[2], 10, 90, 1090);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 3u);
  checkData(Point, Intervals[0], 10, 90, 1090);
  checkData(Point, Intervals[1], 20, 80, 2080);
  checkData(Point, Intervals[2], 30, 70, 3070);

  Point = 35;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 3u);
  checkData(Point, Intervals[0], 10, 90, 1090);
  checkData(Point, Intervals[1], 20, 80, 2080);
  checkData(Point, Intervals[2], 30, 70, 3070);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 3u);
  checkData(Point, Intervals[0], 30, 70, 3070);
  checkData(Point, Intervals[1], 20, 80, 2080);
  checkData(Point, Intervals[2], 10, 90, 1090);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 3u);
  checkData(Point, Intervals[0], 10, 90, 1090);
  checkData(Point, Intervals[1], 20, 80, 2080);
  checkData(Point, Intervals[2], 30, 70, 3070);

  Point = 40;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 10, 90, 1090);
  checkData(Point, Intervals[1], 20, 80, 2080);
  checkData(Point, Intervals[2], 30, 70, 3070);
  checkData(Point, Intervals[3], 40, 60, 4060);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 40, 60, 4060);
  checkData(Point, Intervals[1], 30, 70, 3070);
  checkData(Point, Intervals[2], 20, 80, 2080);
  checkData(Point, Intervals[3], 10, 90, 1090);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 10, 90, 1090);
  checkData(Point, Intervals[1], 20, 80, 2080);
  checkData(Point, Intervals[2], 30, 70, 3070);
  checkData(Point, Intervals[3], 40, 60, 4060);

  Point = 50;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 10, 90, 1090);
  checkData(Point, Intervals[1], 20, 80, 2080);
  checkData(Point, Intervals[2], 30, 70, 3070);
  checkData(Point, Intervals[3], 40, 60, 4060);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 40, 60, 4060);
  checkData(Point, Intervals[1], 30, 70, 3070);
  checkData(Point, Intervals[2], 20, 80, 2080);
  checkData(Point, Intervals[3], 10, 90, 1090);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 10, 90, 1090);
  checkData(Point, Intervals[1], 20, 80, 2080);
  checkData(Point, Intervals[2], 30, 70, 3070);
  checkData(Point, Intervals[3], 40, 60, 4060);

  Point = 60;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 10, 90, 1090);
  checkData(Point, Intervals[1], 20, 80, 2080);
  checkData(Point, Intervals[2], 30, 70, 3070);
  checkData(Point, Intervals[3], 40, 60, 4060);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 40, 60, 4060);
  checkData(Point, Intervals[1], 30, 70, 3070);
  checkData(Point, Intervals[2], 20, 80, 2080);
  checkData(Point, Intervals[3], 10, 90, 1090);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 10, 90, 1090);
  checkData(Point, Intervals[1], 20, 80, 2080);
  checkData(Point, Intervals[2], 30, 70, 3070);
  checkData(Point, Intervals[3], 40, 60, 4060);

  Point = 65;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 3u);
  checkData(Point, Intervals[0], 10, 90, 1090);
  checkData(Point, Intervals[1], 20, 80, 2080);
  checkData(Point, Intervals[2], 30, 70, 3070);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 3u);
  checkData(Point, Intervals[0], 30, 70, 3070);
  checkData(Point, Intervals[1], 20, 80, 2080);
  checkData(Point, Intervals[2], 10, 90, 1090);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 3u);
  checkData(Point, Intervals[0], 10, 90, 1090);
  checkData(Point, Intervals[1], 20, 80, 2080);
  checkData(Point, Intervals[2], 30, 70, 3070);

  Point = 70;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 3u);
  checkData(Point, Intervals[0], 10, 90, 1090);
  checkData(Point, Intervals[1], 20, 80, 2080);
  checkData(Point, Intervals[2], 30, 70, 3070);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 3u);
  checkData(Point, Intervals[0], 30, 70, 3070);
  checkData(Point, Intervals[1], 20, 80, 2080);
  checkData(Point, Intervals[2], 10, 90, 1090);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 3u);
  checkData(Point, Intervals[0], 10, 90, 1090);
  checkData(Point, Intervals[1], 20, 80, 2080);
  checkData(Point, Intervals[2], 30, 70, 3070);

  Point = 75;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 2u);
  checkData(Point, Intervals[0], 10, 90, 1090);
  checkData(Point, Intervals[1], 20, 80, 2080);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 2u);
  checkData(Point, Intervals[0], 20, 80, 2080);
  checkData(Point, Intervals[1], 10, 90, 1090);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 2u);
  checkData(Point, Intervals[0], 10, 90, 1090);
  checkData(Point, Intervals[1], 20, 80, 2080);

  Point = 80;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 2u);
  checkData(Point, Intervals[0], 10, 90, 1090);
  checkData(Point, Intervals[1], 20, 80, 2080);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 2u);
  checkData(Point, Intervals[0], 20, 80, 2080);
  checkData(Point, Intervals[1], 10, 90, 1090);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 2u);
  checkData(Point, Intervals[0], 10, 90, 1090);
  checkData(Point, Intervals[1], 20, 80, 2080);

  Point = 85;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  checkData(Point, Intervals[0], 10, 90, 1090);

  Point = 90;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  checkData(Point, Intervals[0], 10, 90, 1090);

  // Invalid interval values: 90] < x
  Point = 95;
  Intervals = Tree.getContaining(Point);
  EXPECT_TRUE(Intervals.empty());
}

// Complex Overlapping.
TEST(IntervalTreeTest, ComplexIntervalsOverlapping) {
  UUAlloc Allocator;
  UUTree Tree(Allocator);
  UUReferences Intervals;
  UUPoint Point;

  // [30, 35] <- (3035)
  // [39, 50] <- (3950)
  // [55, 61] <- (5561)
  // [31, 56] <- (3156)
  // [12, 21] <- (1221)
  // [25, 41] <- (2541)
  // [49, 65] <- (4965)
  // [71, 79] <- (7179)
  // [11, 16] <- (1116)
  // [20, 30] <- (2030)
  // [36, 54] <- (3654)
  // [60, 70] <- (6070)
  // [74, 80] <- (7480)
  // [15, 40] <- (1540)
  // [43, 45] <- (4345)
  // [50, 75] <- (5075)
  // [10, 85] <- (1085)

  //                    30--35  39------------50  55----61
  //                      31------------------------56
  //     12--------21 25------------41      49-------------65   71-----79
  //   11----16  20-----30    36----------------54    60------70  74---- 80
  //       15---------------------40  43--45  50--------------------75
  // 10----------------------------------------------------------------------85

  Tree.insert(30, 35, 3035);
  Tree.insert(39, 50, 3950);
  Tree.insert(55, 61, 5561);
  Tree.insert(31, 56, 3156);
  Tree.insert(12, 21, 1221);
  Tree.insert(25, 41, 2541);
  Tree.insert(49, 65, 4965);
  Tree.insert(71, 79, 7179);
  Tree.insert(11, 16, 1116);
  Tree.insert(20, 30, 2030);
  Tree.insert(36, 54, 3654);
  Tree.insert(60, 70, 6070);
  Tree.insert(74, 80, 7480);
  Tree.insert(15, 40, 1540);
  Tree.insert(43, 45, 4345);
  Tree.insert(50, 75, 5075);
  Tree.insert(10, 85, 1085);
  Tree.create();

  // Find valid interval values.
  Point = 30;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 25, 41, 2541);
  checkData(Point, Intervals[2], 15, 40, 1540);
  checkData(Point, Intervals[3], 20, 30, 2030);
  checkData(Point, Intervals[4], 30, 35, 3035);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 30, 35, 3035);
  checkData(Point, Intervals[1], 20, 30, 2030);
  checkData(Point, Intervals[2], 25, 41, 2541);
  checkData(Point, Intervals[3], 15, 40, 1540);
  checkData(Point, Intervals[4], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 15, 40, 1540);
  checkData(Point, Intervals[2], 25, 41, 2541);
  checkData(Point, Intervals[3], 20, 30, 2030);
  checkData(Point, Intervals[4], 30, 35, 3035);

  Point = 35;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 31, 56, 3156);
  checkData(Point, Intervals[2], 25, 41, 2541);
  checkData(Point, Intervals[3], 15, 40, 1540);
  checkData(Point, Intervals[4], 30, 35, 3035);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 30, 35, 3035);
  checkData(Point, Intervals[1], 25, 41, 2541);
  checkData(Point, Intervals[2], 31, 56, 3156);
  checkData(Point, Intervals[3], 15, 40, 1540);
  checkData(Point, Intervals[4], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 31, 56, 3156);
  checkData(Point, Intervals[2], 15, 40, 1540);
  checkData(Point, Intervals[3], 25, 41, 2541);
  checkData(Point, Intervals[4], 30, 35, 3035);

  Point = 39;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 6u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 31, 56, 3156);
  checkData(Point, Intervals[2], 36, 54, 3654);
  checkData(Point, Intervals[3], 39, 50, 3950);
  checkData(Point, Intervals[4], 25, 41, 2541);
  checkData(Point, Intervals[5], 15, 40, 1540);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 6u);
  checkData(Point, Intervals[0], 39, 50, 3950);
  checkData(Point, Intervals[1], 25, 41, 2541);
  checkData(Point, Intervals[2], 36, 54, 3654);
  checkData(Point, Intervals[3], 31, 56, 3156);
  checkData(Point, Intervals[4], 15, 40, 1540);
  checkData(Point, Intervals[5], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 6u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 31, 56, 3156);
  checkData(Point, Intervals[2], 15, 40, 1540);
  checkData(Point, Intervals[3], 36, 54, 3654);
  checkData(Point, Intervals[4], 25, 41, 2541);
  checkData(Point, Intervals[5], 39, 50, 3950);

  Point = 50;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 6u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 31, 56, 3156);
  checkData(Point, Intervals[2], 36, 54, 3654);
  checkData(Point, Intervals[3], 39, 50, 3950);
  checkData(Point, Intervals[4], 49, 65, 4965);
  checkData(Point, Intervals[5], 50, 75, 5075);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 6u);
  checkData(Point, Intervals[0], 39, 50, 3950);
  checkData(Point, Intervals[1], 49, 65, 4965);
  checkData(Point, Intervals[2], 36, 54, 3654);
  checkData(Point, Intervals[3], 31, 56, 3156);
  checkData(Point, Intervals[4], 50, 75, 5075);
  checkData(Point, Intervals[5], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 6u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 31, 56, 3156);
  checkData(Point, Intervals[2], 50, 75, 5075);
  checkData(Point, Intervals[3], 36, 54, 3654);
  checkData(Point, Intervals[4], 49, 65, 4965);
  checkData(Point, Intervals[5], 39, 50, 3950);

  Point = 55;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 31, 56, 3156);
  checkData(Point, Intervals[2], 49, 65, 4965);
  checkData(Point, Intervals[3], 50, 75, 5075);
  checkData(Point, Intervals[4], 55, 61, 5561);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 55, 61, 5561);
  checkData(Point, Intervals[1], 49, 65, 4965);
  checkData(Point, Intervals[2], 31, 56, 3156);
  checkData(Point, Intervals[3], 50, 75, 5075);
  checkData(Point, Intervals[4], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 31, 56, 3156);
  checkData(Point, Intervals[2], 50, 75, 5075);
  checkData(Point, Intervals[3], 49, 65, 4965);
  checkData(Point, Intervals[4], 55, 61, 5561);

  Point = 61;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 49, 65, 4965);
  checkData(Point, Intervals[2], 50, 75, 5075);
  checkData(Point, Intervals[3], 55, 61, 5561);
  checkData(Point, Intervals[4], 60, 70, 6070);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 55, 61, 5561);
  checkData(Point, Intervals[1], 60, 70, 6070);
  checkData(Point, Intervals[2], 49, 65, 4965);
  checkData(Point, Intervals[3], 50, 75, 5075);
  checkData(Point, Intervals[4], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 50, 75, 5075);
  checkData(Point, Intervals[2], 49, 65, 4965);
  checkData(Point, Intervals[3], 60, 70, 6070);
  checkData(Point, Intervals[4], 55, 61, 5561);

  Point = 31;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 31, 56, 3156);
  checkData(Point, Intervals[2], 25, 41, 2541);
  checkData(Point, Intervals[3], 15, 40, 1540);
  checkData(Point, Intervals[4], 30, 35, 3035);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 30, 35, 3035);
  checkData(Point, Intervals[1], 25, 41, 2541);
  checkData(Point, Intervals[2], 31, 56, 3156);
  checkData(Point, Intervals[3], 15, 40, 1540);
  checkData(Point, Intervals[4], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 31, 56, 3156);
  checkData(Point, Intervals[2], 15, 40, 1540);
  checkData(Point, Intervals[3], 25, 41, 2541);
  checkData(Point, Intervals[4], 30, 35, 3035);

  Point = 56;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 31, 56, 3156);
  checkData(Point, Intervals[2], 49, 65, 4965);
  checkData(Point, Intervals[3], 50, 75, 5075);
  checkData(Point, Intervals[4], 55, 61, 5561);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 55, 61, 5561);
  checkData(Point, Intervals[1], 49, 65, 4965);
  checkData(Point, Intervals[2], 31, 56, 3156);
  checkData(Point, Intervals[3], 50, 75, 5075);
  checkData(Point, Intervals[4], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 31, 56, 3156);
  checkData(Point, Intervals[2], 50, 75, 5075);
  checkData(Point, Intervals[3], 49, 65, 4965);
  checkData(Point, Intervals[4], 55, 61, 5561);

  Point = 12;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 3u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 11, 16, 1116);
  checkData(Point, Intervals[2], 12, 21, 1221);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 3u);
  checkData(Point, Intervals[0], 11, 16, 1116);
  checkData(Point, Intervals[1], 12, 21, 1221);
  checkData(Point, Intervals[2], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 3u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 12, 21, 1221);
  checkData(Point, Intervals[2], 11, 16, 1116);

  Point = 21;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 15, 40, 1540);
  checkData(Point, Intervals[2], 20, 30, 2030);
  checkData(Point, Intervals[3], 12, 21, 1221);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 12, 21, 1221);
  checkData(Point, Intervals[1], 20, 30, 2030);
  checkData(Point, Intervals[2], 15, 40, 1540);
  checkData(Point, Intervals[3], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 15, 40, 1540);
  checkData(Point, Intervals[2], 20, 30, 2030);
  checkData(Point, Intervals[3], 12, 21, 1221);

  Point = 25;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 15, 40, 1540);
  checkData(Point, Intervals[2], 20, 30, 2030);
  checkData(Point, Intervals[3], 25, 41, 2541);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 20, 30, 2030);
  checkData(Point, Intervals[1], 25, 41, 2541);
  checkData(Point, Intervals[2], 15, 40, 1540);
  checkData(Point, Intervals[3], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 15, 40, 1540);
  checkData(Point, Intervals[2], 25, 41, 2541);
  checkData(Point, Intervals[3], 20, 30, 2030);

  Point = 41;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 31, 56, 3156);
  checkData(Point, Intervals[2], 36, 54, 3654);
  checkData(Point, Intervals[3], 39, 50, 3950);
  checkData(Point, Intervals[4], 25, 41, 2541);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 39, 50, 3950);
  checkData(Point, Intervals[1], 25, 41, 2541);
  checkData(Point, Intervals[2], 36, 54, 3654);
  checkData(Point, Intervals[3], 31, 56, 3156);
  checkData(Point, Intervals[4], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 31, 56, 3156);
  checkData(Point, Intervals[2], 36, 54, 3654);
  checkData(Point, Intervals[3], 25, 41, 2541);
  checkData(Point, Intervals[4], 39, 50, 3950);

  Point = 49;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 31, 56, 3156);
  checkData(Point, Intervals[2], 36, 54, 3654);
  checkData(Point, Intervals[3], 39, 50, 3950);
  checkData(Point, Intervals[4], 49, 65, 4965);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 39, 50, 3950);
  checkData(Point, Intervals[1], 49, 65, 4965);
  checkData(Point, Intervals[2], 36, 54, 3654);
  checkData(Point, Intervals[3], 31, 56, 3156);
  checkData(Point, Intervals[4], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 31, 56, 3156);
  checkData(Point, Intervals[2], 36, 54, 3654);
  checkData(Point, Intervals[3], 49, 65, 4965);
  checkData(Point, Intervals[4], 39, 50, 3950);

  Point = 65;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 50, 75, 5075);
  checkData(Point, Intervals[2], 60, 70, 6070);
  checkData(Point, Intervals[3], 49, 65, 4965);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 60, 70, 6070);
  checkData(Point, Intervals[1], 49, 65, 4965);
  checkData(Point, Intervals[2], 50, 75, 5075);
  checkData(Point, Intervals[3], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 50, 75, 5075);
  checkData(Point, Intervals[2], 49, 65, 4965);
  checkData(Point, Intervals[3], 60, 70, 6070);

  Point = 71;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 3u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 50, 75, 5075);
  checkData(Point, Intervals[2], 71, 79, 7179);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 3u);
  checkData(Point, Intervals[0], 71, 79, 7179);
  checkData(Point, Intervals[1], 50, 75, 5075);
  checkData(Point, Intervals[2], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 3u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 50, 75, 5075);
  checkData(Point, Intervals[2], 71, 79, 7179);

  Point = 79;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 3u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 74, 80, 7480);
  checkData(Point, Intervals[2], 71, 79, 7179);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 3u);
  checkData(Point, Intervals[0], 74, 80, 7480);
  checkData(Point, Intervals[1], 71, 79, 7179);
  checkData(Point, Intervals[2], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 3u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 71, 79, 7179);
  checkData(Point, Intervals[2], 74, 80, 7480);

  Point = 11;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 2u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 11, 16, 1116);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 2u);
  checkData(Point, Intervals[0], 11, 16, 1116);
  checkData(Point, Intervals[1], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 2u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 11, 16, 1116);

  Point = 16;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 15, 40, 1540);
  checkData(Point, Intervals[2], 12, 21, 1221);
  checkData(Point, Intervals[3], 11, 16, 1116);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 11, 16, 1116);
  checkData(Point, Intervals[1], 12, 21, 1221);
  checkData(Point, Intervals[2], 15, 40, 1540);
  checkData(Point, Intervals[3], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 15, 40, 1540);
  checkData(Point, Intervals[2], 12, 21, 1221);
  checkData(Point, Intervals[3], 11, 16, 1116);

  Point = 20;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 15, 40, 1540);
  checkData(Point, Intervals[2], 20, 30, 2030);
  checkData(Point, Intervals[3], 12, 21, 1221);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 12, 21, 1221);
  checkData(Point, Intervals[1], 20, 30, 2030);
  checkData(Point, Intervals[2], 15, 40, 1540);
  checkData(Point, Intervals[3], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 15, 40, 1540);
  checkData(Point, Intervals[2], 20, 30, 2030);
  checkData(Point, Intervals[3], 12, 21, 1221);

  Point = 30;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 25, 41, 2541);
  checkData(Point, Intervals[2], 15, 40, 1540);
  checkData(Point, Intervals[3], 20, 30, 2030);
  checkData(Point, Intervals[4], 30, 35, 3035);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 30, 35, 3035);
  checkData(Point, Intervals[1], 20, 30, 2030);
  checkData(Point, Intervals[2], 25, 41, 2541);
  checkData(Point, Intervals[3], 15, 40, 1540);
  checkData(Point, Intervals[4], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 15, 40, 1540);
  checkData(Point, Intervals[2], 25, 41, 2541);
  checkData(Point, Intervals[3], 20, 30, 2030);
  checkData(Point, Intervals[4], 30, 35, 3035);

  Point = 36;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 31, 56, 3156);
  checkData(Point, Intervals[2], 36, 54, 3654);
  checkData(Point, Intervals[3], 25, 41, 2541);
  checkData(Point, Intervals[4], 15, 40, 1540);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 25, 41, 2541);
  checkData(Point, Intervals[1], 36, 54, 3654);
  checkData(Point, Intervals[2], 31, 56, 3156);
  checkData(Point, Intervals[3], 15, 40, 1540);
  checkData(Point, Intervals[4], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 31, 56, 3156);
  checkData(Point, Intervals[2], 15, 40, 1540);
  checkData(Point, Intervals[3], 36, 54, 3654);
  checkData(Point, Intervals[4], 25, 41, 2541);

  Point = 54;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 31, 56, 3156);
  checkData(Point, Intervals[2], 36, 54, 3654);
  checkData(Point, Intervals[3], 49, 65, 4965);
  checkData(Point, Intervals[4], 50, 75, 5075);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 49, 65, 4965);
  checkData(Point, Intervals[1], 36, 54, 3654);
  checkData(Point, Intervals[2], 31, 56, 3156);
  checkData(Point, Intervals[3], 50, 75, 5075);
  checkData(Point, Intervals[4], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 31, 56, 3156);
  checkData(Point, Intervals[2], 50, 75, 5075);
  checkData(Point, Intervals[3], 36, 54, 3654);
  checkData(Point, Intervals[4], 49, 65, 4965);

  Point = 60;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 49, 65, 4965);
  checkData(Point, Intervals[2], 50, 75, 5075);
  checkData(Point, Intervals[3], 55, 61, 5561);
  checkData(Point, Intervals[4], 60, 70, 6070);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 55, 61, 5561);
  checkData(Point, Intervals[1], 60, 70, 6070);
  checkData(Point, Intervals[2], 49, 65, 4965);
  checkData(Point, Intervals[3], 50, 75, 5075);
  checkData(Point, Intervals[4], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 50, 75, 5075);
  checkData(Point, Intervals[2], 49, 65, 4965);
  checkData(Point, Intervals[3], 60, 70, 6070);
  checkData(Point, Intervals[4], 55, 61, 5561);

  Point = 70;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 3u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 50, 75, 5075);
  checkData(Point, Intervals[2], 60, 70, 6070);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 3u);
  checkData(Point, Intervals[0], 60, 70, 6070);
  checkData(Point, Intervals[1], 50, 75, 5075);
  checkData(Point, Intervals[2], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 3u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 50, 75, 5075);
  checkData(Point, Intervals[2], 60, 70, 6070);

  Point = 74;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 50, 75, 5075);
  checkData(Point, Intervals[2], 71, 79, 7179);
  checkData(Point, Intervals[3], 74, 80, 7480);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 74, 80, 7480);
  checkData(Point, Intervals[1], 71, 79, 7179);
  checkData(Point, Intervals[2], 50, 75, 5075);
  checkData(Point, Intervals[3], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 50, 75, 5075);
  checkData(Point, Intervals[2], 71, 79, 7179);
  checkData(Point, Intervals[3], 74, 80, 7480);

  Point = 80;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 2u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 74, 80, 7480);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 2u);
  checkData(Point, Intervals[0], 74, 80, 7480);
  checkData(Point, Intervals[1], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 2u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 74, 80, 7480);

  Point = 15;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 15, 40, 1540);
  checkData(Point, Intervals[2], 11, 16, 1116);
  checkData(Point, Intervals[3], 12, 21, 1221);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 11, 16, 1116);
  checkData(Point, Intervals[1], 12, 21, 1221);
  checkData(Point, Intervals[2], 15, 40, 1540);
  checkData(Point, Intervals[3], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 15, 40, 1540);
  checkData(Point, Intervals[2], 12, 21, 1221);
  checkData(Point, Intervals[3], 11, 16, 1116);

  Point = 40;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 6u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 31, 56, 3156);
  checkData(Point, Intervals[2], 36, 54, 3654);
  checkData(Point, Intervals[3], 39, 50, 3950);
  checkData(Point, Intervals[4], 25, 41, 2541);
  checkData(Point, Intervals[5], 15, 40, 1540);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 6u);
  checkData(Point, Intervals[0], 39, 50, 3950);
  checkData(Point, Intervals[1], 25, 41, 2541);
  checkData(Point, Intervals[2], 36, 54, 3654);
  checkData(Point, Intervals[3], 31, 56, 3156);
  checkData(Point, Intervals[4], 15, 40, 1540);
  checkData(Point, Intervals[5], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 6u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 31, 56, 3156);
  checkData(Point, Intervals[2], 15, 40, 1540);
  checkData(Point, Intervals[3], 36, 54, 3654);
  checkData(Point, Intervals[4], 25, 41, 2541);
  checkData(Point, Intervals[5], 39, 50, 3950);

  Point = 43;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 31, 56, 3156);
  checkData(Point, Intervals[2], 36, 54, 3654);
  checkData(Point, Intervals[3], 39, 50, 3950);
  checkData(Point, Intervals[4], 43, 45, 4345);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 43, 45, 4345);
  checkData(Point, Intervals[1], 39, 50, 3950);
  checkData(Point, Intervals[2], 36, 54, 3654);
  checkData(Point, Intervals[3], 31, 56, 3156);
  checkData(Point, Intervals[4], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 31, 56, 3156);
  checkData(Point, Intervals[2], 36, 54, 3654);
  checkData(Point, Intervals[3], 39, 50, 3950);
  checkData(Point, Intervals[4], 43, 45, 4345);

  Point = 45;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 31, 56, 3156);
  checkData(Point, Intervals[2], 36, 54, 3654);
  checkData(Point, Intervals[3], 39, 50, 3950);
  checkData(Point, Intervals[4], 43, 45, 4345);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 43, 45, 4345);
  checkData(Point, Intervals[1], 39, 50, 3950);
  checkData(Point, Intervals[2], 36, 54, 3654);
  checkData(Point, Intervals[3], 31, 56, 3156);
  checkData(Point, Intervals[4], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 5u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 31, 56, 3156);
  checkData(Point, Intervals[2], 36, 54, 3654);
  checkData(Point, Intervals[3], 39, 50, 3950);
  checkData(Point, Intervals[4], 43, 45, 4345);

  Point = 50;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 6u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 31, 56, 3156);
  checkData(Point, Intervals[2], 36, 54, 3654);
  checkData(Point, Intervals[3], 39, 50, 3950);
  checkData(Point, Intervals[4], 49, 65, 4965);
  checkData(Point, Intervals[5], 50, 75, 5075);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 6u);
  checkData(Point, Intervals[0], 39, 50, 3950);
  checkData(Point, Intervals[1], 49, 65, 4965);
  checkData(Point, Intervals[2], 36, 54, 3654);
  checkData(Point, Intervals[3], 31, 56, 3156);
  checkData(Point, Intervals[4], 50, 75, 5075);
  checkData(Point, Intervals[5], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 6u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 31, 56, 3156);
  checkData(Point, Intervals[2], 50, 75, 5075);
  checkData(Point, Intervals[3], 36, 54, 3654);
  checkData(Point, Intervals[4], 49, 65, 4965);
  checkData(Point, Intervals[5], 39, 50, 3950);

  Point = 75;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 50, 75, 5075);
  checkData(Point, Intervals[2], 74, 80, 7480);
  checkData(Point, Intervals[3], 71, 79, 7179);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Ascending);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 74, 80, 7480);
  checkData(Point, Intervals[1], 71, 79, 7179);
  checkData(Point, Intervals[2], 50, 75, 5075);
  checkData(Point, Intervals[3], 10, 85, 1085);
  Intervals = Tree.getContaining(Point);
  Tree.sortIntervals(Intervals, UUSorting::Descending);
  ASSERT_EQ(Intervals.size(), 4u);
  checkData(Point, Intervals[0], 10, 85, 1085);
  checkData(Point, Intervals[1], 50, 75, 5075);
  checkData(Point, Intervals[2], 71, 79, 7179);
  checkData(Point, Intervals[3], 74, 80, 7480);

  Point = 10;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  checkData(Point, Intervals[0], 10, 85, 1085);

  Point = 85;
  Intervals = Tree.getContaining(Point);
  ASSERT_EQ(Intervals.size(), 1u);
  checkData(Point, Intervals[0], 10, 85, 1085);

  // Invalid interval values.
  Point = 5;
  Intervals = Tree.getContaining(Point);
  EXPECT_TRUE(Intervals.empty());
  Point = 90;
  Intervals = Tree.getContaining(Point);
  EXPECT_TRUE(Intervals.empty());
}

// Four items tree tests. Overlapping. Check mapped values and iterators.
TEST(IntervalTreeTest, MappedValuesIteratorsTree) {
  UUAlloc Allocator;
  UUTree Tree(Allocator);
  UUPoint Point;

  // [10, 20] <- (1020)
  // [15, 25] <- (1525)
  // [50, 60] <- (5060)
  // [55, 65] <- (5565)
  //
  //    [10.........20]
  //          [15.........25]
  //                            [50.........60]
  //                                  [55.........65]
  Tree.insert(10, 20, 1020);
  Tree.insert(15, 25, 1525);
  Tree.insert(50, 60, 5060);
  Tree.insert(55, 65, 5565);
  Tree.create();

  // Iterators.
  {
    // Start searching for '10'.
    Point = 10;
    UUIter Iter = Tree.find(Point);
    EXPECT_NE(Iter, Tree.find_end());
    checkData(Point, Iter, 10, 20, 1020);
    ++Iter;
    EXPECT_EQ(Iter, Tree.find_end());
  }
  {
    // Start searching for '15'.
    Point = 15;
    UUIter Iter = Tree.find(Point);
    ASSERT_TRUE(Iter != Tree.find_end());
    checkData(Point, Iter, 15, 25, 1525);
    ++Iter;
    ASSERT_TRUE(Iter != Tree.find_end());
    checkData(Point, Iter, 10, 20, 1020);
    ++Iter;
    EXPECT_EQ(Iter, Tree.find_end());
  }
  {
    // Start searching for '20'.
    Point = 20;
    UUIter Iter = Tree.find(Point);
    ASSERT_TRUE(Iter != Tree.find_end());
    checkData(Point, Iter, 15, 25, 1525);
    ++Iter;
    ASSERT_TRUE(Iter != Tree.find_end());
    checkData(Point, Iter, 10, 20, 1020);
    ++Iter;
    EXPECT_EQ(Iter, Tree.find_end());
  }
  {
    // Start searching for '25'.
    Point = 25;
    UUIter Iter = Tree.find(Point);
    ASSERT_TRUE(Iter != Tree.find_end());
    checkData(Point, Iter, 15, 25, 1525);
    ++Iter;
    EXPECT_EQ(Iter, Tree.find_end());
  }
  // Invalid interval values.
  {
    Point = 5;
    UUIter Iter = Tree.find(Point);
    EXPECT_EQ(Iter, Tree.find_end());
  }
  {
    Point = 45;
    UUIter Iter = Tree.find(Point);
    EXPECT_EQ(Iter, Tree.find_end());
  }
  {
    Point = 70;
    UUIter Iter = Tree.find(Point);
    EXPECT_EQ(Iter, Tree.find_end());
  }
}

} // namespace
