#include "llvm/CodeGen/MachineScheduler.h"
#include "gtest/gtest.h"

using namespace llvm;

#ifndef NDEBUG
TEST(ResourceSegmentsDeath, OverwriteOnRight) {
  auto X = ResourceSegments({{10, 20}});
  EXPECT_DEATH(X.add({15, 30}), "A resource is being overwritten");
}

TEST(ResourceSegmentsDeath, OverwriteOnLeft) {
  auto X = ResourceSegments({{10, 20}});
  EXPECT_DEATH(X.add({5, 11}), "A resource is being overwritten");
  ;
}

TEST(ResourceSegmentsDeath, FullOverwrite) {
  auto X = ResourceSegments({{10, 20}});
  EXPECT_DEATH(X.add({15, 18}), "A resource is being overwritten");
}

TEST(ResourceSegmentsDeath, ZeroSizeIntervalsNotAllowed) {
  auto X = ResourceSegments({{10, 20}});
  EXPECT_DEATH(X.add({20, 30}, 0), "0-size interval history has no use.");
}
#endif // NDEBUG

TEST(ResourceSegments, ConsecutiveLeftNoOverlap) {
  auto X = ResourceSegments({{10, 20}});
  X.add({7, 9});
  EXPECT_EQ(X, ResourceSegments({{7, 9}, {10, 20}}));
}

TEST(ResourceSegments, ConsecutiveLeftWithOverlap) {
  auto X = ResourceSegments({{10, 20}});
  X.add({7, 10});
  EXPECT_EQ(X, ResourceSegments({{7, 20}}));
}

TEST(ResourceSegments, ConsecutiveRightNoOverlap) {
  auto X = ResourceSegments({{10, 20}});
  X.add({21, 22});
  EXPECT_EQ(X, ResourceSegments({{10, 20}, {21, 22}}));
}

TEST(ResourceSegments, ConsecutiveRightWithOverlap) {
  auto X = ResourceSegments({{10, 20}});
  X.add({20, 22});
  EXPECT_EQ(X, ResourceSegments({{10, 22}}));
}

TEST(ResourceSegments, Disjoint) {
  auto X = ResourceSegments({{10, 20}});
  X.add({22, 23});
  EXPECT_EQ(X, ResourceSegments({{10, 20}, {22, 23}}));
}

TEST(ResourceSegments, SortAfterAdd) {
  auto X = ResourceSegments({{10, 20}, {3, 4}});
  X.add({6, 8});
  EXPECT_EQ(X, ResourceSegments({{3, 4}, {6, 8}, {10, 20}}));
}

TEST(ResourceSegments, AddWithCutOff) {
  auto X = ResourceSegments({{1, 2}, {3, 4}});
  X.add({6, 8}, 2);
  EXPECT_EQ(X, ResourceSegments({{3, 4}, {6, 8}}));
}

TEST(ResourceSegments, add_01) {
  auto X = ResourceSegments({{10, 20}, {30, 40}});
  X.add({21, 29});
  EXPECT_EQ(X, ResourceSegments({{10, 20}, {21, 29}, {30, 40}}));
}

TEST(ResourceSegments, add_02) {
  auto X = ResourceSegments({{10, 20}, {30, 40}});
  X.add({22, 29});
  EXPECT_EQ(X, ResourceSegments({{10, 20}, {22, 29}, {30, 40}}));
  X.add({29, 30});
  EXPECT_EQ(X, ResourceSegments({{10, 20}, {22, 40}}));
}

#ifndef NDEBUG
TEST(ResourceSegmentsDeath, add_empty) {
  auto X = ResourceSegments({{10, 20}, {30, 40}});
  EXPECT_DEATH(X.add({22, 22}), "Cannot add empty resource usage");
}
#endif

TEST(ResourceSegments, sort_two) {
  EXPECT_EQ(ResourceSegments({{30, 40}, {10, 28}}),
            ResourceSegments({{10, 28}, {30, 40}}));
}

TEST(ResourceSegments, sort_three) {
  EXPECT_EQ(ResourceSegments({{30, 40}, {71, 200}, {10, 29}}),
            ResourceSegments({{10, 29}, {30, 40}, {71, 200}}));
}

TEST(ResourceSegments, merge_two) {
  EXPECT_EQ(ResourceSegments({{10, 33}, {30, 40}}),
            ResourceSegments({{10, 40}}));
  EXPECT_EQ(ResourceSegments({{10, 30}, {30, 40}}),
            ResourceSegments({{10, 40}}));
  // Cycle 29 is resource free, so the interval is disjoint.
  EXPECT_EQ(ResourceSegments({{10, 29}, {30, 40}}),
            ResourceSegments({{10, 29}, {30, 40}}));
}

TEST(ResourceSegments, merge_three) {
  EXPECT_EQ(ResourceSegments({{10, 29}, {30, 40}, {71, 200}}),
            ResourceSegments({{10, 29}, {30, 40}, {71, 200}}));
  EXPECT_EQ(ResourceSegments({{10, 29}, {30, 40}, {41, 200}}),
            ResourceSegments({{10, 29}, {30, 40}, {41, 200}}));
  EXPECT_EQ(ResourceSegments({{10, 30}, {30, 40}, {40, 200}}),
            ResourceSegments({{10, 200}}));
  EXPECT_EQ(ResourceSegments({{10, 28}, {30, 71}, {71, 200}}),
            ResourceSegments({{10, 28}, {30, 200}}));
}

////////////////////////////////////////////////////////////////////////////////
// Intersection
TEST(ResourceSegments, intersects) {
  // no intersect
  EXPECT_FALSE(ResourceSegments::intersects({0, 1}, {3, 4}));
  EXPECT_FALSE(ResourceSegments::intersects({3, 4}, {0, 1}));
  EXPECT_FALSE(ResourceSegments::intersects({0, 3}, {3, 4}));
  EXPECT_FALSE(ResourceSegments::intersects({3, 4}, {0, 3}));

  // Share one boundary
  EXPECT_TRUE(ResourceSegments::intersects({5, 6}, {5, 10}));
  EXPECT_TRUE(ResourceSegments::intersects({5, 10}, {5, 6}));

  // full intersect
  EXPECT_TRUE(ResourceSegments::intersects({1, 2}, {0, 3}));
  EXPECT_TRUE(ResourceSegments::intersects({1, 2}, {0, 2}));
  EXPECT_TRUE(ResourceSegments::intersects({0, 3}, {1, 2}));
  EXPECT_TRUE(ResourceSegments::intersects({0, 2}, {1, 2}));

  // right intersect
  EXPECT_TRUE(ResourceSegments::intersects({2, 4}, {0, 3}));
  EXPECT_TRUE(ResourceSegments::intersects({0, 3}, {2, 4}));

  // left intersect
  EXPECT_TRUE(ResourceSegments::intersects({2, 4}, {3, 5}));
  EXPECT_TRUE(ResourceSegments::intersects({3, 5}, {2, 4}));
}

////////////////////////////////////////////////////////////////////////////////
// TOP-DOWN getFirstAvailableAt
TEST(ResourceSegments, getFirstAvailableAtFromTop_oneCycle) {
  auto X = ResourceSegments({{2, 5}});
  //       0 1 2 3 4 5 6 7
  //  Res      X X X
  //    ...X...
  EXPECT_EQ(X.getFirstAvailableAtFromTop(0, 0, 1), 0U);
  EXPECT_EQ(X.getFirstAvailableAtFromTop(1, 0, 1), 1U);
  // Skip to five when hitting cycle 2
  EXPECT_EQ(X.getFirstAvailableAtFromTop(2, 0, 1), 5U);
}

TEST(ResourceSegments, getFirstAvailableAtFromTop_twoCycles) {
  auto X = ResourceSegments({{4, 5}});
  //       0 1 2 3 4 5 6 7
  //  Res          X
  //    ...X X....
  EXPECT_EQ(X.getFirstAvailableAtFromTop(0, 0, 2), 0U);
  EXPECT_EQ(X.getFirstAvailableAtFromTop(1, 0, 2), 1U);
  EXPECT_EQ(X.getFirstAvailableAtFromTop(2, 0, 2), 2U);
  // Skip to cycle 5
  EXPECT_EQ(X.getFirstAvailableAtFromTop(3, 0, 2), 5U);
}

TEST(ResourceSegments, getFirstAvailableAtFromTop_twoCycles_Shifted) {
  auto X = ResourceSegments({{4, 5}});
  //       0 1 2 3 4 5 6 7
  //  Res          X
  //    ...c X X...
  EXPECT_EQ(X.getFirstAvailableAtFromTop(0, 1, 3), 0U);
  EXPECT_EQ(X.getFirstAvailableAtFromTop(1, 1, 3), 1U);
  // Skip to cycle 4
  EXPECT_EQ(X.getFirstAvailableAtFromTop(2, 1, 3), 4U);
  // Stay con cycle 4
  //       0 1 2 3 4 5 6 7
  //  Res          X
  //            ...c X X...
  EXPECT_EQ(X.getFirstAvailableAtFromTop(3, 1, 3), 4U);
  //
  EXPECT_EQ(X.getFirstAvailableAtFromTop(4, 1, 3), 4U);
  EXPECT_EQ(X.getFirstAvailableAtFromTop(5, 1, 3), 5U);
}

TEST(ResourceSegments, getFirstAvailableAtFromTop_twoCycles_Shifted_withGap) {
  auto X = ResourceSegments({{4, 5}, {7, 9}});
  //       0 1 2 3 4 5 6 7 8 9
  //  Res          X     X X
  //         c X X
  EXPECT_EQ(X.getFirstAvailableAtFromTop(1, 1, 3), 1U);
  //       0 1 2 3 4 5 6 7 8 9
  //  Res          X     X X
  //           c X X --> moves to 4
  EXPECT_EQ(X.getFirstAvailableAtFromTop(2, 1, 3), 4U);
  //       0 1 2 3 4 5 6 7 8 9
  //  Res          X     X X
  //             c X X --> moves to 4
  EXPECT_EQ(X.getFirstAvailableAtFromTop(3, 1, 3), 4U);
  //       0 1 2 3 4 5 6 7 8 9
  //  Res          X     X X
  //               c X X --> stays on 4
  EXPECT_EQ(X.getFirstAvailableAtFromTop(4, 1, 3), 4U);
  //       0 1 2 3 4 5 6 7 8 9
  //  Res          X     X X
  //                 c X X --> skips to 8
  EXPECT_EQ(X.getFirstAvailableAtFromTop(5, 1, 3), 8U);
}

TEST(ResourceSegments, getFirstAvailableAtFromTop_basic) {
  auto X = ResourceSegments({{5, 10}, {30, 40}});

  EXPECT_EQ(X.getFirstAvailableAtFromTop(0, 3, 4), 0U);
  EXPECT_EQ(X.getFirstAvailableAtFromTop(1, 3, 4), 1U);
  EXPECT_EQ(X.getFirstAvailableAtFromTop(2, 3, 4), 7U);
  EXPECT_EQ(X.getFirstAvailableAtFromTop(3, 3, 4), 7U);
  EXPECT_EQ(X.getFirstAvailableAtFromTop(4, 3, 4), 7U);
  EXPECT_EQ(X.getFirstAvailableAtFromTop(5, 3, 4), 7U);
  EXPECT_EQ(X.getFirstAvailableAtFromTop(6, 3, 4), 7U);
  EXPECT_EQ(X.getFirstAvailableAtFromTop(7, 3, 4), 7U);
  // Check the empty range between the two intervals of X.
  EXPECT_EQ(X.getFirstAvailableAtFromTop(15, 3, 4), 15U);
  // Overlap the second interval.
  EXPECT_EQ(X.getFirstAvailableAtFromTop(28, 3, 4), 37U);
}

TEST(ResourceSegments, getFirstAvailableAtFromTop_advanced) {
  auto X = ResourceSegments({{3, 6}, {7, 9}, {11, 14}, {30, 33}});

  EXPECT_EQ(X.getFirstAvailableAtFromTop(2, 4, 5), 2U);

  EXPECT_EQ(X.getFirstAvailableAtFromTop(2, 3, 4), 3U);
  // Can schedule at 7U because the interval [14, 19[ does not
  // overlap any of the intervals in X.
  EXPECT_EQ(X.getFirstAvailableAtFromTop(1, 7, 12), 7U);
}

////////////////////////////////////////////////////////////////////////////////
// BOTTOM-UP getFirstAvailableAt
TEST(ResourceSegments, getFirstAvailableAtFromBottom) {
  // Scheduling cycles move to the left...
  //
  //    41 40 39 ... 31 30 29 ... 21 20 19 ... 11 10 9 8 7 6 ... 1 0
  // Res       X   X  X  X               X   X  X  X
  //                                                               X X X X X X
  // Time (relative to instruction execution)                      0 1 2 3 4 5
  auto X = ResourceSegments({{10, 20}, {30, 40}});
  // .. but time (instruction cycle) moves to the right. Therefore, it
  // is always possible to llocate a resource to the right of 0 if 0
  // is not taken, because the right side of the scheduling cycles is
  // empty.
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(0, 0, 1), 0U);
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(0, 0, 9), 0U);
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(0, 0, 10), 0U);
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(0, 0, 20), 0U);
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(0, 0, 21), 0U);
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(0, 0, 22), 0U);
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(0, 0, 29), 0U);
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(0, 0, 30), 0U);
}

TEST(ResourceSegments, getFirstAvailableAtFromBottom_01) {
  auto X = ResourceSegments({{3, 7}});
  // 10 9 8 7 6 5 4 3 2 1 0
  //          X X X X
  //     ...X...           <- one cycle resource placement
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(0, 0, 1), 0U);
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(1, 0, 1), 1U);
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(2, 0, 1), 2U);
  // Skip to 7
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(3, 0, 1), 7U);
}

TEST(ResourceSegments, getFirstAvailableAtFromBottom_02) {
  auto X = ResourceSegments({{3, 7}});
  // 10 9 8 7 6 5 4 3 2 1 0
  //          X X X X
  //   ...X X...           <- two cycles resource placement
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(0, 0, 2), 0U);
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(1, 0, 2), 1U);
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(2, 0, 2), 2U);
  // skip to 8
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(3, 0, 2), 8U);
}

TEST(ResourceSegments, getFirstAvailableAtFromBottom_02_shifted) {
  auto X = ResourceSegments({{3, 7}});
  // 10 9 8 7 6 5 4 3 2 1 0
  //          X X X X
  //    c X X              <- two cycles resource placement but shifted by 1
  //    0 1 2              <- cycles relative to the execution of the
  //    instruction
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(0, 1, 3), 0U);
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(1, 1, 3), 1U);
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(2, 1, 3), 2U);
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(3, 1, 3), 3U);
  // 10 9 8 7 6 5 4 3 2 1 0
  //          X X X X
  //              c X X -> skip to 9
  //              0 1 2
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(4, 1, 3), 9U);
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(5, 1, 3), 9U);
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(6, 1, 3), 9U);
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(7, 1, 3), 9U);
  // 10 9 8 7 6 5 4 3 2 1 0
  //          X X X X
  //      c X X   <- skip to 9
  //      0 1 2
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(8, 1, 3), 9U);
  // 10 9 8 7 6 5 4 3 2 1 0
  //          X X X X
  //    c X X
  //    0 1 2
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(9, 1, 3), 9U);
  // 10 9 8 7 6 5 4 3 2 1 0
  //          X X X X
  //  c X X
  //  0 1 2
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(10, 1, 3), 10U);
}

TEST(ResourceSegments, getFirstAvailableAtFromBottom_03) {
  auto X = ResourceSegments({{1, 2}, {3, 7}});
  //  14 13 12 11 10 9 8 7 6 5 4 3 2 1 0
  //                       X X X X   X
  //                                   X
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(0, 0, 1), 0U);
  //  14 13 12 11 10 9 8 7 6 5 4 3 2 1 0
  //                       X X X X   X
  //                               X
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(1, 0, 1), 2U);
  //  14 13 12 11 10 9 8 7 6 5 4 3 2 1 0
  //                       X X X X   X
  //                               X
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(2, 0, 1), 2U);
  //  14 13 12 11 10 9 8 7 6 5 4 3 2 1 0
  //                       X X X X   X
  //           X  X  X X X
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(2, 0, 5), 11U);
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(3, 0, 5), 11U);
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(5, 0, 5), 11U);
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(11, 0, 5), 11U);
  //  14 13 12 11 10 9 8 7 6 5 4 3 2 1 0
  //                       X X X X   X
  //        X  X  X  X X
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(12, 0, 5), 12U);
}

TEST(ResourceSegments, getFirstAvailableAtFromBottom_03_shifted) {
  //  14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 -1 -2 -3
  //                 X     X X X X   X       X  X
  auto X = ResourceSegments({{-3, -1}, {1, 2}, {3, 7}, {9, 10}});

  EXPECT_EQ(X.getFirstAvailableAtFromBottom(0, 1, 2), 0U);
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(0, 0, 2), 0U);

  //  14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 -1 -2 -3
  //                 X     X X X X   X       X  X
  //                                   X  X  X -> skip to cycle 12
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(0, 0, 3), 12U);
  //  14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 -1 -2 -3
  //                 X     X X X X   X       X  X
  //                                   X  X
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(0, 1, 3), 1U);
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(0, 1, 4), 13U);
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(12, 1, 4), 13U);
  //  14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 -1 -2 -3
  //                 X     X X X X   X       X  X
  //      c  X  X  X
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(13, 1, 4), 13U);
  //  14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 -1 -2 -3
  //                 X     X X X X   X       X  X
  //                                   X  X
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(1, 1, 3), 1U);
  //  14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 -1 -2 -3
  //                 X     X X X X   X       X  X
  //                               C X  X 0 -> skip to cycle 9
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(2, 1, 3), 9U);
  //  16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 -1 -2 -3
  //                       X     X X X X   X       X  X
  //                                   C C X X  X  X  X -> skip to cycle 16
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(3, 2, 7), 16U);
}
TEST(ResourceSegments, getFirstAvailableAtFromBottom_empty) {
  // Empty resource usage can accept schediling at any cycle
  auto X = ResourceSegments();
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(0, 0, 1), 0U);
  EXPECT_EQ(X.getFirstAvailableAtFromBottom(17, 1, 22), 17U);
}
