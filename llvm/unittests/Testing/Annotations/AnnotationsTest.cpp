//===----- unittests/AnnotationsTest.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "llvm/Testing/Annotations/Annotations.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::ResultOf;
using ::testing::UnorderedElementsAre;

namespace {
MATCHER_P2(pair, first_matcher, second_matcher, "") {
  return testing::ExplainMatchResult(
      AllOf(ResultOf([](const auto &entry) { return entry.getKey(); },
                     first_matcher),
            ResultOf([](const auto &entry) { return entry.getValue(); },
                     second_matcher)),
      arg, result_listener);
}

llvm::Annotations::Range range(size_t Begin, size_t End) {
  llvm::Annotations::Range R;
  R.Begin = Begin;
  R.End = End;
  return R;
}

TEST(AnnotationsTest, CleanedCode) {
  EXPECT_EQ(llvm::Annotations("foo^bar$nnn[[baz$^[[qux]]]]").code(),
            "foobarbazqux");
}

TEST(AnnotationsTest, Points) {
  // A single point.
  EXPECT_EQ(llvm::Annotations("^ab").point(), 0u);
  EXPECT_EQ(llvm::Annotations("a^b").point(), 1u);
  EXPECT_EQ(llvm::Annotations("ab^").point(), 2u);

  // Multiple points.
  EXPECT_THAT(llvm::Annotations("^a^bc^d^").points(),
              ElementsAre(0u, 1u, 3u, 4u));

  // No points.
  EXPECT_THAT(llvm::Annotations("ab[[cd]]").points(), IsEmpty());

  // Consecutive points.
  EXPECT_THAT(llvm::Annotations("ab^^^cd").points(), ElementsAre(2u, 2u, 2u));
}

TEST(AnnotationsTest, AllPoints) {
  // Multiple points.
  EXPECT_THAT(llvm::Annotations("0$p1^123$p2^456$p1^$p1^78^9").all_points(),
              UnorderedElementsAre(pair("", ElementsAre(9u)),
                                   pair("p1", ElementsAre(1u, 7u, 7u)),
                                   pair("p2", ElementsAre(4u))));

  // No points.
  EXPECT_THAT(llvm::Annotations("ab[[cd]]").all_points(), IsEmpty());
}

TEST(AnnotationsTest, Ranges) {
  // A single range.
  EXPECT_EQ(llvm::Annotations("[[a]]bc").range(), range(0, 1));
  EXPECT_EQ(llvm::Annotations("a[[bc]]d").range(), range(1, 3));
  EXPECT_EQ(llvm::Annotations("ab[[cd]]").range(), range(2, 4));

  // Empty range.
  EXPECT_EQ(llvm::Annotations("[[]]ab").range(), range(0, 0));
  EXPECT_EQ(llvm::Annotations("a[[]]b").range(), range(1, 1));
  EXPECT_EQ(llvm::Annotations("ab[[]]").range(), range(2, 2));

  // Multiple ranges.
  EXPECT_THAT(llvm::Annotations("[[a]][[b]]cd[[ef]]ef").ranges(),
              ElementsAre(range(0, 1), range(1, 2), range(4, 6)));

  // No ranges.
  EXPECT_THAT(llvm::Annotations("ab^c^defef").ranges(), IsEmpty());
}

TEST(AnnotationsTest, AllRanges) {
  // Multiple ranges.
  EXPECT_THAT(
      llvm::Annotations("[[]]01$outer[[2[[[[$inner[[3]]]]]]456]]7$outer[[89]]")
          .all_ranges(),
      UnorderedElementsAre(
          pair("", ElementsAre(range(0, 0), range(3, 4), range(3, 4))),
          pair("outer", ElementsAre(range(2, 7), range(8, 10))),
          pair("inner", ElementsAre(range(3, 4)))));

  // No ranges.
  EXPECT_THAT(llvm::Annotations("ab^c^defef").all_ranges(), IsEmpty());
}

TEST(AnnotationsTest, Nested) {
  llvm::Annotations Annotated("a[[f^oo^bar[[b[[a]]z]]]]bcdef");
  EXPECT_THAT(Annotated.points(), ElementsAre(2u, 4u));
  EXPECT_THAT(Annotated.ranges(),
              ElementsAre(range(8, 9), range(7, 10), range(1, 10)));
}

TEST(AnnotationsTest, Payload) {
  // // A single unnamed point or range with unspecified payload
  EXPECT_THAT(llvm::Annotations("a$^b").pointWithPayload(), Pair(1u, ""));
  EXPECT_THAT(llvm::Annotations("a$[[b]]cdef").rangeWithPayload(),
              Pair(range(1, 2), ""));

  // A single unnamed point or range with empty payload
  EXPECT_THAT(llvm::Annotations("a$()^b").pointWithPayload(), Pair(1u, ""));
  EXPECT_THAT(llvm::Annotations("a$()[[b]]cdef").rangeWithPayload(),
              Pair(range(1, 2), ""));

  // A single unnamed point or range with payload.
  EXPECT_THAT(llvm::Annotations("a$(foo)^b").pointWithPayload(),
              Pair(1u, "foo"));
  EXPECT_THAT(llvm::Annotations("a$(foo)[[b]]cdef").rangeWithPayload(),
              Pair(range(1, 2), "foo"));

  // A single named point or range with payload
  EXPECT_THAT(llvm::Annotations("a$name(foo)^b").pointWithPayload("name"),
              Pair(1u, "foo"));
  EXPECT_THAT(
      llvm::Annotations("a$name(foo)[[b]]cdef").rangeWithPayload("name"),
      Pair(range(1, 2), "foo"));

  // Multiple named points with payload.
  llvm::Annotations Annotated("a$p1(p1)^bcd$p2(p2)^123$p1^345");
  EXPECT_THAT(Annotated.points(), IsEmpty());
  EXPECT_THAT(Annotated.pointsWithPayload("p1"),
              ElementsAre(Pair(1u, "p1"), Pair(7u, "")));
  EXPECT_THAT(Annotated.pointWithPayload("p2"), Pair(4u, "p2"));
}

TEST(AnnotationsTest, Named) {
  // A single named point or range.
  EXPECT_EQ(llvm::Annotations("a$foo^b").point("foo"), 1u);
  EXPECT_EQ(llvm::Annotations("a$foo[[b]]cdef").range("foo"), range(1, 2));

  // Empty names should also work.
  EXPECT_EQ(llvm::Annotations("a$^b").point(""), 1u);
  EXPECT_EQ(llvm::Annotations("a$[[b]]cdef").range(""), range(1, 2));

  // Multiple named points.
  llvm::Annotations Annotated("a$p1^bcd$p2^123$p1^345");
  EXPECT_THAT(Annotated.points(), IsEmpty());
  EXPECT_THAT(Annotated.points("p1"), ElementsAre(1u, 7u));
  EXPECT_EQ(Annotated.point("p2"), 4u);
}

TEST(AnnotationsTest, Errors) {
  // Annotations use llvm_unreachable, it will only crash in debug mode.
#ifndef NDEBUG
  // point() and range() crash on zero or multiple ranges.
  EXPECT_DEATH(llvm::Annotations("ab[[c]]def").point(),
               "expected exactly one point");
  EXPECT_DEATH(llvm::Annotations("a^b^cdef").point(),
               "expected exactly one point");

  EXPECT_DEATH(llvm::Annotations("a^bcdef").range(),
               "expected exactly one range");
  EXPECT_DEATH(llvm::Annotations("a[[b]]c[[d]]ef").range(),
               "expected exactly one range");

  EXPECT_DEATH(llvm::Annotations("$foo^a$foo^a").point("foo"),
               "expected exactly one point");
  EXPECT_DEATH(llvm::Annotations("$foo[[a]]bc$foo[[a]]").range("foo"),
               "expected exactly one range");

  // Parsing failures.
  EXPECT_DEATH(llvm::Annotations("ff[[fdfd"), "unmatched \\[\\[");
  EXPECT_DEATH(llvm::Annotations("ff[[fdjsfjd]]xxx]]"), "unmatched \\]\\]");
  EXPECT_DEATH(llvm::Annotations("ff$fdsfd"), "unterminated \\$name");
  EXPECT_DEATH(llvm::Annotations("ff$("), "unterminated payload");
  EXPECT_DEATH(llvm::Annotations("ff$name("), "unterminated payload");
#endif
}
} // namespace
