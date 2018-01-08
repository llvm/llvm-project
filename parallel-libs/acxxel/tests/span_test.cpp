//===--- span_test.cpp - Tests for the span class -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "span.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <array>
#include <vector>

namespace {

template <typename T, size_t N> size_t arraySize(T (&)[N]) { return N; }

TEST(Span, NullConstruction) {
  acxxel::Span<int> Span0;
  EXPECT_EQ(nullptr, Span0.data());
  EXPECT_EQ(0, Span0.size());

  acxxel::Span<int> Span1(nullptr);
  EXPECT_EQ(nullptr, Span1.data());
  EXPECT_EQ(0, Span1.size());
}

TEST(Span, PtrSizeConstruction) {
  int ZeroSize = 0;
  acxxel::Span<int> Span0(nullptr, ZeroSize);
  EXPECT_EQ(Span0.data(), nullptr);
  EXPECT_EQ(Span0.size(), 0);

  int Values[] = {0, 1, 2};
  acxxel::Span<int> Span1(Values, arraySize(Values));
  EXPECT_EQ(Span1.data(), Values);
  EXPECT_EQ(static_cast<size_t>(Span1.size()), arraySize(Values));

  acxxel::Span<int> Span2(Values, ZeroSize);
  EXPECT_EQ(Span2.data(), Values);
  EXPECT_EQ(Span2.size(), 0);
}

TEST(Span, PtrSizeConstruction_NegativeCount) {
  int Values[] = {0, 1, 2};
  EXPECT_DEATH(acxxel::Span<int> Span0(Values, -1), "terminate");
}

TEST(Span, PtrSizeConstruction_NullptrNonzeroSize) {
  EXPECT_DEATH(acxxel::Span<int> Span0(nullptr, 1), "terminate");
}

TEST(Span, FirstLastConstruction) {
  int Values[] = {0, 1, 2};

  acxxel::Span<int> Span0(Values, Values);
  EXPECT_EQ(Span0.data(), Values);
  EXPECT_EQ(Span0.size(), 0);

  acxxel::Span<int> Span(Values, Values + 2);
  EXPECT_EQ(Span.data(), Values);
  EXPECT_EQ(Span.size(), 2);
}

TEST(Span, FirstLastConstruction_LastBeforeFirst) {
  int Values[] = {0, 1, 2};
  EXPECT_DEATH(acxxel::Span<int> Span(Values + 2, Values), "terminate");
}

TEST(Span, ArrayConstruction) {
  int Array[] = {0, 1, 2};
  acxxel::Span<int> Span(Array);
  EXPECT_EQ(Span.data(), Array);
  EXPECT_EQ(Span.size(), 3);
}

TEST(Span, StdArrayConstruction) {
  std::array<int, 3> Array{{0, 1, 2}};
  acxxel::Span<int> Span(Array);
  EXPECT_EQ(Span.data(), Array.data());
  EXPECT_EQ(static_cast<size_t>(Span.size()), Array.size());

  std::array<const int, 3> ConstArray{{0, 1, 2}};
  acxxel::Span<const int> ConstSpan(ConstArray);
  EXPECT_EQ(ConstSpan.data(), ConstArray.data());
  EXPECT_EQ(static_cast<size_t>(ConstSpan.size()), ConstArray.size());
}

TEST(Span, ContainerConstruction) {
  std::vector<int> Vector = {0, 1, 2};
  acxxel::Span<int> Span(Vector);
  EXPECT_EQ(Span.data(), &Vector[0]);
  EXPECT_EQ(static_cast<size_t>(Span.size()), Vector.size());
}

TEST(Span, CopyConstruction) {
  int Values[] = {0, 1, 2};
  acxxel::Span<int> Span0(Values);
  acxxel::Span<int> Span1(Span0);
  EXPECT_EQ(Span1.data(), Values);
  EXPECT_EQ(static_cast<size_t>(Span1.size()), arraySize(Values));
}

TEST(Span, CopyAssignment) {
  int Values[] = {0, 1, 2};
  acxxel::Span<int> Span0(Values);
  acxxel::Span<int> Span1;
  Span1 = Span0;
  EXPECT_EQ(Span1.data(), Values);
  EXPECT_EQ(static_cast<size_t>(Span1.size()), arraySize(Values));
}

TEST(Span, CopyConstFromNonConst) {
  int Values[] = {0, 1, 2};
  acxxel::Span<int> Span0(Values);
  acxxel::Span<const int> Span1(Span0);
  EXPECT_EQ(Span1.data(), Values);
  EXPECT_EQ(static_cast<size_t>(Span1.size()), arraySize(Values));
}

TEST(Span, FirstMethod) {
  int Values[] = {0, 1, 2};
  acxxel::Span<int> Span(Values);
  acxxel::Span<int> Span0 = Span.first(0);
  acxxel::Span<int> Span1 = Span.first(1);
  acxxel::Span<int> Span2 = Span.first(2);
  acxxel::Span<int> Span3 = Span.first(3);

  EXPECT_EQ(Span0.data(), Values);
  EXPECT_EQ(Span1.data(), Values);
  EXPECT_EQ(Span2.data(), Values);
  EXPECT_EQ(Span3.data(), Values);

  EXPECT_TRUE(Span0.empty());

  EXPECT_THAT(Span1, ::testing::ElementsAre(0));
  EXPECT_THAT(Span2, ::testing::ElementsAre(0, 1));
  EXPECT_THAT(Span3, ::testing::ElementsAre(0, 1, 2));
}

TEST(Span, FirstMethod_IllegalArguments) {
  int Values[] = {0, 1, 2};
  acxxel::Span<int> Span(Values);

  EXPECT_DEATH(Span.first(-1), "terminate");
  EXPECT_DEATH(Span.first(4), "terminate");
}

TEST(Span, LastMethod) {
  int Values[] = {0, 1, 2};
  acxxel::Span<int> Span(Values);
  acxxel::Span<int> Span0 = Span.last(0);
  acxxel::Span<int> Span1 = Span.last(1);
  acxxel::Span<int> Span2 = Span.last(2);
  acxxel::Span<int> Span3 = Span.last(3);

  EXPECT_EQ(Span0.data(), Values);
  EXPECT_EQ(Span1.data(), Values + 2);
  EXPECT_EQ(Span2.data(), Values + 1);
  EXPECT_EQ(Span3.data(), Values);

  EXPECT_TRUE(Span0.empty());

  EXPECT_THAT(Span1, ::testing::ElementsAre(2));
  EXPECT_THAT(Span2, ::testing::ElementsAre(1, 2));
  EXPECT_THAT(Span3, ::testing::ElementsAre(0, 1, 2));
}

TEST(Span, LastMethod_IllegalArguments) {
  int Values[] = {0, 1, 2};
  acxxel::Span<int> Span(Values);

  EXPECT_DEATH(Span.last(-1), "terminate");
  EXPECT_DEATH(Span.last(4), "terminate");
}

TEST(Span, SubspanMethod) {
  int Values[] = {0, 1, 2};
  acxxel::Span<int> Span(Values);

  acxxel::Span<int> Span0 = Span.subspan(0);
  acxxel::Span<int> Span0e = Span.subspan(0, acxxel::dynamic_extent);
  acxxel::Span<int> Span00 = Span.subspan(0, 0);
  acxxel::Span<int> Span01 = Span.subspan(0, 1);
  acxxel::Span<int> Span02 = Span.subspan(0, 2);
  acxxel::Span<int> Span03 = Span.subspan(0, 3);

  acxxel::Span<int> Span1 = Span.subspan(1);
  acxxel::Span<int> Span1e = Span.subspan(1, acxxel::dynamic_extent);
  acxxel::Span<int> Span10 = Span.subspan(1, 0);
  acxxel::Span<int> Span11 = Span.subspan(1, 1);
  acxxel::Span<int> Span12 = Span.subspan(1, 2);

  acxxel::Span<int> Span2 = Span.subspan(2);
  acxxel::Span<int> Span2e = Span.subspan(2, acxxel::dynamic_extent);
  acxxel::Span<int> Span20 = Span.subspan(2, 0);
  acxxel::Span<int> Span21 = Span.subspan(2, 1);

  acxxel::Span<int> Span3 = Span.subspan(3);
  acxxel::Span<int> Span3e = Span.subspan(3, acxxel::dynamic_extent);
  acxxel::Span<int> Span30 = Span.subspan(3, 0);

  EXPECT_EQ(Span0.data(), Values);
  EXPECT_EQ(Span0e.data(), Values);
  EXPECT_EQ(Span00.data(), Values);
  EXPECT_EQ(Span01.data(), Values);
  EXPECT_EQ(Span02.data(), Values);
  EXPECT_EQ(Span03.data(), Values);

  EXPECT_EQ(Span1.data(), Values + 1);
  EXPECT_EQ(Span1e.data(), Values + 1);
  EXPECT_EQ(Span10.data(), Values + 1);
  EXPECT_EQ(Span11.data(), Values + 1);
  EXPECT_EQ(Span12.data(), Values + 1);

  EXPECT_EQ(Span2.data(), Values + 2);
  EXPECT_EQ(Span2e.data(), Values + 2);
  EXPECT_EQ(Span20.data(), Values + 2);
  EXPECT_EQ(Span21.data(), Values + 2);

  EXPECT_EQ(Span3.data(), Values + 3);
  EXPECT_EQ(Span3e.data(), Values + 3);
  EXPECT_EQ(Span30.data(), Values + 3);

  EXPECT_TRUE(Span00.empty());
  EXPECT_TRUE(Span10.empty());
  EXPECT_TRUE(Span20.empty());
  EXPECT_TRUE(Span30.empty());

  EXPECT_THAT(Span0, ::testing::ElementsAre(0, 1, 2));
  EXPECT_THAT(Span0e, ::testing::ElementsAre(0, 1, 2));
  EXPECT_THAT(Span01, ::testing::ElementsAre(0));
  EXPECT_THAT(Span02, ::testing::ElementsAre(0, 1));
  EXPECT_THAT(Span03, ::testing::ElementsAre(0, 1, 2));

  EXPECT_THAT(Span1, ::testing::ElementsAre(1, 2));
  EXPECT_THAT(Span1e, ::testing::ElementsAre(1, 2));
  EXPECT_THAT(Span11, ::testing::ElementsAre(1));
  EXPECT_THAT(Span12, ::testing::ElementsAre(1, 2));

  EXPECT_THAT(Span2, ::testing::ElementsAre(2));
  EXPECT_THAT(Span2e, ::testing::ElementsAre(2));
  EXPECT_THAT(Span21, ::testing::ElementsAre(2));

  EXPECT_TRUE(Span3.empty());
  EXPECT_TRUE(Span3e.empty());
}

TEST(Span, SubspanMethod_IllegalArguments) {
  int Values[] = {0, 1, 2};
  acxxel::Span<int> Span(Values);
  EXPECT_DEATH(Span.subspan(-1, 0), "terminate");
  EXPECT_DEATH(Span.subspan(0, -2), "terminate");
  EXPECT_DEATH(Span.subspan(0, 4), "terminate");
  EXPECT_DEATH(Span.subspan(1, 3), "terminate");
  EXPECT_DEATH(Span.subspan(2, 2), "terminate");
  EXPECT_DEATH(Span.subspan(3, 1), "terminate");
  EXPECT_DEATH(Span.subspan(4, 0), "terminate");
}

TEST(Span, ElementAccess) {
  int Values[] = {0, 1, 2};
  acxxel::Span<int> Span(Values);

  EXPECT_EQ(&Span[0], Values);
  EXPECT_EQ(&Span[1], Values + 1);
  EXPECT_EQ(&Span[2], Values + 2);
  EXPECT_EQ(&Span(0), Values);
  EXPECT_EQ(&Span(1), Values + 1);
  EXPECT_EQ(&Span(2), Values + 2);

  Span[0] = 5;
  EXPECT_EQ(Values[0], 5);

  Span(0) = 0;
  EXPECT_EQ(Values[0], 0);

  const int ConstValues[] = {0, 1, 2};
  acxxel::Span<const int> ConstSpan(ConstValues);

  EXPECT_EQ(&ConstSpan[0], ConstValues);
  EXPECT_EQ(&ConstSpan[1], ConstValues + 1);
  EXPECT_EQ(&ConstSpan[2], ConstValues + 2);
  EXPECT_EQ(&ConstSpan(0), ConstValues);
  EXPECT_EQ(&ConstSpan(1), ConstValues + 1);
  EXPECT_EQ(&ConstSpan(2), ConstValues + 2);
}

} // namespace
