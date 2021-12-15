//===- IntegerPolyhedron.cpp - Tests for IntegerPolyhedron class ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/IntegerPolyhedron.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {

using testing::ElementsAre;

/// Construct a IntegerPolyhedron from a set of inequality and
/// equality constraints.
static IntegerPolyhedron
makeSetFromConstraints(unsigned ids, ArrayRef<SmallVector<int64_t, 4>> ineqs,
                       ArrayRef<SmallVector<int64_t, 4>> eqs,
                       unsigned syms = 0) {
  IntegerPolyhedron set(ineqs.size(), eqs.size(), ids + 1, ids - syms, syms,
                        /*numLocals=*/0);
  for (const auto &eq : eqs)
    set.addEquality(eq);
  for (const auto &ineq : ineqs)
    set.addInequality(ineq);
  return set;
}

TEST(IntegerPolyhedronTest, removeInequality) {
  IntegerPolyhedron set =
      makeSetFromConstraints(1, {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}}, {});

  set.removeInequalityRange(0, 0);
  EXPECT_EQ(set.getNumInequalities(), 5u);

  set.removeInequalityRange(1, 3);
  EXPECT_EQ(set.getNumInequalities(), 3u);
  EXPECT_THAT(set.getInequality(0), ElementsAre(0, 0));
  EXPECT_THAT(set.getInequality(1), ElementsAre(3, 3));
  EXPECT_THAT(set.getInequality(2), ElementsAre(4, 4));

  set.removeInequality(1);
  EXPECT_EQ(set.getNumInequalities(), 2u);
  EXPECT_THAT(set.getInequality(0), ElementsAre(0, 0));
  EXPECT_THAT(set.getInequality(1), ElementsAre(4, 4));
}

TEST(IntegerPolyhedronTest, removeEquality) {
  IntegerPolyhedron set =
      makeSetFromConstraints(1, {}, {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}});

  set.removeEqualityRange(0, 0);
  EXPECT_EQ(set.getNumEqualities(), 5u);

  set.removeEqualityRange(1, 3);
  EXPECT_EQ(set.getNumEqualities(), 3u);
  EXPECT_THAT(set.getEquality(0), ElementsAre(0, 0));
  EXPECT_THAT(set.getEquality(1), ElementsAre(3, 3));
  EXPECT_THAT(set.getEquality(2), ElementsAre(4, 4));

  set.removeEquality(1);
  EXPECT_EQ(set.getNumEqualities(), 2u);
  EXPECT_THAT(set.getEquality(0), ElementsAre(0, 0));
  EXPECT_THAT(set.getEquality(1), ElementsAre(4, 4));
}

TEST(IntegerPolyhedronTest, clearConstraints) {
  IntegerPolyhedron set = makeSetFromConstraints(1, {}, {});

  set.addInequality({1, 0});
  EXPECT_EQ(set.atIneq(0, 0), 1);
  EXPECT_EQ(set.atIneq(0, 1), 0);

  set.clearConstraints();

  set.addInequality({1, 0});
  EXPECT_EQ(set.atIneq(0, 0), 1);
  EXPECT_EQ(set.atIneq(0, 1), 0);
}

TEST(IntegerPolyhedronTest, removeIdRange) {
  IntegerPolyhedron set(3, 2, 1);

  set.addInequality({10, 11, 12, 20, 21, 30, 40});
  set.removeId(IntegerPolyhedron::IdKind::Symbol, 1);
  EXPECT_THAT(set.getInequality(0),
              testing::ElementsAre(10, 11, 12, 20, 30, 40));

  set.removeIdRange(IntegerPolyhedron::IdKind::Dimension, 0, 2);
  EXPECT_THAT(set.getInequality(0), testing::ElementsAre(12, 20, 30, 40));

  set.removeIdRange(IntegerPolyhedron::IdKind::Local, 1, 1);
  EXPECT_THAT(set.getInequality(0), testing::ElementsAre(12, 20, 30, 40));

  set.removeIdRange(IntegerPolyhedron::IdKind::Local, 0, 1);
  EXPECT_THAT(set.getInequality(0), testing::ElementsAre(12, 20, 40));
}

} // namespace mlir
