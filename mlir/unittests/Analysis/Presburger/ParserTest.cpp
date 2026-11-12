//===- PresbugerParserTest.cpp - Presburger parsing unit tests --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains tests for parsing IntegerSets to IntegerPolyhedron.
// The tests with invalid input check that the parser only accepts well-formed
// IntegerSets. The tests with well-formed input compare the returned FACs to
// manually constructed FACs with a PresburgerSet equality check.
//
//===----------------------------------------------------------------------===//

#include "Parser.h"

#include <gtest/gtest.h>

using namespace mlir;
using namespace presburger;

/// Construct a IntegerPolyhedron from a set of inequality, equality, and
/// division constraints.
static IntegerPolyhedron makeFACFromConstraints(
    unsigned dims, unsigned syms, ArrayRef<SmallVector<int64_t, 4>> ineqs,
    ArrayRef<SmallVector<int64_t, 4>> eqs = {},
    ArrayRef<std::pair<SmallVector<int64_t, 4>, int64_t>> divs = {}) {
  IntegerPolyhedron fac(ineqs.size(), eqs.size(), dims + syms + 1,
                        PresburgerSpace::getSetSpace(dims, syms, 0));
  for (const auto &div : divs)
    fac.addLocalFloorDiv(div.first, div.second);
  for (const auto &eq : eqs)
    fac.addEquality(eq);
  for (const auto &ineq : ineqs)
    fac.addInequality(ineq);
  return fac;
}

/// Parses and compares the `str` to the `ex`. The equality check is performed
/// by using PresburgerSet::isEqual
static bool parseAndCompare(StringRef str, const IntegerPolyhedron &ex) {
  IntegerPolyhedron poly = parseIntegerPolyhedron(str);
  return PresburgerSet(poly).isEqual(PresburgerSet(ex));
}

TEST(ParseFACTest, ParseAndCompareTest) {
  // constant-fold addition
  EXPECT_TRUE(parseAndCompare("() : (4 + 3 >= 0)",
                              makeFACFromConstraints(0, 0, {}, {})));

  // constant-fold addition + multiplication
  EXPECT_TRUE(parseAndCompare("()[a] : (4 * 3 == 10 + 2)",
                              makeFACFromConstraints(0, 1, {}, {})));

  // constant-fold ceildiv + floordiv
  EXPECT_TRUE(parseAndCompare("(x) : (11 ceildiv 3 == 13 floordiv 3)",
                              makeFACFromConstraints(1, 0, {}, {})));

  // simple ineq
  EXPECT_TRUE(parseAndCompare("(x)[] : (x >= 0)",
                              makeFACFromConstraints(1, 0, {{1, 0}})));

  // simple eq
  EXPECT_TRUE(parseAndCompare("(x)[] : (x == 0)",
                              makeFACFromConstraints(1, 0, {}, {{1, 0}})));

  // multiple constraints
  EXPECT_TRUE(parseAndCompare("(x)[] : (7 * x >= 0, -7 * x + 5 >= 0)",
                              makeFACFromConstraints(1, 0, {{7, 0}, {-7, 5}})));

  // multiplication distribution
  EXPECT_TRUE(
      parseAndCompare("(x) : (2 * x >= 2, (-7 + x * 9) * 5 >= 0)",
                      makeFACFromConstraints(1, 0, {{2, -2}, {45, -35}})));

  // multiple dimensions
  EXPECT_TRUE(parseAndCompare("(x,y,z)[] : (x + y - z >= 0)",
                              makeFACFromConstraints(3, 0, {{1, 1, -1, 0}})));

  // dimensions and symbols
  EXPECT_TRUE(
      parseAndCompare("(x,y,z)[a,b] : (x + y - z + 2 * a - 15 * b >= 0)",
                      makeFACFromConstraints(3, 2, {{1, 1, -1, 2, -15, 0}})));

  // only symbols
  EXPECT_TRUE(parseAndCompare("()[a] : (2 * a - 4 == 0)",
                              makeFACFromConstraints(0, 1, {}, {{2, -4}})));

  // no linear terms
  EXPECT_TRUE(parseAndCompare(
      "(x, y) : (26 * (x floordiv 6) == y floordiv 3)",
      makeFACFromConstraints(2, 0, {}, {{0, 0, 26, -1, 0}},
                             {{{1, 0, 0}, 6}, {{0, 1, 0, 0}, 3}})));

  // simple floordiv
  EXPECT_TRUE(parseAndCompare(
      "(x, y) : (y - 3 * ((x + y - 13) floordiv 3) - 42 == 0)",
      makeFACFromConstraints(2, 0, {}, {{0, 1, -3, -42}}, {{{1, 1, -13}, 3}})));

  // simple ceildiv
  EXPECT_TRUE(parseAndCompare(
      "(x, y) : (y - 3 * ((x + y - 13) ceildiv 3) - 42 == 0)",
      makeFACFromConstraints(2, 0, {}, {{0, 1, -3, -42}}, {{{1, 1, -11}, 3}})));

  // simple mod
  EXPECT_TRUE(parseAndCompare(
      "(x, y) : (y - 3 * ((x + y - 13) mod 3) - 42 == 0)",
      makeFACFromConstraints(2, 0, {}, {{-3, -2, 9, -3}}, {{{1, 1, -13}, 3}})));

  // multiple floordiv
  EXPECT_TRUE(parseAndCompare(
      "(x, y) : (y - x floordiv 3 - y floordiv 2 == 0)",
      makeFACFromConstraints(2, 0, {}, {{0, 1, -1, -1, 0}},
                             {{{1, 0, 0}, 3}, {{0, 1, 0, 0}, 2}})));

  // multiple ceildiv
  EXPECT_TRUE(parseAndCompare(
      "(x, y) : (y - x ceildiv 3 - y ceildiv 2 == 0)",
      makeFACFromConstraints(2, 0, {}, {{0, 1, -1, -1, 0}},
                             {{{1, 0, 2}, 3}, {{0, 1, 0, 1}, 2}})));

  // multiple mod
  EXPECT_TRUE(parseAndCompare(
      "(x, y) : (y - x mod 3 - y mod 2 == 0)",
      makeFACFromConstraints(2, 0, {}, {{-1, 0, 3, 2, 0}},
                             {{{1, 0, 0}, 3}, {{0, 1, 0, 0}, 2}})));

  // nested floordiv
  EXPECT_TRUE(parseAndCompare(
      "(x, y) : (y - (x + y floordiv 2) floordiv 3 == 0)",
      makeFACFromConstraints(2, 0, {}, {{0, 1, 0, -1, 0}},
                             {{{0, 1, 0}, 2}, {{1, 0, 1, 0}, 3}})));

  // nested mod
  EXPECT_TRUE(parseAndCompare(
      "(x, y) : (y - (x + y mod 2) mod 3 == 0)",
      makeFACFromConstraints(2, 0, {}, {{-1, 0, 2, 3, 0}},
                             {{{0, 1, 0}, 2}, {{1, 1, -2, 0}, 3}})));

  // nested floordiv + ceildiv + mod
  EXPECT_TRUE(parseAndCompare(
      "(x, y) : ((2 * x + 3 * (y floordiv 2) + x mod 7 + 1) ceildiv 3 == 42)",
      makeFACFromConstraints(
          2, 0, {}, {{0, 0, 0, 0, 1, -42}},
          {{{0, 1, 0}, 2}, {{1, 0, 0, 0}, 7}, {{3, 0, 3, -7, 3}, 3}})));
}
