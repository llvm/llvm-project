//===- TestOMPTraitParser.cpp - Tests for OMP Trait Parser ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kmp_traits.h"
#include "gtest/gtest.h"

namespace {

//===----------------------------------------------------------------------===//
// Helper to parse and auto-cleanup
//===----------------------------------------------------------------------===//

class ParserTest : public ::testing::Test {
protected:
  kmp_trait_context *context = nullptr;

  void parse(const char *spec, const char *dbg_name = nullptr) {
    context = kmp_trait_context::parse_from_spec(kmp_str_ref(spec), dbg_name);
  }

  void TearDown() override {
    if (context) {
      delete context;
      context = nullptr;
    }
  }
};

template <bool expected_result>
static void check_result_single(kmp_trait_context *context,
                                const kmp_vector<int> &result,
                                int expected_device_num) {
  EXPECT_EQ(context->match(expected_device_num), expected_result);
  EXPECT_EQ(result.contains(expected_device_num), expected_result);
}

template <bool expected_result, int... device_nums>
static void check_result(kmp_trait_context *context,
                         const kmp_vector<int> &result) {
  (check_result_single<expected_result>(context, result, device_nums), ...);
}

template <bool expected_result, int... device_nums>
static void check_result(kmp_trait_context *context) {
  const kmp_vector<int> &result = context->evaluate();
  check_result<expected_result, device_nums...>(context, result);
}

//===----------------------------------------------------------------------===//
// Literal Device Numbers
//===----------------------------------------------------------------------===//

TEST_F(ParserTest, SingleLiteral) {
  parse("5");

  ASSERT_NE(context, nullptr);
  // Device 5 is out of range (mock has 4 devices: 0-3), so match returns false

  EXPECT_EQ(context->evaluate().size(), 0u);
  check_result<false, 5, 0, 4, 6>(context);
}

TEST_F(ParserTest, ZeroLiteral) {
  parse("0");

  ASSERT_NE(context, nullptr);
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 1u);
  check_result<true, 0>(context, result);
  check_result<false, 1>(context, result);
}

TEST_F(ParserTest, MultipleLiterals) {
  parse("1,2,3");

  ASSERT_NE(context, nullptr);
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 3u);
  check_result<true, 1, 2, 3>(context, result);
  check_result<false, 0, 4>(context, result);
}

TEST_F(ParserTest, LiteralsWithSpaces) {
  parse("1, 2, 3");

  ASSERT_NE(context, nullptr);
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 3u);
  check_result<true, 1, 2, 3>(context, result);
  check_result<false, 0, 4>(context, result);
}

TEST_F(ParserTest, LiteralsWithLeadingSpaces) {
  parse("  1,  2,  3");

  ASSERT_NE(context, nullptr);
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 3u);
  check_result<true, 1, 2, 3>(context, result);
  check_result<false, 0, 4>(context, result);
}

TEST_F(ParserTest, LargeLiteral) {
  parse("12345");

  ASSERT_NE(context, nullptr);
  // Device 12345 is out of range, so match returns false
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 0u);
  check_result<false, 12345, 0>(context, result);
}

//===----------------------------------------------------------------------===//
// Wildcard
//===----------------------------------------------------------------------===//

TEST_F(ParserTest, Wildcard) {
  parse("*");

  ASSERT_NE(context, nullptr);
  // Wildcard matches all 4 mock devices
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 4u);
  check_result<true, 0, 1, 2, 3>(context, result);
  check_result<false, 100>(context, result);
}

TEST_F(ParserTest, WildcardWithLiterals) {
  parse("1, *, 3");

  ASSERT_NE(context, nullptr);
  // Wildcard makes all in-range devices match
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 4u);
  check_result<true, 0, 1, 2, 3>(context, result);
  check_result<false, 100>(context, result);
}

//===----------------------------------------------------------------------===//
// UID Traits
//===----------------------------------------------------------------------===//

TEST_F(ParserTest, UIDTrait) {
  parse("uid(device-0)");

  ASSERT_NE(context, nullptr);
  // Uses mock: device-0 is at index 0
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 1u);
  check_result<true, 0>(context, result);
  check_result<false, 1>(context, result);
}

TEST_F(ParserTest, UIDTraitWithUnderscore) {
  parse("uid(my_device_123)");

  ASSERT_NE(context, nullptr);
  // This UID doesn't match any mock device
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 0u);
  check_result<false, 0, 1>(context, result);
}

TEST_F(ParserTest, UIDTraitWithDash) {
  parse("uid(device-2)");

  ASSERT_NE(context, nullptr);
  // Uses mock: device-2 is at index 2
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 1u);
  check_result<true, 2>(context, result);
  check_result<false, 0, 1, 3>(context, result);
}

TEST_F(ParserTest, MultipleUIDTraits) {
  parse("uid(device-1), uid(device-3)");

  ASSERT_NE(context, nullptr);
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 2u);
  check_result<true, 1, 3>(context, result);
  check_result<false, 0, 2>(context, result);
}

TEST_F(ParserTest, MixedLiteralsAndUIDs) {
  parse("0, uid(device-2), 1, uid(device-3)");

  ASSERT_NE(context, nullptr);
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 4u);
  check_result<true, 0, 1, 2, 3>(context, result);
}

//===----------------------------------------------------------------------===//
// Negation
//===----------------------------------------------------------------------===//

TEST_F(ParserTest, NegatedUID) {
  parse("!uid(device-0)");

  ASSERT_NE(context, nullptr);
  // Negated: everything except device-0 matches
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 3u);
  check_result<false, 0>(context, result);
  check_result<true, 1, 2, 3>(context, result);
}

//===----------------------------------------------------------------------===//
// Grouping with Parentheses
//===----------------------------------------------------------------------===//

TEST_F(ParserTest, SimpleGroup) {
  parse("(uid(device-1))");

  ASSERT_NE(context, nullptr);
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 1u);
  check_result<false, 0>(context, result);
  check_result<true, 1>(context, result);
}

TEST_F(ParserTest, GroupWithOR) {
  parse("(uid(device-0) || uid(device-2))");

  ASSERT_NE(context, nullptr);
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 2u);
  check_result<true, 0, 2>(context, result);
  check_result<false, 1, 3>(context, result);
}

TEST_F(ParserTest, GroupWithAND) {
  parse("(uid(device-0) && uid(device-0))");

  ASSERT_NE(context, nullptr);
  // Both refer to same device, so AND passes for device 0
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 1u);
  check_result<true, 0>(context, result);
  check_result<false, 1>(context, result);
}

TEST_F(ParserTest, NegatedGroup) {
  parse("!(uid(device-0) || uid(device-1))");

  ASSERT_NE(context, nullptr);
  // Negated: matches devices NOT in {0, 1}
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 2u);
  check_result<false, 0, 1>(context, result);
  check_result<true, 2, 3>(context, result);
}

//===----------------------------------------------------------------------===//
// Complex Expressions
//===----------------------------------------------------------------------===//

TEST_F(ParserTest, ComplexMixed) {
  parse("0, 1, uid(device-2), *");

  ASSERT_NE(context, nullptr);
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 4u);
  check_result<true, 0, 1, 2, 3>(context, result);
  check_result<false, 100>(context, result);
}

TEST_F(ParserTest, MultipleORGroups) {
  parse("(uid(device-0) || uid(device-1)), (uid(device-2) || uid(device-3))");

  ASSERT_NE(context, nullptr);
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 4u);
  check_result<true, 0, 1, 2, 3>(context, result);
}

//===----------------------------------------------------------------------===//
// Complex Boolean Operators
//===----------------------------------------------------------------------===//

TEST_F(ParserTest, ThreeWayOR) {
  // Three UIDs combined with OR
  parse("(uid(device-0) || uid(device-1) || uid(device-2))");

  ASSERT_NE(context, nullptr);
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 3u);
  check_result<true, 0, 1, 2>(context, result);
  check_result<false, 3>(context, result);
}

TEST_F(ParserTest, FourWayOR) {
  // All four mock devices via OR
  parse("(uid(device-0) || uid(device-1) || uid(device-2) || uid(device-3))");

  ASSERT_NE(context, nullptr);
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 4u);
  check_result<true, 0, 1, 2, 3>(context, result);
}

TEST_F(ParserTest, ThreeWayAND) {
  // Three identical UIDs with AND - all must match same device
  parse("(uid(device-1) && uid(device-1) && uid(device-1))");

  ASSERT_NE(context, nullptr);
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 1u);
  check_result<true, 1>(context, result);
  check_result<false, 0, 2, 3>(context, result);
}

TEST_F(ParserTest, ANDWithDifferentUIDs) {
  // AND with different UIDs - can never match (device can't have two UIDs)
  parse("(uid(device-0) && uid(device-1))");

  ASSERT_NE(context, nullptr);
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 0u);
  check_result<false, 0, 1, 2, 3>(context, result);
}

TEST_F(ParserTest, NegatedThreeWayOR) {
  // Negate a group of three UIDs - matches devices NOT in {0, 1, 2}
  parse("!(uid(device-0) || uid(device-1) || uid(device-2))");

  ASSERT_NE(context, nullptr);
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 1u);
  check_result<false, 0, 1, 2>(context, result);
  check_result<true, 3>(context, result);
}

TEST_F(ParserTest, NegatedAND) {
  // Negate an AND group - since AND never matches, negation matches all
  parse("!(uid(device-0) && uid(device-1))");

  ASSERT_NE(context, nullptr);
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 4u);
  check_result<true, 0, 1, 2, 3>(context, result);
}

TEST_F(ParserTest, NegatedANDWithSameUID) {
  // Negate an AND that matches device-0 - matches everything except 0
  parse("!(uid(device-0) && uid(device-0))");

  ASSERT_NE(context, nullptr);
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 3u);
  check_result<false, 0>(context, result);
  check_result<true, 1, 2, 3>(context, result);
}

TEST_F(ParserTest, NestedParensWithOR) {
  // Nested parentheses around OR
  parse("((uid(device-0) || uid(device-1)))");

  ASSERT_NE(context, nullptr);
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 2u);
  check_result<true, 0, 1>(context, result);
  check_result<false, 2, 3>(context, result);
}

TEST_F(ParserTest, NestedParensWithAND) {
  // Nested parentheses around AND
  parse("((uid(device-2) && uid(device-2)))");

  ASSERT_NE(context, nullptr);
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 1u);
  check_result<true, 2>(context, result);
  check_result<false, 0, 1, 3>(context, result);
}

TEST_F(ParserTest, DoubleNegation) {
  // Double negation: !!uid(device-0) should match device-0
  parse("!(!uid(device-0))");

  ASSERT_NE(context, nullptr);
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 1u);
  check_result<true, 0>(context, result);
  check_result<false, 1, 2, 3>(context, result);
}

TEST_F(ParserTest, NegatedNestedOR) {
  // Negate nested OR group
  parse("!((uid(device-0) || uid(device-1)))");

  ASSERT_NE(context, nullptr);
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 2u);
  check_result<false, 0, 1>(context, result);
  check_result<true, 2, 3>(context, result);
}

TEST_F(ParserTest, MultipleNegatedExprs) {
  // Multiple negated clauses - OR semantics between clauses
  parse("!uid(device-0), !uid(device-1)");

  ASSERT_NE(context, nullptr);
  // First clause matches 1,2,3; Second clause matches 0,2,3
  // OR between clauses: union = all devices
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 4u);
  check_result<true, 0, 1, 2, 3>(context, result);
}

TEST_F(ParserTest, MixedNegatedAndNonNegated) {
  // Mix negated and non-negated clauses
  parse("uid(device-0), !uid(device-0)");

  ASSERT_NE(context, nullptr);
  // First matches 0, second matches 1,2,3 -> union = all
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 4u);
  check_result<true, 0, 1, 2, 3>(context, result);
}

TEST_F(ParserTest, ComplexORGroupsInSeparateExprs) {
  // Complex OR groups as separate clauses
  parse("(uid(device-0) || uid(device-1)), (uid(device-2) || uid(device-3))");

  ASSERT_NE(context, nullptr);
  // First matches 0,1; Second matches 2,3 -> union = all
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 4u);
  check_result<true, 0, 1, 2, 3>(context, result);
}

TEST_F(ParserTest, NegatedORGroupWithLiteral) {
  // Negated OR group combined with literal in separate clauses
  parse("!(uid(device-0) || uid(device-1)), 0");

  ASSERT_NE(context, nullptr);
  // First matches 2,3; Second matches 0 -> union = 0,2,3
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 3u);
  check_result<true, 0, 2, 3>(context, result);
  check_result<false, 1>(context, result);
}

TEST_F(ParserTest, DeeplyNestedWithOperators) {
  // Deeply nested with operators
  parse("(((uid(device-0) || uid(device-1))))");

  ASSERT_NE(context, nullptr);
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 2u);
  check_result<true, 0, 1>(context, result);
  check_result<false, 2, 3>(context, result);
}

TEST_F(ParserTest, ORWithSpacesAroundOperators) {
  // OR with lots of whitespace
  parse("(  uid(device-0)   ||   uid(device-2)   ||   uid(device-3)  )");

  ASSERT_NE(context, nullptr);
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 3u);
  check_result<true, 0, 2, 3>(context, result);
  check_result<false, 1>(context, result);
}

TEST_F(ParserTest, ANDWithSpacesAroundOperators) {
  // AND with lots of whitespace
  parse("(  uid(device-1)   &&   uid(device-1)  )");

  ASSERT_NE(context, nullptr);
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 1u);
  check_result<true, 1>(context, result);
  check_result<false, 0, 2, 3>(context, result);
}

//===----------------------------------------------------------------------===//
// Mixed && and || (in separate clauses/groups)
//===----------------------------------------------------------------------===//

TEST_F(ParserTest, ORExprAndANDExpr) {
  // OR group in first clause, AND group in second clause
  parse("(uid(device-0) || uid(device-1)), (uid(device-2) && uid(device-2))");

  ASSERT_NE(context, nullptr);
  // First clause matches 0,1; Second clause matches 2 -> union = 0,1,2
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 3u);
  check_result<true, 0, 1, 2>(context, result);
  check_result<false, 3>(context, result);
}

TEST_F(ParserTest, ANDExprAndORExpr) {
  // AND group first, OR group second
  parse("(uid(device-0) && uid(device-0)), (uid(device-2) || uid(device-3))");

  ASSERT_NE(context, nullptr);
  // First clause matches 0; Second clause matches 2,3 -> union = 0,2,3
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 3u);
  check_result<true, 0, 2, 3>(context, result);
  check_result<false, 1>(context, result);
}

TEST_F(ParserTest, MultipleANDAndORExprs) {
  // Multiple clauses alternating between AND and OR
  parse("(uid(device-0) && uid(device-0)), (uid(device-1) || uid(device-2)), "
        "(uid(device-3) && uid(device-3))");

  ASSERT_NE(context, nullptr);
  // Expr 1 matches 0; Expr 2 matches 1,2; Expr 3 matches 3 -> all
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 4u);
  check_result<true, 0, 1, 2, 3>(context, result);
}

TEST_F(ParserTest, NegatedORWithAND) {
  // Negated OR clause combined with AND clause
  parse("!(uid(device-0) || uid(device-1)), (uid(device-0) && uid(device-0))");

  ASSERT_NE(context, nullptr);
  // First clause matches 2,3; Second clause matches 0 -> union = 0,2,3
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 3u);
  check_result<true, 0, 2, 3>(context, result);
  check_result<false, 1>(context, result);
}

TEST_F(ParserTest, NegatedANDWithOR) {
  // Negated AND clause combined with OR clause
  parse("!(uid(device-0) && uid(device-0)), (uid(device-0) || uid(device-1))");

  ASSERT_NE(context, nullptr);
  // First clause matches 1,2,3; Second clause matches 0,1 -> all
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 4u);
  check_result<true, 0, 1, 2, 3>(context, result);
}

TEST_F(ParserTest, ComplexMixedOperators) {
  // Complex mix: OR, AND, negated OR, literal
  parse("(uid(device-0) || uid(device-1)), (uid(device-2) && uid(device-2)), "
        "!(uid(device-0) || uid(device-1) || uid(device-2)), 0");

  ASSERT_NE(context, nullptr);
  // Expr 1: 0,1; Expr 2: 2; Expr 3: NOT(0,1,2) = 3; Expr 4: 0
  // Union = all
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 4u);
  check_result<true, 0, 1, 2, 3>(context, result);
}

TEST_F(ParserTest, ANDNeverMatchesWithOR) {
  // AND that never matches combined with OR that does
  parse("(uid(device-0) && uid(device-1)), (uid(device-2) || uid(device-3))");

  ASSERT_NE(context, nullptr);
  // First clause: never matches (different UIDs); Second: 2,3
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 2u);
  check_result<false, 0, 1>(context, result);
  check_result<true, 2, 3>(context, result);
}

TEST_F(ParserTest, ORNeverMatchesWithAND) {
  // OR with non-existent UIDs combined with AND that matches
  parse("(uid(nonexistent-a) || uid(nonexistent-b)), (uid(device-0) && "
        "uid(device-0))");

  ASSERT_NE(context, nullptr);
  // First clause: no match; Second: 0
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 1u);
  check_result<true, 0>(context, result);
  check_result<false, 1, 2, 3>(context, result);
}

TEST_F(ParserTest, ThreeWayORAndThreeWayAND) {
  // Three-way OR and three-way AND in separate clauses
  parse("(uid(device-0) || uid(device-1) || uid(device-2)), (uid(device-3) && "
        "uid(device-3) && uid(device-3))");

  ASSERT_NE(context, nullptr);
  // First: 0,1,2; Second: 3 -> all
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 4u);
  check_result<true, 0, 1, 2, 3>(context, result);
}

TEST_F(ParserTest, NegatedMixedExprs) {
  // Both clauses negated with different operators
  parse("!(uid(device-0) || uid(device-1)), !(uid(device-2) && uid(device-2))");

  ASSERT_NE(context, nullptr);
  // First: NOT(0,1) = 2,3; Second: NOT(2) = 0,1,3
  // Union = all
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 4u);
  check_result<true, 0, 1, 2, 3>(context, result);
}

TEST_F(ParserTest, LiteralsWithMixedOperatorExprs) {
  // Literals combined with both OR and AND clauses
  parse("0, (uid(device-1) || uid(device-2)), 3, (uid(device-0) && "
        "uid(device-0))");

  ASSERT_NE(context, nullptr);
  // Literals: 0,3; OR clause: 1,2; AND clause: 0
  // Union = all
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 4u);
  check_result<true, 0, 1, 2, 3>(context, result);
}

//===----------------------------------------------------------------------===//
// Nested Mixed Operators (|| and && at different nesting levels)
//===----------------------------------------------------------------------===//

TEST_F(ParserTest, ORContainingANDGroup) {
  // Outer OR with inner AND group: (A || (B && C))
  // For (B && C) to match, both B and C must match same device
  parse("(uid(device-0) || (uid(device-1) && uid(device-1)))");

  ASSERT_NE(context, nullptr);
  // device-0 matches via first operand
  // device-1 matches via (device-1 && device-1)
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 2u);
  check_result<true, 0, 1>(context, result);
  check_result<false, 2, 3>(context, result);
}

TEST_F(ParserTest, ANDContainingORGroup) {
  // Outer AND with inner OR group: (A && (B || C))
  // Both the trait A and the group (B || C) must match
  // Since A is uid(device-0), only device-0 can satisfy A
  // (B || C) must also match device-0 for AND to succeed
  parse("(uid(device-0) && (uid(device-0) || uid(device-1)))");

  ASSERT_NE(context, nullptr);
  // device-0: uid(device-0) matches AND (uid(device-0) || uid(device-1))
  // matches -> true device-1: uid(device-0) doesn't match -> false
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 1u);
  check_result<true, 0>(context, result);
  check_result<false, 1, 2, 3>(context, result);
}

TEST_F(ParserTest, ORWithTwoANDGroups) {
  // ((A && B) || (C && D)) - OR of two AND groups
  parse(
      "((uid(device-0) && uid(device-0)) || (uid(device-2) && uid(device-2)))");

  ASSERT_NE(context, nullptr);
  // First AND matches device-0; Second AND matches device-2
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 2u);
  check_result<true, 0, 2>(context, result);
  check_result<false, 1, 3>(context, result);
}

TEST_F(ParserTest, ANDWithTwoORGroups) {
  // ((A || B) && (C || D)) - AND of two OR groups
  // For a device to match: must match (A || B) AND must match (C || D)
  parse(
      "((uid(device-0) || uid(device-1)) && (uid(device-0) || uid(device-2)))");

  ASSERT_NE(context, nullptr);
  // device-0: matches (0||1) AND matches (0||2) -> true
  // device-1: matches (0||1) but NOT (0||2) -> false
  // device-2: NOT (0||1) -> false
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 1u);
  check_result<true, 0>(context, result);
  check_result<false, 1, 2, 3>(context, result);
}

TEST_F(ParserTest, ORWithNestedANDContainingOR) {
  // (A || (B && (C || D))) - three levels of nesting
  parse(
      "(uid(device-3) || (uid(device-0) && (uid(device-0) || uid(device-1))))");

  ASSERT_NE(context, nullptr);
  // device-0: inner (0||1) matches, uid(device-0) matches -> AND matches; OR
  // satisfied device-1: inner (0||1) matches, but uid(device-0) doesn't -> AND
  // fails; outer uid(device-3) fails device-3: outer uid(device-3) matches
  // directly
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 2u);
  check_result<true, 0, 3>(context, result);
  check_result<false, 1, 2>(context, result);
}

TEST_F(ParserTest, ANDWithNestedORContainingAND) {
  // (A && (B || (C && D))) - three levels of nesting
  parse(
      "(uid(device-0) && (uid(device-0) || (uid(device-1) && uid(device-1))))");

  ASSERT_NE(context, nullptr);
  // device-0: uid(device-0) matches; inner (uid(device-0) || ...) matches ->
  // AND satisfied device-1: uid(device-0) doesn't match -> AND fails
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 1u);
  check_result<true, 0>(context, result);
  check_result<false, 1, 2, 3>(context, result);
}

TEST_F(ParserTest, NegatedORContainingAND) {
  // !(A || (B && C)) - negated complex expression
  parse("!(uid(device-0) || (uid(device-1) && uid(device-1)))");

  ASSERT_NE(context, nullptr);
  // Without negation: matches 0, 1
  // With negation: matches 2, 3
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 2u);
  check_result<false, 0, 1>(context, result);
  check_result<true, 2, 3>(context, result);
}

TEST_F(ParserTest, NegatedANDContainingOR) {
  // !(A && (B || C)) - negated complex expression
  parse("!(uid(device-0) && (uid(device-0) || uid(device-1)))");

  ASSERT_NE(context, nullptr);
  // Without negation: matches only 0
  // With negation: matches 1, 2, 3
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 3u);
  check_result<false, 0>(context, result);
  check_result<true, 1, 2, 3>(context, result);
}

TEST_F(ParserTest, ComplexNestedWithAllDevices) {
  // ((A || B) && (C || D)) where union covers all but AND restricts
  parse(
      "((uid(device-0) || uid(device-1)) && (uid(device-1) || uid(device-2)))");

  ASSERT_NE(context, nullptr);
  // device-0: (0||1)=true, (1||2)=false -> AND=false
  // device-1: (0||1)=true, (1||2)=true -> AND=true
  // device-2: (0||1)=false -> AND=false
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 1u);
  check_result<true, 1>(context, result);
  check_result<false, 0, 2, 3>(context, result);
}

TEST_F(ParserTest, TripleNestedMixedOperators) {
  // (((A || B) && C) || D) - deeply nested with alternating operators
  parse(
      "(((uid(device-0) || uid(device-1)) && uid(device-0)) || uid(device-3))");

  ASSERT_NE(context, nullptr);
  // Inner (0||1): matches 0, 1
  // Middle ((0||1) && 0): matches only 0
  // Outer (... || 3): matches 0, 3
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 2u);
  check_result<true, 0, 3>(context, result);
  check_result<false, 1, 2>(context, result);
}

TEST_F(ParserTest, ANDChainWithNestedOR) {
  // (A && (B || C) && D) - wait, this mixes operators at same level
  // Actually: ((A && (B || C)) is valid - let's do that
  // Let's do: (uid(device-0) && (uid(device-0) || uid(device-1)) &&
  // uid(device-0)) This is three-way AND where middle operand is an OR group
  parse("(uid(device-0) && (uid(device-0) || uid(device-1)) && uid(device-0))");

  ASSERT_NE(context, nullptr);
  // All three must match: uid(device-0), (0||1), uid(device-0)
  // Only device-0 satisfies all
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 1u);
  check_result<true, 0>(context, result);
  check_result<false, 1, 2, 3>(context, result);
}

TEST_F(ParserTest, ORChainWithNestedAND) {
  // (A || (B && C) || D) - three-way OR where middle is AND group
  parse("(uid(device-0) || (uid(device-1) && uid(device-1)) || uid(device-3))");

  ASSERT_NE(context, nullptr);
  // Any of: device-0, (device-1 && device-1), device-3
  // Matches: 0, 1, 3
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 3u);
  check_result<true, 0, 1, 3>(context, result);
  check_result<false, 2>(context, result);
}

TEST_F(ParserTest, NestedMixedWithSpaces) {
  // Nested mixed operators with lots of whitespace
  parse("(  uid(device-0)  ||  ( uid(device-1)  &&  uid(device-1) )  ||  "
        "uid(device-2)  )");

  ASSERT_NE(context, nullptr);
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 3u);
  check_result<true, 0, 1, 2>(context, result);
  check_result<false, 3>(context, result);
}

//===----------------------------------------------------------------------===//
// Empty Input
//===----------------------------------------------------------------------===//

TEST_F(ParserTest, EmptyString) {
  parse("");

  ASSERT_NE(context, nullptr);
  // Empty context matches nothing
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 0u);
  check_result<false, 0, 1>(context, result);
}

TEST_F(ParserTest, OnlyWhitespace) {
  parse("   ");

  ASSERT_NE(context, nullptr);
  kmp_vector<int> result = context->evaluate();

  EXPECT_EQ(result.size(), 0u);
  check_result<false, 0>(context, result);
}

//===----------------------------------------------------------------------===//
// Error Cases
//===----------------------------------------------------------------------===//

TEST_F(ParserTest, OnlyComma) {
  ASSERT_DEATH(
      parse(",", "test_only_comma"),
      "OMP: Error #[0-9]+: trait parser while parsing test_only_comma: "
      "failed to parse trait specification \\(,\\)");
}

TEST_F(ParserTest, OnlyCommaNullDbgName) {
  ASSERT_DEATH(parse(","),
               "OMP: Error #[0-9]+: trait parser while parsing \\(null\\): "
               "failed to parse trait specification \\(,\\)");
}

TEST_F(ParserTest, MixedAndOrSameLevel) {
  // OpenMP 6.0 explicitly excludes "&&" and "||" from appearing in the same
  // grouping level.
  ASSERT_DEATH(
      parse("uid(a) && uid(b) || uid(c)", "mixed_and_or_same_level"),
      "OMP: Error #[0-9]+: trait parser while parsing mixed_and_or_same_level: "
      "failed to parse trait specification "
      "\\(\\|\\| uid\\(c\\)\\)");
}

TEST_F(ParserTest, MixedOrAndSameLevel) {
  ASSERT_DEATH(
      parse("uid(a) || uid(b) && uid(c)", "mixed_or_and_same_level"),
      "OMP: Error #[0-9]+: trait parser while parsing mixed_or_and_same_level: "
      "failed to parse trait specification "
      "\\(&& uid\\(c\\)\\)");
}

TEST_F(ParserTest, InvalidUID) {
  // Empty UID is not allowed
  ASSERT_DEATH(parse("uid()", "invalid_uid"),
               "OMP: Error #[0-9]+: trait parser while parsing invalid_uid: "
               "invalid uid \\(\\)");
}

TEST_F(ParserTest, UnclosedParenthesis) {
  ASSERT_DEATH(
      parse("(uid(a)", "unclosed_parenthesis"),
      "OMP: Error #[0-9]+: trait parser while parsing unclosed_parenthesis: "
      "failed to parse trait specification \\(\\(uid\\(a\\)\\)");
}

TEST_F(ParserTest, UnmatchedClosingParenthesis) {
  ASSERT_DEATH(parse("uid(a))", "unmatched_closing_parenthesis"),
               "OMP: Error #[0-9]+: trait parser while parsing "
               "unmatched_closing_parenthesis: "
               "failed to parse trait specification \\(\\)\\)");
}

TEST_F(ParserTest, EmptyParentheses) {
  ASSERT_DEATH(
      parse("()", "empty_parentheses"),
      "OMP: Error #[0-9]+: trait parser while parsing empty_parentheses: "
      "failed to parse trait specification \\(\\(\\)\\)");
}

TEST_F(ParserTest, TrailingOperator) {
  ASSERT_DEATH(
      parse("uid(a) &&", "trailing_operator"),
      "OMP: Error #[0-9]+: trait parser while parsing trailing_operator: "
      "failed to parse trait specification \\(uid\\(a\\) &&\\)");
}

TEST_F(ParserTest, LeadingOperator) {
  ASSERT_DEATH(
      parse("&& uid(a)", "leading_operator"),
      "OMP: Error #[0-9]+: trait parser while parsing leading_operator: "
      "failed to parse trait specification \\(&& uid\\(a\\)\\)");
}

TEST_F(ParserTest, DoubleComma) {
  ASSERT_DEATH(parse("uid(a),,uid(b)", "double_comma"),
               "OMP: Error #[0-9]+: trait parser while parsing double_comma: "
               "failed to parse trait specification \\(,,uid\\(b\\)\\)");
}

//===----------------------------------------------------------------------===//
// parse_single_device Tests
//===----------------------------------------------------------------------===//

TEST(ParseSingleDeviceTest, ValidSingleDigit) {
  int result = kmp_trait_context::parse_single_device(kmp_str_ref("5"), 10);
  EXPECT_EQ(result, 5);
}

TEST(ParseSingleDeviceTest, ValidMultiDigit) {
  int result = kmp_trait_context::parse_single_device(kmp_str_ref("123"), 200);
  EXPECT_EQ(result, 123);
}

TEST(ParseSingleDeviceTest, Zero) {
  int result = kmp_trait_context::parse_single_device(kmp_str_ref("0"), 10);
  EXPECT_EQ(result, 0);
}

TEST(ParseSingleDeviceTest, AtLimit) {
  int result = kmp_trait_context::parse_single_device(kmp_str_ref("10"), 10);
  EXPECT_EQ(result, 10);
}

TEST(ParseSingleDeviceTest, AboveLimit) {
  ASSERT_DEATH(kmp_trait_context::parse_single_device(kmp_str_ref("11"), 10,
                                                      "above_limit"),
               "OMP: Error #[0-9]+: trait parser while parsing above_limit: "
               "value 11 above limit \\(10\\)");
}

TEST(ParseSingleDeviceTest, NonInteger) {
  ASSERT_DEATH(kmp_trait_context::parse_single_device(kmp_str_ref("abc"), 10,
                                                      "non_integer"),
               "OMP: Error #[0-9]+: trait parser while parsing non_integer: "
               "failed to parse trait specification \\(abc\\)");
}

TEST(ParseSingleDeviceTest, EmptyString) {
  ASSERT_DEATH(
      kmp_trait_context::parse_single_device(kmp_str_ref(""), 10, "empty"),
      "OMP: Error #[0-9]+: trait parser while parsing empty: "
      "failed to parse trait specification \\(\\)");
}

TEST(ParseSingleDeviceTest, LeadingSpaces) {
  // consume_integer skips leading spaces
  int result = kmp_trait_context::parse_single_device(kmp_str_ref("  7"), 10);
  EXPECT_EQ(result, 7);
}

TEST(ParseSingleDeviceTest, LargeNumber) {
  int result =
      kmp_trait_context::parse_single_device(kmp_str_ref("999999"), 1000000);
  EXPECT_EQ(result, 999999);
}

} // namespace
