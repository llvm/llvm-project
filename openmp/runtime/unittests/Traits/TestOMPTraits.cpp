//===- TestOMPTraits.cpp - Tests for OMP Trait classes -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kmp_traits.h"
#include "gtest/gtest.h"

using namespace kmp_traits;

namespace {

//===----------------------------------------------------------------------===//
// kmp_wildcard_trait Tests
//===----------------------------------------------------------------------===//

TEST(kmp_wildcard_trait_test, MatchesAnyDevice) {
  kmp_wildcard_trait *trait = new kmp_wildcard_trait();

  EXPECT_TRUE(trait->match(0));
  EXPECT_TRUE(trait->match(1));
  EXPECT_TRUE(trait->match(100));
  EXPECT_TRUE(trait->match(-1));

  delete trait;
}

TEST(kmp_wildcard_trait_test, Equality) {
  kmp_wildcard_trait *t1 = new kmp_wildcard_trait();
  kmp_wildcard_trait *t2 = new kmp_wildcard_trait();

  EXPECT_TRUE(*t1 == *t2);

  delete t1;
  delete t2;
}

//===----------------------------------------------------------------------===//
// kmp_literal_trait Tests
//===----------------------------------------------------------------------===//

TEST(kmp_literal_trait_test, MatchesExactDevice) {
  kmp_literal_trait *trait = new kmp_literal_trait(5);

  EXPECT_TRUE(trait->match(5));
  EXPECT_FALSE(trait->match(0));
  EXPECT_FALSE(trait->match(4));
  EXPECT_FALSE(trait->match(6));

  delete trait;
}

TEST(kmp_literal_trait_test, MatchesZero) {
  kmp_literal_trait *trait = new kmp_literal_trait(0);

  EXPECT_TRUE(trait->match(0));
  EXPECT_FALSE(trait->match(1));

  delete trait;
}

#ifndef NDEBUG
TEST(kmp_literal_trait_test, MatchesNegative) {
  EXPECT_DEATH(new kmp_literal_trait(-1), "Device number must be non-negative");
}
#endif

TEST(kmp_literal_trait_test, EqualitySameValue) {
  kmp_literal_trait *t1 = new kmp_literal_trait(42);
  kmp_literal_trait *t2 = new kmp_literal_trait(42);

  EXPECT_TRUE(*t1 == *t2);

  delete t1;
  delete t2;
}

TEST(kmp_literal_trait_test, EqualityDifferentValue) {
  kmp_literal_trait *t1 = new kmp_literal_trait(1);
  kmp_literal_trait *t2 = new kmp_literal_trait(2);

  EXPECT_FALSE(*t1 == *t2);

  delete t1;
  delete t2;
}

//===----------------------------------------------------------------------===//
// kmp_uid_trait Tests
//===----------------------------------------------------------------------===//

TEST(kmp_uid_trait_test, Construction) {
  kmp_uid_trait *trait = new kmp_uid_trait(kmp_str_ref("test-uid"));

  // Just verify it can be constructed without crashing
  delete trait;
}

TEST(kmp_uid_trait_test, MatchWithMock) {
  kmp_uid_trait *trait = new kmp_uid_trait(kmp_str_ref("device-0"));

  // Uses the mock omp_get_uid_from_device
  EXPECT_TRUE(trait->match(0)); // device-0 matches
  EXPECT_FALSE(trait->match(1)); // device-1 doesn't match
  EXPECT_FALSE(trait->match(2)); // device-2 doesn't match

  delete trait;
}

TEST(kmp_uid_trait_test, MatchWithCustomMock) {
  kmp_uid_trait *trait = new kmp_uid_trait(kmp_str_ref("custom-uid"));

  // Set a custom mock function
  trait->set_uid_from_device([](int device) -> const char * {
    return device == 2 ? "custom-uid" : "other";
  });

  EXPECT_FALSE(trait->match(0));
  EXPECT_FALSE(trait->match(1));
  EXPECT_TRUE(trait->match(2)); // custom-uid matches device 2
  EXPECT_FALSE(trait->match(3));

  delete trait;
}

TEST(kmp_uid_trait_test, EqualitySameUID) {
  kmp_uid_trait *t1 = new kmp_uid_trait(kmp_str_ref("my-device"));
  kmp_uid_trait *t2 = new kmp_uid_trait(kmp_str_ref("my-device"));

  EXPECT_TRUE(*t1 == *t2);

  delete t1;
  delete t2;
}

TEST(kmp_uid_trait_test, EqualityDifferentUID) {
  kmp_uid_trait *t1 = new kmp_uid_trait(kmp_str_ref("device-a"));
  kmp_uid_trait *t2 = new kmp_uid_trait(kmp_str_ref("device-b"));

  EXPECT_FALSE(*t1 == *t2);

  delete t1;
  delete t2;
}

//===----------------------------------------------------------------------===//
// kmp_trait_expr_single Tests
//===----------------------------------------------------------------------===//

TEST(kmp_trait_expr_single_test, CreateAndDestroy) {
  kmp_trait_expr_single *expr = new kmp_trait_expr_single();
  EXPECT_NE(expr, nullptr);
  delete expr;
}

TEST(kmp_trait_expr_single_test, CreateWithTrait) {
  kmp_trait_expr_single *expr =
      new kmp_trait_expr_single(new kmp_literal_trait(2));

  // Mock: 4 devices
  expr->set_num_devices([]() { return 4; });

  EXPECT_TRUE(expr->match(2));
  EXPECT_FALSE(expr->match(0));
  EXPECT_FALSE(expr->match(1));
  EXPECT_FALSE(expr->match(5)); // Out of range

  delete expr;
}

TEST(kmp_trait_expr_single_test, SetTrait) {
  kmp_trait_expr_single *expr = new kmp_trait_expr_single();
  expr->set_trait(new kmp_literal_trait(3));

  // Mock: 4 devices
  expr->set_num_devices([]() { return 4; });

  EXPECT_TRUE(expr->match(3));
  EXPECT_FALSE(expr->match(0));

  delete expr;
}

TEST(kmp_trait_expr_single_test, DefaultNotNegated) {
  kmp_trait_expr_single *expr = new kmp_trait_expr_single();

  EXPECT_FALSE(expr->is_negated());

  delete expr;
}

TEST(kmp_trait_expr_single_test, SetNegated) {
  kmp_trait_expr_single *expr = new kmp_trait_expr_single();

  expr->set_negated(true);
  EXPECT_TRUE(expr->is_negated());

  expr->set_negated(false);
  EXPECT_FALSE(expr->is_negated());

  delete expr;
}

TEST(kmp_trait_expr_single_test, MatchNegated) {
  kmp_trait_expr_single *expr =
      new kmp_trait_expr_single(new kmp_literal_trait(2));
  expr->set_negated(true);

  // Mock: 4 devices
  expr->set_num_devices([]() { return 4; });

  // Without negation: matches 2
  // With negation: matches everything in-range except 2
  EXPECT_FALSE(expr->match(2));
  EXPECT_TRUE(expr->match(0));
  EXPECT_TRUE(expr->match(1));
  EXPECT_TRUE(expr->match(3));
  // Out of range devices return false regardless of negation
  EXPECT_FALSE(expr->match(5));

  delete expr;
}

TEST(kmp_trait_expr_single_test, MatchWildcard) {
  kmp_trait_expr_single *expr =
      new kmp_trait_expr_single(new kmp_wildcard_trait());

  // Mock: 4 devices
  expr->set_num_devices([]() { return 4; });

  // Wildcard matches any in-range device
  EXPECT_TRUE(expr->match(0));
  EXPECT_TRUE(expr->match(3));
  // Out of range devices return false
  EXPECT_FALSE(expr->match(100));

  delete expr;
}

TEST(kmp_trait_expr_single_test, Equality) {
  kmp_trait_expr_single *e1 =
      new kmp_trait_expr_single(new kmp_literal_trait(1));
  kmp_trait_expr_single *e2 =
      new kmp_trait_expr_single(new kmp_literal_trait(1));

  EXPECT_TRUE(*e1 == *e2);

  delete e1;
  delete e2;
}

TEST(kmp_trait_expr_single_test, EqualityDifferentTrait) {
  kmp_trait_expr_single *e1 =
      new kmp_trait_expr_single(new kmp_literal_trait(1));
  kmp_trait_expr_single *e2 =
      new kmp_trait_expr_single(new kmp_literal_trait(2));

  EXPECT_FALSE(*e1 == *e2);

  delete e1;
  delete e2;
}

TEST(kmp_trait_expr_single_test, EqualityDifferentNegation) {
  kmp_trait_expr_single *e1 =
      new kmp_trait_expr_single(new kmp_literal_trait(1));
  kmp_trait_expr_single *e2 =
      new kmp_trait_expr_single(new kmp_literal_trait(1));
  e2->set_negated(true);

  EXPECT_FALSE(*e1 == *e2);

  delete e1;
  delete e2;
}

//===----------------------------------------------------------------------===//
// kmp_trait_expr_group Tests
//===----------------------------------------------------------------------===//

TEST(kmp_trait_expr_group_test, CreateAndDestroy) {
  kmp_trait_expr_group *group = new kmp_trait_expr_group();
  EXPECT_NE(group, nullptr);
  delete group;
}

TEST(kmp_trait_expr_group_test, DefaultTypeIsOR) {
  kmp_trait_expr_group *group = new kmp_trait_expr_group();

  EXPECT_EQ(group->get_group_type(), kmp_trait_expr_group::OR);

  delete group;
}

TEST(kmp_trait_expr_group_test, SetTypeAND) {
  kmp_trait_expr_group *group = new kmp_trait_expr_group();

  group->set_group_type(kmp_trait_expr_group::AND);
  EXPECT_EQ(group->get_group_type(), kmp_trait_expr_group::AND);

  delete group;
}

TEST(kmp_trait_expr_group_test, DefaultNotNegated) {
  kmp_trait_expr_group *group = new kmp_trait_expr_group();

  EXPECT_FALSE(group->is_negated());

  delete group;
}

TEST(kmp_trait_expr_group_test, SetNegated) {
  kmp_trait_expr_group *group = new kmp_trait_expr_group();

  group->set_negated(true);
  EXPECT_TRUE(group->is_negated());

  group->set_negated(false);
  EXPECT_FALSE(group->is_negated());

  delete group;
}

TEST(kmp_trait_expr_group_test, AddTraitDirectly) {
  kmp_trait_expr_group *group = new kmp_trait_expr_group();

  group->add_expr(new kmp_wildcard_trait());

  // Mock: 4 devices
  group->set_num_devices([]() { return 4; });

  // Wildcard matches any in-range device
  EXPECT_TRUE(group->match(0));
  EXPECT_TRUE(group->match(3));
  // Out of range devices return false
  EXPECT_FALSE(group->match(100));

  delete group;
}

TEST(kmp_trait_expr_group_test, AddExpr) {
  kmp_trait_expr_group *group = new kmp_trait_expr_group();

  group->add_expr(new kmp_trait_expr_single(new kmp_literal_trait(2)));

  // Mock: 4 devices
  group->set_num_devices([]() { return 4; });

  EXPECT_TRUE(group->match(2));
  EXPECT_FALSE(group->match(0));
  EXPECT_FALSE(group->match(5)); // Out of range

  delete group;
}

TEST(kmp_trait_expr_group_test, MatchORSemantics) {
  kmp_trait_expr_group *group = new kmp_trait_expr_group();
  group->set_group_type(kmp_trait_expr_group::OR);

  group->add_expr(new kmp_literal_trait(1));
  group->add_expr(new kmp_literal_trait(2));
  group->add_expr(new kmp_literal_trait(3));

  // Mock: 5 devices
  group->set_num_devices([]() { return 5; });

  // OR: matches if ANY trait matches
  EXPECT_TRUE(group->match(1));
  EXPECT_TRUE(group->match(2));
  EXPECT_TRUE(group->match(3));
  EXPECT_FALSE(group->match(0));
  EXPECT_FALSE(group->match(4));

  delete group;
}

TEST(kmp_trait_expr_group_test, MatchANDSemantics) {
  kmp_trait_expr_group *group = new kmp_trait_expr_group();
  group->set_group_type(kmp_trait_expr_group::AND);

  // For AND to pass, ALL traits must match the same device
  // A single literal only matches one device
  group->add_expr(new kmp_literal_trait(2));

  // Mock: 4 devices
  group->set_num_devices([]() { return 4; });

  EXPECT_TRUE(group->match(2));
  EXPECT_FALSE(group->match(0));
  // Out of range
  EXPECT_FALSE(group->match(5));

  delete group;
}

TEST(kmp_trait_expr_group_test, MatchANDWithWildcard) {
  kmp_trait_expr_group *group = new kmp_trait_expr_group();
  group->set_group_type(kmp_trait_expr_group::AND);

  group->add_expr(new kmp_wildcard_trait());
  group->add_expr(new kmp_literal_trait(2));

  // Mock: 4 devices
  group->set_num_devices([]() { return 4; });

  // Wildcard matches all, literal matches 2
  // AND: both must match
  EXPECT_TRUE(group->match(2));
  EXPECT_FALSE(group->match(0));
  // Out of range
  EXPECT_FALSE(group->match(5));

  delete group;
}

TEST(kmp_trait_expr_group_test, MatchNegated) {
  kmp_trait_expr_group *group = new kmp_trait_expr_group();

  group->add_expr(new kmp_literal_trait(2));
  group->set_negated(true);

  // Mock: 4 devices
  group->set_num_devices([]() { return 4; });

  // Without negation: matches 2
  // With negation: matches everything in-range except 2
  EXPECT_FALSE(group->match(2));
  EXPECT_TRUE(group->match(0));
  EXPECT_TRUE(group->match(1));
  EXPECT_TRUE(group->match(3));
  // Out of range devices return false regardless of negation
  EXPECT_FALSE(group->match(5));

  delete group;
}

TEST(kmp_trait_expr_group_test, MatchEmptyGroupOR) {
  kmp_trait_expr_group *group = new kmp_trait_expr_group();
  group->set_group_type(kmp_trait_expr_group::OR);

  // Mock: 4 devices
  group->set_num_devices([]() { return 4; });

  // Empty OR: no traits match, so result is false
  EXPECT_FALSE(group->match(0));
  EXPECT_FALSE(group->match(1));

  delete group;
}

TEST(kmp_trait_expr_group_test, MatchEmptyGroupAND) {
  kmp_trait_expr_group *group = new kmp_trait_expr_group();
  group->set_group_type(kmp_trait_expr_group::AND);

  // Mock: 4 devices
  group->set_num_devices([]() { return 4; });

  // Empty AND: vacuously true (0 out of 0 traits match)
  EXPECT_TRUE(group->match(0));
  EXPECT_TRUE(group->match(1));

  delete group;
}

TEST(kmp_trait_expr_group_test, Equality) {
  kmp_trait_expr_group *g1 = new kmp_trait_expr_group();
  kmp_trait_expr_group *g2 = new kmp_trait_expr_group();

  g1->add_expr(new kmp_literal_trait(1));
  g2->add_expr(new kmp_literal_trait(1));

  EXPECT_TRUE(*g1 == *g2);

  delete g1;
  delete g2;
}

TEST(kmp_trait_expr_group_test, EqualityDifferentNegation) {
  kmp_trait_expr_group *g1 = new kmp_trait_expr_group();
  kmp_trait_expr_group *g2 = new kmp_trait_expr_group();

  g1->add_expr(new kmp_literal_trait(1));
  g2->add_expr(new kmp_literal_trait(1));
  g2->set_negated(true);

  EXPECT_FALSE(*g1 == *g2);

  delete g1;
  delete g2;
}

TEST(kmp_trait_expr_group_test, NestedGroups) {
  kmp_trait_expr_group *outer = new kmp_trait_expr_group();
  outer->set_group_type(kmp_trait_expr_group::OR);

  kmp_trait_expr_group *inner = new kmp_trait_expr_group();
  inner->set_group_type(kmp_trait_expr_group::AND);
  inner->add_expr(new kmp_literal_trait(1));
  inner->add_expr(new kmp_wildcard_trait());

  outer->add_expr(inner);
  outer->add_expr(new kmp_literal_trait(2));

  // Mock: 4 devices
  outer->set_num_devices([]() { return 4; });

  // Inner matches device 1 (literal 1 AND wildcard)
  // Outer matches 1 OR 2
  EXPECT_TRUE(outer->match(1));
  EXPECT_TRUE(outer->match(2));
  EXPECT_FALSE(outer->match(0));
  EXPECT_FALSE(outer->match(3));

  delete outer;
}

//===----------------------------------------------------------------------===//
// kmp_trait_clause Tests
//===----------------------------------------------------------------------===//

TEST(kmp_trait_clause_test, CreateAndDestroy) {
  kmp_trait_clause *clause = new kmp_trait_clause();
  EXPECT_NE(clause, nullptr);
  delete clause;
}

TEST(kmp_trait_clause_test, SetExprWithTrait) {
  kmp_trait_clause *clause = new kmp_trait_clause();
  clause->set_expr(new kmp_literal_trait(2));

  // The trait is wrapped in kmp_trait_expr_single internally
  kmp_trait_expr *expr = clause->get_expr();
  EXPECT_NE(expr, nullptr);

  delete clause;
}

TEST(kmp_trait_clause_test, SetExprWithExpr) {
  kmp_trait_clause *clause = new kmp_trait_clause();
  kmp_trait_expr_group *group = new kmp_trait_expr_group();
  group->add_expr(new kmp_literal_trait(1));
  clause->set_expr(group);

  EXPECT_EQ(clause->get_expr(), group);

  delete clause;
}

TEST(kmp_trait_clause_test, Equality) {
  kmp_trait_clause *c1 = new kmp_trait_clause();
  kmp_trait_clause *c2 = new kmp_trait_clause();

  c1->set_expr(new kmp_literal_trait(1));
  c2->set_expr(new kmp_literal_trait(1));

  EXPECT_TRUE(*c1 == *c2);

  delete c1;
  delete c2;
}

TEST(kmp_trait_clause_test, EqualityDifferentExprs) {
  kmp_trait_clause *c1 = new kmp_trait_clause();
  kmp_trait_clause *c2 = new kmp_trait_clause();

  c1->set_expr(new kmp_literal_trait(1));
  c2->set_expr(new kmp_literal_trait(2));

  EXPECT_FALSE(*c1 == *c2);

  delete c1;
  delete c2;
}

//===----------------------------------------------------------------------===//
// kmp_trait_context Tests
//===----------------------------------------------------------------------===//

TEST(kmp_trait_context_test, CreateAndDestroy) {
  kmp_trait_context *context = new kmp_trait_context();
  EXPECT_NE(context, nullptr);
  delete context;
}

TEST(kmp_trait_context_test, AddClause) {
  kmp_trait_context *context = new kmp_trait_context();
  kmp_trait_clause *clause = new kmp_trait_clause();
  clause->set_expr(new kmp_literal_trait(2));
  context->add_clause(clause);

  // Mock: 4 devices
  context->set_num_devices([]() { return 4; });

  EXPECT_TRUE(context->match(2));
  EXPECT_FALSE(context->match(0));
  // Out of range
  EXPECT_FALSE(context->match(5));

  delete context;
}

TEST(kmp_trait_context_test, MultipleClauses) {
  kmp_trait_context *context = new kmp_trait_context();

  kmp_trait_clause *c1 = new kmp_trait_clause();
  c1->set_expr(new kmp_literal_trait(1));
  context->add_clause(c1);

  kmp_trait_clause *c2 = new kmp_trait_clause();
  c2->set_expr(new kmp_literal_trait(2));
  context->add_clause(c2);

  kmp_trait_clause *c3 = new kmp_trait_clause();
  c3->set_expr(new kmp_literal_trait(3));
  context->add_clause(c3);

  // Mock: 5 devices
  context->set_num_devices([]() { return 5; });

  // Context uses OR semantics between clauses
  EXPECT_TRUE(context->match(1));
  EXPECT_TRUE(context->match(2));
  EXPECT_TRUE(context->match(3));
  EXPECT_FALSE(context->match(0));
  EXPECT_FALSE(context->match(4));

  delete context;
}

TEST(kmp_trait_context_test, EmptyContextMatchesNothing) {
  kmp_trait_context *context = new kmp_trait_context();

  // Mock: 4 devices
  context->set_num_devices([]() { return 4; });

  EXPECT_FALSE(context->match(0));
  EXPECT_FALSE(context->match(1));

  delete context;
}

TEST(kmp_trait_context_test, WildcardClause) {
  kmp_trait_context *context = new kmp_trait_context();
  kmp_trait_clause *clause = new kmp_trait_clause();
  clause->set_expr(new kmp_wildcard_trait());
  context->add_clause(clause);

  // Mock: 4 devices
  context->set_num_devices([]() { return 4; });

  // In-range devices match
  EXPECT_TRUE(context->match(0));
  EXPECT_TRUE(context->match(3));
  // Out of range devices return false
  EXPECT_FALSE(context->match(100));
  EXPECT_FALSE(context->match(-1));

  delete context;
}

TEST(kmp_trait_context_test, EvaluateWithMock) {
  kmp_trait_context *context = new kmp_trait_context();

  // Mock: 5 devices
  context->set_num_devices([]() { return 5; });

  kmp_trait_clause *c1 = new kmp_trait_clause();
  c1->set_expr(new kmp_literal_trait(1));
  context->add_clause(c1);

  kmp_trait_clause *c2 = new kmp_trait_clause();
  c2->set_expr(new kmp_literal_trait(3));
  context->add_clause(c2);

  kmp_vector<int> result = context->evaluate();
  EXPECT_EQ(result.size(), 2u);
  EXPECT_TRUE(result.contains(1));
  EXPECT_TRUE(result.contains(3));
  EXPECT_FALSE(result.contains(0));
  EXPECT_FALSE(result.contains(2));
  EXPECT_FALSE(result.contains(4));

  delete context;
}

TEST(kmp_trait_context_test, Equality) {
  kmp_trait_context *ctx1 = new kmp_trait_context();
  kmp_trait_context *ctx2 = new kmp_trait_context();

  kmp_trait_clause *c1 = new kmp_trait_clause();
  c1->set_expr(new kmp_literal_trait(1));
  ctx1->add_clause(c1);

  kmp_trait_clause *c2 = new kmp_trait_clause();
  c2->set_expr(new kmp_literal_trait(1));
  ctx2->add_clause(c2);

  EXPECT_TRUE(*ctx1 == *ctx2);

  delete ctx1;
  delete ctx2;
}

TEST(kmp_trait_context_test, EqualityDifferentClauses) {
  kmp_trait_context *ctx1 = new kmp_trait_context();
  kmp_trait_context *ctx2 = new kmp_trait_context();

  kmp_trait_clause *c1 = new kmp_trait_clause();
  c1->set_expr(new kmp_literal_trait(1));
  ctx1->add_clause(c1);

  kmp_trait_clause *c2 = new kmp_trait_clause();
  c2->set_expr(new kmp_literal_trait(2));
  ctx2->add_clause(c2);

  EXPECT_FALSE(*ctx1 == *ctx2);

  delete ctx1;
  delete ctx2;
}

//===----------------------------------------------------------------------===//
// kmp_trait_context Iterator Tests
//===----------------------------------------------------------------------===//

TEST(kmp_trait_context_test, IteratorRangeBasedFor) {
  kmp_trait_context *context = new kmp_trait_context();

  // Mock: 5 devices
  context->set_num_devices([]() { return 5; });

  kmp_trait_clause *c1 = new kmp_trait_clause();
  c1->set_expr(new kmp_literal_trait(1));
  context->add_clause(c1);

  kmp_trait_clause *c2 = new kmp_trait_clause();
  c2->set_expr(new kmp_literal_trait(3));
  context->add_clause(c2);

  // Use range-based for loop (should auto-evaluate)
  kmp_vector<int> collected;
  for (int d : *context) {
    collected.push_back(d);
  }

  EXPECT_EQ(collected.size(), 2u);
  EXPECT_TRUE(collected.contains(1));
  EXPECT_TRUE(collected.contains(3));

  delete context;
}

TEST(kmp_trait_context_test, IteratorAutoEvaluates) {
  kmp_trait_context *context = new kmp_trait_context();

  // Mock: 4 devices
  context->set_num_devices([]() { return 4; });

  kmp_trait_clause *clause = new kmp_trait_clause();
  clause->set_expr(new kmp_wildcard_trait());
  context->add_clause(clause);

  // Directly use begin()/end() without calling evaluate() first
  int count = 0;
  for (const int *it = context->begin(); it != context->end(); ++it) {
    EXPECT_GE(*it, 0);
    EXPECT_LT(*it, 4);
    count++;
  }

  EXPECT_EQ(count, 4);

  delete context;
}

TEST(kmp_trait_context_test, IteratorEmptyContext) {
  kmp_trait_context *context = new kmp_trait_context();

  // Mock: 4 devices
  context->set_num_devices([]() { return 4; });

  // Empty context - no clauses added
  int count = 0;
  for (int d : *context) {
    (void)d;
    count++;
  }

  EXPECT_EQ(count, 0);
  EXPECT_EQ(context->begin(), context->end());

  delete context;
}

TEST(kmp_trait_context_test, IteratorBeginEnd) {
  kmp_trait_context *context = new kmp_trait_context();

  // Mock: 3 devices
  context->set_num_devices([]() { return 3; });

  kmp_trait_clause *clause = new kmp_trait_clause();
  clause->set_expr(new kmp_literal_trait(2));
  context->add_clause(clause);

  // Test begin/end directly
  const int *b = context->begin();
  const int *e = context->end();

  EXPECT_EQ(e - b, 1); // Should have exactly 1 element
  EXPECT_EQ(*b, 2);

  delete context;
}

TEST(kmp_trait_context_test, IteratorMultipleDevices) {
  kmp_trait_context *context = new kmp_trait_context();

  // Add clauses for devices 0, 2, 4
  for (int i = 0; i < 6; i += 2) {
    kmp_trait_clause *clause = new kmp_trait_clause();
    clause->set_expr(new kmp_literal_trait(i));
    context->add_clause(clause);
  }

  // Mock: 6 devices (must be set after adding clauses to propagate to them)
  context->set_num_devices([]() { return 6; });

  // Collect via iterator
  kmp_vector<int> collected;
  for (int d : *context) {
    collected.push_back(d);
  }

  EXPECT_EQ(collected.size(), 3u);
  EXPECT_TRUE(collected.contains(0));
  EXPECT_TRUE(collected.contains(2));
  EXPECT_TRUE(collected.contains(4));

  delete context;
}

TEST(kmp_trait_context_test, IteratorConsistentWithEvaluate) {
  kmp_trait_context *context = new kmp_trait_context();

  // Mock: 5 devices
  context->set_num_devices([]() { return 5; });

  kmp_trait_clause *c1 = new kmp_trait_clause();
  c1->set_expr(new kmp_literal_trait(1));
  context->add_clause(c1);

  kmp_trait_clause *c2 = new kmp_trait_clause();
  c2->set_expr(new kmp_literal_trait(4));
  context->add_clause(c2);

  // Get result via evaluate()
  const kmp_vector<int> &eval_result = context->evaluate();

  // Collect via iterator
  kmp_vector<int> iter_result;
  for (int d : *context) {
    iter_result.push_back(d);
  }

  // Both should give the same results
  EXPECT_EQ(eval_result.size(), iter_result.size());
  for (size_t i = 0; i < eval_result.size(); i++) {
    EXPECT_EQ(eval_result[i], iter_result[i]);
  }

  delete context;
}

TEST(kmp_trait_context_test, EvaluateReturnsByReference) {
  kmp_trait_context *context = new kmp_trait_context();

  // Mock: 3 devices
  context->set_num_devices([]() { return 3; });

  kmp_trait_clause *clause = new kmp_trait_clause();
  clause->set_expr(new kmp_wildcard_trait());
  context->add_clause(clause);

  // Multiple calls to evaluate() should return reference to the same data
  const kmp_vector<int> &result1 = context->evaluate();
  const kmp_vector<int> &result2 = context->evaluate();

  EXPECT_EQ(&result1, &result2);

  delete context;
}

//===----------------------------------------------------------------------===//
// get_num_devices Propagation Tests
//===----------------------------------------------------------------------===//

TEST(kmp_trait_context_test, PropagationToClausesAddedAfterSetNumDevices) {
  kmp_trait_context *context = new kmp_trait_context();

  // Set mock BEFORE adding clauses - propagation should still work
  context->set_num_devices([]() { return 6; });

  // Add clauses for devices 0, 2, 4 (all require 6 devices to be in range)
  for (int i = 0; i < 6; i += 2) {
    kmp_trait_clause *clause = new kmp_trait_clause();
    clause->set_expr(new kmp_literal_trait(i));
    context->add_clause(clause);
  }

  // All three devices should match because propagation worked
  kmp_vector<int> collected;
  for (int d : *context) {
    collected.push_back(d);
  }

  EXPECT_EQ(collected.size(), 3u);
  EXPECT_TRUE(collected.contains(0));
  EXPECT_TRUE(collected.contains(2));
  EXPECT_TRUE(collected.contains(4));

  delete context;
}

TEST(kmp_trait_context_test, PropagationToClausesAddedBeforeSetNumDevices) {
  kmp_trait_context *context = new kmp_trait_context();

  // Add clauses BEFORE setting mock
  for (int i = 0; i < 6; i += 2) {
    kmp_trait_clause *clause = new kmp_trait_clause();
    clause->set_expr(new kmp_literal_trait(i));
    context->add_clause(clause);
  }

  // Set mock AFTER adding clauses
  context->set_num_devices([]() { return 6; });

  // All three devices should match
  kmp_vector<int> collected;
  for (int d : *context) {
    collected.push_back(d);
  }

  EXPECT_EQ(collected.size(), 3u);
  EXPECT_TRUE(collected.contains(0));
  EXPECT_TRUE(collected.contains(2));
  EXPECT_TRUE(collected.contains(4));

  delete context;
}

TEST(kmp_trait_expr_group_test, PropagationToExprsAddedAfterSetNumDevices) {
  kmp_trait_expr_group *group = new kmp_trait_expr_group();

  // Set mock BEFORE adding expressions
  group->set_num_devices([]() { return 8; });

  // Add expressions for devices 5, 6, 7 (require 8 devices)
  group->add_expr(new kmp_literal_trait(5));
  group->add_expr(new kmp_literal_trait(6));
  group->add_expr(new kmp_literal_trait(7));

  // All should match
  EXPECT_TRUE(group->match(5));
  EXPECT_TRUE(group->match(6));
  EXPECT_TRUE(group->match(7));
  EXPECT_FALSE(group->match(4));

  delete group;
}

TEST(kmp_trait_expr_group_test, PropagationToExprsAddedBeforeSetNumDevices) {
  kmp_trait_expr_group *group = new kmp_trait_expr_group();

  // Add expressions BEFORE setting mock
  group->add_expr(new kmp_literal_trait(5));
  group->add_expr(new kmp_literal_trait(6));
  group->add_expr(new kmp_literal_trait(7));

  // Set mock AFTER adding expressions
  group->set_num_devices([]() { return 8; });

  // All should match
  EXPECT_TRUE(group->match(5));
  EXPECT_TRUE(group->match(6));
  EXPECT_TRUE(group->match(7));
  EXPECT_FALSE(group->match(4));

  delete group;
}

TEST(kmp_trait_expr_group_test, PropagationToNestedGroups) {
  kmp_trait_expr_group *outer = new kmp_trait_expr_group();
  outer->set_group_type(kmp_trait_expr_group::OR);

  // Set mock on outer group FIRST
  outer->set_num_devices([]() { return 10; });

  // Create inner group and add high-numbered devices
  kmp_trait_expr_group *inner = new kmp_trait_expr_group();
  inner->set_group_type(kmp_trait_expr_group::OR);
  inner->add_expr(new kmp_literal_trait(8));
  inner->add_expr(new kmp_literal_trait(9));

  // Add inner to outer - should propagate mock to inner and its children
  outer->add_expr(inner);

  // Add another expression directly to outer
  outer->add_expr(new kmp_literal_trait(7));

  // All should match with 10 devices
  EXPECT_TRUE(outer->match(7));
  EXPECT_TRUE(outer->match(8));
  EXPECT_TRUE(outer->match(9));
  EXPECT_FALSE(outer->match(10)); // Out of range

  delete outer;
}

TEST(kmp_trait_expr_group_test, PropagationToDeeplyNestedGroups) {
  // Create a deeply nested structure: outer -> middle -> inner
  kmp_trait_expr_group *outer = new kmp_trait_expr_group();
  outer->set_group_type(kmp_trait_expr_group::OR);

  // Set mock on outer
  outer->set_num_devices([]() { return 12; });

  kmp_trait_expr_group *middle = new kmp_trait_expr_group();
  middle->set_group_type(kmp_trait_expr_group::OR);

  kmp_trait_expr_group *inner = new kmp_trait_expr_group();
  inner->set_group_type(kmp_trait_expr_group::OR);
  inner->add_expr(new kmp_literal_trait(10));
  inner->add_expr(new kmp_literal_trait(11));

  middle->add_expr(inner);
  middle->add_expr(new kmp_literal_trait(9));

  outer->add_expr(middle);
  outer->add_expr(new kmp_literal_trait(8));

  // All devices 8-11 should match (requires 12 devices)
  EXPECT_TRUE(outer->match(8));
  EXPECT_TRUE(outer->match(9));
  EXPECT_TRUE(outer->match(10));
  EXPECT_TRUE(outer->match(11));
  EXPECT_FALSE(outer->match(12)); // Out of range

  delete outer;
}

TEST(kmp_trait_context_test, PropagationToNestedGroupsInClauses) {
  kmp_trait_context *context = new kmp_trait_context();

  // Set mock on context FIRST
  context->set_num_devices([]() { return 10; });

  // Create a group with nested structure
  kmp_trait_expr_group *group = new kmp_trait_expr_group();
  group->set_group_type(kmp_trait_expr_group::OR);

  kmp_trait_expr_group *inner = new kmp_trait_expr_group();
  inner->set_group_type(kmp_trait_expr_group::OR);
  inner->add_expr(new kmp_literal_trait(8));
  inner->add_expr(new kmp_literal_trait(9));

  group->add_expr(inner);
  group->add_expr(new kmp_literal_trait(7));

  // Create clause with the group
  kmp_trait_clause *clause = new kmp_trait_clause();
  clause->set_expr(group);

  // Add clause to context - should propagate to group and inner
  context->add_clause(clause);

  // All should match
  kmp_vector<int> collected;
  for (int d : *context) {
    collected.push_back(d);
  }

  EXPECT_EQ(collected.size(), 3u);
  EXPECT_TRUE(collected.contains(7));
  EXPECT_TRUE(collected.contains(8));
  EXPECT_TRUE(collected.contains(9));

  delete context;
}

TEST(kmp_trait_context_test, PropagationMixedOrder) {
  // Test a complex scenario with mixed ordering
  kmp_trait_context *context = new kmp_trait_context();

  // Add first clause before set_num_devices
  kmp_trait_clause *c1 = new kmp_trait_clause();
  c1->set_expr(new kmp_literal_trait(5));
  context->add_clause(c1);

  // Set mock
  context->set_num_devices([]() { return 8; });

  // Add second clause after set_num_devices
  kmp_trait_clause *c2 = new kmp_trait_clause();
  c2->set_expr(new kmp_literal_trait(6));
  context->add_clause(c2);

  // Add third clause with a group
  kmp_trait_expr_group *group = new kmp_trait_expr_group();
  group->add_expr(new kmp_literal_trait(7));

  kmp_trait_clause *c3 = new kmp_trait_clause();
  c3->set_expr(group);
  context->add_clause(c3);

  // All three should match
  kmp_vector<int> collected;
  for (int d : *context) {
    collected.push_back(d);
  }

  EXPECT_EQ(collected.size(), 3u);
  EXPECT_TRUE(collected.contains(5));
  EXPECT_TRUE(collected.contains(6));
  EXPECT_TRUE(collected.contains(7));

  delete context;
}

} // namespace
