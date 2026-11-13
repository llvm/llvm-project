//===- ValueBoundsOpInterfaceTest.cpp - Value bounds tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"
#include <tuple>

using namespace mlir;

namespace {
using Cmp = ValueBoundsConstraintSet::ComparisonOperator;
using Variable = ValueBoundsConstraintSet::Variable;

// Exercise the constraint-set API directly on function arguments. No custom
// ValueBoundsOpInterface models are needed to supply facts or register rules.
class TestConstraintSet : public ValueBoundsConstraintSet {
public:
  explicit TestConstraintSet(MLIRContext *ctx, bool conservative = false)
      : ValueBoundsConstraintSet(
            ctx,
            [](Value, std::optional<int64_t>, ValueBoundsConstraintSet &) {
              return false;
            },
            {}, conservative) {}

  using ValueBoundsConstraintSet::getExpr;
  using ValueBoundsConstraintSet::processWorklist;

  auto getCounts() const {
    return std::make_tuple(cstr.getNumVars(), cstr.getNumEqualities(),
                           cstr.getNumInequalities());
  }
  size_t getNumPendingRules() const { return pendingConditionalBounds.size(); }
};

class ConditionalValueBoundsTest : public testing::Test {
protected:
  ConditionalValueBoundsTest() {
    ctx.loadDialect<func::FuncDialect>();
    module = parseSourceString<ModuleOp>(R"mlir(
      func.func @test(%a: index, %b: index, %result: index) {
        return
      }
    )mlir",
                                         &ctx);
    assert(module);
    auto func = *module->getOps<func::FuncOp>().begin();
    a = func.getArgument(0);
    b = func.getArgument(1);
    result = func.getArgument(2);
  }

  OpFoldResult constant(int64_t value) {
    return Builder(&ctx).getIndexAttr(value);
  }

  MLIRContext ctx;
  OwningOpRef<ModuleOp> module;
  Value a, b, result;
};

TEST_F(ConditionalValueBoundsTest, SimpleGuardsAreMaterializedOnce) {
  TestConstraintSet cstr(&ctx);
  (void)cstr.getExpr(a);
  (void)cstr.getExpr(b);
  (void)cstr.getExpr(result);
  auto counts = cstr.getCounts();
  cstr.bound(result).when({{a, Cmp::GE, constant(0)},
                           {constant(0), Cmp::LE, b},
                           {a, Cmp::LE, b},
                           {constant(2), Cmp::LT, a},
                           {b, Cmp::LE, constant(4)},
                           {constant(1), Cmp::LT, constant(2)}}) >= 5;
  EXPECT_NE(cstr.getCounts(), counts);
  counts = cstr.getCounts();
  cstr.processWorklist();
  EXPECT_EQ(cstr.getCounts(), counts);
  EXPECT_EQ(cstr.getNumPendingRules(), 1u);
  cstr.bound(a) == 3;
  cstr.bound(b) == 4;
  EXPECT_TRUE(cstr.populateAndCompare(result, Cmp::GE, constant(5)));
  EXPECT_EQ(cstr.getNumPendingRules(), 0u);
}

TEST_F(ConditionalValueBoundsTest, FalseConstantGuardIsDiscarded) {
  TestConstraintSet cstr(&ctx);
  (void)cstr.getExpr(result);
  cstr.bound(result).when({{constant(0), Cmp::GT, constant(1)}}) >= 5;
  auto counts = cstr.getCounts();
  cstr.processWorklist();
  EXPECT_EQ(cstr.getCounts(), counts);
  EXPECT_EQ(cstr.getNumPendingRules(), 0u);
  EXPECT_FALSE(cstr.populateAndCompare(result, Cmp::GE, constant(5)));
}

TEST_F(ConditionalValueBoundsTest, ShapedGuardsAreMaterializedOnce) {
  auto shapedModule = parseSourceString<ModuleOp>(R"mlir(
    func.func @shaped(%tensor: tensor<?x?xf32>, %result: index) { return }
  )mlir",
                                                  &ctx);
  ASSERT_TRUE(shapedModule);
  auto func = *shapedModule->getOps<func::FuncOp>().begin();
  Value tensor = func.getArgument(0);
  Value target = func.getArgument(1);
  TestConstraintSet cstr(&ctx);
  (void)cstr.getExpr(tensor, 0);
  (void)cstr.getExpr(tensor, 1);
  (void)cstr.getExpr(target);
  auto counts = cstr.getCounts();
  cstr.bound(target).when({{{tensor, 0}, Cmp::LE, {tensor, 1}}}) >= 5;
  EXPECT_NE(cstr.getCounts(), counts);
  counts = cstr.getCounts();
  cstr.processWorklist();
  EXPECT_EQ(cstr.getCounts(), counts);
  cstr.bound(tensor)[0] == 3;
  cstr.bound(tensor)[1] == 4;
  EXPECT_TRUE(cstr.populateAndCompare(target, Cmp::GE, constant(5)));
}

TEST_F(ConditionalValueBoundsTest, ConjunctionWaitsForEveryCondition) {
  TestConstraintSet cstr(&ctx);
  (void)cstr.getExpr(result);
  cstr.bound(result).when(
      {{a, Cmp::GE, constant(0)}, {b, Cmp::GE, constant(0)}}) >= 5;
  cstr.bound(a) >= 0;
  EXPECT_FALSE(cstr.populateAndCompare(result, Cmp::GE, constant(5)));
  EXPECT_EQ(cstr.getNumPendingRules(), 1u);

  cstr.bound(b) >= 0;
  EXPECT_TRUE(cstr.populateAndCompare(result, Cmp::GE, constant(5)));
  EXPECT_EQ(cstr.getNumPendingRules(), 0u);
}

TEST_F(ConditionalValueBoundsTest, FalseGuardAfterUnknownRetiresConjunction) {
  TestConstraintSet cstr(&ctx);
  (void)cstr.getExpr(result);
  auto conditional = cstr.bound(result).when(
      {{a, Cmp::GE, constant(0)}, {b, Cmp::GE, constant(0)}});
  conditional >= 5;
  cstr.bound(b) <= -1;
  auto counts = cstr.getCounts();
  cstr.processWorklist();
  EXPECT_EQ(cstr.getNumPendingRules(), 0u);
  EXPECT_EQ(cstr.getCounts(), counts);
  EXPECT_FALSE(cstr.populateAndCompare(result, Cmp::GE, constant(5)));

  // Retained builders must discard new consequences of a disproven rule.
  conditional >= 9;
  conditional.when({{a, Cmp::GE, constant(0)}}) >= 11;
  cstr.bound(a) >= 0;
  cstr.processWorklist();
  EXPECT_EQ(cstr.getNumPendingRules(), 0u);
  EXPECT_FALSE(cstr.populateAndCompare(result, Cmp::GE, constant(5)));
}

TEST_F(ConditionalValueBoundsTest, DisprovenComparisonsAreRetired) {
  // Exercise every operator, including equality disproven in either direction.
  for (auto [cmp, value] : {std::pair{Cmp::LT, 1},
                            {Cmp::LE, 1},
                            {Cmp::EQ, 1},
                            {Cmp::EQ, -1},
                            {Cmp::GT, -1},
                            {Cmp::GE, -1}}) {
    SCOPED_TRACE(testing::Message()
                 << "comparison " << cmp << ", value " << value);
    TestConstraintSet cstr(&ctx);
    (void)cstr.getExpr(result);
    cstr.bound(result).when({{a, cmp, constant(0)}}) >= 5;
    cstr.bound(a) == value;
    EXPECT_FALSE(cstr.populateAndCompare(result, Cmp::GE, constant(5)));
    EXPECT_EQ(cstr.getNumPendingRules(), 0u);
  }
}

TEST_F(ConditionalValueBoundsTest, WhenLeavesOriginalBuilderUnconditional) {
  TestConstraintSet cstr(&ctx);
  (void)cstr.getExpr(result);
  auto bounds = cstr.bound(result);
  const auto &original = bounds;
  auto conditionalA = original.when({{a, Cmp::GE, constant(0)}});
  auto conditionalB = original.when({{b, Cmp::GE, constant(0)}});
  conditionalA >= 5;
  conditionalB >= 7;
  bounds >= 0;

  EXPECT_TRUE(cstr.populateAndCompare(result, Cmp::GE, constant(0)));
  EXPECT_FALSE(cstr.populateAndCompare(result, Cmp::GE, constant(5)));
  EXPECT_EQ(cstr.getNumPendingRules(), 2u);

  cstr.bound(b) >= 0;
  EXPECT_TRUE(cstr.populateAndCompare(result, Cmp::GE, constant(7)));
  EXPECT_EQ(cstr.getNumPendingRules(), 1u);
  cstr.bound(a) >= 0;
  cstr.processWorklist();
  EXPECT_EQ(cstr.getNumPendingRules(), 0u);
}

TEST_F(ConditionalValueBoundsTest, ChainedConditionsPreserveOriginalBuilder) {
  TestConstraintSet cstr(&ctx);
  (void)cstr.getExpr(result);
  auto conditionalA = cstr.bound(result).when({{a, Cmp::GE, constant(0)}});
  auto conditionalAB = conditionalA.when({{b, Cmp::GE, constant(0)}});
  conditionalA >= 3;
  conditionalAB >= 7;

  cstr.bound(a) >= 0;
  EXPECT_TRUE(cstr.populateAndCompare(result, Cmp::GE, constant(3)));
  EXPECT_FALSE(cstr.populateAndCompare(result, Cmp::GE, constant(7)));

  // Chaining after the original rule fires must still require the new guard.
  conditionalA.when({{b, Cmp::GE, constant(0)}}) >= 9;
  EXPECT_FALSE(cstr.populateAndCompare(result, Cmp::GE, constant(9)));
  cstr.bound(b) >= 0;
  EXPECT_TRUE(cstr.populateAndCompare(result, Cmp::GE, constant(9)));
  EXPECT_EQ(cstr.getNumPendingRules(), 0u);
}

TEST_F(ConditionalValueBoundsTest, ChainedConditionsRetainEarlierGuards) {
  TestConstraintSet cstr(&ctx);
  (void)cstr.getExpr(result);
  cstr.bound(result)
          .when({{a, Cmp::GE, constant(0)}})
          .when({{b, Cmp::GE, constant(0)}}) >= 7;
  cstr.bound(b) >= 0;
  EXPECT_FALSE(cstr.populateAndCompare(result, Cmp::GE, constant(7)));
  cstr.bound(a) >= 0;
  EXPECT_TRUE(cstr.populateAndCompare(result, Cmp::GE, constant(7)));
  EXPECT_EQ(cstr.getNumPendingRules(), 0u);
}

TEST_F(ConditionalValueBoundsTest, UnknownCycleDoesNotProveItsOwnGuards) {
  TestConstraintSet cstr(&ctx);
  (void)cstr.getExpr(a);
  (void)cstr.getExpr(b);
  cstr.bound(b).when({{a, Cmp::GE, constant(0)}}) >= 0;
  cstr.bound(a).when({{b, Cmp::GE, constant(0)}}) >= 0;
  EXPECT_FALSE(cstr.populateAndCompare(a, Cmp::GE, constant(0)));
  EXPECT_FALSE(cstr.populateAndCompare(b, Cmp::GE, constant(0)));
  EXPECT_EQ(cstr.getNumPendingRules(), 2u);

  cstr.bound(a) >= 0;
  EXPECT_TRUE(cstr.populateAndCompare(b, Cmp::GE, constant(0)));
  EXPECT_EQ(cstr.getNumPendingRules(), 0u);
}

TEST_F(ConditionalValueBoundsTest, ConsequencesSurviveColumnInsertions) {
  TestConstraintSet cstr(&ctx);
  (void)cstr.getExpr(result);
  auto bounds = cstr.bound(result).when({{a, Cmp::GE, constant(0)}});
  bounds == cstr.getExpr(a) + 7;
  // Comparisons insert anonymous dimensions ahead of mapped symbol columns.
  EXPECT_FALSE(cstr.populateAndCompare(result, Cmp::GE, constant(7)));
  (void)cstr.getExpr(b);
  cstr.bound(a) >= 0;
  EXPECT_TRUE(cstr.populateAndCompare(result, Cmp::GE, constant(7)));
  EXPECT_TRUE(cstr.populateAndCompare(
      result, Cmp::EQ,
      Variable(AffineMap::get(0, 1, Builder(&ctx).getAffineSymbolExpr(0) + 7),
               ValueRange{a})));
  // A retained builder may still add bounds after its guard has been proven.
  bounds >= 8;
  EXPECT_TRUE(cstr.populateAndCompare(result, Cmp::GE, constant(8)));
}

TEST_F(ConditionalValueBoundsTest, GuardExpressionsAreMaterializedOnce) {
  TestConstraintSet cstr(&ctx);
  (void)cstr.getExpr(result);
  (void)cstr.getExpr(b);
  // Materializing a guard with a different floor division must preserve the
  // existing local variable. Rechecking the guard must not add more columns,
  // locals, or constraints.
  cstr.bound(b) == cstr.getExpr(a).floorDiv(5);
  Variable quotient(
      AffineMap::get(0, 1, Builder(&ctx).getAffineSymbolExpr(0).floorDiv(3)),
      ValueRange{a});
  cstr.bound(result).when({{quotient, Cmp::GE, constant(0)}}) >= 5;
  auto counts = cstr.getCounts();
  cstr.processWorklist();
  EXPECT_EQ(counts, cstr.getCounts());
  cstr.processWorklist();
  EXPECT_EQ(counts, cstr.getCounts());
  EXPECT_EQ(cstr.getNumPendingRules(), 1u);

  cstr.bound(a) == 10;
  EXPECT_TRUE(cstr.populateAndCompare(result, Cmp::GE, constant(5)));
  EXPECT_TRUE(cstr.populateAndCompare(b, Cmp::EQ, constant(2)));
  EXPECT_EQ(cstr.getNumPendingRules(), 0u);
}

TEST_F(ConditionalValueBoundsTest, GuardDoesNotIntroduceSemiAffineAssumptions) {
  TestConstraintSet cstr(&ctx, /*conservative=*/true);
  (void)cstr.getExpr(result);
  Builder builder(&ctx);
  Variable remainder(AffineMap::get(0, 2,
                                    builder.getAffineSymbolExpr(0) %
                                        builder.getAffineSymbolExpr(1)),
                     ValueRange{a, b});
  cstr.bound(result).when({{remainder, Cmp::GE, constant(0)}}) >= 5;
  // Conservative flattening can assume a positive divisor to bound a mod
  // expression. Materializing a guard must not introduce that assumption.
  EXPECT_FALSE(cstr.populateAndCompare(result, Cmp::GE, constant(5)));
  EXPECT_EQ(cstr.getNumPendingRules(), 1u);
}
} // namespace
