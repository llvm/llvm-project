//===- PatternBenefit.cpp - RewritePattern benefit unit tests -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "gtest/gtest.h"

using namespace mlir;

namespace {
TEST(PatternBenefitTest, BenefitOrder) {
  // There was a bug which caused low-benefit op-specific patterns to never be
  // called in presence of high-benefit op-agnostic pattern

  MLIRContext context;

  OpBuilder builder(&context);
  auto module = ModuleOp::create(builder.getUnknownLoc());

  struct Pattern1 : public OpRewritePattern<ModuleOp> {
    Pattern1(mlir::MLIRContext *context, bool *called)
        : OpRewritePattern<ModuleOp>(context, /*benefit*/ 1), called(called) {}

    mlir::LogicalResult
    matchAndRewrite(ModuleOp /*op*/,
                    mlir::PatternRewriter & /*rewriter*/) const override {
      *called = true;
      return failure();
    }

  private:
    bool *called;
  };

  struct Pattern2 : public RewritePattern {
    Pattern2(bool *called)
        : RewritePattern(/*benefit*/ 2, MatchAnyOpTypeTag{}), called(called) {}

    mlir::LogicalResult
    matchAndRewrite(Operation * /*op*/,
                    mlir::PatternRewriter & /*rewriter*/) const override {
      *called = true;
      return failure();
    }

  private:
    bool *called;
  };

  OwningRewritePatternList patterns(&context);

  bool called1 = false;
  bool called2 = false;

  patterns.insert<Pattern1>(&context, &called1);
  patterns.insert<Pattern2>(&called2);

  FrozenRewritePatternList frozenPatterns(std::move(patterns));
  PatternApplicator pa(frozenPatterns);
  pa.applyDefaultCostModel();

  class MyPatternRewriter : public PatternRewriter {
  public:
    MyPatternRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}
  };

  MyPatternRewriter rewriter(&context);
  (void)pa.matchAndRewrite(module, rewriter);

  EXPECT_TRUE(called1);
  EXPECT_TRUE(called2);
}
} // namespace
