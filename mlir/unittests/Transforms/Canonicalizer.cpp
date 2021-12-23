//===- DialectConversion.cpp - Dialect conversion unit tests --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "gtest/gtest.h"

using namespace mlir;

namespace {

struct DisabledPattern : public RewritePattern {
  DisabledPattern(MLIRContext *context)
      : RewritePattern("test.foo", /*benefit=*/0, context,
                       /*generatedNamed=*/{}) {
    setDebugName("DisabledPattern");
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1)
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

struct EnabledPattern : public RewritePattern {
  EnabledPattern(MLIRContext *context)
      : RewritePattern("test.foo", /*benefit=*/0, context,
                       /*generatedNamed=*/{}) {
    setDebugName("EnabledPattern");
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() == 1)
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

struct TestDialect : public Dialect {
  static StringRef getDialectNamespace() { return "test"; }

  TestDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context, TypeID::get<TestDialect>()) {
    allowUnknownOperations();
  }

  void getCanonicalizationPatterns(RewritePatternSet &results) const override {
    results.insert<DisabledPattern, EnabledPattern>(results.getContext());
  }
};

TEST(CanonicalizerTest, TestDisablePatterns) {
  MLIRContext context;
  context.getOrLoadDialect<TestDialect>();
  PassManager mgr(&context);
  mgr.addPass(
      createCanonicalizerPass(GreedyRewriteConfig(), {"DisabledPattern"}));

  const char *const code = R"mlir(
    %0:2 = "test.foo"() {sym_name = "A"} : () -> (i32, i32)
    %1 = "test.foo"() {sym_name = "B"} : () -> (f32)
  )mlir";

  OwningModuleRef module = mlir::parseSourceString(code, &context);
  ASSERT_TRUE(succeeded(mgr.run(*module)));

  EXPECT_TRUE(module->lookupSymbol("B"));
  EXPECT_FALSE(module->lookupSymbol("A"));
}

} // end anonymous namespace
