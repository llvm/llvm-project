//===- PatternMatchTest.cpp - PatternMatch unit tests ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "gtest/gtest.h"

#include "../../test/lib/Dialect/Test/TestDialect.h"
#include "../../test/lib/Dialect/Test/TestOps.h"

using namespace mlir;

namespace {
struct AnOpRewritePattern : OpRewritePattern<test::OpA> {
  AnOpRewritePattern(MLIRContext *context)
      : OpRewritePattern(context, /*benefit=*/1,
                         /*generatedNames=*/{test::OpB::getOperationName()}) {}
};
TEST(OpRewritePatternTest, GetGeneratedNames) {
  MLIRContext context;
  AnOpRewritePattern pattern(&context);
  ArrayRef<OperationName> ops = pattern.getGeneratedOps();

  ASSERT_EQ(ops.size(), 1u);
  ASSERT_EQ(ops.front().getStringRef(), test::OpB::getOperationName());
}
} // end anonymous namespace

namespace {
LogicalResult anOpRewritePatternFunc(test::OpA op, PatternRewriter &rewriter) {
  return failure();
}
TEST(AnOpRewritePatternTest, PatternFuncAttributes) {
  MLIRContext context;
  RewritePatternSet patterns(&context);

  patterns.add(anOpRewritePatternFunc, /*benefit=*/3,
               /*generatedNames=*/{test::OpB::getOperationName()});
  ASSERT_EQ(patterns.getNativePatterns().size(), 1U);
  auto &pattern = patterns.getNativePatterns().front();
  ASSERT_EQ(pattern->getBenefit(), 3);
  ASSERT_EQ(pattern->getGeneratedOps().size(), 1U);
  ASSERT_EQ(pattern->getGeneratedOps().front().getStringRef(),
            test::OpB::getOperationName());
}
} // end anonymous namespace
