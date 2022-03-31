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
