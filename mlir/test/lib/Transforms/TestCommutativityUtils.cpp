//===- TestCommutativityUtils.cpp - Pass to test the commutativity utility-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass tests the functionality of the commutativity utility pattern.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/CommutativityUtils.h"

#include "TestDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct CommutativityUtils
    : public PassWrapper<CommutativityUtils, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CommutativityUtils)

  StringRef getArgument() const final { return "test-commutativity-utils"; }
  StringRef getDescription() const final {
    return "Test the functionality of the commutativity utility";
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();

    RewritePatternSet patterns(context);
    populateCommutativityUtilsPatterns(patterns);

    (void)applyPatternsGreedily(func, std::move(patterns));
  }
};
} // namespace

namespace mlir {
namespace test {
void registerCommutativityUtils() { PassRegistration<CommutativityUtils>(); }
} // namespace test
} // namespace mlir
