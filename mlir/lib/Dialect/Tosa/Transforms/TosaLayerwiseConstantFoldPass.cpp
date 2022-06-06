//===- TosaLayerwiseConstantFoldPass.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements constant folding transformations on TOSA operations
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/PassDetail.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::tosa;

namespace {

struct TosaLayerwiseConstantFoldPass
    : public TosaLayerwiseConstantFoldPassBase<TosaLayerwiseConstantFoldPass> {
  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    auto func = getOperation();

    mlir::tosa::populateTosaFoldConstantTransposePatterns(ctx, patterns);
    mlir::tosa::populateTosaOpsCanonicalizationPatterns(ctx, patterns);

    if (applyPatternsAndFoldGreedily(func, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::tosa::createTosaLayerwiseConstantFoldPass() {
  return std::make_unique<TosaLayerwiseConstantFoldPass>();
}
