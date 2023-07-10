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

#include "mlir/Dialect/Tosa/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace tosa {
#define GEN_PASS_DEF_TOSALAYERWISECONSTANTFOLDPASS
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"
} // namespace tosa
} // namespace mlir

using namespace mlir;
using namespace mlir::tosa;

namespace {

template <typename... Args>
void addOpsCanonicalizations(MLIRContext *ctx, RewritePatternSet &patterns) {
  (Args::getCanonicalizationPatterns(patterns, ctx), ...);
}

void populateTosaOpsCanonicalizationPatterns(MLIRContext *ctx,
                                             RewritePatternSet &patterns) {
  addOpsCanonicalizations<
#define GET_OP_LIST
#include "mlir/Dialect/Tosa/IR/TosaOps.cpp.inc"
      >(ctx, patterns);
}

struct TosaLayerwiseConstantFoldPass
    : public tosa::impl::TosaLayerwiseConstantFoldPassBase<
          TosaLayerwiseConstantFoldPass> {
  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    auto func = getOperation();

    mlir::tosa::populateTosaFoldConstantReciprocalPatterns(ctx, patterns);
    mlir::tosa::populateTosaFoldConstantTransposePatterns(ctx, patterns);
    populateTosaOpsCanonicalizationPatterns(ctx, patterns);

    if (applyPatternsAndFoldGreedily(func, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::tosa::createTosaLayerwiseConstantFoldPass() {
  return std::make_unique<TosaLayerwiseConstantFoldPass>();
}
