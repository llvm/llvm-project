//===- AffineLoopNormalize.cpp - AffineLoopNormalize Pass -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a normalizer for affine loop-like ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
#define GEN_PASS_DEF_AFFINELOOPNORMALIZEPASS
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// Normalize affine.parallel ops so that lower bounds are 0 and steps are 1.
/// As currently implemented, this pass cannot fail, but it might skip over ops
/// that are already in a normalized form.
struct AffineLoopNormalizePass
    : public impl::AffineLoopNormalizePassBase<AffineLoopNormalizePass> {

  void runOnOperation() override {
    getOperation().walk([](Operation *op) {
      if (auto affineParallel = dyn_cast<AffineParallelOp>(op))
        normalizeAffineParallel(affineParallel);
      else if (auto affineFor = dyn_cast<AffineForOp>(op))
        (void)normalizeAffineFor(affineFor);
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createAffineLoopNormalizePass() {
  return std::make_unique<AffineLoopNormalizePass>();
}
