//===- LoopInvariantCodeMotion.cpp - Code to perform loop fusion-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop invariant code motion.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/SideEffectUtils.h"

using namespace mlir;

namespace {
/// Loop invariant code motion (LICM) pass.
struct LoopInvariantCodeMotion
    : public LoopInvariantCodeMotionBase<LoopInvariantCodeMotion> {
  void runOnOperation() override;
};
} // namespace

void LoopInvariantCodeMotion::runOnOperation() {
  // Walk through all loops in a function in innermost-loop-first order. This
  // way, we first LICM from the inner loop, and place the ops in
  // the outer loop, which in turn can be further LICM'ed.
  getOperation()->walk(
      [&](LoopLikeOpInterface loopLike) { moveLoopInvariantCode(loopLike); });
}

std::unique_ptr<Pass> mlir::createLoopInvariantCodeMotionPass() {
  return std::make_unique<LoopInvariantCodeMotion>();
}
