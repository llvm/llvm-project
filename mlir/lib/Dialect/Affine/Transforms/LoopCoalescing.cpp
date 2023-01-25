//===- LoopCoalescing.cpp - Pass transforming loop nests into single loops-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_LOOPCOALESCING
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace mlir

#define PASS_NAME "loop-coalescing"
#define DEBUG_TYPE PASS_NAME

using namespace mlir;

namespace {
struct LoopCoalescingPass
    : public impl::LoopCoalescingBase<LoopCoalescingPass> {

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    func.walk<WalkOrder::PreOrder>([](Operation *op) {
      if (auto scfForOp = dyn_cast<scf::ForOp>(op))
        (void)coalescePerfectlyNestedLoops(scfForOp);
      else if (auto affineForOp = dyn_cast<AffineForOp>(op))
        (void)coalescePerfectlyNestedLoops(affineForOp);
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createLoopCoalescingPass() {
  return std::make_unique<LoopCoalescingPass>();
}
