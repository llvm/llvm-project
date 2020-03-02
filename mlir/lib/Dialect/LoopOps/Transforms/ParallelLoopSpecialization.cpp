//===- ParallelLoopSpecialization.cpp - loop.parallel specializeation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Specializes parallel loops for easier unrolling and vectorization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/LoopOps/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using loop::ParallelOp;

/// Rewrite a loop with bounds defined by an affine.min with a constant into 2
/// loops after checking if the bounds are equal to that constant. This is
/// beneficial if the loop will almost always have the constant bound and that
/// version can be fully unrolled and vectorized.
static void specializeLoopForUnrolling(ParallelOp op) {
  SmallVector<int64_t, 2> constantIndices;
  constantIndices.reserve(op.upperBound().size());
  for (auto bound : op.upperBound()) {
    auto minOp = dyn_cast_or_null<AffineMinOp>(bound.getDefiningOp());
    if (!minOp)
      return;
    int64_t minConstant = std::numeric_limits<int64_t>::max();
    for (auto expr : minOp.map().getResults()) {
      if (auto constantIndex = expr.dyn_cast<AffineConstantExpr>())
        minConstant = std::min(minConstant, constantIndex.getValue());
    }
    if (minConstant == std::numeric_limits<int64_t>::max())
      return;
    constantIndices.push_back(minConstant);
  }

  OpBuilder b(op);
  BlockAndValueMapping map;
  Value cond;
  for (auto bound : llvm::zip(op.upperBound(), constantIndices)) {
    Value constant = b.create<ConstantIndexOp>(op.getLoc(), std::get<1>(bound));
    Value cmp = b.create<CmpIOp>(op.getLoc(), CmpIPredicate::eq,
                                 std::get<0>(bound), constant);
    cond = cond ? b.create<AndOp>(op.getLoc(), cond, cmp) : cmp;
    map.map(std::get<0>(bound), constant);
  }
  auto ifOp = b.create<loop::IfOp>(op.getLoc(), cond, /*withElseRegion=*/true);
  ifOp.getThenBodyBuilder().clone(*op.getOperation(), map);
  ifOp.getElseBodyBuilder().clone(*op.getOperation());
  op.erase();
}

namespace {
struct ParallelLoopSpecialization
    : public FunctionPass<ParallelLoopSpecialization> {
  void runOnFunction() override {
    getFunction().walk([](ParallelOp op) { specializeLoopForUnrolling(op); });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createParallelLoopSpecializationPass() {
  return std::make_unique<ParallelLoopSpecialization>();
}

static PassRegistration<ParallelLoopSpecialization>
    pass("parallel-loop-specialization",
         "Specialize parallel loops for vectorization.");
