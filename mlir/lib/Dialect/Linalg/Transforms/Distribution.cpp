//===- Distibution.cpp - linalg named ops to generic ops  --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Linalg distibution pass. It updates `tiled_loop`
// control variables depending on the distribution type.
//
//===----------------------------------------------------------------------===//
//
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "linalg-distribution"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace mlir;
using namespace mlir::linalg;

namespace {

struct DistributeTiledLoopPattern
    : public OpRewritePattern<linalg::TiledLoopOp> {
  DistributeTiledLoopPattern(MLIRContext *context,
                             LinalgLoopDistributionOptions options,
                             LinalgTransformationFilter marker)
      : OpRewritePattern<linalg::TiledLoopOp>(context), options(options),
        marker(marker) {}
  LogicalResult matchAndRewrite(linalg::TiledLoopOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(marker.checkAndNotify(rewriter, op)))
      return failure();
    if (!op.distribution_types().hasValue())
      return failure();

    Location loc = op.getLoc();
    SmallVector<Value, 2> newLowerBounds = op.lowerBound();
    SmallVector<Value, 2> newUpperBounds = op.upperBound();
    SmallVector<Value, 2> newSteps = op.step();

    // Update bounds and steps.
    auto distributionTypes = op.distribution_types().getValue();
    for (int i = 0, e = op.getNumLoops(); i < e; ++i) {
      StringRef type = distributionTypes[i].cast<StringAttr>().getValue();
      auto procInfoCallback = options.procInfoMap.find(type);
      if (procInfoCallback == options.procInfoMap.end())
        continue;

      if (!isParallelIteratorType(op.iterator_types()[i])) {
        op.emitOpError("only support for parallel loops is implemented");
        return failure();
      }
      ProcInfo info = procInfoCallback->second(rewriter, loc);
      updateBoundsForCyclicDistribution(rewriter, loc, info.procId, info.nprocs,
                                        newLowerBounds[i], newUpperBounds[i],
                                        newSteps[i]);
    }
    rewriter.updateRootInPlace(op, [&] {
      op.setLowerBounds(newLowerBounds);
      op.setUpperBounds(newUpperBounds);
      op.setSteps(newSteps);
    });
    marker.replaceLinalgTransformationFilter(rewriter, op);
    return success();
  }

private:
  LinalgLoopDistributionOptions options;
  LinalgTransformationFilter marker;
};

} // namespace

void mlir::linalg::populateLinalgDistributeTiledLoopPattern(
    RewritePatternSet &patterns, const LinalgLoopDistributionOptions &opts,
    const LinalgTransformationFilter &marker) {
  patterns.add<DistributeTiledLoopPattern>(patterns.getContext(), opts, marker);
}
