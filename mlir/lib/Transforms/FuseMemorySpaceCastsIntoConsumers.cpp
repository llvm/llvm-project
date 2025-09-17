//===- FuseMemorySpaceCastsIntoConsumers.cpp - Fuse casts transform -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/FuseMemorySpaceCastsIntoConsumers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/MemOpInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_FUSEMEMORYSPACECASTSINTOCONSUMERS
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

namespace {
//===----------------------------------------------------------------------===//
// FuseCastsPattern pattern
//===----------------------------------------------------------------------===//
/// Pattern to fuse casts into consumer operations.
struct FuseCastsPattern
    : public OpInterfaceRewritePattern<FuseMemorySpaceCastConsumerOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(FuseMemorySpaceCastConsumerOpInterface op,
                                PatternRewriter &rewriter) const override {
    bool modifiedInPlace = false;
    FailureOr<SmallVector<Value>> results =
        op.fuseCastOperands(rewriter, modifiedInPlace);
    assert((!failed(results) || !modifiedInPlace) &&
           "expected `modifiedInPlace` to be false on fusion failure");
    if (failed(results))
      return failure();
    if (modifiedInPlace) {
      rewriter.modifyOpInPlace(op, []() {});
      return success();
    }
    rewriter.replaceOp(op, *results);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// FuseMemorySpaceCastsIntoConsumers pass
//===----------------------------------------------------------------------===//

struct FuseMemorySpaceCastsIntoConsumers
    : public impl::FuseMemorySpaceCastsIntoConsumersBase<
          FuseMemorySpaceCastsIntoConsumers> {
  using impl::FuseMemorySpaceCastsIntoConsumersBase<
      FuseMemorySpaceCastsIntoConsumers>::FuseMemorySpaceCastsIntoConsumersBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateFuseMemorySpaceCastIntoConsumersPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void mlir::populateFuseMemorySpaceCastIntoConsumersPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FuseCastsPattern>(patterns.getContext());
}
