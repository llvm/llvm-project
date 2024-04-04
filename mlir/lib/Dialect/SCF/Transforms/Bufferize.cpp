//===- Bufferize.cpp - scf bufferize pass ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_SCFBUFFERIZE
#define GEN_PASS_DEF_SCFLOOPBUFFERIZATIONPREPROCESSING
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;

namespace {
struct SCFBufferizePass : public impl::SCFBufferizeBase<SCFBufferizePass> {
  void runOnOperation() override {
    auto *func = getOperation();
    auto *context = &getContext();

    bufferization::BufferizeTypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    bufferization::populateBufferizeMaterializationLegality(target);
    populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns,
                                                    target);
    if (failed(applyPartialConversion(func, target, std::move(patterns))))
      return signalPassFailure();
  };
};

struct SCFLoopBufferizationPreprocessingPass
    : public impl::SCFLoopBufferizationPreprocessingBase<
          SCFLoopBufferizationPreprocessingPass> {
  void runOnOperation() override {
    OpBuilder builder(getOperation()->getContext());
    getOperation()->walk([&](scf::YieldOp yieldOp) {
      builder.setInsertionPoint(yieldOp);
      // TODO: Support scf.while.
      auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp());
      if (!forOp)
        return WalkResult::skip();
      for (OpOperand &operand : yieldOp->getOpOperands()) {
        auto tensorType = dyn_cast<TensorType>(operand.get().getType());
        if (!tensorType)
          continue;
        auto bbArg = forOp.getRegionIterArgs()[operand.getOperandNumber()];
        Value materialized =
            builder
                .create<bufferization::MaterializeInDestinationOp>(
                    yieldOp.getLoc(), tensorType, operand.get(), bbArg)
                .getResult();
        operand.set(materialized);
      }
      return WalkResult::advance();
    });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createSCFBufferizePass() {
  return std::make_unique<SCFBufferizePass>();
}

std::unique_ptr<Pass> mlir::createSCFLoopBufferizationPreprocessingPass() {
  return std::make_unique<SCFLoopBufferizationPreprocessingPass>();
}
