//===- TestLinalgFuseConsumer.cpp - Test Linalg fuse consumer  ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass for testing fuse consumer of Linalg ops.
// This is a temporary pass used to verify the correctness of the tiling
// interface in linalg op and the related interface of fuse consumer. It should
// be replaced with that implementation when the corresponding fusion transform
// op is completed.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

#define DEBUG_TYPE "fuse-consumer"

namespace {
struct TestLinalgFuseConsumer
    : public PassWrapper<TestLinalgFuseConsumer, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLinalgFuseConsumer)

  TestLinalgFuseConsumer() = default;
  TestLinalgFuseConsumer(const TestLinalgFuseConsumer &pass)
      : PassWrapper(pass){};
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }
  StringRef getArgument() const final { return "test-linalg-fuse-consumer"; }
  StringRef getDescription() const final {
    return "Test Linalg fuse consumer interface";
  }

  void runOnOperation() override {
    Operation *consumerOp = nullptr, *containingOp = nullptr;
    auto walkRes = getOperation()->walk([&](Operation *op) {
      if (op->hasAttr("consumer")) {
        if (consumerOp) {
          return WalkResult::interrupt();
        }
        consumerOp = op;
      }
      if (op->hasAttr("containing")) {
        if (containingOp) {
          return WalkResult::interrupt();
        }
        containingOp = op;
      }
      return WalkResult::advance();
    });

    if (!consumerOp || !containingOp || walkRes.wasInterrupted()) {
      getOperation()->emitError() << "expect 1 consumer and 1 containing op.";
      return signalPassFailure();
    }

    // Check consumer has tiling interface.
    auto tileableConsumer = dyn_cast<TilingInterface>(consumerOp);
    if (!tileableConsumer) {
      consumerOp->emitError()
          << "consumer is not a TileableInterface: " << *consumerOp;
      return signalPassFailure();
    }

    // Check containing op is "scf::ForallOp".
    auto forallOp = dyn_cast<scf::ForallOp>(containingOp);
    if (!forallOp) {
      containingOp->emitError()
          << "containing op is not a scf.forall: " << containingOp;
      return signalPassFailure();
    }

    // Check dominance.
    DominanceInfo domInfo(getOperation());
    if (llvm::any_of(consumerOp->getOperands(), [&](Value v) {
          return v.getDefiningOp() != containingOp &&
                 !domInfo.properlyDominates(v, containingOp);
        })) {
      consumerOp->emitError()
          << "consumer's operand can't dominate containing op";
      return signalPassFailure();
    }

    // Check consumer don't use more than one result of containingOp.
    Value bridge(nullptr);
    unsigned operandNum;
    for (OpOperand &opd : consumerOp->getOpOperands()) {
      if (opd.get().getDefiningOp() == containingOp) {
        operandNum = opd.getOperandNumber();
        if (!bridge) {
          bridge = opd.get();
        } else if (bridge != opd.get()) {
          consumerOp->emitError()
              << "consumer's operands use more than one containingOp's result";
          return signalPassFailure();
        }
      }
    }

    // Check consumer has DestinationStyleOpInterface.
    auto dstOp = dyn_cast<DestinationStyleOpInterface>(consumerOp);
    if (!dstOp) {
      consumerOp->emitError()
          << "consumer op should have destination style op interface";
      return signalPassFailure();
    }

    // Check consumer doon't use scf.forall's output as init.
    SmallVector<Value> dpsInits =
        llvm::map_to_vector(dstOp.getDpsInits(), [](Value v) { return v; });
    if (llvm::is_contained(dpsInits, bridge)) {
      consumerOp->emitError()
          << "consumer op take result of scf.forall as init";
      return signalPassFailure();
    }

    // Check result was inserted only once.
    int64_t bridgeResultIdx = cast<OpResult>(bridge).getResultNumber();
    Value bridgeBlockArg = forallOp.getRegionOutArgs()[bridgeResultIdx];
    scf::InParallelOp terminatorOp = forallOp.getTerminator();

    tensor::ParallelInsertSliceOp targetInsertOp(nullptr);
    for (Operation &op : terminatorOp.getRegion().front().getOperations()) {
      auto parallelInsertSliceOp = cast<tensor::ParallelInsertSliceOp>(op);
      if (parallelInsertSliceOp.getDest() == bridgeBlockArg) {
        if (!targetInsertOp) {
          targetInsertOp = parallelInsertSliceOp;
        } else {
          containingOp->emitError()
              << "containingOp's result inserted multiple time";
          return signalPassFailure();
        }
      }
    }

    if (!targetInsertOp) {
      containingOp->emitError() << "containingOp's result was not inserted";
      return signalPassFailure();
    }

    SmallVector<OpFoldResult> offsets = targetInsertOp.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = targetInsertOp.getMixedSizes();
    SmallVector<OpFoldResult> strides = targetInsertOp.getMixedStrides();

    // Check all insert stride is 1.
    if (llvm::any_of(strides, [](OpFoldResult foldRes) {
          if (auto attr = foldRes.dyn_cast<Attribute>()) {
            return cast<IntegerAttr>(attr).getInt() != 1;
          }
          return true;
        })) {
      containingOp->emitError() << "containingOp's result yield with stride";
      return signalPassFailure();
    }

    IRRewriter rewriter(terminatorOp);
    Location loc = forallOp.getLoc();

    SmallVector<OpFoldResult> iterDomainOffsets, iterDomainSizes;

    // Try to get iter domain position from input position.
    if (failed(tileableConsumer.getIterationDomainTileFromOperandTile(
            rewriter, operandNum, offsets, sizes, iterDomainOffsets,
            iterDomainSizes))) {
      consumerOp->emitError()
          << "can't get iter domain position from input position";
      return signalPassFailure();
    }

    // Try to get all containing op result's position from iter domain position.
    SmallVector<std::pair<SmallVector<OpFoldResult>, SmallVector<OpFoldResult>>>
        resultPositions(consumerOp->getNumResults());
    for (auto [idx, v] : llvm::enumerate(consumerOp->getResults())) {
      if (failed(tileableConsumer.getResultTilePosition(
              rewriter, idx, iterDomainOffsets, iterDomainSizes,
              resultPositions[idx].first, resultPositions[idx].second))) {
        consumerOp->emitError()
            << "can't get result domain position from iter domain position";
        return signalPassFailure();
      }
    }

    // All check passed, try to fuse consumer.
    // Create tiled implementation of containing op.
    FailureOr<TilingResult> tileAndFuseResult =
        tileableConsumer.getTiledImplementationFromOperandTile(
            rewriter, operandNum, offsets, sizes);
    if (failed(tileAndFuseResult)) {
      consumerOp->emitError() << "get tiled implementation failed";
      return signalPassFailure();
    }

    SmallVector<Operation *> tiledOps = tileAndFuseResult->tiledOps;
    if (tiledOps.size() != 1) {
      consumerOp->emitError()
          << "failed to tile consumer op: " << *tileableConsumer;
      return signalPassFailure();
    }

    // Replace tiled op's operand.
    tiledOps[0]->setOperand(operandNum, targetInsertOp.getSource());
    rewriter.replaceUsesWithIf(bridge, forallOp.getOutputs()[bridgeResultIdx],
                               [&](OpOperand &use) {
                                 Operation *op = use.getOwner();
                                 return forallOp->isProperAncestor(op);
                               });

    SmallVector<Value> newOuts(forallOp.getOutputs());
    newOuts.append(dpsInits);

    // Create new scf.forall op.
    rewriter.setInsertionPoint(forallOp);
    auto newforallOp = rewriter.create<scf::ForallOp>(
        loc, forallOp.getMixedLowerBound(), forallOp.getMixedUpperBound(),
        forallOp.getMixedStep(), newOuts, forallOp.getMapping());
    rewriter.eraseBlock(newforallOp.getBody());
    newforallOp.getRegion().takeBody(forallOp.getRegion());

    for (auto v : dpsInits) {
      newforallOp.getBody()->addArgument(v.getType(), v.getLoc());
      auto bbArgs = newforallOp.getBody()->getArguments();
      rewriter.replaceUsesWithIf(v, bbArgs.back(), [&](OpOperand &use) {
        Operation *op = use.getOwner();
        return newforallOp->isProperAncestor(op);
      });
    }

    // Fix terminator.
    scf::InParallelOp newTerminatorOp = newforallOp.getTerminator();
    SmallVector<Operation *> yieldingOps = llvm::map_to_vector(
        newTerminatorOp.getYieldingOps(), [](Operation &op) { return &op; });
    Operation *firstYieldOp = yieldingOps.front();
    rewriter.setInsertionPoint(firstYieldOp);
    auto bbArgs = newforallOp.getBody()->getArguments();
    for (auto [idx, v] : llvm::enumerate(tiledOps[0]->getResults())) {
      SmallVector<OpFoldResult> strides(resultPositions[idx].first.size(),
                                        rewriter.getIndexAttr(1));
      rewriter.create<tensor::ParallelInsertSliceOp>(
          firstYieldOp->getLoc(), v,
          bbArgs[forallOp.getRank() + forallOp.getOutputs().size() + idx],
          resultPositions[idx].first, resultPositions[idx].second, strides);
    }

    // Replace the result of forall and consumer op.
    for (auto result : llvm::enumerate(forallOp.getResults())) {
      rewriter.replaceAllUsesWith(result.value(),
                                  newforallOp->getResult(result.index()));
    }

    for (auto consumerResult : llvm::enumerate(consumerOp->getResults())) {
      rewriter.replaceAllUsesWith(
          consumerResult.value(),
          newforallOp->getResult(forallOp.getOutputs().size() +
                                 consumerResult.index()));
    }
    rewriter.eraseOp(forallOp);
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestLinalgFuseConsumer() {
  PassRegistration<TestLinalgFuseConsumer>();
}
} // namespace test
} // namespace mlir
