//===- TestLinalgFusionTransforms.cpp - Test Linalg fusion patterns -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for testing Linalg fusion patterns.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::linalg;

static LogicalResult fuseLinalgOpsGreedily(func::FuncOp f) {
  OpBuilder b(f);
  DenseSet<Operation *> eraseSet;

  // Save original Linalg ops, we only want to make a pass over those.
  SmallVector<LinalgOp, 8> linalgOps;
  f.walk([&](LinalgOp op) {
    // TODO: support multi-results.
    if (op->getNumResults() <= 1)
      linalgOps.push_back(op);
  });

  // Tile and Fuse for tensors inputs (TODO: all tensor operands).
  bool changed = false;
  for (LinalgOp linalgOp : llvm::reverse(linalgOps)) {
    for (OpOperand &opOperand : linalgOp->getOpOperands()) {
      if (opOperand.get().getType().isa<MemRefType>()) {
        // TODO: LinalgDependenceGraph should be able to update itself.
        // The current naive and expensive reconstruction of the graph should be
        // removed.
        linalg::Aliases aliases;
        linalg::LinalgDependenceGraph graph(aliases, linalgOps);
        auto info = fuseProducerOfBuffer(b, opOperand, graph);
        if (failed(info))
          continue;
        auto *originalOp = info->originalProducer.getOperation();
        eraseSet.insert(originalOp);
        auto *originalOpInLinalgOpsVector =
            std::find(linalgOps.begin(), linalgOps.end(), originalOp);
        *originalOpInLinalgOpsVector = info->fusedProducer.getOperation();
        changed = true;
      } else if (opOperand.get().getType().isa<RankedTensorType>()) {
        // Tile and Fuse tensor input.
        if (opOperand.getOperandNumber() >= linalgOp.getNumDpsInputs())
          continue;
        auto info = fuseProducerOfTensor(b, opOperand);
        if (failed(info))
          continue;
        auto *originalOp = info->originalProducer.getOperation();
        auto *originalOpInLinalgOpsVector =
            std::find(linalgOps.begin(), linalgOps.end(), originalOp);
        *originalOpInLinalgOpsVector = info->fusedProducer.getOperation();
        // Don't mark for erasure in the tensor case, let DCE handle this.
        changed = true;
      }
    }
  }
  // The `fuseProducerOfBuffer` function performs structural checks and in
  // particular that no covering read or write exist between the consumer and
  // the producer. As a consequence, the only fusions that may occur preserve
  // subsequent dependences and are guaranteed by construction to produce the
  // whole view. We may thus erase the producer once it is fused.
  for (auto *e : eraseSet)
    e->erase();

  return changed ? success() : failure();
}

namespace {
struct TestLinalgGreedyFusion
    : public PassWrapper<TestLinalgGreedyFusion, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLinalgGreedyFusion)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect>();
  }
  StringRef getArgument() const final { return "test-linalg-greedy-fusion"; }
  StringRef getDescription() const final {
    return "Test Linalg fusion by applying a greedy test transformation.";
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns =
        linalg::getLinalgTilingCanonicalizationPatterns(context);
    patterns.add<ExtractSliceOfPadTensorSwapPattern>(context);
    scf::populateSCFForLoopCanonicalizationPatterns(patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    OpPassManager pm(func::FuncOp::getOperationName());
    pm.addPass(createLoopInvariantCodeMotionPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    do {
      (void)applyPatternsAndFoldGreedily(getOperation(), frozenPatterns);
      if (failed(runPipeline(pm, getOperation())))
        this->signalPassFailure();
    } while (succeeded(fuseLinalgOpsGreedily(getOperation())));
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestLinalgGreedyFusion() {
  PassRegistration<TestLinalgGreedyFusion>();
}

} // namespace test
} // namespace mlir
