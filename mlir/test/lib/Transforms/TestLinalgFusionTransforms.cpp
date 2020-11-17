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

#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {
struct TestLinalgFusionTransforms
    : public PassWrapper<TestLinalgFusionTransforms, FunctionPass> {
  TestLinalgFusionTransforms() = default;
  TestLinalgFusionTransforms(const TestLinalgFusionTransforms &pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, linalg::LinalgDialect, scf::SCFDialect,
                    StandardOpsDialect>();
  }

  void runOnFunction() override;
};
} // namespace

static void fillFusionPatterns(MLIRContext *context,
                               const LinalgDependenceGraph &dependenceGraph,
                               OwningRewritePatternList &patterns) {
  patterns.insert<LinalgTileAndFusePattern<MatmulOp>>(
      context, dependenceGraph,
      LinalgTilingOptions()
          .setTileSizes({32, 64, 16})
          .setLoopType(LinalgTilingLoopType::ParallelLoops),
      LinalgFusionOptions().setIndicesToFuse({2}),
      LinalgMarker(Identifier::get("basic_fusion", context),
                   Identifier::get("after_basic_fusion", context)),
      LinalgMarker(ArrayRef<Identifier>(),
                   Identifier::get("after_basic_fusion_producer", context)),
      LinalgMarker(ArrayRef<Identifier>(),
                   Identifier::get("after_basic_fusion_original", context)));

  patterns.insert<LinalgTileAndFusePattern<MatmulOp>>(
      context, dependenceGraph,
      LinalgTilingOptions()
          .setTileSizes({32, 64, 16})
          .setLoopType(LinalgTilingLoopType::ParallelLoops),
      LinalgFusionOptions().setIndicesToFuse({0}),
      LinalgMarker(Identifier::get("lhs_fusion", context),
                   Identifier::get("after_lhs_fusion", context)),
      LinalgMarker(ArrayRef<Identifier>(),
                   Identifier::get("after_lhs_fusion_producer", context)),
      LinalgMarker(ArrayRef<Identifier>(),
                   Identifier::get("after_lhs_fusion_original", context)));

  patterns.insert<LinalgTileAndFusePattern<MatmulOp>>(
      context, dependenceGraph,
      LinalgTilingOptions()
          .setTileSizes({32, 64, 16})
          .setLoopType(LinalgTilingLoopType::ParallelLoops),
      LinalgFusionOptions().setIndicesToFuse({1}),
      LinalgMarker(Identifier::get("rhs_fusion", context),
                   Identifier::get("after_rhs_fusion", context)),
      LinalgMarker(ArrayRef<Identifier>(),
                   Identifier::get("after_rhs_fusion_producer", context)),
      LinalgMarker(ArrayRef<Identifier>(),
                   Identifier::get("after_rhs_fusion_original", context)));

  patterns.insert<LinalgTileAndFusePattern<MatmulOp>>(
      context, dependenceGraph,
      LinalgTilingOptions()
          .setTileSizes({32, 64, 16})
          .setLoopType(LinalgTilingLoopType::ParallelLoops),
      LinalgFusionOptions().setIndicesToFuse({0, 2}),
      LinalgMarker(Identifier::get("two_operand_fusion", context),
                   Identifier::get("after_two_operand_fusion", context)),
      LinalgMarker(
          ArrayRef<Identifier>(),
          Identifier::get("after_two_operand_fusion_producer", context)),
      LinalgMarker(
          ArrayRef<Identifier>(),
          Identifier::get("after_two_operand_fusion_original", context)));

  patterns.insert<LinalgTileAndFusePattern<GenericOp>>(
      context, dependenceGraph,
      LinalgTilingOptions().setTileSizes({32, 64}).setLoopType(
          LinalgTilingLoopType::ParallelLoops),
      LinalgFusionOptions().setIndicesToFuse({0, 1}),
      LinalgMarker(Identifier::get("transpose_fusion", context),
                   Identifier::get("after_transpose_fusion", context)),
      LinalgMarker(ArrayRef<Identifier>(),
                   Identifier::get("after_transpose_fusion_producer", context)),
      LinalgMarker(
          ArrayRef<Identifier>(),
          Identifier::get("after_transpose_fusion_original", context)));
}

static void applyFusionPatterns(MLIRContext *context, FuncOp funcOp) {
  OwningRewritePatternList fusionPatterns;
  Aliases alias;
  LinalgDependenceGraph dependenceGraph =
      LinalgDependenceGraph::buildDependenceGraph(alias, funcOp);
  fillFusionPatterns(context, dependenceGraph, fusionPatterns);
  applyPatternsAndFoldGreedily(funcOp, std::move(fusionPatterns));
}

void TestLinalgFusionTransforms::runOnFunction() {
  applyFusionPatterns(&getContext(), getFunction());
}

static LogicalResult fuseLinalgOpsGreedily(FuncOp f) {
  OpBuilder b(f);
  DenseSet<Operation *> eraseSet;

  // Save original Linalg ops, we only want to make a pass over those.
  SmallVector<LinalgOp, 8> linalgOps;
  f.walk([&](LinalgOp op) {
    // TODO: support multi-results.
    if (op.getOperation()->getNumResults() <= 1)
      linalgOps.push_back(op);
  });

  // Tile and Fuse for tensors inputs (TODO: all tensor operands).
  bool changed = false;
  for (LinalgOp linalgOp : llvm::reverse(linalgOps)) {
    for (auto en : llvm::enumerate(linalgOp.getShapedOperands())) {
      if (en.value().getType().isa<MemRefType>()) {
        // TODO: LinalgDependenceGraph should be able to update itself.
        // The current naive and expensive reconstruction of the graph should be
        // removed.
        linalg::Aliases aliases;
        linalg::LinalgDependenceGraph graph(aliases, linalgOps);
        if (auto info = fuseProducerOfBuffer(b, linalgOp, en.index(), graph)) {
          auto *originalOp = info->originalProducer.getOperation();
          eraseSet.insert(originalOp);
          auto *originalOpInLinalgOpsVector =
              std::find(linalgOps.begin(), linalgOps.end(), originalOp);
          *originalOpInLinalgOpsVector = info->fusedProducer.getOperation();
          changed = true;
        }
      } else {
        assert(en.value().getType().isa<RankedTensorType>());
        // Tile and Fuse tensor input (TODO: init_tensors too).
        if (en.index() >= linalgOp.getNumInputs())
          continue;
        if (auto info = fuseProducerOfTensor(b, linalgOp, en.index())) {
          auto *originalOp = info->originalProducer.getOperation();
          auto *originalOpInLinalgOpsVector =
              std::find(linalgOps.begin(), linalgOps.end(), originalOp);
          *originalOpInLinalgOpsVector = info->fusedProducer.getOperation();
          // Don't mark for erasure in the tensor case, let DCE handle this.
          changed = true;
        }
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
    : public PassWrapper<TestLinalgGreedyFusion, FunctionPass> {
  void runOnFunction() override {
    MLIRContext *context = &getContext();
    OwningRewritePatternList patterns =
        linalg::getLinalgTilingCanonicalizationPatterns(context);
    patterns.insert<AffineMinSCFCanonicalizationPattern>(context);
    FrozenRewritePatternList frozenPatterns(std::move(patterns));
    while (succeeded(fuseLinalgOpsGreedily(getFunction()))) {
      applyPatternsAndFoldGreedily(getFunction(), frozenPatterns);
      PassManager pm(context);
      pm.addPass(createLoopInvariantCodeMotionPass());
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createCSEPass());
      LogicalResult res = pm.run(getFunction().getParentOfType<ModuleOp>());
      if (failed(res))
        this->signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestLinalgFusionTransforms() {
  PassRegistration<TestLinalgFusionTransforms> testFusionTransformsPass(
      "test-linalg-fusion-transform-patterns",
      "Test Linalg fusion transformation patterns by applying them greedily.");
}
void registerTestLinalgGreedyFusion() {
  PassRegistration<TestLinalgGreedyFusion> testFusionTransformsPass(
      "test-linalg-greedy-fusion",
      "Test Linalg fusion by applying a greedy test transformation.");
}
} // namespace test
} // namespace mlir
