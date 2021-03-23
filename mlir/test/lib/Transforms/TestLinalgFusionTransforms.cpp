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

template <LinalgTilingLoopType LoopType>
static void fillFusionPatterns(MLIRContext *context,
                               const LinalgDependenceGraph &dependenceGraph,
                               RewritePatternSet &patterns) {
  patterns.add<LinalgTileAndFusePattern<MatmulOp>,
               LinalgTileAndFusePattern<ConvOp>>(
      context, dependenceGraph,
      LinalgTilingOptions().setTileSizes({32, 64, 16}).setLoopType(LoopType),
      LinalgFusionOptions().setIndicesToFuse({2}),
      LinalgTransformationFilter(
          Identifier::get("basic_fusion", context),
          Identifier::get("after_basic_fusion", context)),
      LinalgTransformationFilter(
          ArrayRef<Identifier>(),
          Identifier::get("after_basic_fusion_producer", context)),
      LinalgTransformationFilter(
          ArrayRef<Identifier>(),
          Identifier::get("after_basic_fusion_original", context)));

  patterns.add<LinalgTileAndFusePattern<MatmulOp>>(
      context, dependenceGraph,
      LinalgTilingOptions().setTileSizes({32, 64, 16}).setLoopType(LoopType),
      LinalgFusionOptions().setIndicesToFuse({0}),
      LinalgTransformationFilter(Identifier::get("lhs_fusion", context),
                                 Identifier::get("after_lhs_fusion", context)),
      LinalgTransformationFilter(
          ArrayRef<Identifier>(),
          Identifier::get("after_lhs_fusion_producer", context)),
      LinalgTransformationFilter(
          ArrayRef<Identifier>(),
          Identifier::get("after_lhs_fusion_original", context)));

  patterns.add<LinalgTileAndFusePattern<MatmulOp>>(
      context, dependenceGraph,
      LinalgTilingOptions().setTileSizes({32, 64, 16}).setLoopType(LoopType),
      LinalgFusionOptions().setIndicesToFuse({1}),
      LinalgTransformationFilter(Identifier::get("rhs_fusion", context),
                                 Identifier::get("after_rhs_fusion", context)),
      LinalgTransformationFilter(
          ArrayRef<Identifier>(),
          Identifier::get("after_rhs_fusion_producer", context)),
      LinalgTransformationFilter(
          ArrayRef<Identifier>(),
          Identifier::get("after_rhs_fusion_original", context)));

  patterns.add<LinalgTileAndFusePattern<MatmulOp>>(
      context, dependenceGraph,
      LinalgTilingOptions().setTileSizes({32, 64, 16}).setLoopType(LoopType),
      LinalgFusionOptions().setIndicesToFuse({0, 2}),
      LinalgTransformationFilter(
          Identifier::get("two_operand_fusion", context),
          Identifier::get("after_two_operand_fusion", context)),
      LinalgTransformationFilter(
          ArrayRef<Identifier>(),
          Identifier::get("after_two_operand_fusion_producer", context)),
      LinalgTransformationFilter(
          ArrayRef<Identifier>(),
          Identifier::get("after_two_operand_fusion_original", context)));

  patterns.add<LinalgTileAndFusePattern<GenericOp>>(
      context, dependenceGraph,
      LinalgTilingOptions().setTileSizes({32, 64}).setLoopType(LoopType),
      LinalgFusionOptions().setIndicesToFuse({0, 1}),
      LinalgTransformationFilter(
          Identifier::get("transpose_fusion", context),
          Identifier::get("after_transpose_fusion", context)),
      LinalgTransformationFilter(
          ArrayRef<Identifier>(),
          Identifier::get("after_transpose_fusion_producer", context)),
      LinalgTransformationFilter(
          ArrayRef<Identifier>(),
          Identifier::get("after_transpose_fusion_original", context)));
}

namespace {
template <LinalgTilingLoopType LoopType = LinalgTilingLoopType::ParallelLoops>
struct TestLinalgFusionTransforms
    : public PassWrapper<TestLinalgFusionTransforms<LoopType>, FunctionPass> {
  TestLinalgFusionTransforms() = default;
  TestLinalgFusionTransforms(const TestLinalgFusionTransforms &pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect, StandardOpsDialect>();
  }

  void runOnFunction() override {
    MLIRContext *context = &this->getContext();
    FuncOp funcOp = this->getFunction();
    RewritePatternSet fusionPatterns(context);
    Aliases alias;
    LinalgDependenceGraph dependenceGraph =
        LinalgDependenceGraph::buildDependenceGraph(alias, funcOp);
    fillFusionPatterns<LoopType>(context, dependenceGraph, fusionPatterns);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(fusionPatterns));
  }
};
} // namespace

static LogicalResult fuseLinalgOpsGreedily(FuncOp f) {
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
    for (OpOperand &opOperand : linalgOp.getShapedOpOperands()) {
      if (opOperand.get().getType().isa<MemRefType>()) {
        // TODO: LinalgDependenceGraph should be able to update itself.
        // The current naive and expensive reconstruction of the graph should be
        // removed.
        linalg::Aliases aliases;
        linalg::LinalgDependenceGraph graph(aliases, linalgOps);
        if (auto info = fuseProducerOfBuffer(b, opOperand, graph)) {
          auto *originalOp = info->originalProducer.getOperation();
          eraseSet.insert(originalOp);
          auto *originalOpInLinalgOpsVector =
              std::find(linalgOps.begin(), linalgOps.end(), originalOp);
          *originalOpInLinalgOpsVector = info->fusedProducer.getOperation();
          changed = true;
        }
      } else {
        assert(opOperand.get().getType().isa<RankedTensorType>());
        // Tile and Fuse tensor input.
        if (opOperand.getOperandNumber() >= linalgOp.getNumInputs())
          continue;
        if (auto info = fuseProducerOfTensor(b, opOperand)) {
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
    RewritePatternSet patterns =
        linalg::getLinalgTilingCanonicalizationPatterns(context);
    patterns.add<AffineMinSCFCanonicalizationPattern>(context);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    while (succeeded(fuseLinalgOpsGreedily(getFunction()))) {
      (void)applyPatternsAndFoldGreedily(getFunction(), frozenPatterns);
      PassManager pm(context);
      pm.addPass(createLoopInvariantCodeMotionPass());
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createCSEPass());
      LogicalResult res = pm.run(getFunction()->getParentOfType<ModuleOp>());
      if (failed(res))
        this->signalPassFailure();
    }
  }
};

/// Pass to test tile and fuse of sequence of operations. Intended only for
/// testing.
struct TestLinalgTileAndFuseSequencePass
    : public PassWrapper<TestLinalgTileAndFuseSequencePass, FunctionPass> {
  TestLinalgTileAndFuseSequencePass() = default;
  TestLinalgTileAndFuseSequencePass(
      const TestLinalgTileAndFuseSequencePass &pass){};

  ListOption<int64_t> tileSizes{
      *this, "tile-sizes", llvm::cl::desc("Tile sizes to use for ops"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect>();
  }

  void runOnFunction() override {
    FuncOp funcOp = getOperation();
    auto &blocks = funcOp.getBody().getBlocks();
    if (!llvm::hasSingleElement(blocks)) {
      return;
    }
    SmallVector<LinalgOp, 2> linalgOps =
        llvm::to_vector<2>(blocks.front().getOps<LinalgOp>());
    Aliases aliases;
    LinalgDependenceGraph dependenceGraph(aliases, linalgOps);
    OpBuilder builder(funcOp.getContext());
    linalg::LinalgTilingLoopType loopType = LinalgTilingLoopType::ParallelLoops;
    if (llvm::any_of(linalgOps, [](LinalgOp linalgOp) {
          return linalgOp.hasTensorSemantics();
        }))
      loopType = LinalgTilingLoopType::Loops;
    Optional<TiledAndFusedLinalgOps> tileAndFuseOps = tileAndFuseLinalgOps(
        builder, linalgOps, dependenceGraph,
        LinalgTilingOptions().setTileSizes(tileSizes).setLoopType(loopType));
    if (!tileAndFuseOps)
      return signalPassFailure();
    if (linalgOps.back().hasTensorSemantics()) {
      linalgOps.back().getOperation()->replaceAllUsesWith(
          tileAndFuseOps->fusedLoops.front());
    }
    for (auto op : linalgOps)
      if (op.hasBufferSemantics())
        op.erase();
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestLinalgFusionTransforms() {
  PassRegistration<TestLinalgFusionTransforms<>> testFusionTransformsPass(
      "test-linalg-fusion-transform-patterns",
      "Test Linalg fusion transformation patterns by applying them greedily.");
}
void registerTestLinalgTensorFusionTransforms() {
  PassRegistration<TestLinalgFusionTransforms<LinalgTilingLoopType::Loops>>
      testTensorFusionTransformsPass(
          "test-linalg-tensor-fusion-transform-patterns",
          "Test Linalg on tensor fusion transformation "
          "patterns by applying them greedily.");
}
void registerTestLinalgGreedyFusion() {
  PassRegistration<TestLinalgGreedyFusion> testFusionTransformsPass(
      "test-linalg-greedy-fusion",
      "Test Linalg fusion by applying a greedy test transformation.");
}
void registerTestLinalgTileAndFuseSequencePass() {
  PassRegistration<TestLinalgTileAndFuseSequencePass>
      testTileAndFuseSequencePass(
          "test-linalg-tile-and-fuse",
          "Test Linalg tiling and fusion of a sequence of Linalg operations.");
}

} // namespace test
} // namespace mlir
