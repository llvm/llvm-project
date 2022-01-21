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
#include "mlir/Dialect/SCF/Transforms.h"
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
               LinalgTileAndFusePattern<Conv2DOp>>(
      context, dependenceGraph,
      LinalgTilingOptions().setTileSizes({32, 64, 16}).setLoopType(LoopType),
      LinalgFusionOptions().setIndicesToFuse({2}),
      LinalgTransformationFilter(
          StringAttr::get(context, "basic_fusion"),
          StringAttr::get(context, "after_basic_fusion")),
      LinalgTransformationFilter(
          ArrayRef<StringAttr>(),
          StringAttr::get(context, "after_basic_fusion_producer")),
      LinalgTransformationFilter(
          ArrayRef<StringAttr>(),
          StringAttr::get(context, "after_basic_fusion_original")));

  patterns.add<LinalgTileAndFusePattern<MatmulOp>>(
      context, dependenceGraph,
      LinalgTilingOptions().setTileSizes({32, 64, 16}).setLoopType(LoopType),
      LinalgFusionOptions().setIndicesToFuse({0}),
      LinalgTransformationFilter(StringAttr::get(context, "lhs_fusion"),
                                 StringAttr::get(context, "after_lhs_fusion")),
      LinalgTransformationFilter(
          ArrayRef<StringAttr>(),
          StringAttr::get(context, "after_lhs_fusion_producer")),
      LinalgTransformationFilter(
          ArrayRef<StringAttr>(),
          StringAttr::get(context, "after_lhs_fusion_original")));

  patterns.add<LinalgTileAndFusePattern<MatmulOp>>(
      context, dependenceGraph,
      LinalgTilingOptions().setTileSizes({32, 64, 16}).setLoopType(LoopType),
      LinalgFusionOptions().setIndicesToFuse({2}),
      LinalgTransformationFilter(StringAttr::get(context, "out_fusion"),
                                 StringAttr::get(context, "after_out_fusion")),
      LinalgTransformationFilter(
          ArrayRef<StringAttr>(),
          StringAttr::get(context, "after_out_fusion_producer")),
      LinalgTransformationFilter(
          ArrayRef<StringAttr>(),
          StringAttr::get(context, "after_out_fusion_original")));

  patterns.add<LinalgTileAndFusePattern<MatmulOp>>(
      context, dependenceGraph,
      LinalgTilingOptions().setTileSizes({32, 64, 16}).setLoopType(LoopType),
      LinalgFusionOptions().setIndicesToFuse({1}),
      LinalgTransformationFilter(StringAttr::get(context, "rhs_fusion"),
                                 StringAttr::get(context, "after_rhs_fusion")),
      LinalgTransformationFilter(
          ArrayRef<StringAttr>(),
          StringAttr::get(context, "after_rhs_fusion_producer")),
      LinalgTransformationFilter(
          ArrayRef<StringAttr>(),
          StringAttr::get(context, "after_rhs_fusion_original")));

  patterns.add<LinalgTileAndFusePattern<MatmulOp>>(
      context, dependenceGraph,
      LinalgTilingOptions().setTileSizes({32, 64, 16}).setLoopType(LoopType),
      LinalgFusionOptions().setIndicesToFuse({0, 2}),
      LinalgTransformationFilter(
          StringAttr::get(context, "two_operand_fusion"),
          StringAttr::get(context, "after_two_operand_fusion")),
      LinalgTransformationFilter(
          ArrayRef<StringAttr>(),
          StringAttr::get(context, "after_two_operand_fusion_producer")),
      LinalgTransformationFilter(
          ArrayRef<StringAttr>(),
          StringAttr::get(context, "after_two_operand_fusion_original")));

  patterns.add<LinalgTileAndFusePattern<GenericOp>>(
      context, dependenceGraph,
      LinalgTilingOptions().setTileSizes({32, 64}).setLoopType(LoopType),
      LinalgFusionOptions().setIndicesToFuse({0, 1}),
      LinalgTransformationFilter(
          StringAttr::get(context, "transpose_fusion"),
          StringAttr::get(context, "after_transpose_fusion")),
      LinalgTransformationFilter(
          ArrayRef<StringAttr>(),
          StringAttr::get(context, "after_transpose_fusion_producer")),
      LinalgTransformationFilter(
          ArrayRef<StringAttr>(),
          StringAttr::get(context, "after_transpose_fusion_original")));
}

namespace {
template <LinalgTilingLoopType LoopType>
struct TestLinalgFusionTransforms
    : public PassWrapper<TestLinalgFusionTransforms<LoopType>,
                         OperationPass<FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect, StandardOpsDialect>();
  }
  TestLinalgFusionTransforms() = default;
  TestLinalgFusionTransforms(const TestLinalgFusionTransforms &pass) {}

  void runOnOperation() override {
    MLIRContext *context = &this->getContext();
    FuncOp funcOp = this->getOperation();
    RewritePatternSet fusionPatterns(context);
    Aliases alias;
    LinalgDependenceGraph dependenceGraph =
        LinalgDependenceGraph::buildDependenceGraph(alias, funcOp);
    fillFusionPatterns<LoopType>(context, dependenceGraph, fusionPatterns);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(fusionPatterns));
  }
};

struct TestLinalgFusionTransformsParallelLoops
    : public TestLinalgFusionTransforms<LinalgTilingLoopType::ParallelLoops> {
  StringRef getArgument() const final {
    return "test-linalg-fusion-transform-patterns";
  }
  StringRef getDescription() const final {
    return "Test Linalg fusion transformation patterns by applying them "
           "greedily.";
  }
};

struct TestLinalgFusionTransformsLoops
    : public TestLinalgFusionTransforms<LinalgTilingLoopType::Loops> {
  StringRef getArgument() const final {
    return "test-linalg-tensor-fusion-transform-patterns";
  }
  StringRef getDescription() const final {
    return "Test Linalg on tensor fusion transformation "
           "patterns by applying them greedily.";
  }
};

struct TestLinalgFusionTransformsTiledLoops
    : public TestLinalgFusionTransforms<LinalgTilingLoopType::TiledLoops> {
  StringRef getArgument() const final {
    return "test-linalg-tiled-loop-fusion-transform-patterns";
  }
  StringRef getDescription() const final {
    return "Test Linalg on tensor fusion transformation "
           "patterns by applying them greedily.";
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
    for (OpOperand *opOperand : linalgOp.getInputAndOutputOperands()) {
      if (opOperand->get().getType().isa<MemRefType>()) {
        // TODO: LinalgDependenceGraph should be able to update itself.
        // The current naive and expensive reconstruction of the graph should be
        // removed.
        linalg::Aliases aliases;
        linalg::LinalgDependenceGraph graph(aliases, linalgOps);
        auto info = fuseProducerOfBuffer(b, *opOperand, graph);
        if (failed(info))
          continue;
        auto *originalOp = info->originalProducer.getOperation();
        eraseSet.insert(originalOp);
        auto *originalOpInLinalgOpsVector =
            std::find(linalgOps.begin(), linalgOps.end(), originalOp);
        *originalOpInLinalgOpsVector = info->fusedProducer.getOperation();
        changed = true;
      } else if (opOperand->get().getType().isa<RankedTensorType>()) {
        // Tile and Fuse tensor input.
        if (opOperand->getOperandNumber() >= linalgOp.getNumInputs())
          continue;
        auto info = fuseProducerOfTensor(b, *opOperand);
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
    : public PassWrapper<TestLinalgGreedyFusion, OperationPass<FuncOp>> {
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
    do {
      (void)applyPatternsAndFoldGreedily(getOperation(), frozenPatterns);
      PassManager pm(context);
      pm.addPass(createLoopInvariantCodeMotionPass());
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createCSEPass());
      LogicalResult res = pm.run(getOperation()->getParentOfType<ModuleOp>());
      if (failed(res))
        this->signalPassFailure();
    } while (succeeded(fuseLinalgOpsGreedily(getOperation())));
  }
};

/// Pass to test tile and fuse of sequence of operations. Intended only for
/// testing.
struct TestLinalgTileAndFuseSequencePass
    : public PassWrapper<TestLinalgTileAndFuseSequencePass,
                         OperationPass<FuncOp>> {
  StringRef getArgument() const final { return "test-linalg-tile-and-fuse"; }
  StringRef getDescription() const final {
    return "Test Linalg tiling and fusion of a sequence of Linalg operations.";
  }
  TestLinalgTileAndFuseSequencePass() = default;
  TestLinalgTileAndFuseSequencePass(
      const TestLinalgTileAndFuseSequencePass &pass)
      : PassWrapper(pass){};

  ListOption<int64_t> tileSizes{
      *this, "tile-sizes", llvm::cl::desc("Tile sizes to use for ops"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override {
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
  PassRegistration<TestLinalgFusionTransformsParallelLoops>();
}
void registerTestLinalgTensorFusionTransforms() {
  PassRegistration<TestLinalgFusionTransformsLoops>();
}
void registerTestLinalgTiledLoopFusionTransforms() {
  PassRegistration<TestLinalgFusionTransformsTiledLoops>();
}
void registerTestLinalgGreedyFusion() {
  PassRegistration<TestLinalgGreedyFusion>();
}
void registerTestLinalgTileAndFuseSequencePass() {
  PassRegistration<TestLinalgTileAndFuseSequencePass>();
}

} // namespace test
} // namespace mlir
