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
      LinalgFusionOptions(),
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
}

static void applyFusionPatterns(MLIRContext *context, FuncOp funcOp) {
  OwningRewritePatternList fusionPatterns;
  Aliases alias;
  LinalgDependenceGraph dependenceGraph =
      LinalgDependenceGraph::buildDependenceGraph(alias, funcOp);
  fillFusionPatterns(context, dependenceGraph, fusionPatterns);
  applyPatternsAndFoldGreedily(funcOp, fusionPatterns);
}

void TestLinalgFusionTransforms::runOnFunction() {
  applyFusionPatterns(&getContext(), getFunction());
}

namespace mlir {
void registerTestLinalgFusionTransforms() {
  PassRegistration<TestLinalgFusionTransforms> testFusionTransformsPass(
      "test-linalg-fusion-transform-patterns",
      "Test Linalg fusion transformation patterns by applying them greedily.");
}
} // namespace mlir
