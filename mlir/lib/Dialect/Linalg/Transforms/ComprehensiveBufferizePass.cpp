//===- ComprehensiveBufferize.cpp - Single pass bufferization -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/AffineInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ArithInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizableOpInterface.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizationInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ComprehensiveBufferize.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/LinalgInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ModuleBufferization.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/SCFInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/StdInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/TensorInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/VectorInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::linalg::comprehensive_bufferize;

namespace {
struct LinalgComprehensiveModuleBufferize
    : public LinalgComprehensiveModuleBufferizeBase<
          LinalgComprehensiveModuleBufferize> {
  LinalgComprehensiveModuleBufferize() = default;

  LinalgComprehensiveModuleBufferize(
      const LinalgComprehensiveModuleBufferize &p) = default;

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<bufferization::BufferizationDialect, linalg::LinalgDialect,
                memref::MemRefDialect, tensor::TensorDialect,
                vector::VectorDialect, scf::SCFDialect,
                arith::ArithmeticDialect, StandardOpsDialect, AffineDialect>();
    affine_ext::registerBufferizableOpInterfaceExternalModels(registry);
    arith_ext::registerBufferizableOpInterfaceExternalModels(registry);
    bufferization_ext::registerBufferizableOpInterfaceExternalModels(registry);
    linalg_ext::registerBufferizableOpInterfaceExternalModels(registry);
    scf_ext::registerBufferizableOpInterfaceExternalModels(registry);
    std_ext::registerModuleBufferizationExternalModels(registry);
    std_ext::registerBufferizableOpInterfaceExternalModels(registry);
    tensor_ext::registerBufferizableOpInterfaceExternalModels(registry);
    vector_ext::registerBufferizableOpInterfaceExternalModels(registry);
  }
};
} // namespace

static void applyEnablingTransformations(ModuleOp moduleOp) {
  RewritePatternSet patterns(moduleOp.getContext());
  patterns.add<GeneralizePadTensorOpPattern>(moduleOp.getContext());
  (void)applyPatternsAndFoldGreedily(moduleOp, std::move(patterns));
}

static FailureOr<Value> allocationFnUsingAlloca(OpBuilder &b, Location loc,
                                                MemRefType type,
                                                ArrayRef<Value> dynShape) {
  Value allocated = b.create<memref::AllocaOp>(
      loc, type, dynShape, b.getI64IntegerAttr(kBufferAlignments));
  return allocated;
}

void LinalgComprehensiveModuleBufferize::runOnOperation() {
  auto options = std::make_unique<BufferizationOptions>();
  if (useAlloca) {
    options->allocationFn = allocationFnUsingAlloca;
    options->deallocationFn = [](OpBuilder &b, Location loc, Value v) {
      return success();
    };
  }

  options->allowReturnMemref = allowReturnMemref;
  options->allowUnknownOps = allowUnknownOps;
  options->analysisFuzzerSeed = analysisFuzzerSeed;
  options->testAnalysisOnly = testAnalysisOnly;
  options->printConflicts = printConflicts;
  options->createDeallocs = createDeallocs;

  // Enable InitTensorOp elimination.
  if (initTensorElimination) {
    options->addPostAnalysisStep<
        linalg_ext::InsertSliceAnchoredInitTensorEliminationStep>();
  }

  // Only certain scf.for ops are supported by the analysis.
  options->addPostAnalysisStep<scf_ext::AssertScfForAliasingProperties>();

  ModuleOp moduleOp = getOperation();
  applyEnablingTransformations(moduleOp);

  if (failed(runComprehensiveBufferize(moduleOp, std::move(options)))) {
    signalPassFailure();
    return;
  }

  if (testAnalysisOnly)
    return;

  OpPassManager cleanupPipeline("builtin.module");
  cleanupPipeline.addPass(createCanonicalizerPass());
  cleanupPipeline.addPass(createCSEPass());
  cleanupPipeline.addPass(createLoopInvariantCodeMotionPass());
  (void)runPipeline(cleanupPipeline, moduleOp);
}

std::unique_ptr<Pass> mlir::createLinalgComprehensiveModuleBufferizePass() {
  return std::make_unique<LinalgComprehensiveModuleBufferize>();
}
