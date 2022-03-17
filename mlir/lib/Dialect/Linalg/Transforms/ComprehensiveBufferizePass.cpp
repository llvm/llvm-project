//===- ComprehensiveBufferize.cpp - Single pass bufferization -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/Arithmetic/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ModuleBufferization.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::linalg;
using namespace mlir::linalg::comprehensive_bufferize;

namespace {
struct LinalgComprehensiveModuleBufferize
    : public LinalgComprehensiveModuleBufferizeBase<
          LinalgComprehensiveModuleBufferize> {
  LinalgComprehensiveModuleBufferize() = default;

  LinalgComprehensiveModuleBufferize(
      const LinalgComprehensiveModuleBufferize &p) = default;

  explicit LinalgComprehensiveModuleBufferize(
      const OneShotBufferizationOptions &options)
      : options(options) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<bufferization::BufferizationDialect, linalg::LinalgDialect,
                memref::MemRefDialect, tensor::TensorDialect,
                vector::VectorDialect, scf::SCFDialect,
                arith::ArithmeticDialect, func::FuncDialect, AffineDialect>();
    arith::registerBufferizableOpInterfaceExternalModels(registry);
    bufferization::registerAllocationOpInterfaceExternalModels(registry);
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
    scf::registerBufferizableOpInterfaceExternalModels(registry);
    std_ext::registerModuleBufferizationExternalModels(registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
    vector::registerBufferizableOpInterfaceExternalModels(registry);
  }

private:
  llvm::Optional<OneShotBufferizationOptions> options;
};
} // namespace

static void applyEnablingTransformations(ModuleOp moduleOp) {
  RewritePatternSet patterns(moduleOp.getContext());
  patterns.add<GeneralizePadOpPattern>(moduleOp.getContext());
  (void)applyPatternsAndFoldGreedily(moduleOp, std::move(patterns));
}

static FailureOr<Value> allocationFnUsingAlloca(OpBuilder &b, Location loc,
                                                MemRefType type,
                                                ValueRange dynShape,
                                                unsigned int bufferAlignment) {
  Value allocated = b.create<memref::AllocaOp>(
      loc, type, dynShape, b.getI64IntegerAttr(bufferAlignment));
  return allocated;
}

void LinalgComprehensiveModuleBufferize::runOnOperation() {
  OneShotBufferizationOptions opt;
  if (!options) {
    // Make new bufferization options if none were provided when creating the
    // pass.
    if (useAlloca) {
      opt.allocationFn = allocationFnUsingAlloca;
      opt.deallocationFn = [](OpBuilder &b, Location loc, Value v) {
        return success();
      };
    }
    opt.allowReturnAllocs = allowReturnAllocs;
    opt.allowUnknownOps = allowUnknownOps;
    opt.analysisFuzzerSeed = analysisFuzzerSeed;
    opt.createDeallocs = createDeallocs;
    opt.fullyDynamicLayoutMaps = fullyDynamicLayoutMaps;
    opt.printConflicts = printConflicts;
    opt.testAnalysisOnly = testAnalysisOnly;
    opt.alwaysAliasingWithDest = alwaysAliasingWithDest;
    if (initTensorElimination) {
      opt.addPostAnalysisStep(insertSliceAnchoredInitTensorEliminationStep);
    }
  } else {
    opt = *options;
  }

  ModuleOp moduleOp = getOperation();
  applyEnablingTransformations(moduleOp);

  if (failed(runModuleBufferize(moduleOp, opt))) {
    signalPassFailure();
    return;
  }

  if (opt.testAnalysisOnly)
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

std::unique_ptr<Pass> mlir::createLinalgComprehensiveModuleBufferizePass(
    const OneShotBufferizationOptions &options) {
  return std::make_unique<LinalgComprehensiveModuleBufferize>(options);
}
