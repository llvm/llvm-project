//===- ComprehensiveBufferize.cpp - Single pass bufferization -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ComprehensiveBufferize.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {
struct LinalgComprehensiveModuleBufferize
    : public LinalgComprehensiveModuleBufferizeBase<
          LinalgComprehensiveModuleBufferize> {
  LinalgComprehensiveModuleBufferize() {}

  LinalgComprehensiveModuleBufferize(
      const LinalgComprehensiveModuleBufferize &p) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, memref::MemRefDialect,
                tensor::TensorDialect, vector::VectorDialect, scf::SCFDialect,
                arith::ArithmeticDialect, StandardOpsDialect>();
    registerBufferizableOpInterfaceExternalModels(registry);
  }
};
} // end namespace

static void applyEnablingTransformations(ModuleOp moduleOp) {
  RewritePatternSet patterns(moduleOp.getContext());
  patterns.add<GeneralizePadTensorOpPattern>(moduleOp.getContext());
  (void)applyPatternsAndFoldGreedily(moduleOp, std::move(patterns));
}

static Optional<Value>
allocationFnUsingAlloca(OpBuilder &b, Location loc, MemRefType type,
                        const SmallVector<Value> &dynShape) {
  Value allocated = b.create<memref::AllocaOp>(
      loc, type, dynShape, b.getI64IntegerAttr(kBufferAlignments));
  return allocated;
}

void LinalgComprehensiveModuleBufferize::runOnOperation() {
  BufferizationOptions options;
  if (useAlloca) {
    options.allocationFns->allocationFn = allocationFnUsingAlloca;
    options.allocationFns->deallocationFn = [](OpBuilder &b, Location loc,
                                               Value v) {};
  }
  options.allowReturnMemref = allowReturnMemref;
  options.analysisFuzzerSeed = analysisFuzzerSeed;
  options.testAnalysisOnly = testAnalysisOnly;

  ModuleOp moduleOp = getOperation();
  applyEnablingTransformations(moduleOp);

  if (failed(runComprehensiveBufferize(moduleOp, options))) {
    signalPassFailure();
    return;
  }

  if (options.testAnalysisOnly)
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
