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
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/AffineInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/LinalgInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ModuleBufferization.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SCF/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/StandardOps/Transforms/BufferizableOpInterfaceImpl.h"
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

  LinalgComprehensiveModuleBufferize(bool linalgCopy) {
    this->useLinalgCopy = linalgCopy;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<bufferization::BufferizationDialect, linalg::LinalgDialect,
                memref::MemRefDialect, tensor::TensorDialect,
                vector::VectorDialect, scf::SCFDialect,
                arith::ArithmeticDialect, StandardOpsDialect, AffineDialect>();
    affine_ext::registerBufferizableOpInterfaceExternalModels(registry);
    arith::registerBufferizableOpInterfaceExternalModels(registry);
    linalg_ext::registerBufferizableOpInterfaceExternalModels(registry);
    scf::registerBufferizableOpInterfaceExternalModels(registry);
    std_ext::registerModuleBufferizationExternalModels(registry);
    mlir::registerBufferizableOpInterfaceExternalModels(registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
    vector::registerBufferizableOpInterfaceExternalModels(registry);
  }
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

/// Create a linalg::GenericOp version of an n-D copy that can further tile,
/// lower to loops or vectorize, unlike the current implementation of
/// memref::CopyOp.
/// Do not depend on linalg::CopyOp that is getting deprecated.
static LogicalResult createLinalgCopyOp(OpBuilder &b, Location loc, Value from,
                                        Value to) {
  auto memrefTypeFrom = from.getType().cast<MemRefType>();
  auto memrefTypeTo = to.getType().cast<MemRefType>();
  if (!memrefTypeFrom || !memrefTypeTo ||
      memrefTypeFrom.getRank() != memrefTypeTo.getRank())
    return failure();
  AffineMap id =
      AffineMap::getMultiDimIdentityMap(memrefTypeTo.getRank(), b.getContext());
  SmallVector<StringRef> iteratorTypes(memrefTypeTo.getRank(),
                                       getParallelIteratorTypeName());
  b.create<linalg::GenericOp>(loc,
                              /*inputs=*/from,
                              /*outputs=*/to,
                              /*indexingMaps=*/llvm::makeArrayRef({id, id}),
                              /*iteratorTypes=*/iteratorTypes,
                              [](OpBuilder &b, Location loc, ValueRange args) {
                                b.create<linalg::YieldOp>(loc, args.front());
                              });
  return success();
}

void LinalgComprehensiveModuleBufferize::runOnOperation() {
  auto options = std::make_unique<AnalysisBufferizationOptions>();
  if (useAlloca) {
    options->allocationFn = allocationFnUsingAlloca;
    options->deallocationFn = [](OpBuilder &b, Location loc, Value v) {
      return success();
    };
  }
  // TODO: atm memref::CopyOp can be 200x slower than linalg::GenericOp.
  // Once this perf bug is fixed more systematically, we can revisit.
  if (useLinalgCopy)
    options->memCpyFn = createLinalgCopyOp;

  options->allowReturnMemref = allowReturnMemref;
  options->allowUnknownOps = allowUnknownOps;
  options->analysisFuzzerSeed = analysisFuzzerSeed;
  options->createDeallocs = createDeallocs;
  options->fullyDynamicLayoutMaps = fullyDynamicLayoutMaps;
  options->printConflicts = printConflicts;
  options->testAnalysisOnly = testAnalysisOnly;

  // Enable InitTensorOp elimination.
  if (initTensorElimination) {
    options->addPostAnalysisStep<
        linalg_ext::InsertSliceAnchoredInitTensorEliminationStep>();
  }

  // Only certain scf.for ops are supported by the analysis.
  options->addPostAnalysisStep<scf::AssertScfForAliasingProperties>();

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

std::unique_ptr<Pass>
mlir::createLinalgComprehensiveModuleBufferizePass(bool useLinalgCopy) {
  return std::make_unique<LinalgComprehensiveModuleBufferize>(useLinalgCopy);
}
