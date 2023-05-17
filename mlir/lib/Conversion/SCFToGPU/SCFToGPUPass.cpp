//===- SCFToGPUPass.cpp - Convert a loop nest to a GPU kernel -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"

#include "mlir/Conversion/SCFToGPU/SCFToGPU.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/CommandLine.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTAFFINEFORTOGPU
#define GEN_PASS_DEF_CONVERTPARALLELLOOPTOGPU
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;

namespace {
// A pass that traverses top-level loops in the function and converts them to
// GPU launch operations.  Nested launches are not allowed, so this does not
// walk the function recursively to avoid considering nested loops.
struct ForLoopMapper : public impl::ConvertAffineForToGPUBase<ForLoopMapper> {
  ForLoopMapper() = default;
  ForLoopMapper(unsigned numBlockDims, unsigned numThreadDims) {
    this->numBlockDims = numBlockDims;
    this->numThreadDims = numThreadDims;
  }

  void runOnOperation() override {
    for (Operation &op : llvm::make_early_inc_range(
             getOperation().getFunctionBody().getOps())) {
      if (auto forOp = dyn_cast<affine::AffineForOp>(&op)) {
        if (failed(convertAffineLoopNestToGPULaunch(forOp, numBlockDims,
                                                    numThreadDims)))
          signalPassFailure();
      }
    }
  }
};

struct ParallelLoopToGpuPass
    : public impl::ConvertParallelLoopToGpuBase<ParallelLoopToGpuPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateParallelLoopToGPUPatterns(patterns);
    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    configureParallelLoopToGPULegality(target);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
    finalizeParallelLoopToGPUConversion(getOperation());
  }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
mlir::createAffineForToGPUPass(unsigned numBlockDims, unsigned numThreadDims) {
  return std::make_unique<ForLoopMapper>(numBlockDims, numThreadDims);
}
std::unique_ptr<InterfacePass<FunctionOpInterface>>
mlir::createAffineForToGPUPass() {
  return std::make_unique<ForLoopMapper>();
}

std::unique_ptr<Pass> mlir::createParallelLoopToGpuPass() {
  return std::make_unique<ParallelLoopToGpuPass>();
}
