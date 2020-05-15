//===- SCFToGPUPass.cpp - Convert a loop nest to a GPU kernel -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "../PassDetail.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPU.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/CommandLine.h"

#define PASS_NAME "convert-scf-to-gpu"
#define LOOPOP_TO_GPU_PASS_NAME "convert-loop-op-to-gpu"

using namespace mlir;
using namespace mlir::scf;

namespace {
// A pass that traverses top-level loops in the function and converts them to
// GPU launch operations.  Nested launches are not allowed, so this does not
// walk the function recursively to avoid considering nested loops.
struct ForLoopMapper : public ConvertSimpleSCFToGPUBase<ForLoopMapper> {
  ForLoopMapper() = default;
  ForLoopMapper(unsigned numBlockDims, unsigned numThreadDims) {
    this->numBlockDims = numBlockDims;
    this->numThreadDims = numThreadDims;
  }

  void runOnFunction() override {
    for (Operation &op : llvm::make_early_inc_range(getFunction().getOps())) {
      if (auto forOp = dyn_cast<AffineForOp>(&op)) {
        if (failed(convertAffineLoopNestToGPULaunch(forOp, numBlockDims,
                                                    numThreadDims)))
          signalPassFailure();
      } else if (auto forOp = dyn_cast<ForOp>(&op)) {
        if (failed(
                convertLoopNestToGPULaunch(forOp, numBlockDims, numThreadDims)))
          signalPassFailure();
      }
    }
  }
};

// A pass that traverses top-level loops in the function and convertes them to
// GPU launch operations. The top-level loops itself does not have to be
// perfectly nested. The only requirement is that there be as many perfectly
// nested loops as the size of `numWorkGroups`. Within these any loop nest has
// to be perfectly nested upto depth equal to size of `workGroupSize`.
struct ImperfectlyNestedForLoopMapper
    : public ConvertSCFToGPUBase<ImperfectlyNestedForLoopMapper> {
  ImperfectlyNestedForLoopMapper() = default;
  ImperfectlyNestedForLoopMapper(ArrayRef<int64_t> numWorkGroups,
                                 ArrayRef<int64_t> workGroupSize) {
    this->numWorkGroups = numWorkGroups;
    this->workGroupSize = workGroupSize;
  }

  void runOnFunction() override {
    // Insert the num work groups and workgroup sizes as constant values. This
    // pass is only used for testing.
    FuncOp funcOp = getFunction();
    OpBuilder builder(funcOp.getOperation()->getRegion(0));
    SmallVector<Value, 3> numWorkGroupsVal, workGroupSizeVal;
    for (auto val : numWorkGroups) {
      auto constOp = builder.create<ConstantOp>(
          funcOp.getLoc(), builder.getIntegerAttr(builder.getIndexType(), val));
      numWorkGroupsVal.push_back(constOp);
    }
    for (auto val : workGroupSize) {
      auto constOp = builder.create<ConstantOp>(
          funcOp.getLoc(), builder.getIntegerAttr(builder.getIndexType(), val));
      workGroupSizeVal.push_back(constOp);
    }
    for (ForOp forOp : llvm::make_early_inc_range(funcOp.getOps<ForOp>())) {
      if (failed(convertLoopToGPULaunch(forOp, numWorkGroupsVal,
                                        workGroupSizeVal))) {
        return signalPassFailure();
      }
    }
  }
};

struct ParallelLoopToGpuPass
    : public ConvertParallelLoopToGpuBase<ParallelLoopToGpuPass> {
  void runOnOperation() override {
    OwningRewritePatternList patterns;
    populateParallelLoopToGPUPatterns(patterns, &getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalDialect<AffineDialect>();
    target.addLegalDialect<gpu::GPUDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addIllegalOp<scf::ParallelOp>();
    if (failed(applyPartialConversion(getOperation(), target, patterns)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createSimpleSCFToGPUPass(unsigned numBlockDims, unsigned numThreadDims) {
  return std::make_unique<ForLoopMapper>(numBlockDims, numThreadDims);
}
std::unique_ptr<OperationPass<FuncOp>> mlir::createSimpleSCFToGPUPass() {
  return std::make_unique<ForLoopMapper>();
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::createLoopToGPUPass(ArrayRef<int64_t> numWorkGroups,
                          ArrayRef<int64_t> workGroupSize) {
  return std::make_unique<ImperfectlyNestedForLoopMapper>(numWorkGroups,
                                                          workGroupSize);
}
std::unique_ptr<OperationPass<FuncOp>> mlir::createLoopToGPUPass() {
  return std::make_unique<ImperfectlyNestedForLoopMapper>();
}

std::unique_ptr<Pass> mlir::createParallelLoopToGpuPass() {
  return std::make_unique<ParallelLoopToGpuPass>();
}
