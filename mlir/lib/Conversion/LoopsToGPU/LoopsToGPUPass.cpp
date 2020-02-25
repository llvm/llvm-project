//===- LoopsToGPUPass.cpp - Convert a loop nest to a GPU kernel -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LoopsToGPU/LoopsToGPUPass.h"
#include "mlir/Conversion/LoopsToGPU/LoopsToGPU.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/CommandLine.h"

#define PASS_NAME "convert-loops-to-gpu"
#define LOOPOP_TO_GPU_PASS_NAME "convert-loop-op-to-gpu"

using namespace mlir;
using namespace mlir::loop;

static llvm::cl::OptionCategory clOptionsCategory(PASS_NAME " options");
static llvm::cl::opt<unsigned>
    clNumBlockDims("gpu-block-dims",
                   llvm::cl::desc("Number of GPU block dimensions for mapping"),
                   llvm::cl::cat(clOptionsCategory), llvm::cl::init(1u));
static llvm::cl::opt<unsigned> clNumThreadDims(
    "gpu-thread-dims",
    llvm::cl::desc("Number of GPU thread dimensions for mapping"),
    llvm::cl::cat(clOptionsCategory), llvm::cl::init(1u));

static llvm::cl::OptionCategory clLoopOpToGPUCategory(LOOPOP_TO_GPU_PASS_NAME
                                                      " options");
static llvm::cl::list<unsigned>
    clNumWorkGroups("gpu-num-workgroups",
                    llvm::cl::desc("Num workgroups in the GPU launch"),
                    llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated,
                    llvm::cl::cat(clLoopOpToGPUCategory));
static llvm::cl::list<unsigned>
    clWorkGroupSize("gpu-workgroup-size",
                    llvm::cl::desc("Workgroup Size in the GPU launch"),
                    llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated,
                    llvm::cl::cat(clLoopOpToGPUCategory));

namespace {
// A pass that traverses top-level loops in the function and converts them to
// GPU launch operations.  Nested launches are not allowed, so this does not
// walk the function recursively to avoid considering nested loops.
struct ForLoopMapper : public FunctionPass<ForLoopMapper> {
  ForLoopMapper(unsigned numBlockDims, unsigned numThreadDims)
      : numBlockDims(numBlockDims), numThreadDims(numThreadDims) {}

  void runOnFunction() override {
    for (Block &block : getFunction())
      for (Operation &op : llvm::make_early_inc_range(block)) {
        if (auto forOp = dyn_cast<AffineForOp>(&op)) {
          if (failed(convertAffineLoopNestToGPULaunch(forOp, numBlockDims,
                                                      numThreadDims)))
            signalPassFailure();
        } else if (auto forOp = dyn_cast<ForOp>(&op)) {
          if (failed(convertLoopNestToGPULaunch(forOp, numBlockDims,
                                                numThreadDims)))
            signalPassFailure();
        }
      }
  }

  unsigned numBlockDims;
  unsigned numThreadDims;
};

// A pass that traverses top-level loops in the function and convertes them to
// GPU launch operations. The top-level loops itself does not have to be
// perfectly nested. The only requirement is that there be as many perfectly
// nested loops as the size of `numWorkGroups`. Within these any loop nest has
// to be perfectly nested upto depth equal to size of `workGroupSize`.
struct ImperfectlyNestedForLoopMapper
    : public FunctionPass<ImperfectlyNestedForLoopMapper> {
  ImperfectlyNestedForLoopMapper(ArrayRef<int64_t> numWorkGroups,
                                 ArrayRef<int64_t> workGroupSize)
      : numWorkGroups(numWorkGroups.begin(), numWorkGroups.end()),
        workGroupSize(workGroupSize.begin(), workGroupSize.end()) {}

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
    for (Block &block : getFunction()) {
      for (Operation &op : llvm::make_early_inc_range(block)) {
        if (auto forOp = dyn_cast<ForOp>(&op)) {
          if (failed(convertLoopToGPULaunch(forOp, numWorkGroupsVal,
                                            workGroupSizeVal))) {
            return signalPassFailure();
          }
        }
      }
    }
  }
  SmallVector<int64_t, 3> numWorkGroups;
  SmallVector<int64_t, 3> workGroupSize;
};

struct ParallelLoopToGpuPass : public OperationPass<ParallelLoopToGpuPass> {
  void runOnOperation() override {
    OwningRewritePatternList patterns;
    populateParallelLoopToGPUPatterns(patterns, &getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalDialect<AffineOpsDialect>();
    target.addLegalDialect<gpu::GPUDialect>();
    target.addLegalDialect<loop::LoopOpsDialect>();
    target.addIllegalOp<loop::ParallelOp>();
    if (failed(applyPartialConversion(getOperation(), target, patterns)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>>
mlir::createSimpleLoopsToGPUPass(unsigned numBlockDims,
                                 unsigned numThreadDims) {
  return std::make_unique<ForLoopMapper>(numBlockDims, numThreadDims);
}

std::unique_ptr<OpPassBase<FuncOp>>
mlir::createLoopToGPUPass(ArrayRef<int64_t> numWorkGroups,
                          ArrayRef<int64_t> workGroupSize) {
  return std::make_unique<ImperfectlyNestedForLoopMapper>(numWorkGroups,
                                                          workGroupSize);
}

std::unique_ptr<Pass> mlir::createParallelLoopToGpuPass() {
  return std::make_unique<ParallelLoopToGpuPass>();
}

static PassRegistration<ForLoopMapper>
    registration(PASS_NAME, "Convert top-level loops to GPU kernels", [] {
      return std::make_unique<ForLoopMapper>(clNumBlockDims.getValue(),
                                             clNumThreadDims.getValue());
    });

static PassRegistration<ImperfectlyNestedForLoopMapper> loopOpToGPU(
    LOOPOP_TO_GPU_PASS_NAME, "Convert top-level loop::ForOp to GPU kernels",
    [] {
      SmallVector<int64_t, 3> numWorkGroups, workGroupSize;
      numWorkGroups.assign(clNumWorkGroups.begin(), clNumWorkGroups.end());
      workGroupSize.assign(clWorkGroupSize.begin(), clWorkGroupSize.end());
      return std::make_unique<ImperfectlyNestedForLoopMapper>(numWorkGroups,
                                                              workGroupSize);
    });

static PassRegistration<ParallelLoopToGpuPass>
    pass("convert-parallel-loops-to-gpu", "Convert mapped loop.parallel ops"
                                          " to gpu launch operations.");
