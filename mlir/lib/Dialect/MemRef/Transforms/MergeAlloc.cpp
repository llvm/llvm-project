//===- MergeAlloc.cpp - General framework for merge-allocation ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/Transforms/MergeAllocTickBased.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/StaticMemoryPlanning.h"

#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace memref {
#define GEN_PASS_DEF_MERGEALLOC
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"

namespace {

LogicalResult passDriver(Operation *op,
                         const memref::MergeAllocationOptions &o) {
  BufferViewFlowAnalysis aliasAnaly{op};
  auto tracesOrFail = o.tracer(op, aliasAnaly, o);
  if (failed(tracesOrFail)) {
    return failure();
  }
  if (o.checkOnly) {
    return success();
  }
  for (auto &traces : (*tracesOrFail).scopeTraces) {
    auto schedule = o.planner(op, *traces, o);
    if (failed(schedule)) {
      return failure();
    }
    if (failed(o.mutator(op, traces->getAllocScope(), *schedule, o))) {
      return failure();
    }
  }
  return success();
}

} // namespace
} // namespace memref

using namespace mlir;
namespace {
class MergeAllocPass : public memref::impl::MergeAllocBase<MergeAllocPass> {
  using parent = memref::impl::MergeAllocBase<MergeAllocPass>;
  void runOnOperation() override {
    memref::MergeAllocationOptions opt;
    if (!options) {
      opt.checkOnly = optionAnalysisOnly;
      opt.plannerOptions = plannerOptions;
      opt.alignment = optionAlignment;
      opt.tracer = memref::TickCollecter();
      opt.planner = memref::tickBasedPlanMemory;
      opt.mutator = memref::MergeAllocDefaultMutator();
    } else {
      opt = options.value();
      if (!opt.tracer)
        opt.tracer = memref::TickCollecter();
      if (!opt.planner)
        opt.planner = memref::tickBasedPlanMemory;
      if (!opt.mutator)
        opt.mutator = memref::MergeAllocDefaultMutator();
    }
    if (opt.alignment <= 0) {
      signalPassFailure();
    }
    auto op = getOperation();
    if (failed(memref::passDriver(op, opt))) {
      signalPassFailure();
    }
  }

  std::optional<memref::MergeAllocationOptions> options;

public:
  MergeAllocPass() = default;
  explicit MergeAllocPass(const memref::MergeAllocationOptions &o)
      : options{std::move(o)} {}
};
} // namespace
} // namespace mlir

std::unique_ptr<mlir::Pass>
mlir::memref::createMergeAllocPass(const memref::MergeAllocationOptions &o) {
  return std::make_unique<MergeAllocPass>(o);
}

std::unique_ptr<mlir::Pass> mlir::memref::createMergeAllocPass() {
  return std::make_unique<MergeAllocPass>();
}
