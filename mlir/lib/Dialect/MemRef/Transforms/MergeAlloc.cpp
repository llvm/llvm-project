//===- MergeAlloc.cpp - Calling convention conversion ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/Transforms/MergeAlloc.h"
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

LogicalResult passDriver(Operation *op, const memref::MergeAllocOptions &o,
                         TraceCollectorFunc tracer, MemoryPlannerFunc planner,
                         MemoryMergeMutatorFunc mutator) {
  BufferViewFlowAnalysis aliasAnaly{op};
  auto tracesOrFail = tracer(op, aliasAnaly, o);
  if (failed(tracesOrFail)) {
    return failure();
  }
  if (o.optionCheck) {
    return success();
  }
  for (auto &[scope, traces] : (*tracesOrFail).scopeToTraces) {
    auto schedule = planner(op, *traces, o);
    if (failed(schedule)) {
      return failure();
    }
    if (failed(mutator(op, scope, *schedule, o))) {
      return failure();
    }
  }
  return success();
}

} // namespace
} // namespace memref

using namespace mlir;
struct MergeAllocPass : memref::impl::MergeAllocBase<MergeAllocPass> {
  using parent = memref::impl::MergeAllocBase<MergeAllocPass>;
  void runOnOperation() override {
    auto op = getOperation();
    if (failed(memref::passDriver(
            op, memref::MergeAllocOptions{optionCheck, optionNoLocality},
            memref::tickBasedCollectMemoryTrace, memref::tickBasedPlanMemory,
            memref::tickBasedMutateAllocations))) {
        signalPassFailure();
    }
  }

public:
  MergeAllocPass(const memref::MergeAllocOptions &o) : parent{o} {}
};
} // namespace mlir

std::unique_ptr<mlir::Pass>
mlir::memref::createMergeAllocPass(const memref::MergeAllocOptions &o) {
  return std::make_unique<MergeAllocPass>(o);
}
