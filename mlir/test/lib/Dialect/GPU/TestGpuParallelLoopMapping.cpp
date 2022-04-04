//===- TestGPUParallelLoopMapping.cpp - Test pass for GPU loop mapping ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the pass testing the utilities for mapping parallel
// loops to gpu hardware ids.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/ParallelLoopMapper.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// Simple pass for testing the mapping of parallel loops to hardware ids using
/// a greedy mapping strategy.
struct TestGpuGreedyParallelLoopMappingPass
    : public PassWrapper<TestGpuGreedyParallelLoopMappingPass,
                         OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestGpuGreedyParallelLoopMappingPass)

  StringRef getArgument() const final {
    return "test-gpu-greedy-parallel-loop-mapping";
  }
  StringRef getDescription() const final {
    return "Greedily maps all parallel loops to gpu hardware ids.";
  }
  void runOnOperation() override {
    for (Region &region : getOperation()->getRegions())
      greedilyMapParallelSCFToGPU(region);
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestGpuParallelLoopMappingPass() {
  PassRegistration<TestGpuGreedyParallelLoopMappingPass>();
}
} // namespace test
} // namespace mlir
