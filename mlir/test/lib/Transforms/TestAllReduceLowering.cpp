//===- TestAllReduceLowering.cpp - Test gpu.all_reduce lowering -----------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains test passes for lowering the gpu.all_reduce op.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct TestAllReduceLoweringPass
    : public ModulePass<TestAllReduceLoweringPass> {
  void runOnModule() override {
    OwningRewritePatternList patterns;
    populateGpuRewritePatterns(&getContext(), patterns);
    applyPatternsGreedily(getModule(), patterns);
  }
};
} // namespace

static PassRegistration<TestAllReduceLoweringPass>
    pass("test-all-reduce-lowering",
         "Lowers gpu.all-reduce ops within the GPU dialect.");
