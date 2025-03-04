//===- TestXeGPUTransforms.cpp - Test XeGPU transforms and lowerings ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::xegpu;
using namespace mlir::vector;

namespace {
struct TestXeGPUSubgroupDistribution
    : public PassWrapper<TestXeGPUSubgroupDistribution,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestXeGPUSubgroupDistribution)

  TestXeGPUSubgroupDistribution() = default;
  TestXeGPUSubgroupDistribution(const TestXeGPUSubgroupDistribution &pass) =
      default;

  StringRef getArgument() const final {
    return "test-xegpu-subgroup-distribute";
  }
  StringRef getDescription() const final {
    return "Test lowering patterns for distributing XeGPU ops to work items";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<xegpu::XeGPUDialect>();
    registry.insert<vector::VectorDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateXeGPUSubgroupDistributePatterns(patterns);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestXeGPUTransforms() {
  PassRegistration<TestXeGPUSubgroupDistribution>();
}
} // namespace test
} // namespace mlir
