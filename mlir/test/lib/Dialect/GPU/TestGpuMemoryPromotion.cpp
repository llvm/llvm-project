//===- TestGPUMemoryPromotionPass.cpp - Test pass for GPU promotion -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the pass testing the utilities for moving data across
// different levels of the GPU memory hierarchy.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/MemoryPromotion.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// Simple pass for testing the promotion to workgroup memory in GPU functions.
/// Promotes all arguments with "gpu.test_promote_workgroup" attribute. This
/// does not check whether the promotion is legal (e.g., amount of memory used)
/// or beneficial (e.g., makes previously uncoalesced loads coalesced).
class TestGpuMemoryPromotionPass
    : public PassWrapper<TestGpuMemoryPromotionPass,
                         OperationPass<gpu::GPUFuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect, scf::SCFDialect>();
  }
  StringRef getArgument() const final { return "test-gpu-memory-promotion"; }
  StringRef getDescription() const final {
    return "Promotes the annotated arguments of gpu.func to workgroup memory.";
  }

  void runOnOperation() override {
    gpu::GPUFuncOp op = getOperation();
    for (unsigned i = 0, e = op.getNumArguments(); i < e; ++i) {
      if (op.getArgAttrOfType<UnitAttr>(i, "gpu.test_promote_workgroup"))
        promoteToWorkgroupMemory(op, i);
    }
  }
};
} // namespace

namespace mlir {
void registerTestGpuMemoryPromotionPass() {
  PassRegistration<TestGpuMemoryPromotionPass>();
}
} // namespace mlir
