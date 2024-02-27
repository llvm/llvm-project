//===- AMDGPUTransformOps.cpp - Implementation of AMDGPU transform ops-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMDGPU/TransformOps/AMDGPUTransformOps.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/AMDGPU/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;
using namespace mlir::amdgpu;
using namespace mlir::transform;
using namespace mlir::func;

#define DEBUG_TYPE "amdgpu-transforms"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

DiagnosedSilenceableFailure
ApplyOptimizeSharedMemoryReadsAndWritesOp::applyToOne(
    TransformRewriter &rewriter, FuncOp funcOp, ApplyToEachResultList &results,
    TransformState &state) {
  optimizeSharedMemoryReadsAndWritesOp(funcOp);
  return DiagnosedSilenceableFailure::success();
}

void ApplyOptimizeSharedMemoryReadsAndWritesOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTarget(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class AMDGPUTransformDialectExtension
    : public TransformDialectExtension<AMDGPUTransformDialectExtension> {
public:
  AMDGPUTransformDialectExtension() {
    declareGeneratedDialect<arith::ArithDialect>();
    declareGeneratedDialect<affine::AffineDialect>();
    declareGeneratedDialect<amdgpu::AMDGPUDialect>();
    declareGeneratedDialect<vector::VectorDialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/AMDGPU/TransformOps/AMDGPUTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/AMDGPU/TransformOps/AMDGPUTransformOps.cpp.inc"

void amdgpu::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<AMDGPUTransformDialectExtension>();
}
