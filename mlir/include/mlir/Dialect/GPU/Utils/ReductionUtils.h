//===- ReductionUtils.h - Reduction Utilities -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_TRANSFORMS_REDUCTIONUTILS_H_
#define MLIR_DIALECT_GPU_TRANSFORMS_REDUCTIONUTILS_H_

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

namespace mlir {

struct ClusterInfo {
  unsigned clusterStride;
  unsigned clusterSize;
  unsigned subgroupSize;
};

FailureOr<ClusterInfo> getAndValidateClusterInfo(gpu::SubgroupReduceOp op,
  unsigned subgroupSize);

FailureOr<Value>
createSubgroupDPPReduction(PatternRewriter &rewriter, gpu::SubgroupReduceOp op,
                           Value input, gpu::AllReduceOperation mode,
                           const ClusterInfo &ci, amdgpu::Chipset chipset,
                           function_ref<Value(Value)> packFn,
                           function_ref<Value(Value)> unpackFn);

} // namespace mlir

#endif // MLIR_DIALECT_GPU_TRANSFORMS_REDUCTIONUTILS_H_
