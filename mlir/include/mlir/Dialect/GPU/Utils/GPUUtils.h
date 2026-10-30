//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines utility functions exposed by the GPU dialect
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_TRANSFORMS_UTILS_H_
#define MLIR_DIALECT_GPU_TRANSFORMS_UTILS_H_

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Support/LLVM.h"

#include <string>

namespace mlir {
class Operation;
class Value;

namespace gpu {
class GPUFuncOp;
class LaunchOp;

/// Returns the matching vector combining kind.
vector::CombiningKind convertReductionKind(gpu::AllReduceOperation mode);

/// Lane layout of the clusters reduced by a `gpu.subgroup_reduce` op.
struct SubgroupReduceClusterInfo {
  unsigned clusterStride;
  unsigned clusterSize;
  unsigned subgroupSize;
};

/// Returns the cluster layout of `op` in a subgroup of `subgroupSize` lanes,
/// which must be a power of two. Emits an op error and fails if the cluster
/// size exceeds the subgroup size or the cluster stride is not less than it.
FailureOr<SubgroupReduceClusterInfo>
getSubgroupReduceClusterInfo(SubgroupReduceOp op, unsigned subgroupSize);
} // namespace gpu

/// Get a gpu.func created from outlining the region of a gpu.launch op with the
/// given `kernelFnName`. The region of the `launchOp` can use values from
/// above. These need to be captured and passed as arguments to the generated
/// gpu.func. The generated function has arguments
/// - corresponding to the values passed in as `operands`, in that order.
/// - any additional values that might be used within the region of the
///   `launchOp` and defined above it. These captured values are appended to the
///   `operands` list.
gpu::GPUFuncOp outlineKernelFunc(gpu::LaunchOp launchOp, StringRef kernelFnName,
                                 SmallVectorImpl<Value> &operands);

/// Sink operations into the `launchOp` to reduce the number of values that are
/// used within the region of the operation, but defined outside of the
/// region.
LogicalResult sinkOperationsIntoLaunchOp(
    gpu::LaunchOp launchOp,
    llvm::function_ref<bool(Operation *)> isSinkingBeneficiary);

} // namespace mlir
#endif // MLIR_DIALECT_GPU_TRANSFORMS_UTILS_H_
