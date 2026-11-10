//===- Utils.cpp - GPU transforms utils -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements GPU dialect transforms utils.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Utils/GPUUtils.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::gpu {

vector::CombiningKind convertReductionKind(gpu::AllReduceOperation mode) {
  switch (mode) {
#define MAP_CASE(X)                                                            \
  case gpu::AllReduceOperation::X:                                             \
    return vector::CombiningKind::X

    MAP_CASE(ADD);
    MAP_CASE(MUL);
    MAP_CASE(MINUI);
    MAP_CASE(MINSI);
    MAP_CASE(MINNUMF);
    MAP_CASE(MAXSI);
    MAP_CASE(MAXUI);
    MAP_CASE(MAXNUMF);
    MAP_CASE(AND);
    MAP_CASE(OR);
    MAP_CASE(XOR);
    MAP_CASE(MINIMUMF);
    MAP_CASE(MAXIMUMF);
    MAP_CASE(MINIMUMNUMF);
    MAP_CASE(MAXIMUMNUMF);

#undef MAP_CASE
  }

  llvm_unreachable("Vector and GPU reduction kinds should match 1:1");
}

FailureOr<SubgroupReduceClusterInfo>
getSubgroupReduceClusterInfo(SubgroupReduceOp op, unsigned subgroupSize) {
  assert(llvm::isPowerOf2_32(subgroupSize));

  std::optional<uint32_t> clusterSize = op.getClusterSize();
  assert(!clusterSize ||
         llvm::isPowerOf2_32(*clusterSize)); // Verifier should've caught this.
  if (clusterSize && *clusterSize > subgroupSize)
    return op.emitOpError()
           << "cluster size " << *clusterSize
           << " is greater than subgroup size " << subgroupSize;
  unsigned effectiveClusterSize = clusterSize.value_or(subgroupSize);

  auto clusterStride = op.getClusterStride();
  assert(llvm::isPowerOf2_32(clusterStride)); // Verifier should've caught this.
  if (clusterStride >= subgroupSize)
    return op.emitOpError()
           << "cluster stride " << clusterStride
           << " is not less than subgroup size " << subgroupSize;

  return SubgroupReduceClusterInfo{clusterStride, effectiveClusterSize,
                                   subgroupSize};
}
} // namespace mlir::gpu
