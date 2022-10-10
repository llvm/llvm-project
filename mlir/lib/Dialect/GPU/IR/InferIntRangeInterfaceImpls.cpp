//===- InferIntRangeInterfaceImpls.cpp - Integer range impls for gpu -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"

using namespace mlir;
using namespace mlir::gpu;

// Maximum grid and block dimensions of all known GPUs are less than 2^32.
static constexpr uint64_t kMaxDim = std::numeric_limits<uint32_t>::max();
// Maximum subgroups are no larger than 128.
static constexpr uint64_t kMaxSubgroupSize = 128;

static ConstantIntRanges getIndexRange(uint64_t umin, uint64_t umax) {
  unsigned width = IndexType::kInternalStorageBitWidth;
  return ConstantIntRanges::fromUnsigned(APInt(width, umin),
                                         APInt(width, umax));
}

void BlockDimOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                   SetIntRangeFn setResultRange) {
  setResultRange(getResult(), getIndexRange(1, kMaxDim));
}

void BlockIdOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                  SetIntRangeFn setResultRange) {
  setResultRange(getResult(), getIndexRange(0, kMaxDim - 1));
}

void GridDimOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                  SetIntRangeFn setResultRange) {
  setResultRange(getResult(), getIndexRange(1, kMaxDim));
}

void ThreadIdOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                   SetIntRangeFn setResultRange) {
  setResultRange(getResult(), getIndexRange(0, kMaxDim - 1));
}

void LaneIdOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                 SetIntRangeFn setResultRange) {
  setResultRange(getResult(), getIndexRange(0, kMaxSubgroupSize - 1));
}

void SubgroupIdOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                     SetIntRangeFn setResultRange) {
  setResultRange(getResult(), getIndexRange(0, kMaxDim - 1));
}

void GlobalIdOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                   SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 getIndexRange(0, std::numeric_limits<int64_t>::max()));
}

void NumSubgroupsOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), getIndexRange(1, kMaxDim));
}

void SubgroupSizeOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), getIndexRange(1, kMaxSubgroupSize));
}

void LaunchOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                 SetIntRangeFn setResultRange) {
  auto setRange = [&](const ConstantIntRanges &argRange, Value dimResult,
                      Value idxResult) {
    if (argRange.umin().getBitWidth() != IndexType::kInternalStorageBitWidth)
      return;
    ConstantIntRanges dimRange =
        argRange.intersection(getIndexRange(1, kMaxDim));
    setResultRange(dimResult, dimRange);
    ConstantIntRanges idxRange =
        getIndexRange(0, dimRange.umax().getZExtValue() - 1);
    setResultRange(idxResult, idxRange);
  };

  argRanges = argRanges.drop_front(getAsyncDependencies().size());
  KernelDim3 gridDims = getGridSize();
  KernelDim3 blockIds = getBlockIds();
  setRange(argRanges[0], gridDims.x, blockIds.x);
  setRange(argRanges[1], gridDims.y, blockIds.y);
  setRange(argRanges[2], gridDims.z, blockIds.z);
  KernelDim3 blockDims = getBlockSize();
  KernelDim3 threadIds = getThreadIds();
  setRange(argRanges[3], blockDims.x, threadIds.x);
  setRange(argRanges[4], blockDims.y, threadIds.y);
  setRange(argRanges[5], blockDims.z, threadIds.z);
}
