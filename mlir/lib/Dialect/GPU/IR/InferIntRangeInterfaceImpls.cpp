//===- InferIntRangeInterfaceImpls.cpp - Integer range impls for gpu -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include <optional>

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

namespace {
enum class LaunchDims : uint32_t { Block = 0, Grid = 1 };
} // end namespace

/// If the operation `op` is in a context that is annotated with maximum
/// launch dimensions (a launch op with constant block or grid
/// sizes or a launch_func op with the appropriate dimensions), return
/// the bound on the maximum size of the dimension that the op is querying.
/// IDs will be one less than this bound.

static Value valueByDim(KernelDim3 dims, Dimension dim) {
  switch (dim) {
  case Dimension::x:
    return dims.x;
  case Dimension::y:
    return dims.y;
  case Dimension::z:
    return dims.z;
  }
  llvm_unreachable("All dimension enum cases handled above");
}

static uint64_t zext(uint32_t arg) { return static_cast<uint64_t>(arg); }

template <typename Op>
static std::optional<uint64_t> getKnownLaunchDim(Op op, LaunchDims type) {
  Dimension dim = op.getDimension();
  if (auto launch = op->template getParentOfType<LaunchOp>()) {
    KernelDim3 bounds;
    switch (type) {
    case LaunchDims::Block:
      bounds = launch.getBlockSizeOperandValues();
      break;
    case LaunchDims::Grid:
      bounds = launch.getGridSizeOperandValues();
      break;
    }
    Value maybeBound = valueByDim(bounds, dim);
    APInt value;
    if (matchPattern(maybeBound, m_ConstantInt(&value)))
      return value.getZExtValue();
  }

  if (auto func = op->template getParentOfType<GPUFuncOp>()) {
    switch (type) {
    case LaunchDims::Block:
      return llvm::transformOptional(func.getKnownBlockSize(dim), zext);
    case LaunchDims::Grid:
      return llvm::transformOptional(func.getKnownGridSize(dim), zext);
    }
  }
  return std::nullopt;
}

void BlockDimOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                   SetIntRangeFn setResultRange) {
  std::optional<uint64_t> knownVal =
      getKnownLaunchDim(*this, LaunchDims::Block);
  if (knownVal)
    setResultRange(getResult(), getIndexRange(*knownVal, *knownVal));
  else
    setResultRange(getResult(), getIndexRange(1, kMaxDim));
}

void BlockIdOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                  SetIntRangeFn setResultRange) {
  uint64_t max = getKnownLaunchDim(*this, LaunchDims::Grid).value_or(kMaxDim);
  setResultRange(getResult(), getIndexRange(0, max - 1ULL));
}

void GridDimOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                  SetIntRangeFn setResultRange) {
  std::optional<uint64_t> knownVal = getKnownLaunchDim(*this, LaunchDims::Grid);
  if (knownVal)
    setResultRange(getResult(), getIndexRange(*knownVal, *knownVal));
  else
    setResultRange(getResult(), getIndexRange(1, kMaxDim));
}

void ThreadIdOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                   SetIntRangeFn setResultRange) {
  uint64_t max = getKnownLaunchDim(*this, LaunchDims::Block).value_or(kMaxDim);
  setResultRange(getResult(), getIndexRange(0, max - 1ULL));
}

void LaneIdOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                 SetIntRangeFn setResultRange) {
  setResultRange(getResult(), getIndexRange(0, kMaxSubgroupSize - 1ULL));
}

void SubgroupIdOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                     SetIntRangeFn setResultRange) {
  setResultRange(getResult(), getIndexRange(0, kMaxDim - 1ULL));
}

void GlobalIdOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                   SetIntRangeFn setResultRange) {
  uint64_t blockDimMax =
      getKnownLaunchDim(*this, LaunchDims::Block).value_or(kMaxDim);
  uint64_t gridDimMax =
      getKnownLaunchDim(*this, LaunchDims::Grid).value_or(kMaxDim);
  setResultRange(getResult(),
                 getIndexRange(0, (blockDimMax * gridDimMax) - 1ULL));
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
