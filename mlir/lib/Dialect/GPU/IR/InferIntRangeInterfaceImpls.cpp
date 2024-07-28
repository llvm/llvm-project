//===- InferIntRangeInterfaceImpls.cpp - Integer range impls for gpu -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include <optional>

using namespace mlir;
using namespace mlir::gpu;

// Maximum grid and block dimensions of all known GPUs are less than 2^32.
static constexpr uint64_t kMaxDim = std::numeric_limits<uint32_t>::max();
// Maximum cluster size
static constexpr uint64_t kMaxClusterDim = 8;
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

static std::optional<uint64_t>
getKnownLaunchAttr(GPUFuncOp func, LaunchDims dims, Dimension dim) {
  DenseI32ArrayAttr bounds;
  switch (dims) {
  case LaunchDims::Block:
    bounds = func.getKnownBlockSizeAttr();
    break;
  case LaunchDims::Grid:
    bounds = func.getKnownGridSizeAttr();
    break;
  }
  if (!bounds)
    return std::nullopt;
  if (bounds.size() < static_cast<uint32_t>(dim))
    return std::nullopt;
  return zext(bounds[static_cast<uint32_t>(dim)]);
}

static std::optional<uint64_t> getKnownLaunchAttr(FunctionOpInterface func,
                                                  StringRef attrName,
                                                  Dimension dim) {
  auto bounds = func.getOperation()->getAttrOfType<DenseI32ArrayAttr>(attrName);
  if (!bounds)
    return std::nullopt;
  if (bounds.size() < static_cast<uint32_t>(dim))
    return std::nullopt;
  return zext(bounds[static_cast<uint32_t>(dim)]);
}

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

  if (auto gpuFunc = op->template getParentOfType<GPUFuncOp>()) {
    auto inherentAttr = getKnownLaunchAttr(gpuFunc, type, dim);
    if (inherentAttr)
      return inherentAttr;
  }
  if (auto func = op->template getParentOfType<FunctionOpInterface>()) {
    StringRef attrName;
    switch (type) {
    case LaunchDims::Block:
      attrName = GPUDialect::KnownBlockSizeAttrHelper::getNameStr();
      break;
    case LaunchDims::Grid:
      attrName = GPUDialect::KnownGridSizeAttrHelper::getNameStr();
      break;
    }
    auto discardableAttr = getKnownLaunchAttr(func, attrName, dim);
    if (discardableAttr)
      return discardableAttr;
  }
  return std::nullopt;
}

void ClusterDimOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                     SetIntRangeFn setResultRange) {
  uint64_t max = kMaxDim;
  if (auto specified = getUpperBound())
    max = specified->getZExtValue();
  setResultRange(getResult(), getIndexRange(1, max));
}

void ClusterDimBlocksOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                           SetIntRangeFn setResultRange) {
  uint64_t max = kMaxClusterDim;
  if (auto specified = getUpperBound())
    max = specified->getZExtValue();
  setResultRange(getResult(), getIndexRange(1, max));
}

void ClusterIdOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                    SetIntRangeFn setResultRange) {
  uint64_t max = kMaxDim;
  if (auto specified = getUpperBound())
    max = specified->getZExtValue();
  setResultRange(getResult(), getIndexRange(0, max - 1ULL));
}

void ClusterBlockIdOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                         SetIntRangeFn setResultRange) {
  uint64_t max = kMaxClusterDim;
  if (auto specified = getUpperBound())
    max = specified->getZExtValue();
  setResultRange(getResult(), getIndexRange(0, max - 1ULL));
}

void BlockDimOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                   SetIntRangeFn setResultRange) {
  std::optional<uint64_t> knownVal =
      getKnownLaunchDim(*this, LaunchDims::Block);
  if (knownVal)
    return setResultRange(getResult(), getIndexRange(*knownVal, *knownVal));
  ;
  uint64_t max = kMaxDim;
  if (auto specified = getUpperBound())
    max = specified->getZExtValue();
  setResultRange(getResult(), getIndexRange(1, max));
}

void BlockIdOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                  SetIntRangeFn setResultRange) {
  uint64_t max = kMaxDim;
  if (auto fromContext = getKnownLaunchDim(*this, LaunchDims::Grid))
    max = fromContext.value();
  if (auto specified = getUpperBound())
    max = specified->getZExtValue();
  setResultRange(getResult(), getIndexRange(0, max - 1ULL));
}

void GridDimOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                  SetIntRangeFn setResultRange) {
  std::optional<uint64_t> knownVal = getKnownLaunchDim(*this, LaunchDims::Grid);
  if (knownVal)
    return setResultRange(getResult(), getIndexRange(*knownVal, *knownVal));
  uint64_t max = kMaxDim;
  if (auto specified = getUpperBound())
    max = specified->getZExtValue();
  setResultRange(getResult(), getIndexRange(1, max));
}

void ThreadIdOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                   SetIntRangeFn setResultRange) {
  uint64_t max = kMaxDim;
  if (auto fromContext = getKnownLaunchDim(*this, LaunchDims::Block))
    max = fromContext.value();
  if (auto specified = getUpperBound())
    max = specified->getZExtValue();
  setResultRange(getResult(), getIndexRange(0, max - 1ULL));
}

void LaneIdOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                 SetIntRangeFn setResultRange) {
  uint64_t max = kMaxSubgroupSize;
  if (auto specified = getUpperBound())
    max = specified->getZExtValue();
  setResultRange(getResult(), getIndexRange(0, max - 1ULL));
}

void SubgroupIdOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                     SetIntRangeFn setResultRange) {
  uint64_t max = kMaxDim;
  if (auto specified = getUpperBound())
    max = specified->getZExtValue();
  setResultRange(getResult(), getIndexRange(0, max - 1ULL));
}

void GlobalIdOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                   SetIntRangeFn setResultRange) {
  if (auto specified = getUpperBound())
    return setResultRange(getResult(),
                          getIndexRange(0, specified->getZExtValue() - 1ULL));

  uint64_t blockDimMax =
      getKnownLaunchDim(*this, LaunchDims::Block).value_or(kMaxDim);
  uint64_t gridDimMax =
      getKnownLaunchDim(*this, LaunchDims::Grid).value_or(kMaxDim);
  setResultRange(getResult(),
                 getIndexRange(0, (blockDimMax * gridDimMax) - 1ULL));
}

void NumSubgroupsOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                       SetIntRangeFn setResultRange) {
  uint64_t max = kMaxDim;
  if (auto specified = getUpperBound())
    max = specified->getZExtValue();
  setResultRange(getResult(), getIndexRange(1, max));
}

void SubgroupSizeOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                       SetIntRangeFn setResultRange) {
  uint64_t max = kMaxSubgroupSize;
  if (auto specified = getUpperBound())
    max = specified->getZExtValue();
  setResultRange(getResult(), getIndexRange(1, max));
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
