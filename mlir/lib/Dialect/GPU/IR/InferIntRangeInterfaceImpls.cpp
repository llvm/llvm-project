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
#include <optional>

using namespace mlir;
using namespace mlir::gpu;

// Maximum grid and block dimensions of all known GPUs are less than 2^32.
static constexpr uint64_t kMaxDim = std::numeric_limits<uint32_t>::max();
// Maximum cluster size.
static constexpr uint64_t kMaxClusterDim = 16;
// Maximum subgroups are no larger than 128.
static constexpr uint64_t kMaxSubgroupSize = 128;

static ConstantIntRanges getIndexRange(uint64_t umin, uint64_t umax) {
  unsigned width = IndexType::kInternalStorageBitWidth;
  return ConstantIntRanges::fromUnsigned(APInt(width, umin),
                                         APInt(width, umax));
}

static uint64_t zext(uint32_t arg) { return static_cast<uint64_t>(arg); }

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

static std::optional<uint32_t>
getKnownLaunchAttr(GPUFuncOp func, DimensionKind dims, Dimension dim) {
  DenseI32ArrayAttr bounds;
  switch (dims) {
  case DimensionKind::Other:
    return std::nullopt;
  case DimensionKind::Block:
    bounds = func.getKnownBlockSizeAttr();
    break;
  case DimensionKind::Grid:
    bounds = func.getKnownGridSizeAttr();
    break;
  case DimensionKind::Cluster:
    bounds = func.getKnownClusterSizeAttr();
    break;
  }
  if (!bounds)
    return std::nullopt;
  if (bounds.size() <= static_cast<uint32_t>(dim))
    return std::nullopt;
  return bounds[static_cast<uint32_t>(dim)];
}

static std::optional<uint32_t> getKnownLaunchAttr(FunctionOpInterface func,
                                                  StringRef attrName,
                                                  Dimension dim) {
  auto bounds = func.getOperation()->getAttrOfType<DenseI32ArrayAttr>(attrName);
  if (!bounds)
    return std::nullopt;
  if (bounds.size() <= static_cast<uint32_t>(dim))
    return std::nullopt;
  return bounds[static_cast<uint32_t>(dim)];
}

std::optional<uint32_t>
mlir::gpu::getKnownDimensionSizeAround(Operation *op, DimensionKind kind,
                                       Dimension dim) {
  if (auto launch = op->getParentOfType<LaunchOp>()) {
    KernelDim3 bounds;
    switch (kind) {
    case DimensionKind::Other:
      return std::nullopt;
    case DimensionKind::Block:
      bounds = launch.getBlockSizeOperandValues();
      break;
    case DimensionKind::Grid:
      bounds = launch.getGridSizeOperandValues();
      break;
    case DimensionKind::Cluster:
      if (launch.hasClusterSize()) {
        auto clusterBounds = launch.getClusterSizeOperandValues();
        if (clusterBounds)
          bounds = *clusterBounds;
      }
      break;
    }
    Value maybeBound = valueByDim(bounds, dim);
    APInt value;
    if (maybeBound && matchPattern(maybeBound, m_ConstantInt(&value)))
      return value.getZExtValue();
  }

  if (auto gpuFunc = op->getParentOfType<GPUFuncOp>()) {
    auto inherentAttr = getKnownLaunchAttr(gpuFunc, kind, dim);
    if (inherentAttr)
      return inherentAttr;
  }
  if (auto func = op->getParentOfType<FunctionOpInterface>()) {
    StringRef attrName;
    switch (kind) {
    case DimensionKind::Other:
      return std::nullopt;
    case DimensionKind::Block:
      attrName = GPUDialect::KnownBlockSizeAttrHelper::getNameStr();
      break;
    case DimensionKind::Grid:
      attrName = GPUDialect::KnownGridSizeAttrHelper::getNameStr();
      break;
    case DimensionKind::Cluster:
      attrName = GPUDialect::KnownClusterSizeAttrHelper::getNameStr();
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
  if (auto known = getKnownDimensionSizeAround(*this, DimensionKind::Cluster,
                                               getDimension()))
    return setResultRange(getResult(),
                          getIndexRange(zext(*known), zext(*known)));

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
  if (auto known = getKnownDimensionSizeAround(*this, DimensionKind::Cluster,
                                               getDimension()))
    max = zext(*known);
  if (auto specified = getUpperBound())
    max = specified->getZExtValue();
  setResultRange(getResult(), getIndexRange(0, max - 1ULL));
}

void BlockDimOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                   SetIntRangeFn setResultRange) {
  std::optional<uint32_t> knownVal =
      getKnownDimensionSizeAround(*this, DimensionKind::Block, getDimension());
  if (knownVal)
    return setResultRange(getResult(),
                          getIndexRange(zext(*knownVal), zext(*knownVal)));

  uint64_t max = kMaxDim;
  if (auto specified = getUpperBound())
    max = specified->getZExtValue();
  setResultRange(getResult(), getIndexRange(1, max));
}

void BlockIdOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                  SetIntRangeFn setResultRange) {
  uint64_t max = kMaxDim;
  if (auto fromContext = getKnownDimensionSizeAround(*this, DimensionKind::Grid,
                                                     getDimension()))
    max = zext(*fromContext);
  if (auto specified = getUpperBound())
    max = specified->getZExtValue();
  setResultRange(getResult(), getIndexRange(0, max - 1ULL));
}

void GridDimOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                  SetIntRangeFn setResultRange) {
  std::optional<uint32_t> knownVal =
      getKnownDimensionSizeAround(*this, DimensionKind::Grid, getDimension());
  if (knownVal)
    return setResultRange(getResult(),
                          getIndexRange(zext(*knownVal), zext(*knownVal)));
  uint64_t max = kMaxDim;
  if (auto specified = getUpperBound())
    max = specified->getZExtValue();
  setResultRange(getResult(), getIndexRange(1, max));
}

void ThreadIdOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                   SetIntRangeFn setResultRange) {
  uint64_t max = kMaxDim;
  if (auto fromContext = getKnownDimensionSizeAround(
          *this, DimensionKind::Block, getDimension()))
    max = zext(*fromContext);
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

  uint64_t blockDimMax = zext(
      getKnownDimensionSizeAround(*this, DimensionKind::Block, getDimension())
          .value_or(kMaxDim));
  uint64_t gridDimMax = zext(
      getKnownDimensionSizeAround(*this, DimensionKind::Grid, getDimension())
          .value_or(kMaxDim));
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
