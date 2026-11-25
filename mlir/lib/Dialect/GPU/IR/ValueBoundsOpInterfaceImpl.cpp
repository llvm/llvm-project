//===- ValueBoundsOpInterfaceImpl.cpp - Impl. of ValueBoundsOpInterface ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/ValueBoundsOpInterfaceImpl.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;
using namespace mlir::gpu;

namespace {
static std::optional<unsigned> getIndexWidth(Operation *op, Type indexType) {
  DataLayout layout = DataLayout::closest(op);
  if (!layout.hasExactTypeEntry(indexType))
    return std::nullopt;
  return layout.getTypeSizeInBits(indexType).getFixedValue();
}

/// Implement ValueBoundsOpInterface (which only works on index-typed values,
/// gathers a set of constraint expressions, and is used for affine analyses)
/// in terms of InferIntRangeInterface (which works
/// on arbitrary integer types, creates [min, max] ranges, and is used in for
/// arithmetic simplification).
template <typename Op>
struct GpuIdOpInterface
    : public ValueBoundsOpInterface::ExternalModel<GpuIdOpInterface<Op>, Op> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto inferrable = cast<InferIntRangeInterface>(op);
    assert(value == op->getResult(0) &&
           "inferring for value that isn't the GPU op's result");
    auto translateConstraint = [&](Value v, const ConstantIntRanges &range) {
      assert(v == value &&
             "GPU ID op inferring values for something that's not its result");
      // The constraint set stores signed 64-bit constants. Its inclusive
      // upper bound must also fit after conversion to an exclusive bound.
      if (range.smin().isSignedIntN(64) &&
          range.smin().sgt(std::numeric_limits<int64_t>::min()))
        cstr.bound(v) >= range.smin().getSExtValue();
      if (range.smax().isSignedIntN(64) &&
          range.smax().slt(std::numeric_limits<int64_t>::max()))
        cstr.bound(v) <= range.smax().getSExtValue();
    };
    assert(inferrable->getNumOperands() == 0 && "ID ops have no operands");
    inferrable.inferResultRanges({}, getIndexWidth(op, value.getType()),
                                 translateConstraint);
  }
};

/// Implement ValueBoundsOpInterface on subgroup broadcast operations to
/// indicate that such a broadcast does not modify the ranges of the values in
/// question. Handles shaped types just in case one wants to broadcast a memref
/// descriptor.
struct SubgroupBroadcastOpInterface
    : public ValueBoundsOpInterface::ExternalModel<SubgroupBroadcastOpInterface,
                                                   SubgroupBroadcastOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto broadcastOp = cast<SubgroupBroadcastOp>(op);
    assert(value == broadcastOp.getResult() && "invalid value");
    cstr.bound(value) == cstr.getExpr(broadcastOp.getSrc());
  }

  void populateBoundsForShapedValueDim(Operation *op, Value value, int64_t dim,
                                       ValueBoundsConstraintSet &cstr) const {
    auto broadcastOp = cast<SubgroupBroadcastOp>(op);
    assert(value == broadcastOp.getResult() && "invalid value");
    cstr.bound(value)[dim] == cstr.getExpr(broadcastOp.getSrc(), dim);
  }
};

struct GpuLaunchOpInterface
    : public ValueBoundsOpInterface::ExternalModel<GpuLaunchOpInterface,
                                                   LaunchOp> {
  static AffineExpr getPositiveDimension(Value dim, unsigned width,
                                         ValueBoundsConstraintSet &cstr) {
    AffineExpr expr = cstr.getExpr(dim);
    if (auto constant = dyn_cast<AffineConstantExpr>(expr)) {
      APInt size = APInt(64, constant.getValue()).sextOrTrunc(width);
      if (!size.isStrictlyPositive())
        return {};
      return cstr.getExpr(size.getSExtValue());
    }
    // Launch dimensions are positive and fit in 32 unsigned bits. At 64
    // bits or more, every valid dimension fits in the signed index range.
    if (width >= 64)
      return expr;
    if (!ValueBoundsConstraintSet::isProvablyPositive(dim, cstr))
      return {};
    int64_t max = APInt::getSignedMaxValue(width).getSExtValue();
    auto maxAttr = Builder(dim.getContext()).getIndexAttr(max);
    if (!cstr.populateAndCompare(dim, ValueBoundsConstraintSet::LE,
                                 OpFoldResult(maxAttr)))
      return {};
    return expr;
  }

  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto launchOp = cast<LaunchOp>(op);

    Value sizeArg = nullptr;
    bool isSize = false;
    KernelDim3 gridSizeArgs = launchOp.getGridSizeOperandValues();
    KernelDim3 blockSizeArgs = launchOp.getBlockSizeOperandValues();
    std::optional<unsigned> width = getIndexWidth(op, value.getType());
    if (!width)
      return;

    auto match = [&](KernelDim3 bodyArgs, KernelDim3 externalArgs,
                     bool areSizeArgs) {
      if (value == bodyArgs.x) {
        sizeArg = externalArgs.x;
        isSize = areSizeArgs;
      }
      if (value == bodyArgs.y) {
        sizeArg = externalArgs.y;
        isSize = areSizeArgs;
      }
      if (value == bodyArgs.z) {
        sizeArg = externalArgs.z;
        isSize = areSizeArgs;
      }
    };
    match(launchOp.getThreadIds(), blockSizeArgs, false);
    match(launchOp.getBlockSize(), blockSizeArgs, true);
    match(launchOp.getBlockIds(), gridSizeArgs, false);
    match(launchOp.getGridSize(), gridSizeArgs, true);
    if (auto clusterIds = launchOp.getClusterIds()) {
      KernelDim3 counts = *launchOp.getClusterSize();
      KernelDim3 blocksPerCluster = *launchOp.getClusterSizeOperandValues();
      for (auto axis : {&KernelDim3::x, &KernelDim3::y, &KernelDim3::z}) {
        bool isCount = value == counts.*axis;
        if (!isCount && value != (*clusterIds).*axis)
          continue;
        AffineExpr grid =
            getPositiveDimension(gridSizeArgs.*axis, *width, cstr);
        AffineExpr cluster =
            getPositiveDimension(blocksPerCluster.*axis, *width, cstr);
        if (!grid || !cluster)
          return;
        AffineExpr count = grid.floorDiv(cluster);
        if (isCount)
          cstr.bound(value) == count;
        else {
          cstr.bound(value) >= 0;
          cstr.bound(value) < count;
        }
        return;
      }
    }

    if (!sizeArg)
      return;
    AffineExpr size = getPositiveDimension(sizeArg, *width, cstr);
    if (!size)
      return;
    if (isSize) {
      cstr.bound(value) == size;
      cstr.bound(value) >= 1;
    } else {
      cstr.bound(value) < size;
      cstr.bound(value) >= 0;
    }
  }
};
} // namespace

void mlir::gpu::registerValueBoundsOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, GPUDialect *dialect) {
#define REGISTER(X) X::attachInterface<GpuIdOpInterface<X>>(*ctx);
    REGISTER(ClusterDimOp)
    REGISTER(ClusterDimBlocksOp)
    REGISTER(ClusterIdOp)
    REGISTER(ClusterBlockIdOp)
    REGISTER(BlockDimOp)
    REGISTER(BlockIdOp)
    REGISTER(GridDimOp)
    REGISTER(ThreadIdOp)
    REGISTER(LaneIdOp)
    REGISTER(SubgroupIdOp)
    REGISTER(GlobalIdOp)
    REGISTER(NumSubgroupsOp)
    REGISTER(SubgroupSizeOp)
#undef REGISTER

    LaunchOp::attachInterface<GpuLaunchOpInterface>(*ctx);
    SubgroupBroadcastOp::attachInterface<SubgroupBroadcastOpInterface>(*ctx);
  });
}
