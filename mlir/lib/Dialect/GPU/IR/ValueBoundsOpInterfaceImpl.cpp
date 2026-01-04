//===- ValueBoundsOpInterfaceImpl.cpp - Impl. of ValueBoundsOpInterface ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/ValueBoundsOpInterfaceImpl.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;
using namespace mlir::gpu;

namespace {
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
      cstr.bound(v) >= range.smin().getSExtValue();
      cstr.bound(v) <= range.smax().getSExtValue();
    };
    assert(inferrable->getNumOperands() == 0 && "ID ops have no operands");
    inferrable.inferResultRanges({}, translateConstraint);
  }
};

struct GpuLaunchOpInterface
    : public ValueBoundsOpInterface::ExternalModel<GpuLaunchOpInterface,
                                                   LaunchOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto launchOp = cast<LaunchOp>(op);

    Value sizeArg = nullptr;
    bool isSize = false;
    KernelDim3 gridSizeArgs = launchOp.getGridSizeOperandValues();
    KernelDim3 blockSizeArgs = launchOp.getBlockSizeOperandValues();

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
    if (launchOp.hasClusterSize()) {
      KernelDim3 clusterSizeArgs = *launchOp.getClusterSizeOperandValues();
      match(*launchOp.getClusterIds(), clusterSizeArgs, false);
      match(*launchOp.getClusterSize(), clusterSizeArgs, true);
    }

    if (!sizeArg)
      return;
    if (isSize) {
      cstr.bound(value) == cstr.getExpr(sizeArg);
      cstr.bound(value) >= 1;
    } else {
      cstr.bound(value) < cstr.getExpr(sizeArg);
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
  });
}
