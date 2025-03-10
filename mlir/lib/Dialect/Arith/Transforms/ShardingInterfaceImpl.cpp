//===- ShardingInterfaceImpl.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Mesh/Interfaces/ShardingInterfaceImpl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/ShardingInterfaceImpl.h"
#include "mlir/Dialect/Mesh/Interfaces/ShardingInterface.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::mesh;

namespace {

// Sharding of arith.constant
// RankedTensor constants can be sharded like any other tensor.
//   %cst = arith.constant dense<0.000000e+00> : tensor<1024x1024xf32>
//   %sharding = mesh.sharding @mesh4x4 split_axes = [[0]] : !mesh.sharding
// Scalar constants are always replicated and need no sharding annotation.

struct ConstantShardingInterface
    : public ShardingInterface::ExternalModel<ConstantShardingInterface,
                                              ConstantOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto ndims = 0;
    if (auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType())) {
      ndims = type.getRank();
    }
    return SmallVector<utils::IteratorType>(ndims,
                                            utils::IteratorType::parallel);
  }

  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    if (auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType())) {
      return SmallVector<AffineMap>(1, {AffineMap::getMultiDimIdentityMap(
                                           type.getRank(), op->getContext())});
    }
    return {};
  }

  // Indicate failure if no result sharding exists.
  // Otherwise mirror result sharding if it is a tensor constant.
  // Otherwise return replication option.
  FailureOr<ShardingOption>
  getShardingOption(Operation *op, ArrayRef<MeshSharding> operandShardings,
                    ArrayRef<MeshSharding> resultShardings) const {
    assert(resultShardings.size() == 1 &&
           "Expecting exactly one result sharding for arith.constant");
    auto resultSharding = resultShardings[0];
    if (!resultSharding) {
      return failure();
    }
    if (auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType())) {
      ShardingArray axesArray(resultSharding.getSplitAxes().size());
      for (auto [i, axes] : llvm::enumerate(resultSharding.getSplitAxes())) {
        axesArray[i].append(axes.asArrayRef().begin(), axes.asArrayRef().end());
      }
      return ShardingOption(axesArray, resultSharding.getMeshAttr());
    }
    return ShardingOption({}, resultSharding.getMeshAttr());
  }

  LogicalResult spmdize(Operation *op, ArrayRef<Value> spmdizedOperands,
                        ArrayRef<MeshSharding> operandShardings,
                        ArrayRef<MeshSharding> resultShardings,
                        IRMapping &spmdizationMap,
                        SymbolTableCollection &symbolTable,
                        OpBuilder &builder) const {
    auto cOp = cast<ConstantOp>(op);
    if (auto value = dyn_cast<DenseIntOrFPElementsAttr>(cOp.getValue())) {
      if (!value.isSplat() || !resultShardings[0]) {
        // Currently non-splat constants are not supported.
        return failure();
      }
      auto sharding = resultShardings[0];
      auto newType = cast<RankedTensorType>(shardType(
          cOp.getType(), getMesh(op, sharding.getMeshAttr(), symbolTable),
          sharding));
      auto newValue = value.resizeSplat(newType);
      auto newOp = builder.create<ConstantOp>(op->getLoc(), newType, newValue);
      spmdizationMap.map(op->getResult(0), newOp.getResult());
      spmdizationMap.map(op, newOp.getOperation());
    } else {
      // `clone` will populate the mapping of old to new results.
      (void)builder.clone(*op, spmdizationMap);
    }
    return success();
  }
};
} // namespace

void mlir::arith::registerShardingInterfaceExternalModels(
    DialectRegistry &registry) {

  registry.addExtension(+[](MLIRContext *ctx, ArithDialect *dialect) {
    ConstantOp::template attachInterface<ConstantShardingInterface>(*ctx);
  });
}
