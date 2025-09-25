//===- ShardingInterfaceImpl.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Shard/Interfaces/ShardingInterface.h"
#include "mlir/Dialect/Shard/Interfaces/ShardingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/ShardingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;
using namespace mlir::tensor;
using namespace mlir::shard;

namespace {

// Sharding of tensor.empty/tensor.splat
template <typename OpTy>
struct CreatorOpShardingInterface
    : public ShardingInterface::ExternalModel<CreatorOpShardingInterface<OpTy>,
                                              OpTy> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto ndims = mlir::cast<ShapedType>(op->getResult(0).getType()).getRank();
    return SmallVector<utils::IteratorType>(ndims,
                                            utils::IteratorType::parallel);
  }

  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    MLIRContext *ctx = op->getContext();
    Value val = op->getResult(0);
    auto type = dyn_cast<RankedTensorType>(val.getType());
    if (!type)
      return {};
    return SmallVector<AffineMap>(
        op->getNumOperands() + op->getNumResults(),
        {AffineMap::getMultiDimIdentityMap(type.getRank(), ctx)});
  }

  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<Sharding> operandShardings,
                          ArrayRef<Sharding> resultShardings,
                          IRMapping &partitionMap,
                          SymbolTableCollection &symbolTable,
                          OpBuilder &builder) const {
    assert(resultShardings.size() == 1);
    auto resType = cast<RankedTensorType>(op->getResult(0).getType());
    mlir::shard::GridOp grid;
    ShapedType shardType;
    if (resType.getRank() > 0) {
      grid = shard::getGrid(op, resultShardings[0].getGridAttr(), symbolTable);
      shardType =
          cast<ShapedType>(shard::shardType(resType, grid, resultShardings[0]));
    } else {
      shardType = resType;
    }
    Operation *newOp = nullptr;
    // if the sharding introduces a new dynamic dimension, we take it from
    // the dynamic sharding info. For now bail out if it's not
    // provided.
    if (!shardType.hasStaticShape()) {
      assert(op->getResult(0).hasOneUse());
      SmallVector<Value> newOperands;
      auto oldType = cast<ShapedType>(resType);
      assert(oldType.getRank() == shardType.getRank());
      int currOldOprndNum = -1;
      shard::ShardShapeOp shapeForDevice;
      ValueRange device;
      Operation *newSharding = nullptr;
      for (auto i = 0; i < oldType.getRank(); ++i) {
        if (!oldType.isDynamicDim(i) && shardType.isDynamicDim(i)) {
          if (!newSharding) {
            newSharding =
                ShardingOp::create(builder, op->getLoc(), resultShardings[0]);
            device =
                shard::ProcessMultiIndexOp::create(builder, op->getLoc(), grid)
                    .getResults();
            shapeForDevice = shard::ShardShapeOp::create(
                builder, op->getLoc(), oldType.getShape(), partitionedOperands,
                newSharding->getResult(0), device);
          }
          newOperands.emplace_back(shapeForDevice.getResult()[i]);
        } else if (oldType.isDynamicDim(i)) {
          assert(shardType.isDynamicDim(i));
          newOperands.emplace_back(partitionedOperands[++currOldOprndNum]);
        }
      }
      newOp = OpTy::create(builder, op->getLoc(), shardType, newOperands);
      partitionMap.map(op->getResult(0), newOp->getResult(0));
    } else {
      // `clone` will populate the mapping of old to new results.
      newOp = builder.clone(*op, partitionMap);
    }
    newOp->getResult(0).setType(shardType);

    return success();
  }
};
} // namespace

void mlir::tensor::registerShardingInterfaceExternalModels(
    DialectRegistry &registry) {

  registry.addExtension(+[](MLIRContext *ctx, TensorDialect *dialect) {
    EmptyOp::template attachInterface<CreatorOpShardingInterface<EmptyOp>>(
        *ctx);
    SplatOp::template attachInterface<CreatorOpShardingInterface<SplatOp>>(
        *ctx);
  });
}
