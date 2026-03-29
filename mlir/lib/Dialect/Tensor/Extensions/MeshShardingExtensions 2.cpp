//===- ShardingInterfaceImpl.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Mesh/Interfaces/ShardingInterface.h"
#include "mlir/Dialect/Mesh/Interfaces/ShardingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/ShardingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tensor-sharding-impl"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;
using namespace mlir::tensor;
using namespace mlir::mesh;

namespace {

// Sharding of tensor.empty
struct EmptyOpShardingInterface
    : public ShardingInterface::ExternalModel<EmptyOpShardingInterface,
                                              tensor::EmptyOp> {
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
    return {AffineMap::getMultiDimIdentityMap(type.getRank(), ctx)};
  }

  LogicalResult spmdize(Operation *op, ArrayRef<Value> spmdizedOperands,
                        ArrayRef<MeshSharding> operandShardings,
                        ArrayRef<MeshSharding> resultShardings,
                        IRMapping &spmdizationMap,
                        SymbolTableCollection &symbolTable,
                        OpBuilder &builder) const {
    auto shardType = cast<ShapedType>(mesh::shardType(
        op->getResult(0).getType(),
        mesh::getMesh(op, resultShardings[0].getMeshAttr(), symbolTable),
        resultShardings[0]));
    Operation *newOp = nullptr;
    // if the sharding introduces a new dynamic dimension, we take it from
    // the dynamic sharding info. For now bail out if it's not
    // provided.
    assert(resultShardings.size() == 1);
    if (!shardType.hasStaticShape()) {
      assert(op->getResult(0).hasOneUse());
      SmallVector<Value> newOperands;
      auto oldType = cast<ShapedType>(op->getResult(0).getType());
      assert(oldType.getRank() == shardType.getRank());
      int currOldOprndNum = -1;
      mesh::ShardShapeOp shapeForDevice;
      Value device;
      Operation *newSharding = nullptr;
      for (auto i = 0; i < oldType.getRank(); ++i) {
        if (!oldType.isDynamicDim(i) && shardType.isDynamicDim(i)) {
          if (!newSharding) {
            newSharding =
                builder.create<ShardingOp>(op->getLoc(), resultShardings[0]);
            device = builder.create<mesh::ProcessLinearIndexOp>(
                op->getLoc(), resultShardings[0].getMesh());
            shapeForDevice = builder.create<mesh::ShardShapeOp>(
                op->getLoc(), oldType.getShape(), newSharding->getResult(0),
                device);
          }
          newOperands.emplace_back(shapeForDevice.getResult()[i]);
        } else if (oldType.isDynamicDim(i)) {
          assert(shardType.isDynamicDim(i));
          newOperands.emplace_back(spmdizedOperands[++currOldOprndNum]);
        }
      }
      newOp =
          builder.create<tensor::EmptyOp>(op->getLoc(), shardType, newOperands);
      spmdizationMap.map(op->getResult(0), newOp->getResult(0));
    } else {
      // `clone` will populate the mapping of old to new results.
      newOp = builder.clone(*op, spmdizationMap);
    }
    newOp->getResult(0).setType(shardType);

    return success();
  }
};
} // namespace

void mlir::tensor::registerShardingInterfaceExternalModels(
    DialectRegistry &registry) {

  registry.addExtension(+[](MLIRContext *ctx, TensorDialect *dialect) {
    EmptyOp::template attachInterface<EmptyOpShardingInterface>(*ctx);
  });
}
