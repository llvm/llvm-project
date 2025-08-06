//===- ShardingInterfaceImpl.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SHARD_INTERFACES_SHARDINGINTERFACEIMPL_H_
#define MLIR_DIALECT_SHARD_INTERFACES_SHARDINGINTERFACEIMPL_H_

#include "mlir/Dialect/Shard/IR/ShardOps.h"
#include "mlir/Dialect/Shard/Interfaces/ShardingInterface.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Value.h"

namespace mlir {

class Operation;
class IRMapping;
class SymbolTableCollection;

namespace shard {

// Retrieve the grid axes corresponding to each operation loop iterator based
// on the provided shardings for the op's operands and results.
// Assumes that the indexingMaps are projected permutations.
ShardingArray getGridAxisAssignmentForLoopIterators(
    ArrayRef<Sharding> operandShardings, ArrayRef<Sharding> resultShardings,
    ArrayRef<utils::IteratorType> loopIteratorTypes,
    ArrayRef<AffineMap> indexingMaps);

bool isAtLeastOneReductionIteratorSharded(
    ArrayRef<utils::IteratorType> loopIteratorTypes,
    ArrayRef<SmallVector<GridAxis>> gridAxisAssignmentForLoopIterators);

// Get the set of grid axes that correspond to reduction loop iterators.
SmallVector<GridAxis> getReductionGridAxes(
    ArrayRef<utils::IteratorType> loopIteratorTypes,
    ArrayRef<SmallVector<GridAxis>> gridAxisAssignmentForLoopIterators);

// Inserts a clone of the operation that has all ranked tensor
// arguments/results sharded.
void partitionTriviallyShardableOperation(Operation &op,
                                          ArrayRef<Value> partitionedOperands,
                                          ArrayRef<Sharding> operandShardings,
                                          ArrayRef<Sharding> resultShardings,
                                          IRMapping &partitionMap,
                                          SymbolTableCollection &symbolTable,
                                          OpBuilder &builder);

// All ranked tensor argument and result dimensions have
// independent parallel loop iterators.
template <typename Op>
struct IndependentParallelIteratorDomainShardingInterface
    : public ShardingInterface::ExternalModel<
          IndependentParallelIteratorDomainShardingInterface<Op>, Op> {
  SmallVector<utils::IteratorType>
  getLoopIteratorTypes(Operation *operation) const {
    SmallVector<utils::IteratorType> iterTypes;
    for (Type t : operation->getOperandTypes()) {
      populateIteratorTypes(t, iterTypes);
    }
    for (Type t : operation->getResultTypes()) {
      populateIteratorTypes(t, iterTypes);
    }
    return iterTypes;
  }

  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    // TODO: implement.
    return SmallVector<AffineMap>();
  }

  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<Sharding> operandShardings,
                          ArrayRef<Sharding> resultShardings,
                          IRMapping &partitionMap,
                          SymbolTableCollection &symbolTable,
                          OpBuilder &builder) const {
    partitionTriviallyShardableOperation(*op, partitionedOperands,
                                         operandShardings, resultShardings,
                                         partitionMap, symbolTable, builder);
    return success();
  }

private:
  void
  populateIteratorTypes(Type t,
                        SmallVector<utils::IteratorType> &iterTypes) const {
    RankedTensorType rankedTensorType = dyn_cast<RankedTensorType>(t);
    if (!rankedTensorType) {
      return;
    }

    iterTypes.reserve(iterTypes.size() + rankedTensorType.getRank());
    for (int64_t i = 0; i < rankedTensorType.getRank(); ++i) {
      iterTypes.push_back(utils::IteratorType::parallel);
    }
  }
};

// Sharding of elementwise operations like tensor addition and multiplication.
template <typename ElemwiseOp>
struct ElementwiseShardingInterface
    : public ShardingInterface::ExternalModel<
          ElementwiseShardingInterface<ElemwiseOp>, ElemwiseOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    Value val = op->getOperand(0);
    auto type = dyn_cast<RankedTensorType>(val.getType());
    if (!type)
      return {};
    SmallVector<utils::IteratorType> types(type.getRank(),
                                           utils::IteratorType::parallel);
    return types;
  }

  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    MLIRContext *ctx = op->getContext();
    Value val = op->getOperand(0);
    auto type = dyn_cast<RankedTensorType>(val.getType());
    if (!type)
      return {};
    int64_t rank = type.getRank();
    int64_t num = op->getNumOperands() + op->getNumResults();
    SmallVector<AffineMap> maps(num,
                                AffineMap::getMultiDimIdentityMap(rank, ctx));
    return maps;
  }

  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<Sharding> operandShardings,
                          ArrayRef<Sharding> resultShardings,
                          IRMapping &partitionMap,
                          SymbolTableCollection &symbolTable,
                          OpBuilder &builder) const {
    partitionTriviallyShardableOperation(*op, partitionedOperands,
                                         operandShardings, resultShardings,
                                         partitionMap, symbolTable, builder);
    return success();
  }
};

} // namespace shard
} // namespace mlir

#endif // MLIR_DIALECT_SHARD_INTERFACES_SHARDINGINTERFACEIMPL_H_
