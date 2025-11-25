//===- ShardOps.h - Shard Dialect Operations --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SHARD_IR_SHARDOPS_H
#define MLIR_DIALECT_SHARD_IR_SHARDOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace shard {

using GridAxis = int16_t;
using GridAxesAttr = DenseI16ArrayAttr;
using ShardShapeAttr = DenseI64ArrayAttr;
using HaloSizePairAttr = DenseI64ArrayAttr;

} // namespace shard
} // namespace mlir

#include "mlir/Dialect/Shard/IR/ShardEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Shard/IR/ShardAttributes.h.inc"

namespace mlir {
namespace shard {

class Sharding {
private:
  ::mlir::FlatSymbolRefAttr grid;
  SmallVector<GridAxesAttr> split_axes;
  SmallVector<int64_t> static_halo_sizes;
  SmallVector<int64_t> static_sharded_dims_offsets;
  SmallVector<Value> dynamic_halo_sizes;
  SmallVector<Value> dynamic_sharded_dims_offsets;

public:
  Sharding(::mlir::FlatSymbolRefAttr grid_ = nullptr);
  Sharding(Value rhs);
  static Sharding get(::mlir::FlatSymbolRefAttr grid_,
                      ArrayRef<GridAxesAttr> split_axes_,
                      ArrayRef<int64_t> static_halo_sizes_ = {},
                      ArrayRef<int64_t> static_sharded_dims_offsets_ = {},
                      ArrayRef<Value> dynamic_halo_sizes_ = {},
                      ArrayRef<Value> dynamic_sharded_dims_offsets_ = {});
  ::mlir::FlatSymbolRefAttr getGridAttr() const { return grid; }
  ::llvm::StringRef getGrid() const { return grid ? grid.getValue() : ""; }
  ArrayRef<GridAxesAttr> getSplitAxes() const { return split_axes; }
  ArrayRef<int64_t> getStaticHaloSizes() const { return static_halo_sizes; }
  ArrayRef<int64_t> getStaticShardedDimsOffsets() const {
    return static_sharded_dims_offsets;
  }
  ArrayRef<Value> getDynamicHaloSizes() const { return dynamic_halo_sizes; }
  ArrayRef<Value> getDynamicShardedDimsOffsets() const {
    return dynamic_sharded_dims_offsets;
  }
  operator bool() const { return (!grid) == false; }
  bool operator==(Value rhs) const;
  bool operator!=(Value rhs) const;
  bool operator==(const Sharding &rhs) const;
  bool operator!=(const Sharding &rhs) const;
  bool equalSplitAxes(const Sharding &rhs) const;
  bool equalHaloAndShardSizes(const Sharding &rhs) const;
  bool equalHaloSizes(const Sharding &rhs) const;
  bool equalShardSizes(const Sharding &rhs) const;
};

} // namespace shard
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Shard/IR/ShardTypes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Shard/IR/ShardOps.h.inc"

namespace mlir {
namespace shard {

inline bool isReductionLoop(utils::IteratorType iType) {
  return iType == utils::IteratorType::reduction;
}

// Remove empty subarrays of `array` until a minimum lengh of one is reached.
template <typename T>
void removeTrailingEmptySubArray(SmallVector<SmallVector<T>> &array) {
  while (array.size() > 1 && array.back().empty())
    array.pop_back();
}

// Is the same tensor replicated on all processes.
inline bool isFullReplication(Sharding sharding) {
  return llvm::all_of(sharding.getSplitAxes(), [](GridAxesAttr axes) {
    return axes.asArrayRef().empty();
  });
}

inline shard::GridOp
getGridOrNull(Operation *op, FlatSymbolRefAttr gridSymbol,
              SymbolTableCollection &symbolTableCollection) {
  if (!gridSymbol)
    return nullptr;
  return symbolTableCollection.lookupNearestSymbolFrom<shard::GridOp>(
      op, gridSymbol);
}

inline shard::GridOp getGrid(Operation *op, FlatSymbolRefAttr gridSymbol,
                             SymbolTableCollection &symbolTableCollection) {
  shard::GridOp gridOp = getGridOrNull(op, gridSymbol, symbolTableCollection);
  assert(gridOp);
  return gridOp;
}

// Get the corresponding grid op using the standard attribute nomenclature.
template <typename Op>
shard::GridOp getGrid(Op op, SymbolTableCollection &symbolTableCollection) {
  return getGrid(op.getOperation(), op.getGridAttr(), symbolTableCollection);
}

template <>
inline shard::GridOp
getGrid<ShardOp>(ShardOp op, SymbolTableCollection &symbolTableCollection) {
  return getGrid(
      op.getOperation(),
      cast<ShardingOp>(op.getSharding().getDefiningOp()).getGridAttr(),
      symbolTableCollection);
}

// Get the number of processes that participate in each group
// induced by `gridAxes`.
template <typename GridAxesRange, typename GridShapeRange>
int64_t collectiveProcessGroupSize(GridAxesRange &&gridAxes,
                                   GridShapeRange &&gridShape) {
  int64_t res = 1;

  for (GridAxis axis : gridAxes) {
    auto axisSize = *(std::begin(gridShape) + axis);
    if (ShapedType::isDynamic(axisSize)) {
      return ShapedType::kDynamic;
    }
    res *= axisSize;
  }

  return res;
}

template <typename GridAxesRange>
int64_t collectiveProcessGroupSize(GridAxesRange &&gridAxes, GridOp grid) {
  return collectiveProcessGroupSize(std::forward<GridAxesRange>(gridAxes),
                                    grid.getShape());
}

// Get the size of a sharded dimension.
inline int64_t shardDimension(int64_t dimSize, int64_t shardCount) {
  if (ShapedType::isDynamic(dimSize) || ShapedType::isDynamic(shardCount))
    return ShapedType::kDynamic;

  assert(dimSize % shardCount == 0);
  return dimSize / shardCount;
}

// Get the size of an unsharded dimension.
inline int64_t gatherDimension(int64_t dimSize, int64_t shardCount) {
  if (ShapedType::isDynamic(dimSize) || ShapedType::isDynamic(shardCount))
    return ShapedType::kDynamic;

  return dimSize * shardCount;
}

// Return the sharded shape `shape` according ot sharding `sharding`.
// The shape for the tensor on each device in the grid.
// Example:
// On a 2x4x? grid with split axes = [[0], [1], [2]] the shape ?x5x1 would
// result in a shape for each shard of ?x2x?.
ShapedType shardShapedType(ShapedType shape, GridOp grid, Sharding sharding);

// If ranked tensor type return its sharded counterpart.
//
// If not ranked tensor type return `type`.
// `sharding` in that case must be null.
Type shardType(Type type, GridOp grid, Sharding sharding);

// Insert shard op if there is not one that already has the same sharding.
// Use newShardOp if it is not null. Otherwise create a new one.
// May insert resharding if required.
// Potentially updates newShardOp.
void maybeInsertTargetShardingAnnotation(Sharding sharding, OpResult result,
                                         OpBuilder &builder);
void maybeInsertSourceShardingAnnotation(Sharding sharding, OpOperand &operand,
                                         OpBuilder &builder);

/// Converts a vector of OpFoldResults (ints) into vector of Values of the
/// provided type.
SmallVector<Value> getMixedAsValues(OpBuilder b, const Location &loc,
                                    llvm::ArrayRef<int64_t> statics,
                                    ValueRange dynamics, Type type = Type());
} // namespace shard
} // namespace mlir

#endif // MLIR_DIALECT_SHARD_IR_SHARDOPS_H
