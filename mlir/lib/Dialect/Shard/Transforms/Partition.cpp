//===- Partition.cpp --------------------------------------------- C++ --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Shard/Transforms/Partition.h"

#include "mlir/Dialect/Shard/IR/ShardDialect.h"
#include "mlir/Dialect/Shard/IR/ShardOps.h"
#include "mlir/Dialect/Shard/Interfaces/ShardingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <iterator>
#include <optional>
#include <tuple>
#include <utility>

namespace mlir::shard {

template <typename SourceAxes, typename TargetAxes>
static bool arePartialAxesCompatible(const SourceAxes &sourceAxes,
                                     const TargetAxes &targetAxes) {
  return llvm::all_of(targetAxes, [&sourceAxes](auto &targetAxis) {
    return sourceAxes.contains(targetAxis);
  });
}

static Sharding targetShardingInSplitLastAxis(MLIRContext *ctx,
                                              const Sharding &sourceSharding,
                                              int64_t splitTensorAxis,
                                              GridAxis splitGridAxis) {
  SmallVector<GridAxesAttr> targetShardingSplitAxes =
      llvm::to_vector(sourceSharding.getSplitAxes());
  while (static_cast<int64_t>(targetShardingSplitAxes.size()) <=
         splitTensorAxis) {
    targetShardingSplitAxes.push_back(GridAxesAttr::get(ctx, {}));
  }
  auto targetSplitAxes =
      llvm::to_vector(targetShardingSplitAxes[splitTensorAxis].asArrayRef());
  targetSplitAxes.push_back(splitGridAxis);
  targetShardingSplitAxes[splitTensorAxis] =
      GridAxesAttr::get(ctx, targetSplitAxes);
  return Sharding::get(sourceSharding.getGridAttr(), targetShardingSplitAxes);
}

// Split a replicated tensor along a grid axis.
// E.g. [[0, 1]] -> [[0, 1, 2]].
// Returns the partitioned target value with its sharding.
static std::tuple<TypedValue<ShapedType>, Sharding>
splitLastAxisInResharding(ImplicitLocOpBuilder &builder,
                          Sharding sourceSharding,
                          TypedValue<ShapedType> sourceShard, GridOp grid,
                          int64_t splitTensorAxis, GridAxis splitGridAxis) {
  TypedValue<ShapedType> targetShard =
      AllSliceOp::create(builder, sourceShard, grid,
                         ArrayRef<GridAxis>(splitGridAxis), splitTensorAxis)
          .getResult();
  Sharding targetSharding = targetShardingInSplitLastAxis(
      builder.getContext(), std::move(sourceSharding), splitTensorAxis,
      splitGridAxis);
  return {targetShard, targetSharding};
}

// Detect if the resharding is of type e.g.
// [[0, 1]] -> [[0, 1, 2]].
// If detected, returns the corresponding tensor axis grid axis pair.
// Does not detect insertions like
// [[0, 1]] -> [[0, 2, 1]].
static std::optional<std::tuple<int64_t, GridAxis>>
detectSplitLastAxisInResharding(const Sharding &sourceSharding,
                                const Sharding &targetSharding) {
  for (size_t tensorAxis = 0; tensorAxis < targetSharding.getSplitAxes().size();
       ++tensorAxis) {
    if (sourceSharding.getSplitAxes().size() > tensorAxis) {
      if (sourceSharding.getSplitAxes()[tensorAxis].size() + 1 !=
          targetSharding.getSplitAxes()[tensorAxis].size()) {
        continue;
      }
      if (!llvm::equal(
              sourceSharding.getSplitAxes()[tensorAxis].asArrayRef(),
              llvm::make_range(
                  targetSharding.getSplitAxes()[tensorAxis]
                      .asArrayRef()
                      .begin(),
                  targetSharding.getSplitAxes()[tensorAxis].asArrayRef().end() -
                      1))) {
        continue;
      }
    } else {
      if (targetSharding.getSplitAxes()[tensorAxis].size() != 1) {
        continue;
      }
    }
    return std::make_tuple(
        tensorAxis,
        targetSharding.getSplitAxes()[tensorAxis].asArrayRef().back());
  }
  return std::nullopt;
}

static std::optional<std::tuple<TypedValue<ShapedType>, Sharding>>
trySplitLastAxisInResharding(ImplicitLocOpBuilder &builder, GridOp grid,
                             const Sharding &sourceSharding,
                             Sharding targetSharding,
                             TypedValue<ShapedType> sourceShard) {
  if (auto detectRes = detectSplitLastAxisInResharding(
          sourceSharding, std::move(targetSharding))) {
    auto [tensorAxis, gridAxis] = detectRes.value();
    return splitLastAxisInResharding(builder, sourceSharding, sourceShard, grid,
                                     tensorAxis, gridAxis);
  }

  return std::nullopt;
}

// Detect if the resharding removes trailing split Axes along a tensor
// dimension, e.g.
// [[0, 1, 2]] -> [[0, 1]], [[0, 1, 2]] -> [0] or [[0, 1, 2]] -> [].
// If detected, returns the corresponding (tensor dim, grid axes) pair, where
// the "grid axes" are the removed trailing split axes.
static std::optional<std::tuple<int64_t, SmallVector<GridAxis>>>
detectUnsplitLastAxesInResharding(const Sharding &sourceSharding,
                                  const Sharding &targetSharding) {
  for (size_t tensorDim = 0; tensorDim < sourceSharding.getSplitAxes().size();
       ++tensorDim) {
    if (targetSharding.getSplitAxes().size() > tensorDim) {
      // No match if the target sharding does not have less split axes than the
      // source sharding along the current tensor dimension.
      if (sourceSharding.getSplitAxes()[tensorDim].size() <=
          targetSharding.getSplitAxes()[tensorDim].size())
        continue;
      // No match if the split axes of the target sharding are different from
      // the first split axes of the source sharding.
      if (!std::equal(
              targetSharding.getSplitAxes()[tensorDim].asArrayRef().begin(),
              targetSharding.getSplitAxes()[tensorDim].asArrayRef().end(),
              sourceSharding.getSplitAxes()[tensorDim].asArrayRef().begin()))
        continue;
    } else {
      // Here the target dimension is replicated; there is nothing to do if the
      // source dimension is also replicated.
      if (sourceSharding.getSplitAxes()[tensorDim].size() == 0)
        continue;
    }
    // This is a match. Return the current tensor dimension and the trailing
    // grid axis of the source sharding along this dimension.
    SmallVector<GridAxis> unsplitAxes;
    size_t dimOff = tensorDim >= targetSharding.getSplitAxes().size()
                        ? 0
                        : targetSharding.getSplitAxes()[tensorDim].size();
    for (auto a =
             sourceSharding.getSplitAxes()[tensorDim].asArrayRef().begin() +
             dimOff;
         a != sourceSharding.getSplitAxes()[tensorDim].asArrayRef().end(); ++a)
      unsplitAxes.push_back(*a);
    return std::make_tuple(tensorDim, unsplitAxes);
  }
  return std::nullopt;
}

// Return the resulting Sharding if the unsplit last axes resharding is applied.
static Sharding targetShardingInUnsplitLastAxes(MLIRContext *ctx,
                                                const Sharding &sourceSharding,
                                                int64_t splitTensorDim,
                                                size_t numUnsplitAxes) {
  SmallVector<GridAxesAttr> resSplitAxes =
      llvm::to_vector(sourceSharding.getSplitAxes());
  assert(static_cast<int64_t>(resSplitAxes.size()) > splitTensorDim);
  ArrayRef<GridAxis> srcSplitAxes = resSplitAxes[splitTensorDim].asArrayRef();
  assert(srcSplitAxes.size() >= numUnsplitAxes);
  size_t numSplitAxes = srcSplitAxes.size() - numUnsplitAxes;
  SmallVector<GridAxis> newSplitAxes(srcSplitAxes.begin(),
                                     srcSplitAxes.begin() + numSplitAxes);
  resSplitAxes[splitTensorDim] = GridAxesAttr::get(ctx, newSplitAxes);
  return Sharding::get(sourceSharding.getGridAttr(), resSplitAxes);
}

// Return the resulting Tensor type after applying the unsplit last axes
// resharding.
static ShapedType allGatherResultTypeInUnsplitLastAxes(
    ShapedType sourceType, int64_t splitTensorDim, ArrayRef<int64_t> gridShape,
    ArrayRef<GridAxis> unsplitAxes) {
  SmallVector<int64_t> targetShape = llvm::to_vector(sourceType.getShape());
  for (GridAxis gridAxis : unsplitAxes)
    targetShape[splitTensorDim] =
        gatherDimension(targetShape[splitTensorDim], gridShape[gridAxis]);
  return sourceType.cloneWith(targetShape, sourceType.getElementType());
}

// Perform the resharding for the unsplit last axes case.
// This basically performs an all-gather along the unsplit grid axes.
static std::tuple<TypedValue<ShapedType>, Sharding> unsplitLastAxesInResharding(
    ImplicitLocOpBuilder &builder, Sharding sourceSharding,
    ShapedType sourceUnshardedShape, TypedValue<ShapedType> sourceShard,
    GridOp grid, int64_t splitTensorDim, ArrayRef<GridAxis> unsplitAxes) {
  MLIRContext *ctx = builder.getContext();
  builder.setInsertionPointAfterValue(sourceShard);

  Sharding targetSharding = targetShardingInUnsplitLastAxes(
      ctx, std::move(sourceSharding), splitTensorDim, unsplitAxes.size());
  ShapedType allGatherResultType = allGatherResultTypeInUnsplitLastAxes(
      sourceShard.getType(), splitTensorDim, grid.getShape(), unsplitAxes);
  Value allGatherResult = AllGatherOp::create(
      builder,
      RankedTensorType::get(allGatherResultType.getShape(),
                            allGatherResultType.getElementType()),
      grid.getSymName(), unsplitAxes, sourceShard, APInt(64, splitTensorDim));
  ShapedType targetType =
      shardShapedType(sourceUnshardedShape, grid, targetSharding);
  TypedValue<ShapedType> targetShard =
      tensor::CastOp::create(builder, targetType, allGatherResult).getResult();
  return {targetShard, targetSharding};
}

static std::optional<std::tuple<TypedValue<ShapedType>, Sharding>>
tryUnsplitLastAxesInResharding(ImplicitLocOpBuilder &builder, GridOp grid,
                               const Sharding &sourceSharding,
                               Sharding targetSharding,
                               ShapedType sourceUnshardedShape,
                               TypedValue<ShapedType> sourceShard) {
  if (auto detectRes = detectUnsplitLastAxesInResharding(
          sourceSharding, std::move(targetSharding))) {
    auto [tensorDim, gridAxes] = detectRes.value();
    return unsplitLastAxesInResharding(builder, sourceSharding,
                                       sourceUnshardedShape, sourceShard, grid,
                                       tensorDim, gridAxes);
  }

  return std::nullopt;
}

// Detect if the resharding is of type e.g.
// [[0, 1], [2]] -> [[0], [1, 2]].
// Only moving the last axis counts.
// If detected, returns the corresponding (source_tensor_axis,
// target_tensor_axis, grid_axis) tuple.
static std::optional<std::tuple<int64_t, int64_t, GridAxis>>
detectMoveLastSplitAxisInResharding(const Sharding &sourceSharding,
                                    const Sharding &targetSharding) {
  for (size_t sourceTensorAxis = 0;
       sourceTensorAxis < sourceSharding.getSplitAxes().size();
       ++sourceTensorAxis) {
    for (size_t targetTensorAxis = 0;
         targetTensorAxis < targetSharding.getSplitAxes().size();
         ++targetTensorAxis) {
      if (sourceTensorAxis == targetTensorAxis)
        continue;
      if (sourceSharding.getSplitAxes()[sourceTensorAxis].empty() ||
          targetSharding.getSplitAxes()[targetTensorAxis].empty() ||
          sourceSharding.getSplitAxes()[sourceTensorAxis].asArrayRef().back() !=
              targetSharding.getSplitAxes()[targetTensorAxis]
                  .asArrayRef()
                  .back())
        continue;
      if (!llvm::equal(
              llvm::make_range(sourceSharding.getSplitAxes()[sourceTensorAxis]
                                   .asArrayRef()
                                   .begin(),
                               sourceSharding.getSplitAxes()[sourceTensorAxis]
                                       .asArrayRef()
                                       .end() -
                                   1),
              llvm::make_range(targetSharding.getSplitAxes()[targetTensorAxis]
                                   .asArrayRef()
                                   .begin(),
                               targetSharding.getSplitAxes()[targetTensorAxis]
                                       .asArrayRef()
                                       .end() -
                                   1)))
        continue;
      return std::make_tuple(
          sourceTensorAxis, targetTensorAxis,
          sourceSharding.getSplitAxes()[sourceTensorAxis].asArrayRef().back());
    }
  }
  return std::nullopt;
}

static Sharding targetShardingInMoveLastAxis(MLIRContext *ctx,
                                             const Sharding &sourceSharding,
                                             int64_t sourceTensorAxis,
                                             int64_t targetTensorAxis) {
  SmallVector<GridAxesAttr> targetShardingSplitAxes =
      llvm::to_vector(sourceSharding.getSplitAxes());
  while (static_cast<int64_t>(targetShardingSplitAxes.size()) <=
         targetTensorAxis) {
    targetShardingSplitAxes.push_back(GridAxesAttr::get(ctx, {}));
  }

  auto sourceSplitAxes =
      llvm::to_vector(targetShardingSplitAxes[sourceTensorAxis].asArrayRef());
  assert(!sourceSplitAxes.empty());
  auto gridAxis = sourceSplitAxes.back();
  sourceSplitAxes.pop_back();
  targetShardingSplitAxes[sourceTensorAxis] =
      GridAxesAttr::get(ctx, sourceSplitAxes);

  auto targetSplitAxes =
      llvm::to_vector(targetShardingSplitAxes[targetTensorAxis].asArrayRef());
  targetSplitAxes.push_back(gridAxis);
  targetShardingSplitAxes[targetTensorAxis] =
      GridAxesAttr::get(ctx, targetSplitAxes);

  return Sharding::get(sourceSharding.getGridAttr(), targetShardingSplitAxes);
}

static ShapedType allToAllResultShapeInMoveLastAxis(ShapedType sourceShape,
                                                    int64_t splitCount,
                                                    int64_t sourceTensorAxis,
                                                    int64_t targetTensorAxis) {
  SmallVector<int64_t> targetShape = llvm::to_vector(sourceShape.getShape());
  targetShape[sourceTensorAxis] =
      gatherDimension(targetShape[sourceTensorAxis], splitCount);
  targetShape[targetTensorAxis] =
      shardDimension(targetShape[targetTensorAxis], splitCount);
  return sourceShape.cloneWith(targetShape, sourceShape.getElementType());
}

static std::tuple<TypedValue<ShapedType>, Sharding>
moveLastSplitAxisInResharding(ImplicitLocOpBuilder &builder, GridOp grid,
                              Sharding sourceSharding,
                              ShapedType sourceUnshardedShape,
                              TypedValue<ShapedType> sourceShard,
                              int64_t sourceTensorAxis,
                              int64_t targetTensorAxis, GridAxis gridAxis) {
  MLIRContext *ctx = builder.getContext();
  builder.setInsertionPointAfterValue(sourceShard);

  Sharding targetSharding = targetShardingInMoveLastAxis(
      ctx, std::move(sourceSharding), sourceTensorAxis, targetTensorAxis);
  ShapedType allToAllResultShape = allToAllResultShapeInMoveLastAxis(
      sourceShard.getType(), grid.getShape()[gridAxis], sourceTensorAxis,
      targetTensorAxis);
  Value allToAllResult = AllToAllOp::create(
      builder,
      RankedTensorType::get(allToAllResultShape.getShape(),
                            allToAllResultShape.getElementType()),
      grid.getSymName(), SmallVector<GridAxis>({gridAxis}), sourceShard,
      APInt(64, targetTensorAxis), APInt(64, sourceTensorAxis));
  ShapedType targetShape =
      shardShapedType(sourceUnshardedShape, grid, targetSharding);
  TypedValue<ShapedType> targetShard =
      tensor::CastOp::create(builder, targetShape, allToAllResult).getResult();
  return {targetShard, targetSharding};
}

static std::optional<std::tuple<TypedValue<ShapedType>, Sharding>>
tryMoveLastSplitAxisInResharding(ImplicitLocOpBuilder &builder, GridOp grid,
                                 const Sharding &sourceSharding,
                                 Sharding targetSharding,
                                 ShapedType sourceUnshardedShape,
                                 TypedValue<ShapedType> sourceShard) {
  if (auto detectRes = detectMoveLastSplitAxisInResharding(
          sourceSharding, std::move(targetSharding))) {
    auto [sourceTensorAxis, targetTensorAxis, gridAxis] = detectRes.value();
    return moveLastSplitAxisInResharding(
        builder, grid, sourceSharding, sourceUnshardedShape, sourceShard,
        sourceTensorAxis, targetTensorAxis, gridAxis);
  }

  return std::nullopt;
}

// Detect a change in the halo size (only) and create necessary operations if
// needed. A changed halo sizes requires copying the "core" of the source tensor
// into the "core" of the destination tensor followed by an update halo
// operation.
static std::optional<std::tuple<TypedValue<ShapedType>, Sharding>>
tryUpdateHaloInResharding(ImplicitLocOpBuilder &builder, GridOp grid,
                          const Sharding &sourceSharding,
                          const Sharding &targetSharding,
                          ShapedType sourceUnshardedShape,
                          TypedValue<ShapedType> sourceShard) {
  // Currently handles only cases where halo sizes differ but everything else
  // stays the same (from source to destination sharding).
  if (!sourceSharding.equalSplitAxes(targetSharding) ||
      !sourceSharding.getStaticShardedDimsOffsets().empty() ||
      !targetSharding.getStaticShardedDimsOffsets().empty() ||
      sourceSharding.equalHaloSizes(targetSharding)) {
    return std::nullopt;
  }

  auto srcHaloSizes = sourceSharding.getStaticHaloSizes();
  auto tgtHaloSizes = targetSharding.getStaticHaloSizes();
  assert(srcHaloSizes.empty() || srcHaloSizes.size() == tgtHaloSizes.size());
  assert(((srcHaloSizes.empty() || ShapedType::isStaticShape(srcHaloSizes)) &&
          ShapedType::isStaticShape(tgtHaloSizes) &&
          sourceShard.getType().hasStaticShape()) &&
         "dynamic shapes/halos are not supported yet for shard-partition");
  auto rank = sourceShard.getType().getRank();
  auto splitAxes = sourceSharding.getSplitAxes();
  SmallVector<int64_t> srcCoreOffs(rank, 0), tgtCoreOffs(rank, 0),
      strides(rank, 1), outShape(sourceShard.getType().getShape()),
      coreShape(sourceShard.getType().getShape());

  // Determine "core" of source and destination.
  // The core is the local part of the shard excluding halo regions.
  for (auto i = 0u; i < rank; ++i) {
    if (i < splitAxes.size() && !splitAxes[i].empty()) {
      if (!srcHaloSizes.empty()) {
        coreShape[i] -= srcHaloSizes[i * 2] + srcHaloSizes[i * 2 + 1];
        srcCoreOffs[i] = srcHaloSizes[i * 2];
      }
      tgtCoreOffs[i] = tgtHaloSizes[i * 2];
      outShape[i] =
          coreShape[i] + tgtHaloSizes[i * 2] + tgtHaloSizes[i * 2 + 1];
    }
  }

  // Extract core from source and copy into destination core.
  auto noVals = ValueRange{};
  auto initVal =
      tensor::EmptyOp::create(builder, sourceShard.getLoc(), outShape,
                              sourceShard.getType().getElementType());
  auto core = tensor::ExtractSliceOp::create(
      builder, sourceShard.getLoc(),
      RankedTensorType::get(coreShape, sourceShard.getType().getElementType()),
      sourceShard, noVals, noVals, noVals, srcCoreOffs, coreShape, strides);
  auto initOprnd = tensor::InsertSliceOp::create(
      builder, sourceShard.getLoc(), core, initVal, noVals, noVals, noVals,
      tgtCoreOffs, coreShape, strides);

  // Finally update the halo.
  auto updateHaloResult =
      UpdateHaloOp::create(
          builder, sourceShard.getLoc(),
          RankedTensorType::get(outShape,
                                sourceShard.getType().getElementType()),
          initOprnd, grid.getSymName(),
          GridAxesArrayAttr::get(builder.getContext(),
                                 sourceSharding.getSplitAxes()),
          targetSharding.getDynamicHaloSizes(),
          targetSharding.getStaticHaloSizes())
          .getResult();
  return std::make_tuple(cast<TypedValue<ShapedType>>(updateHaloResult),
                         targetSharding);
}

// In most cases the sharded tensor axes must be exactly divisible by the single
// grid axis size. Only halo size changes can deal with non-divisible cases.
static TypedValue<ShapedType>
reshard(ImplicitLocOpBuilder &builder, GridOp grid,
        const Sharding &sourceSharding, const Sharding &targetSharding,
        TypedValue<ShapedType> sourceUnshardedValue,
        TypedValue<ShapedType> sourceShard) {
  // If source and destination sharding are the same, no need to do anything.
  if (sourceSharding == targetSharding || (isFullReplication(sourceSharding) &&
                                           isFullReplication(targetSharding))) {
    return sourceShard;
  }

  // Tries to handle the case where the resharding is needed because the halo
  // sizes are different. Supports arbitrary grid dimensionality.
  if (auto tryRes = tryUpdateHaloInResharding(
          builder, grid, sourceSharding, targetSharding,
          sourceUnshardedValue.getType(), sourceShard)) {
    return std::get<0>(tryRes.value()); // targetShard
  }

  assert(sourceShard.getType() ==
         shardShapedType(sourceUnshardedValue.getType(), grid, sourceSharding));
  [[maybe_unused]] ShapedType targetShardType =
      shardShapedType(sourceUnshardedValue.getType(), grid, targetSharding);
  assert(sourceShard.getType().getRank() == targetShardType.getRank());

  TypedValue<ShapedType> targetShard;
  Sharding actualTargetSharding;
  if (sourceSharding.getStaticShardedDimsOffsets().empty() &&
      targetSharding.getStaticShardedDimsOffsets().empty() &&
      sourceSharding.getStaticHaloSizes().empty() &&
      targetSharding.getStaticHaloSizes().empty()) {
    if (auto tryRes = tryMoveLastSplitAxisInResharding(
            builder, grid, sourceSharding, targetSharding,
            sourceUnshardedValue.getType(), sourceShard)) {
      std::tie(targetShard, actualTargetSharding) = tryRes.value();
    } else if (auto tryRes =
                   trySplitLastAxisInResharding(builder, grid, sourceSharding,
                                                targetSharding, sourceShard)) {
      std::tie(targetShard, actualTargetSharding) = tryRes.value();
    } else if (auto tryRes = tryUnsplitLastAxesInResharding(
                   builder, grid, sourceSharding, targetSharding,
                   sourceUnshardedValue.getType(), sourceShard)) {
      std::tie(targetShard, actualTargetSharding) = tryRes.value();
    }
  }

  assert(targetShard && "Did not find any pattern to apply.");
  assert(actualTargetSharding == targetSharding);
  assert(targetShard.getType() == targetShardType);
  return targetShard;
}

TypedValue<ShapedType> reshard(OpBuilder &builder, GridOp grid, ShardOp source,
                               ShardOp target,
                               TypedValue<ShapedType> sourceShardValue) {
  assert(source.getResult() == target.getSrc());
  auto sourceSharding = source.getSharding();
  auto targetSharding = target.getSharding();
  ImplicitLocOpBuilder implicitLocOpBuilder(target->getLoc(), builder);
  return reshard(implicitLocOpBuilder, grid, sourceSharding, targetSharding,
                 source.getSrc(), sourceShardValue);
}

TypedValue<ShapedType> reshard(OpBuilder &builder, ShardOp source,
                               ShardOp target,
                               TypedValue<ShapedType> sourceShardValue,
                               SymbolTableCollection &symbolTableCollection) {
  GridOp srcGrid = getGrid(source, symbolTableCollection);
  assert(srcGrid && srcGrid == getGrid(target, symbolTableCollection));
  return reshard(builder, srcGrid, source, target, sourceShardValue);
}

void reshardingRegisterDependentDialects(DialectRegistry &registry) {
  registry.insert<shard::ShardDialect, tensor::TensorDialect>();
}

#define GEN_PASS_DEF_PARTITION
#include "mlir/Dialect/Shard/Transforms/Passes.h.inc"

using UnshardedToShardedValueMap = DenseMap<Value, Value>;

// Get the types of block arguments for an partitioned block.
// Reads the sharding annotations of the arguments to deduce the sharded types.
// Types that are not ranked tensors are left unchanged.
static SmallVector<Type>
shardedBlockArgumentTypes(Block &block,
                          SymbolTableCollection &symbolTableCollection) {
  SmallVector<Type> res;
  llvm::transform(
      block.getArguments(), std::back_inserter(res),
      [&symbolTableCollection](BlockArgument arg) {
        auto rankedTensorArg = dyn_cast<TypedValue<RankedTensorType>>(arg);
        if (!rankedTensorArg || rankedTensorArg.getType().getRank() == 0) {
          return arg.getType();
        }

        assert(rankedTensorArg.hasOneUse());
        Operation *useOp = *rankedTensorArg.getUsers().begin();
        ShardOp shardOp = llvm::dyn_cast<ShardOp>(useOp);
        assert(shardOp);
        GridOp grid = getGrid(shardOp, symbolTableCollection);
        return cast<Type>(shardShapedType(rankedTensorArg.getType(), grid,
                                          shardOp.getSharding()));
      });
  return res;
}

static LogicalResult
partitionOperation(Operation &op, ArrayRef<Value> partitionedOperands,
                   ArrayRef<Sharding> operandShardings,
                   ArrayRef<Sharding> resultShardings, IRMapping &partitionMap,
                   SymbolTableCollection &symbolTableCollection,
                   OpBuilder &builder) {
  ShardingInterface shardingInterface = llvm::dyn_cast<ShardingInterface>(op);
  if (!shardingInterface) {
    // If there is no sharding interface we are conservative and assume that
    // the op should be fully replicated no all devices.
    partitionFullyReplicatedOperation(op, partitionedOperands, operandShardings,
                                      resultShardings, partitionMap,
                                      symbolTableCollection, builder);
  } else {
    if (failed(shardingInterface.partition(
            partitionedOperands, operandShardings, resultShardings,
            partitionMap, symbolTableCollection, builder))) {
      return failure();
    }
  }

  assert(llvm::all_of(op.getResults(), [&partitionMap](OpResult result) {
    return partitionMap.contains(result);
  }));

  return success();
}

// Retrieve the sharding annotations for the operands of the given operation.
// If the type is not a ranked tensor it is not require to have an annotation.
static std::vector<Sharding> getOperandShardings(Operation &op) {
  std::vector<Sharding> res;
  res.reserve(op.getNumOperands());
  llvm::transform(op.getOperands(), std::back_inserter(res), [](Value operand) {
    TypedValue<RankedTensorType> rankedTensor =
        dyn_cast<TypedValue<RankedTensorType>>(operand);
    if (!rankedTensor || rankedTensor.getType().getRank() == 0) {
      return Sharding();
    }

    Operation *definingOp = operand.getDefiningOp();
    assert(definingOp);
    ShardOp shardOp = llvm::cast<ShardOp>(definingOp);
    return Sharding(shardOp.getSharding());
  });
  return res;
}

// Retrieve the sharding annotations for the results of the given operation.
// If the type is not a ranked tensor it is not require to have an annotation.
static std::vector<Sharding> getResultShardings(Operation &op) {
  std::vector<Sharding> res;
  res.reserve(op.getNumResults());
  llvm::transform(
      op.getResults(), std::back_inserter(res), [&op](OpResult result) {
        if (!result.hasOneUse() || result.use_empty()) {
          return Sharding();
        }
        TypedValue<RankedTensorType> rankedTensor =
            dyn_cast<TypedValue<RankedTensorType>>(result);
        if (!rankedTensor) {
          return Sharding();
        }
        Operation *userOp = *result.getUsers().begin();
        ShardOp shardOp = llvm::dyn_cast<ShardOp>(userOp);
        if (shardOp) {
          return Sharding(shardOp.getSharding());
        }
        if (rankedTensor.getType().getRank() == 0) {
          // This is a 0d tensor result without explicit sharding.
          // Find grid symbol from operands, if any.
          // Shardings without grid are not always fully supported yet.
          for (auto operand : op.getOperands()) {
            if (auto sharding = operand.getDefiningOp<ShardingOp>()) {
              return Sharding(sharding.getGridAttr());
            }
          }
        }
        return Sharding();
      });
  return res;
}

static LogicalResult
partitionOperation(ShardOp shardOp, IRMapping &partitionMap,
                   SymbolTableCollection &symbolTableCollection,
                   OpBuilder &builder) {
  Value targetPartitionValue;

  // Check if 2 shard ops are chained. If not there is no need for resharding
  // as the source and target shared the same sharding.
  ShardOp srcShardOp = shardOp.getSrc().getDefiningOp<ShardOp>();
  if (!srcShardOp) {
    targetPartitionValue = partitionMap.lookup(shardOp.getSrc());
  } else {
    // Insert resharding.
    TypedValue<ShapedType> srcPartitionValue =
        cast<TypedValue<ShapedType>>(partitionMap.lookup(srcShardOp));
    targetPartitionValue = reshard(builder, srcShardOp, shardOp,
                                   srcPartitionValue, symbolTableCollection);
  }

  assert(!partitionMap.contains(shardOp.getResult()));
  partitionMap.map(shardOp.getResult(), targetPartitionValue);
  return success();
}

// Check if the block args are correctly annotated with sharding information:
//   - non-tensor and 0d-tensor args are ignored
//   - each tensor arg must have exactly one use, which must be a shard.shard
//   operation
static LogicalResult checkFullyAnnotated(Block &block) {
  for (const BlockArgument &arg : block.getArguments()) {
    auto rankedTensorArg = dyn_cast<TypedValue<RankedTensorType>>(arg);
    if (!rankedTensorArg || rankedTensorArg.getType().getRank() == 0)
      continue;

    if (rankedTensorArg.getNumUses() > 1)
      return emitError(block.getParent()->getLoc())
             << "Cannot partition: expected a single use for block argument "
             << arg.getArgNumber() << " in block "
             << block.computeBlockNumber();
    Operation *useOp = *rankedTensorArg.getUsers().begin();
    auto shardOp = dyn_cast<ShardOp>(useOp);
    if (!shardOp)
      return emitError(block.getParent()->getLoc())
             << "Cannot partition: expected a shard.shard op for block "
             << "argument " << arg.getArgNumber() << " in block "
             << block.computeBlockNumber();
  }
  return success();
}

// Check if the operation is correctly and fully annotated with sharding
// information:
//   - Operation results must have exactly one use (e.g. the shard operation).
//   - All operands and all results must be annotated, e.g. they must be
//     produced by/consumed by a shard.shard operation.
//   - Result annotations must not include the 'annotate_for_users' attribute.
//   - Operand annotations must include the 'annotate_for_users' attribute.
// raises an error if the operation is not correctly and fully annotated.
static LogicalResult checkFullyAnnotated(Operation *op) {
  // constant ops do not need to have sharding annotations
  if (op->hasTrait<OpTrait::ConstantLike>())
    return success();

  for (OpOperand &operand : op->getOpOperands()) {
    // non-tensor and 0d-tensor operands are ignored
    auto rankedTT = dyn_cast<RankedTensorType>(operand.get().getType());
    if (!rankedTT || rankedTT.getRank() == 0)
      continue;

    auto shard = operand.get().getDefiningOp<ShardOp>();
    if (!shard)
      return op->emitError() << "Cannot partition: tensor operand "
                             << operand.getOperandNumber()
                             << " must be defined by a shard.shard operation.";
    if (!shard.getAnnotateForUsers())
      return op->emitError()
             << "Cannot partition: shard.shard for operand "
             << operand.getOperandNumber() << " must set 'annotate_for_users'.";
  }
  for (const OpResult &result : op->getResults()) {
    if (!result.hasOneUse())
      return op->emitError()
             << "Cannot partition: result " << result.getResultNumber()
             << " must have exactly one use.";
    auto shard = dyn_cast<ShardOp>(*result.user_begin());
    if (!shard)
      return op->emitError()
             << "Cannot partition: user of result " << result.getResultNumber()
             << " must be shard.shard operation.";
    if (shard.getAnnotateForUsers())
      return op->emitError() << "Cannot partition: shard.shard for result "
                             << result.getResultNumber()
                             << " must not set 'annotate_for_users'.";
  }
  return success();
}

static LogicalResult
partitionOperation(Operation &op, IRMapping &partitionMap,
                   SymbolTableCollection &symbolTableCollection,
                   OpBuilder &builder) {
  if (isa<ShardingOp>(op)) {
    return success();
  }

  if (auto getShardingOp = dyn_cast<GetShardingOp>(op)) {
    auto shardOp = getShardingOp.getSource().getDefiningOp<ShardOp>();
    if (!shardOp) {
      return op.emitError("expected a shard op as source of get_sharding");
    }
    auto newSharding = builder.clone(*shardOp.getSharding().getDefiningOp());
    partitionMap.map(op.getResult(0), newSharding->getResult(0));
    return success();
  }

  ShardOp shardOp = llvm::dyn_cast<ShardOp>(op);
  if (shardOp) {
    return partitionOperation(shardOp, partitionMap, symbolTableCollection,
                              builder);
  }

  // Check if operation is correctly and fully annotated.
  if (failed(checkFullyAnnotated(&op)))
    return failure();

  SmallVector<Value> partitionedOperands;
  llvm::transform(op.getOperands(), std::back_inserter(partitionedOperands),
                  [&partitionMap](Value operand) {
                    assert(partitionMap.contains(operand));
                    return partitionMap.lookup(operand);
                  });
  return partitionOperation(op, partitionedOperands, getOperandShardings(op),
                            getResultShardings(op), partitionMap,
                            symbolTableCollection, builder);
}

static LogicalResult
partitionBlock(Block &block, IRMapping &partitionMap,
               SymbolTableCollection &symbolTableCollection,
               OpBuilder &builder) {

  if (failed(checkFullyAnnotated(block)))
    return failure();

  SmallVector<Location> argLocations;
  llvm::transform(block.getArguments(), std::back_inserter(argLocations),
                  [](BlockArgument arg) { return arg.getLoc(); });
  Block *newBlock = builder.createBlock(
      block.getParent(), {},
      shardedBlockArgumentTypes(block, symbolTableCollection), argLocations);
  for (auto [unshardedBlockArg, partitionedBlockArg] :
       llvm::zip(block.getArguments(), newBlock->getArguments())) {
    partitionMap.map(unshardedBlockArg, partitionedBlockArg);
  }

  OpBuilder::InsertionGuard insertionGuard(builder);
  builder.setInsertionPointToEnd(newBlock);
  for (Operation &op : block.getOperations()) {
    if (failed(partitionOperation(op, partitionMap, symbolTableCollection,
                                  builder))) {
      return failure();
    }
  }

  return success();
}

static LogicalResult
partitionFuncOp(FunctionOpInterface op, IRMapping &partitionMap,
                SymbolTableCollection &symbolTableCollection) {
  OpBuilder builder(op.getFunctionBody());

  // Snapshot the original blocks to not mess up the iteration when adding new
  // blocks.
  SmallVector<Block *> originalBlocks;
  for (Block &b : op.getBlocks()) {
    if (llvm::any_of(b.getOperations(),
                     [](Operation &op) { return isa<ShardOp>(op); })) {
      originalBlocks.push_back(&b);
    }
  }

  for (Block *block : originalBlocks) {
    if (failed(partitionBlock(*block, partitionMap, symbolTableCollection,
                              builder))) {
      return failure();
    }
  }

  for (Block *block : originalBlocks) {
    block->erase();
  }

  // Find a return op and change the function results signature to its operands
  // signature.
  Operation *returnOp = nullptr;
  for (Block &block : op.getFunctionBody()) {
    if (block.empty()) {
      continue;
    }

    if (block.back().hasTrait<OpTrait::ReturnLike>()) {
      returnOp = &block.back();
      break;
    }
  }
  if (returnOp) {
    op.setType(FunctionType::get(
        op->getContext(), op.getFunctionBody().front().getArgumentTypes(),
        returnOp->getOperandTypes()));
  }

  return success();
}

namespace {

struct Partition : public impl::PartitionBase<Partition> {
  void runOnOperation() override {
    IRMapping partitionMap;
    SymbolTableCollection symbolTableCollection;
    if (failed(partitionFuncOp(getOperation(), partitionMap,
                               symbolTableCollection))) {
      return signalPassFailure();
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    reshardingRegisterDependentDialects(registry);
    registry.insert<shard::ShardDialect>();
  }
};

} // namespace

} // namespace mlir::shard
