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
#include <array>
#include <iterator>
#include <memory>
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

/// Base class for resharding patterns.
/// Subclasses implement `tryApply` to detect and apply a specific resharding.
class ReshardingPattern {
public:
  virtual ~ReshardingPattern() = default;

  /// Try to apply this resharding pattern. Returns the resharded value and
  /// resulting sharding on success, or std::nullopt if the pattern doesn't
  /// match.
  virtual std::optional<std::tuple<TypedValue<ShapedType>, Sharding>>
  tryApply(ImplicitLocOpBuilder &builder, GridOp grid, int64_t tensorDim,
           const Sharding &srcSharding, const Sharding &tgtSharding,
           ShapedType srcUnshardedType, TypedValue<ShapedType> srcShard) = 0;

protected:
  /// Returns true if either sharding has non-empty static sharded dims offsets.
  static bool hasStaticOffsets(const Sharding &srcSharding,
                               const Sharding &tgtSharding) {
    return !srcSharding.getStaticShardedDimsOffsets().empty() ||
           !tgtSharding.getStaticShardedDimsOffsets().empty();
  }

  /// Returns true if either sharding has non-empty static sharded dims offsets
  /// or non-empty static halo sizes.
  static bool hasStaticOffsetsOrHalos(const Sharding &srcSharding,
                                      const Sharding &tgtSharding) {
    return hasStaticOffsets(srcSharding, tgtSharding) ||
           !srcSharding.getStaticHaloSizes().empty() ||
           !tgtSharding.getStaticHaloSizes().empty();
  }
};

/// Split a replicated axis: e.g. [[0, 1]] -> [[0, 1, 2]].
class SplitLastAxisPattern : public ReshardingPattern {
  static Sharding tgtSharding(MLIRContext *ctx, const Sharding &srcSharding,
                              int64_t splitTensorDim, GridAxis splitGridAxis) {
    SmallVector<GridAxesAttr> tgtShardingSplitAxes =
        llvm::to_vector(srcSharding.getSplitAxes());
    while (static_cast<int64_t>(tgtShardingSplitAxes.size()) <=
           splitTensorDim) {
      tgtShardingSplitAxes.push_back(GridAxesAttr::get(ctx, {}));
    }
    auto tgtSplitAxes =
        llvm::to_vector(tgtShardingSplitAxes[splitTensorDim].asArrayRef());
    tgtSplitAxes.push_back(splitGridAxis);
    tgtShardingSplitAxes[splitTensorDim] = GridAxesAttr::get(ctx, tgtSplitAxes);
    return Sharding::get(srcSharding.getGridAttr(), tgtShardingSplitAxes);
  }

  // Split a replicated tensor along a grid axis.
  // E.g. [[0, 1]] -> [[0, 1, 2]].
  // Returns the partitioned target value with its sharding.
  static std::tuple<TypedValue<ShapedType>, Sharding>
  apply(ImplicitLocOpBuilder &builder, Sharding srcSharding,
        TypedValue<ShapedType> srcShard, GridOp grid, int64_t splitTensorDim,
        GridAxis splitGridAxis) {
    TypedValue<ShapedType> tgtShard =
        AllSliceOp::create(builder, srcShard, grid,
                           ArrayRef<GridAxis>(splitGridAxis), splitTensorDim)
            .getResult();
    Sharding resultSharding =
        tgtSharding(builder.getContext(), std::move(srcSharding),
                    splitTensorDim, splitGridAxis);
    return {tgtShard, resultSharding};
  }

  // Detect if the resharding is of type e.g.
  // [[0, 1]] -> [[0, 1, 2]].
  // If detected, returns the corresponding grid axis.
  // Does not detect insertions like
  // [[0, 1]] -> [[0, 2, 1]].
  static std::optional<GridAxis> detect(const Sharding &srcSharding,
                                        const Sharding &tgtSharding,
                                        int64_t tensorDim) {
    if (static_cast<size_t>(tensorDim) >= tgtSharding.getSplitAxes().size())
      return std::nullopt;
    auto tgtAxes = tgtSharding.getSplitAxes()[tensorDim].asArrayRef();
    if (srcSharding.getSplitAxes().size() > static_cast<size_t>(tensorDim)) {
      auto srcAxes = srcSharding.getSplitAxes()[tensorDim].asArrayRef();
      if (srcAxes.size() + 1 != tgtAxes.size())
        return std::nullopt;
      if (!llvm::equal(srcAxes,
                       llvm::make_range(tgtAxes.begin(), tgtAxes.end() - 1)))
        return std::nullopt;
    } else {
      if (tgtAxes.size() != 1)
        return std::nullopt;
    }
    return tgtAxes.back();
  }

public:
  std::optional<std::tuple<TypedValue<ShapedType>, Sharding>>
  tryApply(ImplicitLocOpBuilder &builder, GridOp grid, int64_t tensorDim,
           const Sharding &srcSharding, const Sharding &tgtSharding,
           ShapedType srcUnshardedType,
           TypedValue<ShapedType> srcShard) override {
    if (hasStaticOffsetsOrHalos(srcSharding, tgtSharding))
      return std::nullopt;
    if (auto gridAxis = detect(srcSharding, tgtSharding, tensorDim))
      return apply(builder, srcSharding, srcShard, grid, tensorDim,
                   gridAxis.value());
    return std::nullopt;
  }
};

/// Unsplit trailing axes: e.g. [[0, 1, 2]] -> [[0, 1]] or [[0, 1, 2]] -> [].
class UnsplitLastAxesPattern : public ReshardingPattern {
  // Detect if the resharding removes trailing split axes along a tensor
  // dimension, e.g.
  // [[0, 1, 2]] -> [[0, 1]], [[0, 1, 2]] -> [0] or [[0, 1, 2]] -> [].
  // If detected, returns the removed trailing split axes (grid axes).
  static std::optional<SmallVector<GridAxis>>
  detect(const Sharding &srcSharding, const Sharding &tgtSharding,
         int64_t tensorDim) {
    if (static_cast<size_t>(tensorDim) >= srcSharding.getSplitAxes().size())
      return std::nullopt;
    size_t dimOff = 0;
    auto srcSplitAxes = srcSharding.getSplitAxes()[tensorDim].asArrayRef();
    if (tgtSharding.getSplitAxes().size() > static_cast<size_t>(tensorDim)) {
      auto tgtSplitAxes = tgtSharding.getSplitAxes()[tensorDim].asArrayRef();
      // No match if the target sharding does not have less split axes than
      // the source sharding along the current tensor dimension.
      if (srcSplitAxes.size() <= tgtSplitAxes.size())
        return std::nullopt;
      // No match if the split axes of the target sharding are different from
      // the first split axes of the source sharding.
      if (!std::equal(tgtSplitAxes.begin(), tgtSplitAxes.end(),
                      srcSplitAxes.begin()))
        return std::nullopt;
      dimOff = tgtSplitAxes.size();
    } else {
      // Here the target dimension is replicated; there is nothing to do if
      // the source dimension is also replicated.
      if (srcSplitAxes.size() == 0)
        return std::nullopt;
      dimOff = 0;
    }
    // This is a match. Return the trailing grid axes of the source sharding
    // along this dimension.
    ArrayRef<GridAxis> trailingAxes = srcSplitAxes.drop_front(dimOff);
    SmallVector<GridAxis> unsplitAxes(trailingAxes.begin(), trailingAxes.end());
    return unsplitAxes;
  }

  // Return the resulting Sharding if the unsplit last axes resharding is
  // applied.
  static Sharding tgtSharding(MLIRContext *ctx, const Sharding &srcSharding,
                              int64_t splitTensorDim, size_t numUnsplitAxes) {
    SmallVector<GridAxesAttr> resSplitAxes =
        llvm::to_vector(srcSharding.getSplitAxes());
    assert(static_cast<int64_t>(resSplitAxes.size()) > splitTensorDim);
    ArrayRef<GridAxis> srcSplitAxes = resSplitAxes[splitTensorDim].asArrayRef();
    assert(srcSplitAxes.size() >= numUnsplitAxes);
    size_t numSplitAxes = srcSplitAxes.size() - numUnsplitAxes;
    SmallVector<GridAxis> newSplitAxes(srcSplitAxes.begin(),
                                       srcSplitAxes.begin() + numSplitAxes);
    resSplitAxes[splitTensorDim] = GridAxesAttr::get(ctx, newSplitAxes);
    return Sharding::get(srcSharding.getGridAttr(), resSplitAxes);
  }

  // Return the resulting Tensor type after applying the unsplit last axes
  // resharding.
  static ShapedType allGatherResultType(ShapedType srcType,
                                        int64_t splitTensorDim,
                                        ArrayRef<int64_t> gridShape,
                                        ArrayRef<GridAxis> unsplitAxes) {
    SmallVector<int64_t> tgtShape = llvm::to_vector(srcType.getShape());
    for (GridAxis gridAxis : unsplitAxes)
      tgtShape[splitTensorDim] =
          gatherDimension(tgtShape[splitTensorDim], gridShape[gridAxis]);
    return srcType.cloneWith(tgtShape, srcType.getElementType());
  }

  // Perform the resharding for the unsplit last axes case.
  // This basically performs an all-gather along the unsplit grid axes.
  static std::tuple<TypedValue<ShapedType>, Sharding>
  apply(ImplicitLocOpBuilder &builder, Sharding srcSharding,
        ShapedType srcUnshardedType, TypedValue<ShapedType> srcShard,
        GridOp grid, int64_t splitTensorDim, ArrayRef<GridAxis> unsplitAxes) {
    MLIRContext *ctx = builder.getContext();
    builder.setInsertionPointAfterValue(srcShard);

    Sharding resultSharding = tgtSharding(ctx, std::move(srcSharding),
                                          splitTensorDim, unsplitAxes.size());
    ShapedType agResultType = allGatherResultType(
        srcShard.getType(), splitTensorDim, grid.getShape(), unsplitAxes);
    Value allGatherResult = AllGatherOp::create(
        builder,
        RankedTensorType::get(agResultType.getShape(),
                              agResultType.getElementType()),
        grid.getSymName(), unsplitAxes, srcShard, APInt(64, splitTensorDim));
    ShapedType tgtType =
        shardShapedType(srcUnshardedType, grid, resultSharding);
    TypedValue<ShapedType> tgtShard =
        tensor::CastOp::create(builder, tgtType, allGatherResult).getResult();
    return {tgtShard, resultSharding};
  }

public:
  std::optional<std::tuple<TypedValue<ShapedType>, Sharding>>
  tryApply(ImplicitLocOpBuilder &builder, GridOp grid, int64_t tensorDim,
           const Sharding &srcSharding, const Sharding &tgtSharding,
           ShapedType srcUnshardedType,
           TypedValue<ShapedType> srcShard) override {
    if (hasStaticOffsetsOrHalos(srcSharding, tgtSharding))
      return std::nullopt;
    if (auto gridAxes = detect(srcSharding, tgtSharding, tensorDim))
      return apply(builder, srcSharding, srcUnshardedType, srcShard, grid,
                   tensorDim, gridAxes.value());
    return std::nullopt;
  }
};

/// Move a split axis between tensor dimensions:
/// e.g. [[0], []] -> [[], [0]].
class MoveSplitAxisPattern : public ReshardingPattern {
  // Detect if the resharding moves a single split axis from one tensor
  // dimension to another tensor dimension. If detected, returns the
  // corresponding (tgt_tensor_dim, grid_axis) pair.
  static std::optional<std::tuple<int64_t, GridAxis>>
  detect(const Sharding &srcSharding, const Sharding &tgtSharding,
         int64_t srcTensorDim) {
    if (static_cast<size_t>(srcTensorDim) >= srcSharding.getSplitAxes().size())
      return std::nullopt;
    auto srcAxes = srcSharding.getSplitAxes()[srcTensorDim].asArrayRef();
    if (srcAxes.size() != 1)
      return std::nullopt;
    for (size_t tgtTensorDim = 0;
         tgtTensorDim < tgtSharding.getSplitAxes().size(); ++tgtTensorDim) {
      if (static_cast<int64_t>(tgtTensorDim) == srcTensorDim)
        continue;
      auto tgtAxes = tgtSharding.getSplitAxes()[tgtTensorDim].asArrayRef();
      if (tgtAxes.size() != 1 || srcAxes.front() != tgtAxes.front())
        continue;
      return std::make_tuple(static_cast<int64_t>(tgtTensorDim),
                             srcAxes.front());
    }
    return std::nullopt;
  }

  static Sharding tgtSharding(MLIRContext *ctx, const Sharding &srcSharding,
                              int64_t srcTensorDim, int64_t tgtTensorDim) {
    SmallVector<GridAxesAttr> tgtShardingSplitAxes =
        llvm::to_vector(srcSharding.getSplitAxes());
    while (static_cast<int64_t>(tgtShardingSplitAxes.size()) <= tgtTensorDim) {
      tgtShardingSplitAxes.push_back(GridAxesAttr::get(ctx, {}));
    }

    auto srcSplitAxes =
        llvm::to_vector(tgtShardingSplitAxes[srcTensorDim].asArrayRef());
    assert(srcSplitAxes.size() == 1);
    auto gridAxis = srcSplitAxes.back();
    srcSplitAxes.pop_back();
    tgtShardingSplitAxes[srcTensorDim] = GridAxesAttr::get(ctx, srcSplitAxes);

    auto tgtSplitAxes =
        llvm::to_vector(tgtShardingSplitAxes[tgtTensorDim].asArrayRef());
    tgtSplitAxes.push_back(gridAxis);
    tgtShardingSplitAxes[tgtTensorDim] = GridAxesAttr::get(ctx, tgtSplitAxes);

    return Sharding::get(srcSharding.getGridAttr(), tgtShardingSplitAxes);
  }

  static ShapedType allToAllResultShape(ShapedType srcShape, int64_t splitCount,
                                        int64_t srcTensorDim,
                                        int64_t tgtTensorDim) {
    SmallVector<int64_t> tgtShape = llvm::to_vector(srcShape.getShape());
    tgtShape[srcTensorDim] =
        gatherDimension(tgtShape[srcTensorDim], splitCount);
    tgtShape[tgtTensorDim] = shardDimension(tgtShape[tgtTensorDim], splitCount);
    return srcShape.cloneWith(tgtShape, srcShape.getElementType());
  }

  static std::tuple<TypedValue<ShapedType>, Sharding>
  apply(ImplicitLocOpBuilder &builder, GridOp grid, Sharding srcSharding,
        ShapedType srcUnshardedType, TypedValue<ShapedType> srcShard,
        int64_t srcTensorDim, int64_t tgtTensorDim, GridAxis gridAxis) {
    MLIRContext *ctx = builder.getContext();
    builder.setInsertionPointAfterValue(srcShard);

    Sharding resultSharding =
        tgtSharding(ctx, std::move(srcSharding), srcTensorDim, tgtTensorDim);
    ShapedType a2aResultShape =
        allToAllResultShape(srcShard.getType(), grid.getShape()[gridAxis],
                            srcTensorDim, tgtTensorDim);
    Value allToAllResult = AllToAllOp::create(
        builder,
        RankedTensorType::get(a2aResultShape.getShape(),
                              a2aResultShape.getElementType()),
        grid.getSymName(), SmallVector<GridAxis>({gridAxis}), srcShard,
        APInt(64, tgtTensorDim), APInt(64, srcTensorDim));
    ShapedType tgtShape =
        shardShapedType(srcUnshardedType, grid, resultSharding);
    TypedValue<ShapedType> tgtShard =
        tensor::CastOp::create(builder, tgtShape, allToAllResult).getResult();
    return {tgtShard, resultSharding};
  }

public:
  std::optional<std::tuple<TypedValue<ShapedType>, Sharding>>
  tryApply(ImplicitLocOpBuilder &builder, GridOp grid, int64_t tensorDim,
           const Sharding &srcSharding, const Sharding &tgtSharding,
           ShapedType srcUnshardedType,
           TypedValue<ShapedType> srcShard) override {
    if (hasStaticOffsetsOrHalos(srcSharding, tgtSharding))
      return std::nullopt;
    if (auto detectRes = detect(srcSharding, tgtSharding, tensorDim)) {
      auto [tgtTensorDim, gridAxis] = detectRes.value();
      return apply(builder, grid, srcSharding, srcUnshardedType, srcShard,
                   tensorDim, tgtTensorDim, gridAxis);
    }
    return std::nullopt;
  }
};

/// Update halo sizes: handles cases where only the halo sizes differ between
/// source and target sharding. Requires copying the "core" of the source tensor
/// into the "core" of the destination tensor followed by an update halo op.
class UpdateHaloPattern : public ReshardingPattern {
public:
  std::optional<std::tuple<TypedValue<ShapedType>, Sharding>>
  tryApply(ImplicitLocOpBuilder &builder, GridOp grid, int64_t tensorDim,
           const Sharding &srcSharding, const Sharding &tgtSharding,
           ShapedType srcUnshardedType,
           TypedValue<ShapedType> srcShard) override {
    // UpdateHaloPattern handles all dimensions at once; only trigger on dim 0.
    if (tensorDim != 0)
      return std::nullopt;
    // Currently handles only cases where halo sizes differ but everything else
    // stays the same (from source to destination sharding).
    if (!srcSharding.equalSplitAxes(tgtSharding) ||
        hasStaticOffsets(srcSharding, tgtSharding) ||
        srcSharding.equalHaloSizes(tgtSharding)) {
      return std::nullopt;
    }

    auto srcHaloSizes = srcSharding.getStaticHaloSizes();
    auto tgtHaloSizes = tgtSharding.getStaticHaloSizes();
    assert(srcHaloSizes.empty() || srcHaloSizes.size() == tgtHaloSizes.size());
    assert(((srcHaloSizes.empty() || ShapedType::isStaticShape(srcHaloSizes)) &&
            ShapedType::isStaticShape(tgtHaloSizes) &&
            srcShard.getType().hasStaticShape()) &&
           "dynamic shapes/halos are not supported yet for shard-partition");
    auto rank = srcShard.getType().getRank();
    auto splitAxes = srcSharding.getSplitAxes();
    SmallVector<int64_t> srcCoreOffs(rank, 0), tgtCoreOffs(rank, 0),
        strides(rank, 1), outShape(srcShard.getType().getShape()),
        coreShape(srcShard.getType().getShape());

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
    auto initVal = tensor::EmptyOp::create(builder, srcShard.getLoc(), outShape,
                                           srcShard.getType().getElementType());
    auto core = tensor::ExtractSliceOp::create(
        builder, srcShard.getLoc(),
        RankedTensorType::get(coreShape, srcShard.getType().getElementType()),
        srcShard, noVals, noVals, noVals, srcCoreOffs, coreShape, strides);
    auto initOprnd = tensor::InsertSliceOp::create(
        builder, srcShard.getLoc(), core, initVal, noVals, noVals, noVals,
        tgtCoreOffs, coreShape, strides);

    // Finally update the halo.
    auto updateHaloResult =
        UpdateHaloOp::create(builder, srcShard.getLoc(),
                             RankedTensorType::get(
                                 outShape, srcShard.getType().getElementType()),
                             initOprnd, grid.getSymName(),
                             GridAxesArrayAttr::get(builder.getContext(),
                                                    srcSharding.getSplitAxes()),
                             tgtSharding.getDynamicHaloSizes(),
                             tgtSharding.getStaticHaloSizes())
            .getResult();
    return std::make_tuple(cast<TypedValue<ShapedType>>(updateHaloResult),
                           tgtSharding);
  }
};

// In most cases the sharded tensor axes must be exactly divisible by the single
// grid axis size. Only halo size changes can deal with non-divisible cases.
static TypedValue<ShapedType> reshard(ImplicitLocOpBuilder &builder,
                                      GridOp grid, const Sharding &srcSharding,
                                      const Sharding &tgtSharding,
                                      TypedValue<ShapedType> unshardedSrc,
                                      TypedValue<ShapedType> shardedSrc) {
  // If source and destination sharding are the same, no need to do anything.
  if (srcSharding == tgtSharding ||
      (isFullReplication(srcSharding) && isFullReplication(tgtSharding))) {
    return shardedSrc;
  }

  assert(shardedSrc.getType() ==
         shardShapedType(unshardedSrc.getType(), grid, srcSharding));
  [[maybe_unused]] ShapedType tgtShardType =
      shardShapedType(unshardedSrc.getType(), grid, tgtSharding);
  assert(shardedSrc.getType().getRank() == tgtShardType.getRank());
  assert(unshardedSrc.getType().getRank() == tgtShardType.getRank());

  // Each pattern's tryApply checks its own applicability preconditions.
  static UpdateHaloPattern updateHaloPattern;
  static MoveSplitAxisPattern moveSplitAxisPattern;
  static SplitLastAxisPattern splitLastAxisPattern;
  static UnsplitLastAxesPattern unsplitLastAxesPattern;
  static ReshardingPattern *patterns[] = {
      &updateHaloPattern, &moveSplitAxisPattern, &splitLastAxisPattern,
      &unsplitLastAxesPattern};
  TypedValue<ShapedType> currentShard = shardedSrc;
  Sharding currentSharding = srcSharding;
  for (int64_t dim = 0;
       dim < tgtShardType.getRank() && currentSharding != tgtSharding; ++dim) {
    for (auto &pattern : patterns) {
      if (auto tryRes = pattern->tryApply(builder, grid, dim, currentSharding,
                                          tgtSharding, unshardedSrc.getType(),
                                          currentShard)) {
        std::tie(currentShard, currentSharding) = tryRes.value();
        break;
      }
    }
  }

  if (currentSharding != tgtSharding ||
      currentShard.getType() != tgtShardType) {
    builder.emitError()
        << "Failed to reshard; probably hitting an unknown resharding pattern:"
        << " got " << currentSharding << " expected " << tgtSharding
        << " got type " << currentShard.getType() << " expected "
        << tgtShardType;
    return TypedValue<ShapedType>();
  }
  return currentShard;
}

TypedValue<ShapedType> reshard(OpBuilder &builder, GridOp grid,
                               ShardOp srcShardOp, ShardOp tgtShardOp,
                               TypedValue<ShapedType> shardedSrc) {
  assert(srcShardOp.getResult() == tgtShardOp.getSrc());
  auto srcSharding = srcShardOp.getSharding();
  auto tgtSharding = tgtShardOp.getSharding();
  ImplicitLocOpBuilder implicitLocOpBuilder(tgtShardOp->getLoc(), builder);
  return reshard(implicitLocOpBuilder, grid, srcSharding, tgtSharding,
                 srcShardOp.getSrc(), shardedSrc);
}

TypedValue<ShapedType> reshard(OpBuilder &builder, ShardOp srcShardOp,
                               ShardOp tgtShardOp,
                               TypedValue<ShapedType> shardedSrc,
                               SymbolTableCollection &symbolTableCollection) {
  GridOp srcGrid = getGrid(srcShardOp, symbolTableCollection);
  assert(srcGrid && srcGrid == getGrid(tgtShardOp, symbolTableCollection));
  return reshard(builder, srcGrid, srcShardOp, tgtShardOp, shardedSrc);
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
  Value tgtPartitionValue;

  // Check if 2 shard ops are chained. If not there is no need for resharding
  // as the source and target shared the same sharding.
  ShardOp srcShardOp = shardOp.getSrc().getDefiningOp<ShardOp>();
  if (!srcShardOp) {
    tgtPartitionValue = partitionMap.lookup(shardOp.getSrc());
  } else {
    // Insert resharding.
    TypedValue<ShapedType> shardedSrc =
        cast<TypedValue<ShapedType>>(partitionMap.lookup(srcShardOp));
    tgtPartitionValue = reshard(builder, srcShardOp, shardOp, shardedSrc,
                                symbolTableCollection);
    if (!tgtPartitionValue) {
      return shardOp.emitError()
             << "Failed to reshard from " << srcShardOp.getSharding() << " to "
             << shardOp.getSharding();
    }
  }

  assert(!partitionMap.contains(shardOp.getResult()));
  partitionMap.map(shardOp.getResult(), tgtPartitionValue);
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
