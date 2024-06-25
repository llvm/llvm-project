//===- Spmdization.cpp --------------------------------------------- C++ --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Mesh/Transforms/Spmdization.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Mesh/IR/MeshDialect.h"
#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/Dialect/Mesh/Interfaces/ShardingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <iterator>
#include <optional>
#include <tuple>
#include <type_traits>

namespace mlir::mesh {

template <typename SourceAxes, typename TargetAxes>
static bool arePartialAxesCompatible(const SourceAxes &sourceAxes,
                                     const TargetAxes &targetAxes) {
  return llvm::all_of(targetAxes, [&sourceAxes](auto &targetAxis) {
    return sourceAxes.contains(targetAxis);
  });
}

// Return the reduced value and its corresponding sharding.
// Example:
// sourceSharding = <@mesh_1d, [[0]], partial = sum[0]>
// targetSharding = <@mesh_1d, [[]]>
// Then will apply all-reduce on the source value
// and return it with the sharding <@mesh_1d, [[0]]>.
static std::tuple<TypedValue<ShapedType>, MeshShardingAttr>
handlePartialAxesDuringResharding(OpBuilder &builder,
                                  MeshShardingAttr sourceSharding,
                                  MeshShardingAttr targetSharding,
                                  TypedValue<ShapedType> sourceShard) {
  if (sourceSharding.getPartialAxes().empty() &&
      targetSharding.getPartialAxes().empty()) {
    return {sourceShard, sourceSharding};
  }
  assert(targetSharding.getPartialAxes().empty() ||
         (!sourceSharding.getPartialAxes().empty() &&
          sourceSharding.getPartialType() == targetSharding.getPartialType()));
  using Axis = std::decay_t<decltype(sourceSharding.getPartialAxes().front())>;
  using AxisSet = llvm::SmallDenseSet<Axis>;
  AxisSet sourceShardingPartialAxesSet(sourceSharding.getPartialAxes().begin(),
                                       sourceSharding.getPartialAxes().end());
  AxisSet targetShardingPartialAxesSet(targetSharding.getPartialAxes().begin(),
                                       targetSharding.getPartialAxes().end());
  assert(arePartialAxesCompatible(sourceShardingPartialAxesSet,
                                  targetShardingPartialAxesSet));
  llvm::SmallVector<MeshAxis> allReduceMeshAxes;
  llvm::copy_if(sourceShardingPartialAxesSet,
                std::back_inserter(allReduceMeshAxes),
                [&targetShardingPartialAxesSet](Axis a) {
                  return !targetShardingPartialAxesSet.contains(a);
                });
  if (allReduceMeshAxes.empty()) {
    return {sourceShard, sourceSharding};
  }

  builder.setInsertionPointAfterValue(sourceShard);
  TypedValue<ShapedType> resultValue = cast<TypedValue<ShapedType>>(
      builder
          .create<AllReduceOp>(sourceShard.getLoc(), sourceShard.getType(),
                               sourceSharding.getMesh().getLeafReference(),
                               allReduceMeshAxes, sourceShard,
                               sourceSharding.getPartialType())
          .getResult());

  llvm::SmallVector<MeshAxis> remainingPartialAxes;
  llvm::copy_if(sourceShardingPartialAxesSet,
                std::back_inserter(allReduceMeshAxes),
                [&targetShardingPartialAxesSet](Axis a) {
                  return targetShardingPartialAxesSet.contains(a);
                });
  MeshShardingAttr resultSharding =
      MeshShardingAttr::get(builder.getContext(), sourceSharding.getMesh(),
                            sourceSharding.getSplitAxes(), remainingPartialAxes,
                            sourceSharding.getPartialType());
  return {resultValue, resultSharding};
}

static MeshShardingAttr
targetShardingInSplitLastAxis(MLIRContext *ctx, MeshShardingAttr sourceSharding,
                              int64_t splitTensorAxis, MeshAxis splitMeshAxis) {
  SmallVector<MeshAxesAttr> targetShardingSplitAxes =
      llvm::to_vector(sourceSharding.getSplitAxes());
  while (static_cast<int64_t>(targetShardingSplitAxes.size()) <=
         splitTensorAxis) {
    targetShardingSplitAxes.push_back(MeshAxesAttr::get(ctx, {}));
  }
  auto targetSplitAxes =
      llvm::to_vector(targetShardingSplitAxes[splitTensorAxis].asArrayRef());
  targetSplitAxes.push_back(splitMeshAxis);
  targetShardingSplitAxes[splitTensorAxis] =
      MeshAxesAttr::get(ctx, targetSplitAxes);
  return MeshShardingAttr::get(
      ctx, sourceSharding.getMesh(), targetShardingSplitAxes,
      sourceSharding.getPartialAxes(), sourceSharding.getPartialType());
}

// Split a replicated tensor along a mesh axis.
// e.g. [[0, 1]] -> [[0, 1, 2]].
// Returns the spmdized target value with its sharding.
static std::tuple<TypedValue<ShapedType>, MeshShardingAttr>
splitLastAxisInResharding(ImplicitLocOpBuilder &builder,
                          MeshShardingAttr sourceSharding,
                          TypedValue<ShapedType> sourceShard, MeshOp mesh,
                          int64_t splitTensorAxis, MeshAxis splitMeshAxis) {
  TypedValue<ShapedType> targetShard = cast<TypedValue<ShapedType>>(
      builder
          .create<AllSliceOp>(sourceShard, mesh,
                              ArrayRef<MeshAxis>(splitMeshAxis),
                              splitTensorAxis)
          .getResult());
  MeshShardingAttr targetSharding = targetShardingInSplitLastAxis(
      builder.getContext(), sourceSharding, splitTensorAxis, splitMeshAxis);
  return {targetShard, targetSharding};
}

// Detect if the resharding is of type e.g.
// [[0, 1]] -> [[0, 1, 2]].
// If detected, returns the corresponding tensor axis mesh axis pair.
// Does not detect insertions like
// [[0, 1]] -> [[0, 2, 1]].
static std::optional<std::tuple<int64_t, MeshAxis>>
detectSplitLastAxisInResharding(MeshShardingAttr sourceSharding,
                                MeshShardingAttr targetSharding) {
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

static std::optional<std::tuple<TypedValue<ShapedType>, MeshShardingAttr>>
trySplitLastAxisInResharding(ImplicitLocOpBuilder &builder, MeshOp mesh,
                             MeshShardingAttr sourceSharding,
                             MeshShardingAttr targetSharding,
                             TypedValue<ShapedType> sourceShard) {
  if (auto detectRes =
          detectSplitLastAxisInResharding(sourceSharding, targetSharding)) {
    auto [tensorAxis, meshAxis] = detectRes.value();
    return splitLastAxisInResharding(builder, sourceSharding, sourceShard, mesh,
                                     tensorAxis, meshAxis);
  }

  return std::nullopt;
}

// Detect if the resharding is of type e.g.
// [[0, 1, 2]] -> [[0, 1]].
// If detected, returns the corresponding tensor axis mesh axis pair.
static std::optional<std::tuple<int64_t, MeshAxis>>
detectUnsplitLastAxisInResharding(MeshShardingAttr sourceSharding,
                                  MeshShardingAttr targetSharding) {
  for (size_t tensorAxis = 0; tensorAxis < sourceSharding.getSplitAxes().size();
       ++tensorAxis) {
    if (targetSharding.getSplitAxes().size() > tensorAxis) {
      if (sourceSharding.getSplitAxes()[tensorAxis].size() !=
          targetSharding.getSplitAxes()[tensorAxis].size() + 1)
        continue;
      if (!llvm::equal(
              llvm::make_range(
                  sourceSharding.getSplitAxes()[tensorAxis]
                      .asArrayRef()
                      .begin(),
                  sourceSharding.getSplitAxes()[tensorAxis].asArrayRef().end() -
                      1),
              targetSharding.getSplitAxes()[tensorAxis].asArrayRef()))
        continue;
    } else {
      if (sourceSharding.getSplitAxes()[tensorAxis].size() != 1)
        continue;
    }
    return std::make_tuple(
        tensorAxis,
        sourceSharding.getSplitAxes()[tensorAxis].asArrayRef().back());
  }
  return std::nullopt;
}

static MeshShardingAttr
targetShardingInUnsplitLastAxis(MLIRContext *ctx,
                                MeshShardingAttr sourceSharding,
                                int64_t splitTensorAxis) {
  SmallVector<MeshAxesAttr> targetShardingSplitAxes =
      llvm::to_vector(sourceSharding.getSplitAxes());
  assert(static_cast<int64_t>(targetShardingSplitAxes.size()) >
         splitTensorAxis);
  auto targetSplitAxes =
      llvm::to_vector(targetShardingSplitAxes[splitTensorAxis].asArrayRef());

  targetSplitAxes.pop_back();
  targetShardingSplitAxes[splitTensorAxis] =
      MeshAxesAttr::get(ctx, targetSplitAxes);
  return MeshShardingAttr::get(
      ctx, sourceSharding.getMesh(), targetShardingSplitAxes,
      sourceSharding.getPartialAxes(), sourceSharding.getPartialType());
}

static ShapedType allGatherResultShapeInUnsplitLastAxis(
    ShapedType sourceShape, int64_t splitCount, int64_t splitTensorAxis) {
  SmallVector<int64_t> targetShape = llvm::to_vector(sourceShape.getShape());
  targetShape[splitTensorAxis] =
      gatherDimension(targetShape[splitTensorAxis], splitCount);
  return sourceShape.cloneWith(targetShape, sourceShape.getElementType());
}

static std::tuple<TypedValue<ShapedType>, MeshShardingAttr>
unsplitLastAxisInResharding(ImplicitLocOpBuilder &builder,
                            MeshShardingAttr sourceSharding,
                            ShapedType sourceUnshardedShape,
                            TypedValue<ShapedType> sourceShard, MeshOp mesh,
                            int64_t splitTensorAxis, MeshAxis splitMeshAxis) {
  MLIRContext *ctx = builder.getContext();
  builder.setInsertionPointAfterValue(sourceShard);

  MeshShardingAttr targetSharding =
      targetShardingInUnsplitLastAxis(ctx, sourceSharding, splitTensorAxis);
  ShapedType allGatherResultShape = allGatherResultShapeInUnsplitLastAxis(
      sourceShard.getType(), mesh.getShape()[splitMeshAxis], splitTensorAxis);
  Value allGatherResult = builder.create<AllGatherOp>(
      RankedTensorType::get(allGatherResultShape.getShape(),
                            allGatherResultShape.getElementType()),
      mesh.getSymName(), SmallVector<MeshAxis>({splitMeshAxis}), sourceShard,
      APInt(64, splitTensorAxis));
  ShapedType targetShape =
      shardShapedType(sourceUnshardedShape, mesh, targetSharding);
  TypedValue<ShapedType> targetShard = cast<TypedValue<ShapedType>>(
      builder.create<tensor::CastOp>(targetShape, allGatherResult).getResult());
  return {targetShard, targetSharding};
}

static std::optional<std::tuple<TypedValue<ShapedType>, MeshShardingAttr>>
tryUnsplitLastAxisInResharding(ImplicitLocOpBuilder &builder, MeshOp mesh,
                               MeshShardingAttr sourceSharding,
                               MeshShardingAttr targetSharding,
                               ShapedType sourceUnshardedShape,
                               TypedValue<ShapedType> sourceShard) {
  if (auto detectRes =
          detectUnsplitLastAxisInResharding(sourceSharding, targetSharding)) {
    auto [tensorAxis, meshAxis] = detectRes.value();
    return unsplitLastAxisInResharding(builder, sourceSharding,
                                       sourceUnshardedShape, sourceShard, mesh,
                                       tensorAxis, meshAxis);
  }

  return std::nullopt;
}

// Detect if the resharding is of type e.g.
// [[0, 1], [2]] -> [[0], [1, 2]].
// Only moving the last axis counts.
// If detected, returns the corresponding (source_tensor_axis,
// target_tensor_axis, mesh_axis) tuple.
static std::optional<std::tuple<int64_t, int64_t, MeshAxis>>
detectMoveLastSplitAxisInResharding(MeshShardingAttr sourceSharding,
                                    MeshShardingAttr targetSharding) {
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

static MeshShardingAttr
targetShardingInMoveLastAxis(MLIRContext *ctx, MeshShardingAttr sourceSharding,
                             int64_t sourceTensorAxis,
                             int64_t targetTensorAxis) {
  SmallVector<MeshAxesAttr> targetShardingSplitAxes =
      llvm::to_vector(sourceSharding.getSplitAxes());
  while (static_cast<int64_t>(targetShardingSplitAxes.size()) <=
         targetTensorAxis) {
    targetShardingSplitAxes.push_back(MeshAxesAttr::get(ctx, {}));
  }

  auto sourceSplitAxes =
      llvm::to_vector(targetShardingSplitAxes[sourceTensorAxis].asArrayRef());
  assert(!sourceSplitAxes.empty());
  auto meshAxis = sourceSplitAxes.back();
  sourceSplitAxes.pop_back();
  targetShardingSplitAxes[sourceTensorAxis] =
      MeshAxesAttr::get(ctx, sourceSplitAxes);

  auto targetSplitAxes =
      llvm::to_vector(targetShardingSplitAxes[targetTensorAxis].asArrayRef());
  targetSplitAxes.push_back(meshAxis);
  targetShardingSplitAxes[targetTensorAxis] =
      MeshAxesAttr::get(ctx, targetSplitAxes);

  return MeshShardingAttr::get(
      ctx, sourceSharding.getMesh(), targetShardingSplitAxes,
      sourceSharding.getPartialAxes(), sourceSharding.getPartialType());
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

static std::tuple<TypedValue<ShapedType>, MeshShardingAttr>
moveLastSplitAxisInResharding(ImplicitLocOpBuilder &builder, MeshOp mesh,
                              MeshShardingAttr sourceSharding,
                              ShapedType sourceUnshardedShape,
                              TypedValue<ShapedType> sourceShard,
                              int64_t sourceTensorAxis,
                              int64_t targetTensorAxis, MeshAxis meshAxis) {
  MLIRContext *ctx = builder.getContext();
  builder.setInsertionPointAfterValue(sourceShard);

  MeshShardingAttr targetSharding = targetShardingInMoveLastAxis(
      ctx, sourceSharding, sourceTensorAxis, targetTensorAxis);
  ShapedType allToAllResultShape = allToAllResultShapeInMoveLastAxis(
      sourceShard.getType(), mesh.getShape()[meshAxis], sourceTensorAxis,
      targetTensorAxis);
  Value allToAllResult = builder.create<AllToAllOp>(
      RankedTensorType::get(allToAllResultShape.getShape(),
                            allToAllResultShape.getElementType()),
      mesh.getSymName(), SmallVector<MeshAxis>({meshAxis}), sourceShard,
      APInt(64, targetTensorAxis), APInt(64, sourceTensorAxis));
  ShapedType targetShape =
      shardShapedType(sourceUnshardedShape, mesh, targetSharding);
  TypedValue<ShapedType> targetShard = cast<TypedValue<ShapedType>>(
      builder.create<tensor::CastOp>(targetShape, allToAllResult).getResult());
  return {targetShard, targetSharding};
}

static std::optional<std::tuple<TypedValue<ShapedType>, MeshShardingAttr>>
tryMoveLastSplitAxisInResharding(ImplicitLocOpBuilder &builder, MeshOp mesh,
                                 MeshShardingAttr sourceSharding,
                                 MeshShardingAttr targetSharding,
                                 ShapedType sourceUnshardedShape,
                                 TypedValue<ShapedType> sourceShard) {
  if (auto detectRes =
          detectMoveLastSplitAxisInResharding(sourceSharding, targetSharding)) {
    auto [sourceTensorAxis, targetTensorAxis, meshAxis] = detectRes.value();
    return moveLastSplitAxisInResharding(
        builder, mesh, sourceSharding, sourceUnshardedShape, sourceShard,
        sourceTensorAxis, targetTensorAxis, meshAxis);
  }

  return std::nullopt;
}

// Handles only resharding on a 1D mesh.
// Currently the sharded tensor axes must be exactly divisible by the single
// mesh axis size.
static TypedValue<ShapedType>
reshardOn1DMesh(ImplicitLocOpBuilder &builder, MeshOp mesh,
                MeshShardingAttr sourceSharding,
                MeshShardingAttr targetSharding,
                TypedValue<ShapedType> sourceUnshardedValue,
                TypedValue<ShapedType> sourceShard) {
  assert(sourceShard.getType() ==
         shardShapedType(sourceUnshardedValue.getType(), mesh, sourceSharding));
  [[maybe_unused]] ShapedType targetShardType =
      shardShapedType(sourceUnshardedValue.getType(), mesh, targetSharding);
  assert(sourceShard.getType().getRank() == targetShardType.getRank());
  assert(mesh.getRank() == 1 && "Only 1D meshes are currently supported.");

  auto [reducedSourceShard, reducedSourceSharding] =
      handlePartialAxesDuringResharding(builder, sourceSharding, targetSharding,
                                        sourceShard);

  if (reducedSourceSharding == targetSharding) {
    return reducedSourceShard;
  }

  TypedValue<ShapedType> targetShard;
  MeshShardingAttr actualTargetSharding;
  if (auto tryRes = tryMoveLastSplitAxisInResharding(
          builder, mesh, reducedSourceSharding, targetSharding,
          sourceUnshardedValue.getType(), reducedSourceShard)) {
    std::tie(targetShard, actualTargetSharding) = tryRes.value();
  } else if (auto tryRes = trySplitLastAxisInResharding(
                 builder, mesh, reducedSourceSharding, targetSharding,
                 reducedSourceShard)) {
    std::tie(targetShard, actualTargetSharding) = tryRes.value();
  } else if (auto tryRes = tryUnsplitLastAxisInResharding(
                 builder, mesh, reducedSourceSharding, targetSharding,
                 sourceUnshardedValue.getType(), reducedSourceShard)) {
    std::tie(targetShard, actualTargetSharding) = tryRes.value();
  } else {
    assert(false && "Did not find any pattern to apply.");
  }

  assert(actualTargetSharding == targetSharding);
  assert(targetShard.getType() == targetShardType);
  return targetShard;
}

TypedValue<ShapedType> reshard(ImplicitLocOpBuilder &builder, MeshOp mesh,
                               MeshShardingAttr sourceSharding,
                               MeshShardingAttr targetSharding,
                               TypedValue<ShapedType> sourceUnshardedValue,
                               TypedValue<ShapedType> sourceShard) {
  // Resort to handling only 1D meshes since the general case is complicated if
  // it needs to be communication efficient in terms of minimizing the data
  // transfered between devices.
  return reshardOn1DMesh(builder, mesh, sourceSharding, targetSharding,
                         sourceUnshardedValue, sourceShard);
}

TypedValue<ShapedType> reshard(OpBuilder &builder, MeshOp mesh, ShardOp source,
                               ShardOp target,
                               TypedValue<ShapedType> sourceShardValue) {
  assert(source.getResult() == target.getOperand());
  ImplicitLocOpBuilder implicitLocOpBuilder(target->getLoc(), builder);
  return reshard(
      implicitLocOpBuilder, mesh, source.getShard(), target.getShard(),
      cast<TypedValue<ShapedType>>(source.getSrc()), sourceShardValue);
}

TypedValue<ShapedType> reshard(OpBuilder &builder, ShardOp source,
                               ShardOp target,
                               TypedValue<ShapedType> sourceShardValue,
                               SymbolTableCollection &symbolTableCollection) {
  MeshOp srcMesh = getMesh(source, symbolTableCollection);
  assert(srcMesh && srcMesh == getMesh(target, symbolTableCollection));
  return reshard(builder, srcMesh, source, target, sourceShardValue);
}

void reshardingRegisterDependentDialects(DialectRegistry &registry) {
  registry.insert<mesh::MeshDialect, tensor::TensorDialect>();
}

#define GEN_PASS_DEF_SPMDIZATION
#include "mlir/Dialect/Mesh/Transforms/Passes.h.inc"

using UnshardedToShardedValueMap = DenseMap<Value, Value>;

// Get the types of block arguments for an spmdized block.
// Reads the sharding annotations of the arguments to deduce the sharded types.
// Types that are not ranked tensors are left unchanged.
SmallVector<Type>
shardedBlockArgumentTypes(Block &block,
                          SymbolTableCollection &symbolTableCollection) {
  SmallVector<Type> res;
  llvm::transform(
      block.getArguments(), std::back_inserter(res),
      [&symbolTableCollection](BlockArgument arg) {
        auto rankedTensorArg = dyn_cast<TypedValue<RankedTensorType>>(arg);
        if (!rankedTensorArg) {
          return arg.getType();
        }

        assert(rankedTensorArg.hasOneUse());
        Operation *useOp = *rankedTensorArg.getUsers().begin();
        ShardOp shardOp = llvm::dyn_cast<ShardOp>(useOp);
        assert(shardOp);
        MeshOp mesh = getMesh(shardOp, symbolTableCollection);
        return cast<Type>(shardShapedType(rankedTensorArg.getType(), mesh,
                                          shardOp.getShardAttr()));
      });
  return res;
}

static LogicalResult spmdizeOperation(
    Operation &op, ArrayRef<Value> spmdizedOperands,
    ArrayRef<MeshShardingAttr> operandShardings,
    ArrayRef<MeshShardingAttr> resultShardings, IRMapping &spmdizationMap,
    SymbolTableCollection &symbolTableCollection, OpBuilder &builder) {
  ShardingInterface shardingInterface = llvm::dyn_cast<ShardingInterface>(op);
  if (!shardingInterface) {
    // If there is no sharding interface we are conservative and assume that
    // the op should be fully replicated no all devices.
    spmdizeFullyReplicatedOperation(op, spmdizedOperands, operandShardings,
                                    resultShardings, spmdizationMap,
                                    symbolTableCollection, builder);
  } else {
    if (failed(shardingInterface.spmdize(spmdizedOperands, operandShardings,
                                         resultShardings, spmdizationMap,
                                         symbolTableCollection, builder))) {
      return failure();
    }
  }

  assert(llvm::all_of(op.getResults(), [&spmdizationMap](OpResult result) {
    return spmdizationMap.contains(result);
  }));

  return success();
}

// Retrieve the sharding annotations for the operands of the given operation.
// If the type is not a ranked tensor it is not require to have an annotation.
static SmallVector<MeshShardingAttr> getOperandShardings(Operation &op) {
  SmallVector<MeshShardingAttr> res;
  res.reserve(op.getNumOperands());
  llvm::transform(op.getOperands(), std::back_inserter(res), [](Value operand) {
    TypedValue<RankedTensorType> rankedTensor =
        dyn_cast<TypedValue<RankedTensorType>>(operand);
    if (!rankedTensor) {
      return MeshShardingAttr();
    }

    Operation *definingOp = operand.getDefiningOp();
    assert(definingOp);
    ShardOp shardOp = llvm::cast<ShardOp>(definingOp);
    return shardOp.getShard();
  });
  return res;
}

// Retrieve the sharding annotations for the results of the given operation.
// If the type is not a ranked tensor it is not require to have an annotation.
static SmallVector<MeshShardingAttr> getResultShardings(Operation &op) {
  SmallVector<MeshShardingAttr> res;
  res.reserve(op.getNumResults());
  llvm::transform(op.getResults(), std::back_inserter(res),
                  [](OpResult result) {
                    TypedValue<RankedTensorType> rankedTensor =
                        dyn_cast<TypedValue<RankedTensorType>>(result);
                    if (!rankedTensor) {
                      return MeshShardingAttr();
                    }

                    assert(result.hasOneUse());
                    Operation *userOp = *result.getUsers().begin();
                    ShardOp shardOp = llvm::cast<ShardOp>(userOp);
                    return shardOp.getShard();
                  });
  return res;
}

static LogicalResult
spmdizeOperation(ShardOp shardOp, IRMapping &spmdizationMap,
                 SymbolTableCollection &symbolTableCollection,
                 OpBuilder &builder) {
  Value targetSpmdValue;

  // Check if 2 shard ops are chained. If not there is no need for resharding
  // as the source and target shared the same sharding.
  ShardOp srcShardOp =
      dyn_cast_or_null<ShardOp>(shardOp.getOperand().getDefiningOp());
  if (!srcShardOp) {
    targetSpmdValue = spmdizationMap.lookup(shardOp.getOperand());
  } else {
    // Insert resharding.
    TypedValue<ShapedType> srcSpmdValue = cast<TypedValue<ShapedType>>(
        spmdizationMap.lookup(srcShardOp.getOperand()));
    targetSpmdValue = reshard(builder, srcShardOp, shardOp, srcSpmdValue,
                              symbolTableCollection);
  }

  assert(!spmdizationMap.contains(shardOp.getResult()));
  spmdizationMap.map(shardOp.getResult(), targetSpmdValue);
  return success();
}

static LogicalResult
spmdizeOperation(Operation &op, IRMapping &spmdizationMap,
                 SymbolTableCollection &symbolTableCollection,
                 OpBuilder &builder) {
  ShardOp shardOp = llvm::dyn_cast<ShardOp>(op);
  if (shardOp) {
    return spmdizeOperation(shardOp, spmdizationMap, symbolTableCollection,
                            builder);
  }

  SmallVector<Value> spmdizedOperands;
  llvm::transform(op.getOperands(), std::back_inserter(spmdizedOperands),
                  [&spmdizationMap](Value operand) {
                    assert(spmdizationMap.contains(operand));
                    return spmdizationMap.lookup(operand);
                  });
  return spmdizeOperation(op, spmdizedOperands, getOperandShardings(op),
                          getResultShardings(op), spmdizationMap,
                          symbolTableCollection, builder);
}

static LogicalResult spmdizeBlock(Block &block, IRMapping &spmdizationMap,
                                  SymbolTableCollection &symbolTableCollection,
                                  OpBuilder &builder) {
  SmallVector<Location> argLocations;
  llvm::transform(block.getArguments(), std::back_inserter(argLocations),
                  [](BlockArgument arg) { return arg.getLoc(); });
  Block *newBlock = builder.createBlock(
      block.getParent(), {},
      shardedBlockArgumentTypes(block, symbolTableCollection), argLocations);
  for (auto [unshardedBlockArg, spmdizedBlockArg] :
       llvm::zip(block.getArguments(), newBlock->getArguments())) {
    spmdizationMap.map(unshardedBlockArg, spmdizedBlockArg);
  }

  OpBuilder::InsertionGuard insertionGuard(builder);
  builder.setInsertionPointToEnd(newBlock);
  for (Operation &op : block.getOperations()) {
    if (failed(spmdizeOperation(op, spmdizationMap, symbolTableCollection,
                                builder))) {
      return failure();
    }
  }

  return success();
}

static LogicalResult
spmdizeFuncOp(FunctionOpInterface op, IRMapping &spmdizationMap,
              SymbolTableCollection &symbolTableCollection) {
  OpBuilder builder(op.getFunctionBody());

  // Snapshot the original blocks to not mess up the iteration when adding new
  // blocks.
  SmallVector<Block *> originalBlocks;
  llvm::transform(op.getBlocks(), std::back_inserter(originalBlocks),
                  [](Block &b) { return &b; });

  for (Block *block : originalBlocks) {
    if (failed(spmdizeBlock(*block, spmdizationMap, symbolTableCollection,
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
  assert(returnOp);
  op.setType(FunctionType::get(op->getContext(),
                               op.getFunctionBody().front().getArgumentTypes(),
                               returnOp->getOperandTypes()));

  return success();
}

namespace {

struct Spmdization : public impl::SpmdizationBase<Spmdization> {
  void runOnOperation() override {
    IRMapping spmdizationMap;
    SymbolTableCollection symbolTableCollection;
    if (failed(spmdizeFuncOp(getOperation(), spmdizationMap,
                             symbolTableCollection))) {
      return signalPassFailure();
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    reshardingRegisterDependentDialects(registry);
    registry.insert<mesh::MeshDialect>();
  }
};

} // namespace

} // namespace mlir::mesh
