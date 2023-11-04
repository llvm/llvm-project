//===- ShardingInterface.cpp -------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Mesh/Interfaces/ShardingInterface.h"
#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <utility>

#define DEBUG_TYPE "sharding-interface"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;
using namespace mlir::mesh;

#include "mlir/Dialect/Mesh/Interfaces/ShardingInterface.cpp.inc"

//===----------------------------------------------------------------------===//
// common util functions
//===----------------------------------------------------------------------===//

static LogicalResult
checkOperandAffineExprRecursively(AffineExpr expr,
                                  SmallVectorImpl<bool> &seenIds) {
  switch (expr.getKind()) {
  case AffineExprKind::Add: {
    auto binOpExpr = expr.cast<AffineBinaryOpExpr>();
    AffineExpr lhs = binOpExpr.getLHS();
    AffineExpr rhs = binOpExpr.getRHS();
    if (failed(checkOperandAffineExprRecursively(lhs, seenIds)))
      return failure();
    if (failed(checkOperandAffineExprRecursively(rhs, seenIds)))
      return failure();
    return success();
  }
  case AffineExprKind::Mul: {
    auto binOpExpr = expr.cast<AffineBinaryOpExpr>();
    AffineExpr lhs = binOpExpr.getLHS();
    AffineExpr rhs = binOpExpr.getRHS();
    AffineExpr dimExpr;
    if (lhs.getKind() == AffineExprKind::DimId &&
        rhs.getKind() == AffineExprKind::Constant) {
      dimExpr = lhs;
    } else if (rhs.getKind() == AffineExprKind::DimId &&
               lhs.getKind() == AffineExprKind::Constant) {
      dimExpr = rhs;
    } else
      return failure();
    unsigned position = dimExpr.cast<AffineDimExpr>().getPosition();
    if ((size_t)position >= seenIds.size() || seenIds[position])
      return failure();
    seenIds[position] = true;
    return success();
  }
  case AffineExprKind::DimId: {
    unsigned position = expr.cast<AffineDimExpr>().getPosition();
    if ((size_t)position >= seenIds.size() || seenIds[position])
      return failure();
    seenIds[position] = true;
    return success();
  }
  default:
    return failure();
  }
}

static FailureOr<llvm::SmallSet<unsigned, 2>>
checkOperandAffineExpr(AffineExpr expr, unsigned numDims) {
  SmallVector<bool> seenIds(numDims, false);
  if (failed(checkOperandAffineExprRecursively(expr, seenIds)))
    return failure();

  llvm::SmallSet<unsigned, 2> positions;
  for (auto it : llvm::enumerate(seenIds)) {
    if (it.value())
      positions.insert((unsigned)it.index());
  }
  return positions;
}

//===----------------------------------------------------------------------===//
// mesh::getMeshShardingAttr
//===----------------------------------------------------------------------===//

FailureOr<std::pair<bool, MeshShardingAttr>>
mesh::getMeshShardingAttr(OpResult result) {
  Value val = result.cast<Value>();
  bool anyShardedForDef = llvm::any_of(val.getUsers(), [](Operation *user) {
    auto shardOp = llvm::dyn_cast<mesh::ShardOp>(user);
    if (!shardOp)
      return false;
    return !shardOp.getAnnotateForUsers();
  });

  if (anyShardedForDef) {
    // expected to have exact one use if it has a use of `mesh.shard` without
    // unit attr annotate_for_users
    if (!val.hasOneUse())
      return failure();
    auto shardOp = llvm::cast<mesh::ShardOp>(*val.getUsers().begin());
    return std::make_pair(false, shardOp.getShard());
  }

  bool anyShardedForUsers = llvm::any_of(val.getUsers(), [](Operation *user) {
    auto shardOp = llvm::dyn_cast<mesh::ShardOp>(user);
    if (!shardOp)
      return false;
    return shardOp.getAnnotateForUsers();
  });
  if (anyShardedForUsers) {
    SmallVector<ShardOp> shardOps;
    for (Operation *user : val.getUsers()) {
      ShardOp shardOp = llvm::dyn_cast<ShardOp>(user);
      if (shardOp)
        shardOps.push_back(shardOp);
    }
    MeshShardingAttr shardForDef = shardOps[0].getShard();
    for (size_t i = 1; i < shardOps.size(); ++i) {
      // TODO: Deduce a reasonable mesh sharding attr for def when they are
      // different
      assert(shardOps[i].getShard() == shardForDef &&
             "only support all shard ops have the same mesh sharding attr");
    }
    return std::make_pair(true, shardForDef);
  }
  return failure();
}

FailureOr<std::pair<bool, MeshShardingAttr>>
mesh::getMeshShardingAttr(OpOperand &opOperand) {
  Value val = opOperand.get();
  if (ShardOp shardOp = val.getDefiningOp<ShardOp>())
    return std::make_pair(shardOp.getAnnotateForUsers(), shardOp.getShard());

  return failure();
}

//===----------------------------------------------------------------------===//
// ShardingInterface::verifyShardingInterfaceImpl
//===----------------------------------------------------------------------===//

LogicalResult mesh::ShardingInterface::verifyShardingInterfaceImpl() {
  Operation *op = getOperation();

  // check operands and results type
  for (Type type : op->getOperandTypes())
    if (!llvm::isa<RankedTensorType>(type))
      return failure();
  for (Type type : op->getResultTypes())
    if (!llvm::isa<RankedTensorType>(type))
      return failure();

  // check loop types
  SmallVector<IteratorType> loopTypes = getLoopIteratorTypes();
  if (loopTypes.size() == 0)
    return failure();

  // check maps
  SmallVector<AffineMap> maps = getIndexingMaps();
  if (maps.size() == 0)
    return failure();
  unsigned numOperands = op->getNumOperands();
  unsigned numResults = op->getNumResults();
  if (numOperands + numResults != maps.size())
    return failure();

  for (OpResult result : op->getResults()) {
    auto resultType = result.getType().dyn_cast<RankedTensorType>();
    if (!resultType)
      return failure();
    AffineMap map = maps[numOperands + result.getResultNumber()];
    if (!map.isProjectedPermutation()) {
      return failure();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ShardingInterface::printLoopTypesAndIndexingMaps
//===----------------------------------------------------------------------===//

void mesh::ShardingInterface::printLoopTypesAndIndexingMaps(raw_ostream &os) {
  os << "print loop types and indexing maps for: \n";
  getOperation()->print(os);
  os << "\n";
  os << "loop types: [";
  for (IteratorType type : getLoopIteratorTypes()) {
    os << stringifyEnum(type) << " ";
  }
  os << "]\n";
  os << "indexing maps: \n";
  for (AffineMap map : getIndexingMaps())
    os << map << "\n";
  os << "\n";
}

//===----------------------------------------------------------------------===//
// detail::defaultGetShardingOption
//===----------------------------------------------------------------------===//

namespace {

// Update the given `shardingOption` according to `meshAxes` and `loopIdx`
static LogicalResult fillShardingOption(Operation *op,
                                        ShardingOption &shardingOption,
                                        SymbolRefAttr cluster,
                                        ArrayRef<int32_t> meshAxes,
                                        unsigned loopIdx) {
  if ((shardingOption.cluster && cluster &&
       shardingOption.cluster != cluster) ||
      (!shardingOption.shardingArray[loopIdx].empty() &&
       shardingOption.shardingArray[loopIdx] != meshAxes)) {
    LLVM_DEBUG(DBGS() << "sharding option conflicts on loop iterator "
                      << loopIdx << "\n");
    return failure();
  }
  for (size_t i = 0; i < shardingOption.shardingArray.size(); ++i) {
    if (i == loopIdx)
      continue;

    for (int32_t axis : meshAxes) {
      if (std::find(shardingOption.shardingArray[i].begin(),
                    shardingOption.shardingArray[i].end(),
                    axis) != shardingOption.shardingArray[i].end()) {
        LLVM_DEBUG(DBGS() << "sharding option conflicts because mesh axes "
                          << axis << " duplicate");
        return failure();
      }
    }
  }
  if (cluster)
    shardingOption.cluster = cluster;
  if (shardingOption.shardingArray[loopIdx].empty())
    shardingOption.shardingArray[loopIdx].append(meshAxes.begin(),
                                                 meshAxes.end());
  return success();
}

} // namespace

FailureOr<ShardingOption> mesh::detail::defaultGetShardingOption(
    Operation *op, ArrayRef<MeshShardingAttr> operandShardings,
    ArrayRef<MeshShardingAttr> resultShardings) {
  ShardingInterface shardingOp = llvm::cast<ShardingInterface>(op);
  ShardingOption shardingOption;

  if (failed(shardingOp.verifyShardingInterfaceImpl()))
    return op->emitOpError() << "invalid sharding interface implementation";
  SmallVector<IteratorType> loopTypes = shardingOp.getLoopIteratorTypes();
  SmallVector<AffineMap> maps = shardingOp.getIndexingMaps();
  unsigned numOperands = op->getNumOperands();
  shardingOption.shardingArray.resize(loopTypes.size());
  llvm::SmallVector<int32_t> partialMeshAxes;
  Partial partialType;
  llvm::SmallSet<unsigned, 4> visitedLoopIndices;
  bool anyShardingInResultsOrOperands = false;

  // 1. Fill sharding option based on op results
  for (auto shardingIt : llvm::enumerate(resultShardings)) {
    MeshShardingAttr shardAttr = shardingIt.value();
    if (!shardAttr)
      continue;
    AffineMap map = maps[numOperands + shardingIt.index()];
    anyShardingInResultsOrOperands = true;
    // Handle the split axes: calculate the corresponding loop index for each
    // split axes sub-array, and then store the sub-array to
    // shardingOption[index]
    for (auto it : llvm::zip(map.getResults(), shardAttr.getSplitAxes())) {
      AffineExpr expr = std::get<0>(it);
      ArrayRef<int32_t> axes = std::get<1>(it).asArrayRef();
      auto dim = expr.cast<AffineDimExpr>();
      unsigned index = dim.getPosition();
      visitedLoopIndices.insert(index);
      if (failed(fillShardingOption(op, shardingOption, shardAttr.getCluster(),
                                    axes, index)))
        return failure();
    }

    // Handle the partial axes: at this stage, the exact loop index/indices
    // cannot be decided because there could be multiple reduction loops.
    ArrayRef<int32_t> partialAxes = shardAttr.getPartialAxes();
    if (!partialAxes.empty()) {
      if (!partialMeshAxes.empty())
        return op->emitOpError() << "at most one result with partial axes is "
                                    "supported at present";
      partialType = shardAttr.getPartialType();
      partialMeshAxes.append(partialAxes.begin(), partialAxes.end());
      // Add all the reduction loop indices to `visitedLoopIndices` if
      // `partialAxes` is not empty
      for (size_t loopIdx = 0; loopIdx < loopTypes.size(); ++loopIdx) {
        if (isReductionLoop(loopTypes[loopIdx]))
          visitedLoopIndices.insert(loopIdx);
      }
    }
  }

  // 2. Fill sharding option based on operands
  for (auto shardingIt : llvm::enumerate(operandShardings)) {
    MeshShardingAttr shardAttr = shardingIt.value();
    if (!shardAttr)
      continue;

    anyShardingInResultsOrOperands = true;
    AffineMap map = maps[shardingIt.index()];
    unsigned numDims = map.getNumDims();

    // Handle the split axes. Partial axes don't need to be handled because they
    // only affect the defining op of the operand.
    //
    // TODO: Change to process the operands with single loop index first and
    // then the operands with multiple loop indices.
    for (auto it : llvm::zip(map.getResults(), shardAttr.getSplitAxes())) {
      AffineExpr expr = std::get<0>(it);
      ArrayRef<int32_t> axes = std::get<1>(it).asArrayRef();
      FailureOr<llvm::SmallSet<unsigned, 2>> loopIndices =
          checkOperandAffineExpr(expr, numDims);
      if (failed(loopIndices))
        return op->emitOpError()
               << "operand's affine expression is restricted to const_i * "
                  "dim_i + const_j + dim_j + ...";
      if (loopIndices->empty())
        continue;
      if (loopIndices->size() == 1) {
        unsigned loopIdx = *loopIndices->begin();
        visitedLoopIndices.insert(loopIdx);
        if (failed(fillShardingOption(op, shardingOption,
                                      shardAttr.getCluster(), axes, loopIdx)))
          return failure();
      }
      // If multiple loop indices correspond to a dimension of an operand, it is
      // difficult to infer which loop indices are responsible for sharding.
      // Therefore, the exact loop index must be specified by others.
      if (loopIndices->size() > 1) {
        bool seenLoopIndices = false;
        for (unsigned loopIdx : *loopIndices) {
          if (visitedLoopIndices.contains(loopIdx)) {
            seenLoopIndices = true;
            break;
          }
        }
        if (!seenLoopIndices)
          return op->emitOpError()
                 << "the operand " << shardingIt.index()
                 << " has multiple loop indices in a dimension, but none of "
                    "them could be found in the exactly specified annotation "
                    "of op results or operands.";
      }
    }
  }

  // 3. Finalize sharding option
  if (!partialMeshAxes.empty()) {
    bool anyNonEmptyReductionLoop = llvm::any_of(
        llvm::enumerate(shardingOption.shardingArray), [&](auto it) {
          SmallVector<int32_t> &subArray = it.value();
          int64_t idx = it.index();
          return isReductionLoop(loopTypes[idx]) && !subArray.empty();
        });
    if (!anyNonEmptyReductionLoop) {
      bool filled = false;
      for (size_t idx = 0; idx < loopTypes.size(); ++idx) {
        if (isReductionLoop(loopTypes[idx]) &&
            areReductionAndPartialMatch(loopTypes[idx], partialType)) {
          std::ignore = fillShardingOption(op, shardingOption, nullptr,
                                           partialMeshAxes, idx);
          filled = true;
          break;
        }
      }
      if (!filled)
        return op->emitOpError() << "no matched reduction loop found for the "
                                    "result's partial type";
    }
  }
  removeTrailingEmptySubArray(shardingOption.shardingArray);
  if (!anyShardingInResultsOrOperands)
    shardingOption.empty = true;
  return shardingOption;
}

//===----------------------------------------------------------------------===//
// detail::defaultAddShardingAnnotations
//===----------------------------------------------------------------------===//

namespace {

// To add a `mesh.shard` op for the given result, based on the details provided
// in `shardingOption`, `map`, and `loopTypes`.
static LogicalResult addShardOp(OpBuilder &b, OpResult result,
                                const ShardingOption &shardingOption,
                                AffineMap map,
                                ArrayRef<IteratorType> loopTypes) {
  FailureOr<std::pair<bool, MeshShardingAttr>> maybeSharding =
      getMeshShardingAttr(result);
  if (succeeded(maybeSharding) && !maybeSharding->first)
    return success();

  auto resultType = result.getType().cast<RankedTensorType>();
  SmallVector<SmallVector<int32_t>> splitAxes(resultType.getRank());
  SmallVector<int32_t> partialAxes;

  // process the split axes
  for (auto it : llvm::enumerate(map.getResults())) {
    AffineExpr expr = it.value();
    // `expr` must be an `AffineDimExpr` because `map` is verified by
    // isProjectedPermutation
    auto dim = expr.cast<AffineDimExpr>();
    unsigned loopIdx = dim.getPosition();
    if (loopIdx < shardingOption.shardingArray.size())
      splitAxes[it.index()].append(shardingOption.shardingArray[loopIdx]);
  }

  // process the partial axes
  // partialType will be ignored if partialAxes is empty
  Partial partialType = Partial::Sum;
  for (auto it : llvm::zip(loopTypes, shardingOption.shardingArray)) {
    IteratorType iType = std::get<0>(it);
    if (isReductionLoop(iType)) {
      Partial curPartialType = getPartialTypeFromReduction(iType);
      if (!partialAxes.empty())
        assert(partialType == curPartialType &&
               "Only one reduction type is supported");
      partialType = curPartialType;
      const SmallVector<int32_t> &axis = std::get<1>(it);
      partialAxes.append(axis);
    }
  }

  removeTrailingEmptySubArray(splitAxes);
  MeshShardingAttr shardAttr =
      MeshShardingAttr::get(b.getContext(), shardingOption.cluster, splitAxes,
                            partialAxes, partialType);
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfterValue(result);
  auto shardOp = b.create<ShardOp>(result.getLoc(), resultType, result,
                                   shardAttr, /*annotate_for_users*/ false);
  result.replaceAllUsesExcept(shardOp, shardOp);
  return success();
}

// To add a `mesh.shard` op for the given operand, based on the details provided
// in `shardingOption`, `map`, and `loopTypes`.
static LogicalResult addShardOp(OpBuilder &b, OpOperand &opOperand,
                                const ShardingOption &shardingOption,
                                AffineMap map,
                                ArrayRef<IteratorType> loopTypes) {
  auto maybeShardingAttr = getMeshShardingAttr(opOperand);
  if (succeeded(maybeShardingAttr) && maybeShardingAttr->first)
    return success();
  Value operand = opOperand.get();
  auto operandType = operand.getType().cast<RankedTensorType>();
  SmallVector<SmallVector<int32_t>> splitAxes(operandType.getRank());
  unsigned numDims = map.getNumDims();
  for (auto it : llvm::enumerate(map.getResults())) {
    int64_t idx = it.index();
    AffineExpr expr = it.value();
    FailureOr<llvm::SmallSet<unsigned, 2>> loopIndices =
        checkOperandAffineExpr(expr, numDims);
    if (failed(loopIndices))
      return failure();
    SmallVector<unsigned> shardedLoopIndices;
    for (unsigned loopIdx : *loopIndices) {
      if ((size_t)loopIdx < shardingOption.shardingArray.size() &&
          !shardingOption.shardingArray[loopIdx].empty())
        shardedLoopIndices.push_back(loopIdx);
    }
    // mostly one sharded loop index is accepted
    if (shardedLoopIndices.size() > 1)
      return failure();
    if (shardedLoopIndices.size() == 1) {
      splitAxes[idx].append(
          shardingOption.shardingArray[shardedLoopIndices[0]]);
    }
  }

  removeTrailingEmptySubArray(splitAxes);
  MeshShardingAttr shardAttr =
      MeshShardingAttr::get(b.getContext(), shardingOption.cluster, splitAxes);
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(opOperand.getOwner());
  auto shardOp = b.create<ShardOp>(operand.getLoc(), operandType, operand,
                                   shardAttr, true);
  opOperand.set(shardOp);

  return success();
}

} // namespace

LogicalResult mesh::detail::defaultAddShardingAnnotations(
    Operation *op, OpBuilder &b, const ShardingOption &shardingOption) {
  ShardingInterface shardingOp = llvm::cast<ShardingInterface>(op);
  SmallVector<IteratorType> loopTypes = shardingOp.getLoopIteratorTypes();
  SmallVector<AffineMap> maps = shardingOp.getIndexingMaps();
  unsigned numOperands = op->getNumOperands();

  // 1. add mesh.shard ops for all op results
  for (OpResult result : op->getResults()) {
    if (failed(addShardOp(b, result, shardingOption,
                          maps[numOperands + result.getResultNumber()],
                          loopTypes)))
      return failure();
  }

  // 2. add mesh.shard ops for all operands
  for (OpOperand &opOperand : op->getOpOperands()) {
    if (failed(addShardOp(b, opOperand, shardingOption,
                          maps[opOperand.getOperandNumber()], loopTypes)))
      return failure();
  }

  return success();
}
