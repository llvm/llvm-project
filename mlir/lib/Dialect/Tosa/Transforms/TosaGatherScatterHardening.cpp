//===- TosaGatherScatterHardening.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that clamps gather and scatter indices to the
// statically known bounds of their indexed tensors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <cstdint>

namespace mlir {
namespace tosa {
#define GEN_PASS_DEF_TOSAGATHERSCATTERHARDENINGPASS
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"
} // namespace tosa
} // namespace mlir

using namespace mlir;
using namespace mlir::tosa;

namespace {

struct IndexUseGroup {
  IndexUseGroup(Value indices, int64_t upperBound)
      : indices(indices), upperBound(upperBound) {}

  Value indices;
  SmallVector<Operation *> users;
  int64_t upperBound;
};

using GroupsForIndices = DenseMap<Value, SmallVector<unsigned>>;

/// Validates one gather/scatter index use and adds it to a compatible group.
static LogicalResult collectIndexUse(Operation *op, Value indices, Value values,
                                     GroupsForIndices &groupsForIndices,
                                     SmallVectorImpl<IndexUseGroup> &groups) {
  auto valuesType = dyn_cast<RankedTensorType>(values.getType());
  if (!valuesType || valuesType.getRank() <= 1 || valuesType.isDynamicDim(1)) {
    return op->emitOpError(
        "requires a statically known indexed dimension for gather/scatter "
        "hardening");
  }

  int64_t indexedSize = valuesType.getDimSize(1);
  if (indexedSize <= 0) {
    return op->emitOpError(
        "requires a non-empty indexed dimension for gather/scatter hardening");
  }

  auto indicesType = dyn_cast<ShapedType>(indices.getType());
  auto elementType = indicesType
                         ? dyn_cast<IntegerType>(indicesType.getElementType())
                         : IntegerType();
  if (!elementType || elementType.getWidth() > 64) {
    return op->emitOpError(
        "requires indices with an integer element type of at most 64 bits for "
        "gather/scatter hardening");
  }

  // Clamp bounds must be representable in the index element type. Group only
  // users whose resulting clamp bounds are identical.
  int64_t maxRepresentable =
      llvm::APInt::getSignedMaxValue(elementType.getWidth()).getSExtValue();
  int64_t upperBound = std::min(indexedSize - 1, maxRepresentable);

  unsigned groupIndex = groups.size();
  SmallVector<unsigned> &candidateGroups = groupsForIndices[indices];
  for (unsigned candidate : candidateGroups) {
    if (groups[candidate].upperBound == upperBound) {
      groupIndex = candidate;
      break;
    }
  }
  if (groupIndex == groups.size()) {
    groups.emplace_back(indices, upperBound);
    candidateGroups.push_back(groupIndex);
  }

  groups[groupIndex].users.push_back(op);
  return success();
}

/// Collects and validates all gather/scatter index uses in the function.
static LogicalResult
collectIndexUseGroups(func::FuncOp funcOp,
                      SmallVectorImpl<IndexUseGroup> &groups) {
  GroupsForIndices groupsForIndices;
  bool analysisFailed = false;

  funcOp.walk([&](Operation *op) {
    llvm::TypeSwitch<Operation *>(op)
        .Case<tosa::GatherOp>([&](tosa::GatherOp gatherOp) {
          analysisFailed |= failed(collectIndexUse(op, gatherOp.getIndices(),
                                                   gatherOp.getValues(),
                                                   groupsForIndices, groups));
        })
        .Case<tosa::ScatterOp>([&](tosa::ScatterOp scatterOp) {
          analysisFailed |= failed(collectIndexUse(op, scatterOp.getIndices(),
                                                   scatterOp.getValuesIn(),
                                                   groupsForIndices, groups));
        });
  });

  return analysisFailed ? failure() : success();
}

/// Returns whether the indices already have sufficiently restrictive bounds.
static bool isAlreadyHardened(Value indices, IntegerAttr maxAttr) {
  auto clampOp = indices.getDefiningOp<tosa::ClampOp>();
  if (!clampOp)
    return false;

  auto existingMin = dyn_cast<IntegerAttr>(clampOp.getMinValAttr());
  auto existingMax = dyn_cast<IntegerAttr>(clampOp.getMaxValAttr());
  return existingMin && existingMax && !existingMin.getValue().isNegative() &&
         existingMax.getValue().sle(maxAttr.getValue());
}

/// Creates a clamp for the group and rewires its gather/scatter users.
static void hardenIndexUseGroup(IndexUseGroup &group, OpBuilder &builder) {
  Value indices = group.indices;
  auto indicesType = cast<ShapedType>(indices.getType());
  auto elementType = cast<IntegerType>(indicesType.getElementType());
  unsigned bitWidth = elementType.getWidth();

  IntegerAttr minAttr =
      IntegerAttr::get(elementType, llvm::APInt::getZero(bitWidth));
  IntegerAttr maxAttr = IntegerAttr::get(
      elementType,
      llvm::APInt(bitWidth, static_cast<uint64_t>(group.upperBound)));

  // Keep the pass idempotent and avoid nesting a second clamp around an
  // existing clamp that is at least as restrictive as the required one.
  if (isAlreadyHardened(indices, maxAttr))
    return;

  if (Operation *definingOp = indices.getDefiningOp())
    builder.setInsertionPointAfter(definingOp);
  else
    builder.setInsertionPointToStart(cast<BlockArgument>(indices).getOwner());

  Value clampedIndices =
      tosa::ClampOp::create(builder, group.users.front()->getLoc(),
                            indices.getType(), indices, minAttr, maxAttr)
          .getResult();

  for (Operation *user : group.users)
    user->setOperand(/*indices=*/1, clampedIndices);
}

struct TosaGatherScatterHardeningPass
    : public tosa::impl::TosaGatherScatterHardeningPassBase<
          TosaGatherScatterHardeningPass> {
  using Base::Base;

  void runOnOperation() override {
    SmallVector<IndexUseGroup> groups;
    // Do not partially harden the function when one operation cannot be made
    // safe using a statically bounded tosa.clamp.
    if (failed(collectIndexUseGroups(getOperation(), groups))) {
      signalPassFailure();
      return;
    }

    OpBuilder builder(&getContext());
    for (IndexUseGroup &group : groups)
      hardenIndexUseGroup(group, builder);
  }
};

} // namespace
