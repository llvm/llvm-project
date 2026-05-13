//===- IndexedAccessOpInterfaceImpl.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implement IndexedAccessOpInterface on vector dialect operations with
// %memref[%i, %j, ...] operands so generic memref-dialect passes can rewrite
// their base/index pairs. Redundant leading unit vector dimensions are omitted
// from the accessed shape and restored with vector.shape_casts when an alias
// rewrite drops those dimensions. Transfer ops keep their
// VectorTransferOpInterface patterns; gather/scatter have tensor-or-memref
// bases and index-vector operands that do not fit IndexedAccessOpInterface's
// rank-matched index contract.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/Transforms/IndexedAccessOpInterfaceImpl.h"

#include "mlir/Dialect/MemRef/IR/MemoryAccessOpInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

#include <type_traits>

using namespace mlir;
using namespace mlir::memref;

/// Return true if this op has the memref semantics expected by this model.
template <typename LoadStoreOp>
static bool hasMemrefSemantics(Operation *op) {
  return llvm::isa<MemRefType>(cast<LoadStoreOp>(op).getBase().getType());
}

/// Return true if this op supports rank-0 vector operands/results.
template <typename LoadStoreOp>
static constexpr bool supportsRankZeroVectorAccess() {
  return std::is_same_v<LoadStoreOp, vector::LoadOp> ||
         std::is_same_v<LoadStoreOp, vector::StoreOp>;
}

/// Return the number of leading static unit dimensions in `vecTy`.
static unsigned getNumLeadingUnitDims(VectorType vecTy) {
  unsigned numLeadingUnitDims = 0;
  for (auto [size, scalable] :
       llvm::zip_equal(vecTy.getShape(), vecTy.getScalableDims())) {
    if (size != 1 || scalable)
      break;
    ++numLeadingUnitDims;
  }
  return numLeadingUnitDims;
}

/// Return the vector shape whose access strides must be preserved, omitting
/// redundant leading static unit dimensions and marking scalable dimensions as
/// dynamic. If the op cannot access rank-0 vectors, preserve one trailing unit
/// dimension instead of returning an empty shape.
static SmallVector<int64_t> getAccessedVectorShape(VectorType vecTy,
                                                   bool supportsRankZero) {
  unsigned numLeadingUnitDims = getNumLeadingUnitDims(vecTy);
  unsigned rank = static_cast<unsigned>(vecTy.getRank());
  if (!supportsRankZero && numLeadingUnitDims == rank)
    --numLeadingUnitDims;
  return llvm::map_to_vector(
      llvm::zip_equal(vecTy.getShape().drop_front(numLeadingUnitDims),
                      vecTy.getScalableDims().drop_front(numLeadingUnitDims)),
      [](auto dim) {
        auto [size, scalable] = dim;
        return scalable ? ShapedType::kDynamic : size;
      });
}

/// Return `vecTy` with `numLeadingDims` dimensions dropped from the front.
static VectorType dropLeadingDims(VectorType vecTy, unsigned numLeadingDims) {
  return VectorType::get(vecTy.getShape().drop_front(numLeadingDims),
                         vecTy.getElementType(),
                         vecTy.getScalableDims().drop_front(numLeadingDims));
}

/// Return the shape-cast type for vector operands that match `vecTy`.
static std::optional<VectorType>
getShapeCastTypeForOperand(Value operand, VectorType vecTy,
                           unsigned numLeadingDims) {
  auto operandTy = dyn_cast<VectorType>(operand.getType());
  if (!operandTy || operandTy.getShape() != vecTy.getShape() ||
      operandTy.getScalableDims() != vecTy.getScalableDims())
    return std::nullopt;
  return dropLeadingDims(operandTy, numLeadingDims);
}

namespace {
template <typename LoadStoreOp>
struct VectorLoadStoreLikeOpImpl final
    : IndexedAccessOpInterface::ExternalModel<
          VectorLoadStoreLikeOpImpl<LoadStoreOp>, LoadStoreOp> {
  TypedValue<MemRefType> getAccessedMemref(Operation *op) const {
    return cast<LoadStoreOp>(op).getBase();
  }

  Operation::operand_range getIndices(Operation *op) const {
    return cast<LoadStoreOp>(op).getIndices();
  }

  SmallVector<int64_t> getAccessedShape(Operation *op) const {
    assert(hasMemrefSemantics<LoadStoreOp>(op) &&
           "expected vector op with memref semantics");
    return getAccessedVectorShape(cast<LoadStoreOp>(op).getVectorType(),
                                  supportsRankZeroVectorAccess<LoadStoreOp>());
  }

  std::optional<SmallVector<Value>>
  updateMemrefAndIndices(Operation *op, RewriterBase &rewriter, Value newMemref,
                         ValueRange newIndices) const {
    assert(hasMemrefSemantics<LoadStoreOp>(op) &&
           "expected vector op with memref semantics");
    assert(llvm::isa<MemRefType>(newMemref.getType()) &&
           "expected replacement memref");

    VectorType vecTy = cast<LoadStoreOp>(op).getVectorType();
    if (static_cast<int64_t>(newIndices.size()) >= vecTy.getRank()) {
      rewriter.modifyOpInPlace(op, [&]() {
        auto concreteOp = cast<LoadStoreOp>(op);
        concreteOp.getBaseMutable().assign(newMemref);
        concreteOp.getIndicesMutable().assign(newIndices);
      });
      return std::nullopt;
    }

    unsigned numLeadingDimsToDrop = static_cast<unsigned>(
        vecTy.getRank() - static_cast<int64_t>(newIndices.size()));
    assert(numLeadingDimsToDrop <= getNumLeadingUnitDims(vecTy) &&
           "expected only redundant leading unit dimensions to be dropped");

    IRMapping dropDimsMap;
    for (Value operand : op->getOperands()) {
      std::optional<VectorType> castTy =
          getShapeCastTypeForOperand(operand, vecTy, numLeadingDimsToDrop);
      if (!castTy || dropDimsMap.lookupOrNull(operand))
        continue;
      Value castedOperand = vector::ShapeCastOp::create(
          rewriter, operand.getLoc(), *castTy, operand);
      dropDimsMap.map(operand, castedOperand);
    }

    if (op->getNumResults() == 1) {
      // Result types cannot be changed in place on the original op because the
      // caller replaces it using the returned value. Clone at the lower rank,
      // then cast the result back to the original vector type.
      VectorType droppedDimsTy = dropLeadingDims(vecTy, numLeadingDimsToDrop);
      Operation *newOp = rewriter.clone(*op, dropDimsMap);
      rewriter.modifyOpInPlace(newOp, [&]() {
        auto concreteOp = cast<LoadStoreOp>(newOp);
        concreteOp.getBaseMutable().assign(newMemref);
        concreteOp.getIndicesMutable().assign(newIndices);
        newOp->getResult(0).setType(droppedDimsTy);
      });
      Value castBack = vector::ShapeCastOp::create(rewriter, newOp->getLoc(),
                                                   vecTy, newOp->getResult(0));
      return {{castBack}};
    }

    // Store-like ops have no results to replace, so update their vector
    // operands and base/index pair in place.
    rewriter.modifyOpInPlace(op, [&]() {
      auto concreteOp = cast<LoadStoreOp>(op);
      concreteOp.getBaseMutable().assign(newMemref);
      concreteOp.getIndicesMutable().assign(newIndices);
      for (OpOperand &operand : op->getOpOperands()) {
        if (Value replacement = dropDimsMap.lookupOrNull(operand.get()))
          operand.set(replacement);
      }
    });
    return std::nullopt;
  }

  // TODO: The various load and store operations, at the very least vector.load
  // and vector.store, should be taught a starts-in-bounds attribute that would
  // let us optimize index generation.
  bool hasInboundsIndices(Operation *op) const {
    assert(hasMemrefSemantics<LoadStoreOp>(op) &&
           "expected vector op with memref semantics");
    return false;
  }
};

template <typename... Ops>
static void attachLoadStoreLike(MLIRContext *ctx) {
  (Ops::template attachInterface<VectorLoadStoreLikeOpImpl<Ops>>(*ctx), ...);
}

} // namespace

void mlir::vector::registerIndexedAccessOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, vector::VectorDialect *dialect) {
    attachLoadStoreLike<vector::LoadOp, vector::StoreOp, vector::MaskedLoadOp,
                        vector::MaskedStoreOp, vector::ExpandLoadOp,
                        vector::CompressStoreOp>(ctx);
  });
}
