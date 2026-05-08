//===- IndexedAccessOpInterfaceImpl.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implement IndexedAccessOpInterface on vector dialect operations with
// %memref[%i, %j, ...] operands so generic memref-dialect passes can rewrite
// their base/index pairs. Transfer ops keep their VectorTransferOpInterface
// patterns; gather/scatter have tensor-or-memref bases and index-vector
// operands that do not fit IndexedAccessOpInterface's rank-matched index
// contract.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/Transforms/IndexedAccessOpInterfaceImpl.h"

#include "mlir/Dialect/MemRef/IR/MemoryAccessOpInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::memref;

namespace {
/// Return true if this op has the memref semantics expected by this model.
template <typename LoadStoreOp>
bool hasMemrefSemantics(Operation *op) {
  return llvm::isa<MemRefType>(cast<LoadStoreOp>(op).getBase().getType());
}

/// Return the vector shape whose access strides must be preserved, marking
/// scalable dimensions as dynamic.
SmallVector<int64_t> getAccessedVectorShape(VectorType vecTy) {
  return llvm::map_to_vector(
      llvm::zip_equal(vecTy.getShape(), vecTy.getScalableDims()), [](auto dim) {
        auto [size, scalable] = dim;
        return scalable ? ShapedType::kDynamic : size;
      });
}

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
    return getAccessedVectorShape(cast<LoadStoreOp>(op).getVectorType());
  }

  std::optional<SmallVector<Value>>
  updateMemrefAndIndices(Operation *op, RewriterBase &rewriter, Value newMemref,
                         ValueRange newIndices) const {
    assert(hasMemrefSemantics<LoadStoreOp>(op) &&
           "expected vector op with memref semantics");
    assert(llvm::isa<MemRefType>(newMemref.getType()) &&
           "expected replacement memref");
    rewriter.modifyOpInPlace(op, [&]() {
      auto concreteOp = cast<LoadStoreOp>(op);
      concreteOp.getBaseMutable().assign(newMemref);
      concreteOp.getIndicesMutable().assign(newIndices);
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
static void attachAll(MLIRContext *ctx) {
  (Ops::template attachInterface<VectorLoadStoreLikeOpImpl<Ops>>(*ctx), ...);
}

} // namespace

void mlir::vector::registerIndexedAccessOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, vector::VectorDialect *dialect) {
    attachAll<vector::LoadOp, vector::StoreOp, vector::MaskedLoadOp,
              vector::MaskedStoreOp, vector::ExpandLoadOp,
              vector::CompressStoreOp>(ctx);
  });
}
