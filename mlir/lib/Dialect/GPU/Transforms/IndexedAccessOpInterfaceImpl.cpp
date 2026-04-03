//===- IndexedAccessOpInterfaceImpl.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Transforms/IndexedAccessOpInterfaceImpl.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemoryAccessOpInterfaces.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::memref;
using namespace mlir::gpu;

/// Given a GPU matrix type that will be loaded or stored, the leading dimension
/// of the matrix in memory, and whether or not the matrix is transposed,
/// compute the size of the linear memory that the load/store spans as
/// dC + leadingDim * (dR - 1) where dR and dC are the non-contiguous and
/// contiguous matrix dimensions, respectively (we get to the dX-1th row and
/// then access the first dY elements of it).
static int64_t get1DAccessSize(MMAMatrixType matrixType, int64_t leadingDim,
                               bool transpose) {
  assert(matrixType.getShape().size() == 2 && "expected matrices to be 2D");

  int64_t c = matrixType.getShape()[1];
  int64_t r = matrixType.getShape()[0];
  if (transpose)
    std::swap(c, r);
  return c + leadingDim * (r - 1);
}

namespace {
struct SubgroupMmaLoadMatrixOpImpl final
    : IndexedAccessOpInterface::ExternalModel<SubgroupMmaLoadMatrixOpImpl,
                                              SubgroupMmaLoadMatrixOp> {
  TypedValue<MemRefType> getAccessedMemref(Operation *op) const {
    return cast<SubgroupMmaLoadMatrixOp>(op).getSrcMemref();
  }

  Operation::operand_range getIndices(Operation *op) const {
    return cast<SubgroupMmaLoadMatrixOp>(op).getIndices();
  }

  /// This returns a 1-D shape so that it's clear that both linearization and
  /// folding in expand/collapse_shape operations are allowed.
  SmallVector<int64_t> getAccessedShape(Operation *op) const {
    auto loadOp = cast<SubgroupMmaLoadMatrixOp>(op);
    return {get1DAccessSize(cast<MMAMatrixType>(loadOp.getRes().getType()),
                            loadOp.getLeadDimension().getZExtValue(),
                            loadOp.getTranspose().value_or(false))};
  }

  std::optional<SmallVector<Value>>
  updateMemrefAndIndices(Operation *op, RewriterBase &rewriter, Value newMemref,
                         ValueRange newIndices) const {
    auto loadOp = cast<SubgroupMmaLoadMatrixOp>(op);
    rewriter.modifyOpInPlace(loadOp, [&]() {
      loadOp.getSrcMemrefMutable().assign(newMemref);
      loadOp.getIndicesMutable().assign(newIndices);
    });
    return std::nullopt;
  }

  bool hasInboundsIndices(Operation *) const { return true; }
};

struct SubgroupMmaStoreMatrixOpImpl final
    : IndexedAccessOpInterface::ExternalModel<SubgroupMmaStoreMatrixOpImpl,
                                              SubgroupMmaStoreMatrixOp> {
  TypedValue<MemRefType> getAccessedMemref(Operation *op) const {
    return cast<SubgroupMmaStoreMatrixOp>(op).getDstMemref();
  }

  Operation::operand_range getIndices(Operation *op) const {
    return cast<SubgroupMmaStoreMatrixOp>(op).getIndices();
  }

  /// This returns a 1-D shape so that it's clear that both linearization and
  /// folding in expand/collapse_shape operations are allowed.
  SmallVector<int64_t> getAccessedShape(Operation *op) const {
    auto storeOp = cast<SubgroupMmaStoreMatrixOp>(op);
    return {get1DAccessSize(storeOp.getSrc().getType(),
                            storeOp.getLeadDimension().getZExtValue(),
                            storeOp.getTranspose().value_or(false))};
  }

  std::optional<SmallVector<Value>>
  updateMemrefAndIndices(Operation *op, RewriterBase &rewriter, Value newMemref,
                         ValueRange newIndices) const {
    auto storeOp = cast<SubgroupMmaStoreMatrixOp>(op);
    rewriter.modifyOpInPlace(storeOp, [&]() {
      storeOp.getDstMemrefMutable().assign(newMemref);
      storeOp.getIndicesMutable().assign(newIndices);
    });
    return std::nullopt;
  }

  bool hasInboundsIndices(Operation *) const { return true; }
};
} // namespace

void mlir::gpu::registerIndexedAccessOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, gpu::GPUDialect *dialect) {
    SubgroupMmaLoadMatrixOp::attachInterface<SubgroupMmaLoadMatrixOpImpl>(*ctx);
    SubgroupMmaStoreMatrixOp::attachInterface<SubgroupMmaStoreMatrixOpImpl>(
        *ctx);
  });
}
