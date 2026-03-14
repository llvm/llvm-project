//===- ValueBoundsOpInterfaceImpl.cpp - Impl. of ValueBoundsOpInterface ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/IR/ValueBoundsOpInterfaceImpl.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;

namespace mlir {
namespace tensor {
namespace {

struct CastOpInterface
    : public ValueBoundsOpInterface::ExternalModel<CastOpInterface, CastOp> {
  void populateBoundsForShapedValueDim(Operation *op, Value value, int64_t dim,
                                       ValueBoundsConstraintSet &cstr) const {
    auto castOp = cast<CastOp>(op);
    assert(value == castOp.getResult() && "invalid value");

    if (llvm::isa<RankedTensorType>(castOp.getResult().getType()) &&
        llvm::isa<RankedTensorType>(castOp.getSource().getType())) {
      cstr.bound(value)[dim] == cstr.getExpr(castOp.getSource(), dim);
    }
  }
};

struct CollapseShapeOpInterface
    : public ValueBoundsOpInterface::ExternalModel<CollapseShapeOpInterface,
                                                   CollapseShapeOp> {
  void populateBoundsForShapedValueDim(Operation *op, Value value, int64_t dim,
                                       ValueBoundsConstraintSet &cstr) const {
    auto collapseOp = cast<CollapseShapeOp>(op);
    assert(value == collapseOp.getResult() && "invalid value");

    // Multiply the expressions for the dimensions in the reassociation group.
    const ReassociationIndices reassocIndices =
        collapseOp.getReassociationIndices()[dim];
    AffineExpr productExpr =
        cstr.getExpr(collapseOp.getSrc(), reassocIndices[0]);
    for (size_t i = 1; i < reassocIndices.size(); ++i) {
      productExpr =
          productExpr * cstr.getExpr(collapseOp.getSrc(), reassocIndices[i]);
    }
    cstr.bound(value)[dim] == productExpr;
  }
};

struct DimOpInterface
    : public ValueBoundsOpInterface::ExternalModel<DimOpInterface, DimOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto dimOp = cast<DimOp>(op);
    assert(value == dimOp.getResult() && "invalid value");

    cstr.bound(value) >= 0;
    auto constIndex = dimOp.getConstantIndex();
    if (!constIndex.has_value())
      return;
    cstr.bound(value) == cstr.getExpr(dimOp.getSource(), *constIndex);
  }
};

struct EmptyOpInterface
    : public ValueBoundsOpInterface::ExternalModel<EmptyOpInterface, EmptyOp> {
  void populateBoundsForShapedValueDim(Operation *op, Value value, int64_t dim,
                                       ValueBoundsConstraintSet &cstr) const {
    auto emptyOp = cast<EmptyOp>(op);
    assert(value == emptyOp.getResult() && "invalid value");

    cstr.bound(value)[dim] == emptyOp.getMixedSizes()[dim];
  }
};

struct ExpandShapeOpInterface
    : public ValueBoundsOpInterface::ExternalModel<ExpandShapeOpInterface,
                                                   ExpandShapeOp> {
  void populateBoundsForShapedValueDim(Operation *op, Value value, int64_t dim,
                                       ValueBoundsConstraintSet &cstr) const {
    auto expandOp = cast<ExpandShapeOp>(op);
    assert(value == expandOp.getResult() && "invalid value");
    cstr.bound(value)[dim] == expandOp.getMixedOutputShape()[dim];
  }
};

struct ExtractSliceOpInterface
    : public ValueBoundsOpInterface::ExternalModel<ExtractSliceOpInterface,
                                                   ExtractSliceOp> {
  void populateBoundsForShapedValueDim(Operation *op, Value value, int64_t dim,
                                       ValueBoundsConstraintSet &cstr) const {
    auto extractSliceOp = cast<ExtractSliceOp>(op);
    assert(value == extractSliceOp.getResult() && "invalid value");

    llvm::SmallBitVector dropped = extractSliceOp.getDroppedDims();
    int64_t ctr = -1;
    for (int64_t i = 0, e = extractSliceOp.getMixedSizes().size(); i < e; ++i) {
      // Skip over rank-reduced dimensions.
      if (!dropped.test(i))
        ++ctr;
      if (ctr == dim) {
        cstr.bound(value)[dim] == extractSliceOp.getMixedSizes()[i];
        return;
      }
    }
    llvm_unreachable("could not find non-rank-reduced dim");
  }
};

struct PadOpInterface
    : public ValueBoundsOpInterface::ExternalModel<PadOpInterface, PadOp> {
  void populateBoundsForShapedValueDim(Operation *op, Value value, int64_t dim,
                                       ValueBoundsConstraintSet &cstr) const {
    auto padOp = cast<PadOp>(op);
    assert(value == padOp.getResult() && "invalid value");

    AffineExpr srcSize = cstr.getExpr(padOp.getSource(), dim);
    AffineExpr lowPad = cstr.getExpr(padOp.getMixedLowPad()[dim]);
    AffineExpr highPad = cstr.getExpr(padOp.getMixedHighPad()[dim]);
    cstr.bound(value)[dim] == srcSize + lowPad + highPad;
  }
};

struct RankOpInterface
    : public ValueBoundsOpInterface::ExternalModel<RankOpInterface, RankOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto rankOp = cast<RankOp>(op);
    assert(value == rankOp.getResult() && "invalid value");

    auto tensorType =
        llvm::dyn_cast<RankedTensorType>(rankOp.getTensor().getType());
    if (!tensorType)
      return;
    cstr.bound(value) == tensorType.getRank();
  }
};

} // namespace
} // namespace tensor
} // namespace mlir

void mlir::tensor::registerValueBoundsOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, tensor::TensorDialect *dialect) {
    tensor::CastOp::attachInterface<tensor::CastOpInterface>(*ctx);
    tensor::CollapseShapeOp::attachInterface<tensor::CollapseShapeOpInterface>(
        *ctx);
    tensor::DimOp::attachInterface<tensor::DimOpInterface>(*ctx);
    tensor::EmptyOp::attachInterface<tensor::EmptyOpInterface>(*ctx);
    tensor::ExpandShapeOp::attachInterface<tensor::ExpandShapeOpInterface>(
        *ctx);
    tensor::ExtractSliceOp::attachInterface<tensor::ExtractSliceOpInterface>(
        *ctx);
    tensor::PadOp::attachInterface<tensor::PadOpInterface>(*ctx);
    tensor::RankOp::attachInterface<tensor::RankOpInterface>(*ctx);
    // Note: ValueBoundsOpInterface implementation is not required for ops that
    // implement `DestinationStyleOpInterface` (for querying shaped OpResults).
  });
}
