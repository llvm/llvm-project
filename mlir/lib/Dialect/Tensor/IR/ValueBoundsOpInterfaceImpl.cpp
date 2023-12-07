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

struct DimOpInterface
    : public ValueBoundsOpInterface::ExternalModel<DimOpInterface, DimOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto dimOp = cast<DimOp>(op);
    assert(value == dimOp.getResult() && "invalid value");

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
    tensor::DimOp::attachInterface<tensor::DimOpInterface>(*ctx);
    tensor::EmptyOp::attachInterface<tensor::EmptyOpInterface>(*ctx);
    tensor::ExtractSliceOp::attachInterface<tensor::ExtractSliceOpInterface>(
        *ctx);
    tensor::InsertOp::attachInterface<
        DstValueBoundsOpInterfaceExternalModel<tensor::InsertOp>>(*ctx);
    tensor::InsertSliceOp::attachInterface<
        DstValueBoundsOpInterfaceExternalModel<tensor::InsertSliceOp>>(*ctx);
    tensor::PadOp::attachInterface<tensor::PadOpInterface>(*ctx);
    tensor::RankOp::attachInterface<tensor::RankOpInterface>(*ctx);
  });
}
