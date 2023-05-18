//===- ValueBoundsOpInterfaceImpl.cpp - Impl. of ValueBoundsOpInterface ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/ValueBoundsOpInterfaceImpl.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;

namespace mlir {
namespace memref {
namespace {

template <typename OpTy>
struct AllocOpInterface
    : public ValueBoundsOpInterface::ExternalModel<AllocOpInterface<OpTy>,
                                                   OpTy> {
  void populateBoundsForShapedValueDim(Operation *op, Value value, int64_t dim,
                                       ValueBoundsConstraintSet &cstr) const {
    auto allocOp = cast<OpTy>(op);
    assert(value == allocOp.getResult() && "invalid value");

    cstr.bound(value)[dim] == allocOp.getMixedSizes()[dim];
  }
};

struct CastOpInterface
    : public ValueBoundsOpInterface::ExternalModel<CastOpInterface, CastOp> {
  void populateBoundsForShapedValueDim(Operation *op, Value value, int64_t dim,
                                       ValueBoundsConstraintSet &cstr) const {
    auto castOp = cast<CastOp>(op);
    assert(value == castOp.getResult() && "invalid value");

    if (llvm::isa<MemRefType>(castOp.getResult().getType()) &&
        llvm::isa<MemRefType>(castOp.getSource().getType())) {
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

struct GetGlobalOpInterface
    : public ValueBoundsOpInterface::ExternalModel<GetGlobalOpInterface,
                                                   GetGlobalOp> {
  void populateBoundsForShapedValueDim(Operation *op, Value value, int64_t dim,
                                       ValueBoundsConstraintSet &cstr) const {
    auto getGlobalOp = cast<GetGlobalOp>(op);
    assert(value == getGlobalOp.getResult() && "invalid value");

    auto type = getGlobalOp.getType();
    assert(!type.isDynamicDim(dim) && "expected static dim");
    cstr.bound(value)[dim] == type.getDimSize(dim);
  }
};

struct RankOpInterface
    : public ValueBoundsOpInterface::ExternalModel<RankOpInterface, RankOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto rankOp = cast<RankOp>(op);
    assert(value == rankOp.getResult() && "invalid value");

    auto memrefType = llvm::dyn_cast<MemRefType>(rankOp.getMemref().getType());
    if (!memrefType)
      return;
    cstr.bound(value) == memrefType.getRank();
  }
};

struct SubViewOpInterface
    : public ValueBoundsOpInterface::ExternalModel<SubViewOpInterface,
                                                   SubViewOp> {
  void populateBoundsForShapedValueDim(Operation *op, Value value, int64_t dim,
                                       ValueBoundsConstraintSet &cstr) const {
    auto subViewOp = cast<SubViewOp>(op);
    assert(value == subViewOp.getResult() && "invalid value");

    llvm::SmallBitVector dropped = subViewOp.getDroppedDims();
    int64_t ctr = -1;
    for (int64_t i = 0, e = subViewOp.getMixedSizes().size(); i < e; ++i) {
      // Skip over rank-reduced dimensions.
      if (!dropped.test(i))
        ++ctr;
      if (ctr == dim) {
        cstr.bound(value)[dim] == subViewOp.getMixedSizes()[i];
        return;
      }
    }
    llvm_unreachable("could not find non-rank-reduced dim");
  }
};

} // namespace
} // namespace memref
} // namespace mlir

void mlir::memref::registerValueBoundsOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, memref::MemRefDialect *dialect) {
    memref::AllocOp::attachInterface<memref::AllocOpInterface<memref::AllocOp>>(
        *ctx);
    memref::AllocaOp::attachInterface<
        memref::AllocOpInterface<memref::AllocaOp>>(*ctx);
    memref::CastOp::attachInterface<memref::CastOpInterface>(*ctx);
    memref::DimOp::attachInterface<memref::DimOpInterface>(*ctx);
    memref::GetGlobalOp::attachInterface<memref::GetGlobalOpInterface>(*ctx);
    memref::RankOp::attachInterface<memref::RankOpInterface>(*ctx);
    memref::SubViewOp::attachInterface<memref::SubViewOpInterface>(*ctx);
  });
}
