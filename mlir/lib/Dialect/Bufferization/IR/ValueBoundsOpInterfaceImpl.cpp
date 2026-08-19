//===- ValueBoundsOpInterfaceImpl.cpp - Impl. of ValueBoundsOpInterface ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/ValueBoundsOpInterfaceImpl.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;

namespace mlir {
namespace bufferization {
namespace {

/// Model for a materialization op that keeps the shape of its source, i.e.,
/// `to_tensor` and `to_buffer`.
template <typename OpTy>
struct MaterializationOpInterface
    : public ValueBoundsOpInterface::ExternalModel<
          MaterializationOpInterface<OpTy>, OpTy> {
  void populateBoundsForShapedValueDim(Operation *op, Value value, int64_t dim,
                                       ValueBoundsConstraintSet &cstr) const {
    static_assert(
        OpTy::template hasTrait<OpTrait::SameOperandsAndResultShape>(),
        "expected the result and the source to have the same shape");
    auto materializationOp = cast<OpTy>(op);
    assert(value == materializationOp.getResult() && "invalid value");

    // The op also accepts tensor-like and buffer-like types that are not
    // shaped, for which no bound can be computed.
    Value source = materializationOp.getOperand();
    if (isa<ShapedType>(value.getType()) && isa<ShapedType>(source.getType()))
      cstr.bound(value)[dim] == cstr.getExpr(source, dim);
  }
};

} // namespace
} // namespace bufferization
} // namespace mlir

void mlir::bufferization::registerValueBoundsOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx,
                            bufferization::BufferizationDialect *dialect) {
    bufferization::ToBufferOp::attachInterface<
        bufferization::MaterializationOpInterface<bufferization::ToBufferOp>>(
        *ctx);
    bufferization::ToTensorOp::attachInterface<
        bufferization::MaterializationOpInterface<bufferization::ToTensorOp>>(
        *ctx);
  });
}
