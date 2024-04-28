//===- PolynomialOps.cpp - Polynomial dialect ops ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polynomial/IR/PolynomialOps.h"
#include "mlir/Dialect/Polynomial/IR/Polynomial.h"
#include "mlir/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "mlir/Dialect/Polynomial/IR/PolynomialTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APInt.h"

using namespace mlir;
using namespace mlir::polynomial;

void FromTensorOp::build(OpBuilder &builder, OperationState &result,
                         Value input, RingAttr ring) {
  TensorType tensorType = dyn_cast<TensorType>(input.getType());
  auto bitWidth = tensorType.getElementTypeBitWidth();
  APInt cmod(1 + bitWidth, 1);
  cmod = cmod << bitWidth;
  Type resultType = PolynomialType::get(builder.getContext(), ring);
  build(builder, result, resultType, input);
}

LogicalResult FromTensorOp::verify() {
  ArrayRef<int64_t> tensorShape = getInput().getType().getShape();
  RingAttr ring = getOutput().getType().getRing();
  unsigned polyDegree = ring.getPolynomialModulus().getPolynomial().getDegree();
  bool compatible = tensorShape.size() == 1 && tensorShape[0] <= polyDegree;
  if (!compatible) {
    InFlightDiagnostic diag = emitOpError()
                              << "input type " << getInput().getType()
                              << " does not match output type "
                              << getOutput().getType();
    diag.attachNote() << "the input type must be a tensor of shape [d] where d "
                         "is at most the degree of the polynomialModulus of "
                         "the output type's ring attribute";
    return diag;
  }

  APInt coefficientModulus = ring.getCoefficientModulus().getValue();
  unsigned cmodBitWidth = coefficientModulus.ceilLogBase2();
  unsigned inputBitWidth = getInput().getType().getElementTypeBitWidth();

  if (inputBitWidth > cmodBitWidth) {
    InFlightDiagnostic diag = emitOpError()
                              << "input tensor element type "
                              << getInput().getType().getElementType()
                              << " is too large to fit in the coefficients of "
                              << getOutput().getType();
    diag.attachNote() << "the input tensor's elements must be rescaled"
                         " to fit before using from_tensor";
    return diag;
  }

  return success();
}

LogicalResult ToTensorOp::verify() {
  ArrayRef<int64_t> tensorShape = getOutput().getType().getShape();
  unsigned polyDegree = getInput()
                            .getType()
                            .getRing()
                            .getPolynomialModulus()
                            .getPolynomial()
                            .getDegree();
  bool compatible = tensorShape.size() == 1 && tensorShape[0] == polyDegree;

  if (compatible)
    return success();

  InFlightDiagnostic diag =
      emitOpError() << "input type " << getInput().getType()
                    << " does not match output type " << getOutput().getType();
  diag.attachNote() << "the output type must be a tensor of shape [d] where d "
                       "is at most the degree of the polynomialModulus of "
                       "the input type's ring attribute";
  return diag;
}

LogicalResult MulScalarOp::verify() {
  Type argType = getPolynomial().getType();
  PolynomialType polyType;

  if (auto shapedPolyType = dyn_cast<ShapedType>(argType)) {
    polyType = cast<PolynomialType>(shapedPolyType.getElementType());
  } else {
    polyType = cast<PolynomialType>(argType);
  }

  Type coefficientType = polyType.getRing().getCoefficientType();

  if (coefficientType != getScalar().getType())
    return emitOpError() << "polynomial coefficient type " << coefficientType
                         << " does not match scalar type "
                         << getScalar().getType();

  return success();
}
