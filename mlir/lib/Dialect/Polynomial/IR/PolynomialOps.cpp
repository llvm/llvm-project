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

/// Test if a value is a primitive nth root of unity modulo cmod.
bool isPrimitiveNthRootOfUnity(const APInt &root, const unsigned n,
                               const APInt &cmod) {
  // Root bitwidth may be 1 less then cmod.
  APInt r = APInt(root).zext(cmod.getBitWidth());
  assert(r.ule(cmod) && "root must be less than cmod");

  APInt a = r;
  for (size_t k = 1; k < n; k++) {
    if (a.isOne())
      return false;
    a = (a * r).urem(cmod);
  }
  return a.isOne();
}

/// Verify that the types involved in an NTT or INTT operation are
/// compatible.
static LogicalResult verifyNTTOp(Operation *op, RingAttr ring,
                                 RankedTensorType tensorType) {
  Attribute encoding = tensorType.getEncoding();
  if (!encoding) {
    return op->emitOpError()
           << "expects a ring encoding to be provided to the tensor";
  }
  auto encodedRing = dyn_cast<RingAttr>(encoding);
  if (!encodedRing) {
    return op->emitOpError()
           << "the provided tensor encoding is not a ring attribute";
  }

  if (encodedRing != ring) {
    return op->emitOpError()
           << "encoded ring type " << encodedRing
           << " is not equivalent to the polynomial ring " << ring;
  }

  unsigned polyDegree = ring.getPolynomialModulus().getPolynomial().getDegree();
  ArrayRef<int64_t> tensorShape = tensorType.getShape();
  bool compatible = tensorShape.size() == 1 && tensorShape[0] == polyDegree;
  if (!compatible) {
    InFlightDiagnostic diag = op->emitOpError()
                              << "tensor type " << tensorType
                              << " does not match output type " << ring;
    diag.attachNote() << "the tensor must have shape [d] where d "
                         "is exactly the degree of the polynomialModulus of "
                         "the polynomial type's ring attribute";
    return diag;
  }

  if (!ring.getPrimitiveRoot()) {
    return op->emitOpError()
           << "ring type " << ring << " does not provide a primitive root "
           << "of unity, which is required to express an NTT";
  }

  if (!isPrimitiveNthRootOfUnity(ring.getPrimitiveRoot().getValue(), polyDegree,
                                 ring.getCoefficientModulus().getValue())) {
    return op->emitOpError()
           << "ring type " << ring << " has a primitiveRoot attribute '"
           << ring.getPrimitiveRoot()
           << "' that is not a primitive root of the coefficient ring";
  }

  return success();
}

LogicalResult NTTOp::verify() {
  auto ring = getInput().getType().getRing();
  auto tensorType = getOutput().getType();
  return verifyNTTOp(this->getOperation(), ring, tensorType);
}

LogicalResult INTTOp::verify() {
  auto tensorType = getInput().getType();
  auto ring = getOutput().getType().getRing();
  return verifyNTTOp(this->getOperation(), ring, tensorType);
}
