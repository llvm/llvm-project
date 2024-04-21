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
  auto tensorShape = getInput().getType().getShape();
  auto ring = getOutput().getType().getRing();
  auto polyDegree = ring.getPolynomialModulus().getPolynomial().getDegree();
  bool compatible = tensorShape.size() == 1 && tensorShape[0] <= polyDegree;
  if (!compatible) {
    return emitOpError()
           << "input type " << getInput().getType()
           << " does not match output type " << getOutput().getType()
           << ". The input type must be a tensor of shape [d] where d "
              "is at most the degree of the polynomialModulus of "
              "the output type's ring attribute.";
  }

  APInt coefficientModulus = ring.getCoefficientModulus().getValue();
  unsigned cmodBitWidth = coefficientModulus.ceilLogBase2();
  unsigned inputBitWidth = getInput().getType().getElementTypeBitWidth();

  if (inputBitWidth > cmodBitWidth) {
    return emitOpError() << "input tensor element type "
                         << getInput().getType().getElementType()
                         << " is too large to fit in the coefficients of "
                         << getOutput().getType()
                         << ". The input tensor's elements must be rescaled"
                            " to fit before using from_tensor.";
  }

  return success();
}

LogicalResult ToTensorOp::verify() {
  auto tensorShape = getOutput().getType().getShape();
  auto polyDegree = getInput()
                        .getType()
                        .getRing()
                        .getPolynomialModulus()
                        .getPolynomial()
                        .getDegree();
  bool compatible = tensorShape.size() == 1 && tensorShape[0] == polyDegree;

  return compatible
             ? success()
             : emitOpError()
                   << "input type " << getInput().getType()
                   << " does not match output type " << getOutput().getType()
                   << ". The input type must be a tensor of shape [d] where d "
                      "is exactly the degree of the polynomialModulus of "
                      "the output type's ring attribute.";
}

LogicalResult MonomialMulOp::verify() {
  auto ring = getInput().getType().getRing();
  auto idealTerms = ring.getPolynomialModulus().getPolynomial().getTerms();
  bool compatible =
      idealTerms.size() == 2 &&
      (idealTerms[0].coefficient == -1 && idealTerms[0].exponent == 0) &&
      (idealTerms[1].coefficient == 1);

  return compatible ? success()
                    : emitOpError()
                          << "ring type " << ring
                          << " is not supported yet. The ring "
                             "must be of the form (x^n - 1) for some n";
}
