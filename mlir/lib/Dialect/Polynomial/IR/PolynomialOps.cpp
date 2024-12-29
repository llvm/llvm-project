//===- PolynomialOps.cpp - Polynomial dialect ops ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polynomial/IR/PolynomialOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Polynomial/IR/Polynomial.h"
#include "mlir/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "mlir/Dialect/Polynomial/IR/PolynomialTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
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
  IntPolynomialAttr polyMod = ring.getPolynomialModulus();
  if (polyMod) {
    unsigned polyDegree = polyMod.getPolynomial().getDegree();
    bool compatible = tensorShape.size() == 1 && tensorShape[0] <= polyDegree;
    if (!compatible) {
      InFlightDiagnostic diag = emitOpError()
                                << "input type " << getInput().getType()
                                << " does not match output type "
                                << getOutput().getType();
      diag.attachNote()
          << "the input type must be a tensor of shape [d] where d "
             "is at most the degree of the polynomialModulus of "
             "the output type's ring attribute";
      return diag;
    }
  }

  unsigned inputBitWidth = getInput().getType().getElementTypeBitWidth();
  if (inputBitWidth > ring.getCoefficientType().getIntOrFloatBitWidth()) {
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
  IntPolynomialAttr polyMod =
      getInput().getType().getRing().getPolynomialModulus();
  if (polyMod) {
    unsigned polyDegree = polyMod.getPolynomial().getDegree();
    bool compatible = tensorShape.size() == 1 && tensorShape[0] == polyDegree;

    if (compatible)
      return success();

    InFlightDiagnostic diag = emitOpError()
                              << "input type " << getInput().getType()
                              << " does not match output type "
                              << getOutput().getType();
    diag.attachNote()
        << "the output type must be a tensor of shape [d] where d "
           "is at most the degree of the polynomialModulus of "
           "the input type's ring attribute";
    return diag;
  }

  return success();
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
bool isPrimitiveNthRootOfUnity(const APInt &root, const APInt &n,
                               const APInt &cmod) {
  // The first or subsequent multiplications, may overflow the input bit width,
  // so scale them up to ensure they do not overflow.
  unsigned requiredBitWidth =
      std::max(root.getActiveBits() * 2, cmod.getActiveBits() * 2);
  APInt r = APInt(root).zextOrTrunc(requiredBitWidth);
  APInt cmodExt = APInt(cmod).zextOrTrunc(requiredBitWidth);
  assert(r.ule(cmodExt) && "root must be less than cmod");
  uint64_t upperBound = n.getZExtValue();

  APInt a = r;
  for (size_t k = 1; k < upperBound; k++) {
    if (a.isOne())
      return false;
    a = (a * r).urem(cmodExt);
  }
  return a.isOne();
}

/// Verify that the types involved in an NTT or INTT operation are
/// compatible.
static LogicalResult verifyNTTOp(Operation *op, RingAttr ring,
                                 RankedTensorType tensorType,
                                 std::optional<PrimitiveRootAttr> root) {
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

  if (root.has_value()) {
    APInt rootValue = root.value().getValue().getValue();
    APInt rootDegree = root.value().getDegree().getValue();
    APInt cmod = ring.getCoefficientModulus().getValue();
    if (!isPrimitiveNthRootOfUnity(rootValue, rootDegree, cmod)) {
      return op->emitOpError()
             << "provided root " << rootValue.getZExtValue()
             << " is not a primitive root "
             << "of unity mod " << cmod.getZExtValue()
             << ", with the specified degree " << rootDegree.getZExtValue();
    }
  }

  return success();
}

LogicalResult NTTOp::verify() {
  return verifyNTTOp(this->getOperation(), getInput().getType().getRing(),
                     getOutput().getType(), getRoot());
}

LogicalResult INTTOp::verify() {
  return verifyNTTOp(this->getOperation(), getOutput().getType().getRing(),
                     getInput().getType(), getRoot());
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  // Using the built-in parser.parseAttribute requires the full
  // #polynomial.typed_int_polynomial syntax, which is excessive.
  // Instead we parse a keyword int to signal it's an integer polynomial
  Type type;
  if (succeeded(parser.parseOptionalKeyword("float"))) {
    Attribute floatPolyAttr = FloatPolynomialAttr::parse(parser, nullptr);
    if (floatPolyAttr) {
      if (parser.parseColon() || parser.parseType(type))
        return failure();
      result.addAttribute("value",
                          TypedFloatPolynomialAttr::get(type, floatPolyAttr));
      result.addTypes(type);
      return success();
    }
  }

  if (succeeded(parser.parseOptionalKeyword("int"))) {
    Attribute intPolyAttr = IntPolynomialAttr::parse(parser, nullptr);
    if (intPolyAttr) {
      if (parser.parseColon() || parser.parseType(type))
        return failure();

      result.addAttribute("value",
                          TypedIntPolynomialAttr::get(type, intPolyAttr));
      result.addTypes(type);
      return success();
    }
  }

  // In the worst case, still accept the verbose versions.
  TypedIntPolynomialAttr typedIntPolyAttr;
  OptionalParseResult res =
      parser.parseOptionalAttribute<TypedIntPolynomialAttr>(
          typedIntPolyAttr, "value", result.attributes);
  if (res.has_value() && succeeded(res.value())) {
    result.addTypes(typedIntPolyAttr.getType());
    return success();
  }

  TypedFloatPolynomialAttr typedFloatPolyAttr;
  res = parser.parseAttribute<TypedFloatPolynomialAttr>(
      typedFloatPolyAttr, "value", result.attributes);
  if (res.has_value() && succeeded(res.value())) {
    result.addTypes(typedFloatPolyAttr.getType());
    return success();
  }

  return failure();
}

void ConstantOp::print(OpAsmPrinter &p) {
  p << " ";
  if (auto intPoly = dyn_cast<TypedIntPolynomialAttr>(getValue())) {
    p << "int";
    intPoly.getValue().print(p);
  } else if (auto floatPoly = dyn_cast<TypedFloatPolynomialAttr>(getValue())) {
    p << "float";
    floatPoly.getValue().print(p);
  } else {
    assert(false && "unexpected attribute type");
  }
  p << " : ";
  p.printType(getOutput().getType());
}

LogicalResult ConstantOp::inferReturnTypes(
    MLIRContext *context, std::optional<mlir::Location> location,
    ConstantOp::Adaptor adaptor,
    llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  Attribute operand = adaptor.getValue();
  if (auto intPoly = dyn_cast<TypedIntPolynomialAttr>(operand)) {
    inferredReturnTypes.push_back(intPoly.getType());
  } else if (auto floatPoly = dyn_cast<TypedFloatPolynomialAttr>(operand)) {
    inferredReturnTypes.push_back(floatPoly.getType());
  } else {
    assert(false && "unexpected attribute type");
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd canonicalization patterns
//===----------------------------------------------------------------------===//

namespace {
#include "PolynomialCanonicalization.inc"
} // namespace

void SubOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<SubAsAdd>(context);
}

void NTTOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<NTTAfterINTT>(context);
}

void INTTOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<INTTAfterNTT>(context);
}
