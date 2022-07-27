//===- ComplexDialect.cpp - MLIR Complex Dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

#include "mlir/Dialect/Complex/IR/ComplexOpsDialect.cpp.inc"

void complex::ComplexDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Complex/IR/ComplexOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Complex/IR/ComplexAttributes.cpp.inc"
      >();
}

Operation *complex::ComplexDialect::materializeConstant(OpBuilder &builder,
                                                        Attribute value,
                                                        Type type,
                                                        Location loc) {
  if (complex::ConstantOp::isBuildableWith(value, type)) {
    return builder.create<complex::ConstantOp>(loc, type,
                                               value.cast<ArrayAttr>());
  }
  if (arith::ConstantOp::isBuildableWith(value, type))
    return builder.create<arith::ConstantOp>(loc, type, value);
  return nullptr;
}

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Complex/IR/ComplexAttributes.cpp.inc"

LogicalResult complex::NumberAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::llvm::APFloat real, ::llvm::APFloat imag, ::mlir::Type type) {

  if (!type.isa<FloatType>())
    return emitError()
           << "element of the complex attribute must be float like type.";

  const auto &typeFloatSemantics = type.cast<FloatType>().getFloatSemantics();
  if (&real.getSemantics() != &typeFloatSemantics)
    return emitError()
           << "type doesn't match the type implied by its `real` value";
  if (&imag.getSemantics() != &typeFloatSemantics)
    return emitError()
           << "type doesn't match the type implied by its `imag` value";

  return success();
}

void complex::NumberAttr::print(AsmPrinter &printer) const {
  printer << "<:" << getType() << " " << getReal() << ", " << getImag() << ">";
}

Attribute complex::NumberAttr::parse(AsmParser &parser, Type odsType) {
  Type type;
  double real, imag;
  if (parser.parseLess() || parser.parseColon() || parser.parseType(type) ||
      parser.parseFloat(real) || parser.parseComma() ||
      parser.parseFloat(imag) || parser.parseGreater())
    return {};

  bool unused = false;
  APFloat realFloat(real);
  realFloat.convert(type.cast<FloatType>().getFloatSemantics(),
                    APFloat::rmNearestTiesToEven, &unused);
  APFloat imagFloat(imag);
  imagFloat.convert(type.cast<FloatType>().getFloatSemantics(),
                    APFloat::rmNearestTiesToEven, &unused);
  return NumberAttr::get(parser.getContext(), realFloat, imagFloat, type);
}
