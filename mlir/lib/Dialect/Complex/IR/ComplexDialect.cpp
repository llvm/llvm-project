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
  printer << "<:";
  printer.printType(getType());
  printer << " ";
  printer.printFloat(getReal());
  printer << ", ";
  printer.printFloat(getImag());
  printer << ">";
}

Attribute complex::NumberAttr::parse(AsmParser &parser, Type odsType) {
  if (failed(parser.parseLess()))
    return {};

  if (failed(parser.parseColon()))
    return {};

  Type type;
  if (failed(parser.parseType(type)))
    return {};

  double real;
  if (failed(parser.parseFloat(real)))
    return {};

  if (failed(parser.parseComma()))
    return {};

  double imag;
  if (failed(parser.parseFloat(imag)))
    return {};

  if (failed(parser.parseGreater()))
    return {};

  bool unused = false;
  auto realFloat = APFloat(real);
  realFloat.convert(type.cast<FloatType>().getFloatSemantics(),
                    APFloat::rmNearestTiesToEven, &unused);
  auto imagFloat = APFloat(imag);
  imagFloat.convert(type.cast<FloatType>().getFloatSemantics(),
                    APFloat::rmNearestTiesToEven, &unused);

  return NumberAttr::get(parser.getContext(), realFloat, imagFloat, type);
}
