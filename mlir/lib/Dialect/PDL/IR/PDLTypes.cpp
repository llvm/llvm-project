//===- PDLTypes.cpp - Pattern Descriptor Language Types -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::pdl;

//===----------------------------------------------------------------------===//
// TableGen'd type method definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/PDL/IR/PDLOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// PDLDialect
//===----------------------------------------------------------------------===//

void PDLDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/PDL/IR/PDLOpsTypes.cpp.inc"
      >();
}

static Type parsePDLType(AsmParser &parser) {
  StringRef typeTag;
  {
    Type genType;
    auto parseResult = generatedTypeParser(parser, &typeTag, genType);
    if (parseResult.hasValue())
      return genType;
  }

  // FIXME: This ends up with a double error being emitted if `RangeType` also
  // emits an error. We should rework the `generatedTypeParser` to better
  // support when the keyword is valid but the individual type parser itself
  // emits an error.
  parser.emitError(parser.getNameLoc(), "invalid 'pdl' type: `")
      << typeTag << "'";
  return Type();
}

//===----------------------------------------------------------------------===//
// PDL Types
//===----------------------------------------------------------------------===//

bool PDLType::classof(Type type) {
  return llvm::isa<PDLDialect>(type.getDialect());
}

//===----------------------------------------------------------------------===//
// RangeType
//===----------------------------------------------------------------------===//

Type RangeType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();

  SMLoc elementLoc = parser.getCurrentLocation();
  Type elementType = parsePDLType(parser);
  if (!elementType || parser.parseGreater())
    return Type();

  if (elementType.isa<RangeType>()) {
    parser.emitError(elementLoc)
        << "element of pdl.range cannot be another range, but got"
        << elementType;
    return Type();
  }
  return RangeType::get(elementType);
}

void RangeType::print(AsmPrinter &printer) const {
  printer << "<";
  (void)generatedTypePrinter(getElementType(), printer);
  printer << ">";
}

LogicalResult RangeType::verify(function_ref<InFlightDiagnostic()> emitError,
                                Type elementType) {
  if (!elementType.isa<PDLType>() || elementType.isa<RangeType>()) {
    return emitError()
           << "expected element of pdl.range to be one of [!pdl.attribute, "
              "!pdl.operation, !pdl.type, !pdl.value], but got "
           << elementType;
  }
  return success();
}
