//===- TestAttributes.cpp - MLIR Test Dialect Attributes --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains attributes defined by the TestDialect for testing various
// features of MLIR.
//
//===----------------------------------------------------------------------===//

#include "TestAttributes.h"
#include "TestDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace test;

//===----------------------------------------------------------------------===//
// AttrWithSelfTypeParamAttr
//===----------------------------------------------------------------------===//

Attribute AttrWithSelfTypeParamAttr::parse(DialectAsmParser &parser,
                                           Type type) {
  Type selfType;
  if (parser.parseType(selfType))
    return Attribute();
  return get(parser.getContext(), selfType);
}

void AttrWithSelfTypeParamAttr::print(DialectAsmPrinter &printer) const {
  printer << "attr_with_self_type_param " << getType();
}

//===----------------------------------------------------------------------===//
// AttrWithTypeBuilderAttr
//===----------------------------------------------------------------------===//

Attribute AttrWithTypeBuilderAttr::parse(DialectAsmParser &parser, Type type) {
  IntegerAttr element;
  if (parser.parseAttribute(element))
    return Attribute();
  return get(parser.getContext(), element);
}

void AttrWithTypeBuilderAttr::print(DialectAsmPrinter &printer) const {
  printer << "attr_with_type_builder " << getAttr();
}

//===----------------------------------------------------------------------===//
// CompoundAAttr
//===----------------------------------------------------------------------===//

Attribute CompoundAAttr::parse(DialectAsmParser &parser, Type type) {
  int widthOfSomething;
  Type oneType;
  SmallVector<int, 4> arrayOfInts;
  if (parser.parseLess() || parser.parseInteger(widthOfSomething) ||
      parser.parseComma() || parser.parseType(oneType) || parser.parseComma() ||
      parser.parseLSquare())
    return Attribute();

  int intVal;
  while (!*parser.parseOptionalInteger(intVal)) {
    arrayOfInts.push_back(intVal);
    if (parser.parseOptionalComma())
      break;
  }

  if (parser.parseRSquare() || parser.parseGreater())
    return Attribute();
  return get(parser.getContext(), widthOfSomething, oneType, arrayOfInts);
}

void CompoundAAttr::print(DialectAsmPrinter &printer) const {
  printer << "cmpnd_a<" << getWidthOfSomething() << ", " << getOneType()
          << ", [";
  llvm::interleaveComma(getArrayOfInts(), printer);
  printer << "]>";
}

//===----------------------------------------------------------------------===//
// CompoundAAttr
//===----------------------------------------------------------------------===//

Attribute TestI64ElementsAttr::parse(DialectAsmParser &parser, Type type) {
  SmallVector<uint64_t> elements;
  if (parser.parseLess() || parser.parseLSquare())
    return Attribute();
  uint64_t intVal;
  while (succeeded(*parser.parseOptionalInteger(intVal))) {
    elements.push_back(intVal);
    if (parser.parseOptionalComma())
      break;
  }

  if (parser.parseRSquare() || parser.parseGreater())
    return Attribute();
  return parser.getChecked<TestI64ElementsAttr>(
      parser.getContext(), type.cast<ShapedType>(), elements);
}

void TestI64ElementsAttr::print(DialectAsmPrinter &printer) const {
  printer << "i64_elements<[";
  llvm::interleaveComma(getElements(), printer);
  printer << "] : " << getType() << ">";
}

LogicalResult
TestI64ElementsAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                            ShapedType type, ArrayRef<uint64_t> elements) {
  if (type.getNumElements() != static_cast<int64_t>(elements.size())) {
    return emitError()
           << "number of elements does not match the provided shape type, got: "
           << elements.size() << ", but expected: " << type.getNumElements();
  }
  if (type.getRank() != 1 || !type.getElementType().isSignlessInteger(64))
    return emitError() << "expected single rank 64-bit shape type, but got: "
                       << type;
  return success();
}

//===----------------------------------------------------------------------===//
// Tablegen Generated Definitions
//===----------------------------------------------------------------------===//

#include "TestAttrInterfaces.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "TestAttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// TestDialect
//===----------------------------------------------------------------------===//

void TestDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "TestAttrDefs.cpp.inc"
      >();
}

Attribute TestDialect::parseAttribute(DialectAsmParser &parser,
                                      Type type) const {
  StringRef attrTag;
  if (failed(parser.parseKeyword(&attrTag)))
    return Attribute();
  {
    Attribute attr;
    auto parseResult = generatedAttributeParser(parser, attrTag, type, attr);
    if (parseResult.hasValue())
      return attr;
  }
  parser.emitError(parser.getNameLoc(), "unknown test attribute");
  return Attribute();
}

void TestDialect::printAttribute(Attribute attr,
                                 DialectAsmPrinter &printer) const {
  if (succeeded(generatedAttributePrinter(attr, printer)))
    return;
}
