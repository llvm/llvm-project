//===- TestAttributesSyntax.cpp - Test attributes in `Syntax` -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the attributes of the test dialect in partition `Syntax`,
// which exercise custom and declarative assembly formats.
//
//===----------------------------------------------------------------------===//

#include "TestAttributesSyntax.h"
#include "TestDialectBase.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/APFloat.h"

using namespace mlir;
using namespace test;

//===----------------------------------------------------------------------===//
// TestDecimalShapeAttr
//===----------------------------------------------------------------------===//

Attribute TestDecimalShapeAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess()) {
    return Attribute();
  }
  SmallVector<int64_t> shape;
  if (parser.parseOptionalGreater()) {
    auto parseDecimal = [&]() {
      shape.emplace_back();
      auto parseResult = parser.parseOptionalDecimalInteger(shape.back());
      if (!parseResult.has_value() || failed(*parseResult)) {
        parser.emitError(parser.getCurrentLocation()) << "expected an integer";
        return failure();
      }
      return success();
    };
    if (failed(parseDecimal())) {
      return Attribute();
    }
    while (failed(parser.parseOptionalGreater())) {
      if (failed(parser.parseXInDimensionList()) || failed(parseDecimal())) {
        return Attribute();
      }
    }
  }
  return get(parser.getContext(), shape);
}

void TestDecimalShapeAttr::print(AsmPrinter &printer) const {
  printer << "<";
  llvm::interleave(getShape(), printer, "x");
  printer << ">";
}

LogicalResult TestAttrWithFormatAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, int64_t one, std::string two,
    IntegerAttr three, ArrayRef<int> four, uint64_t five, ArrayRef<int> six,
    ArrayRef<AttrWithTypeBuilderAttr> arrayOfAttrs) {
  if (four.size() != static_cast<unsigned>(one))
    return emitError() << "expected 'one' to equal 'four.size()'";
  return success();
}

//===----------------------------------------------------------------------===//
// Utility Functions for Generated Attributes
//===----------------------------------------------------------------------===//

static FailureOr<SmallVector<int>> parseIntArray(AsmParser &parser) {
  SmallVector<int> ints;
  if (parser.parseLSquare() || parser.parseCommaSeparatedList([&]() {
        ints.push_back(0);
        return parser.parseInteger(ints.back());
      }) ||
      parser.parseRSquare())
    return failure();
  return ints;
}

static void printIntArray(AsmPrinter &printer, ArrayRef<int> ints) {
  printer << '[';
  llvm::interleaveComma(ints, printer);
  printer << ']';
}

//===----------------------------------------------------------------------===//
// TestCustomAnchorAttr
//===----------------------------------------------------------------------===//

static ParseResult parseTrueFalse(AsmParser &p, std::optional<int> &result) {
  bool b;
  if (p.parseInteger(b))
    return failure();
  result = b;
  return success();
}

static void printTrueFalse(AsmPrinter &p, std::optional<int> result) {
  p << (*result ? "true" : "false");
}

//===----------------------------------------------------------------------===//
// TestConditionalAliasAttr
//===----------------------------------------------------------------------===//

/// Attempt to parse the conditionally-aliased string attribute as a keyword or
/// string, else try to parse an alias.
static ParseResult parseConditionalAlias(AsmParser &p, StringAttr &value) {
  std::string str;
  if (succeeded(p.parseOptionalKeywordOrString(&str))) {
    value = StringAttr::get(p.getContext(), str);
    return success();
  }
  return p.parseAttribute(value);
}

/// Print the string attribute as an alias if it has one, otherwise print it as
/// a keyword if possible.
static void printConditionalAlias(AsmPrinter &p, StringAttr value) {
  if (succeeded(p.printAlias(value)))
    return;
  p.printKeywordOrString(value);
}

//===----------------------------------------------------------------------===//
// Custom Float Attribute
//===----------------------------------------------------------------------===//

static void printCustomFloatAttr(AsmPrinter &p, StringAttr typeStrAttr,
                                 APFloat value) {
  p << typeStrAttr << " : " << value;
}

static ParseResult parseCustomFloatAttr(AsmParser &p, StringAttr &typeStrAttr,
                                        FailureOr<APFloat> &value) {

  std::string str;
  if (p.parseString(&str))
    return failure();

  typeStrAttr = StringAttr::get(p.getContext(), str);

  if (p.parseColon())
    return failure();

  const llvm::fltSemantics *semantics;
  if (str == "float")
    semantics = &llvm::APFloat::IEEEsingle();
  else if (str == "double")
    semantics = &llvm::APFloat::IEEEdouble();
  else if (str == "fp80")
    semantics = &llvm::APFloat::x87DoubleExtended();
  else
    return p.emitError(p.getCurrentLocation(), "unknown float type, expected "
                                               "'float', 'double' or 'fp80'");

  APFloat parsedValue(0.0);
  if (p.parseFloat(*semantics, parsedValue))
    return failure();

  value.emplace(parsedValue);
  return success();
}

//===----------------------------------------------------------------------===//
// TestCustomStructAttr
//===----------------------------------------------------------------------===//

static void printCustomStructAttr(AsmPrinter &p, int64_t value) {
  if (ShapedType::isDynamic(value)) {
    p << "?";
  } else {
    p.printStrippedAttrOrType(value);
  }
}

static ParseResult parseCustomStructAttr(AsmParser &p, int64_t &value) {
  if (succeeded(p.parseOptionalQuestion())) {
    value = ShapedType::kDynamic;
    return success();
  }
  return p.parseInteger(value);
}

static void printCustomOptStructFieldAttr(AsmPrinter &p, ArrayAttr attr) {
  if (attr && attr.size() == 1 && isa<IntegerAttr>(attr[0])) {
    p << cast<IntegerAttr>(attr[0]).getInt();
  } else {
    p.printStrippedAttrOrType(attr);
  }
}

static ParseResult parseCustomOptStructFieldAttr(AsmParser &p,
                                                 ArrayAttr &attr) {
  int64_t value;
  OptionalParseResult result = p.parseOptionalInteger(value);
  if (result.has_value()) {
    if (failed(result.value()))
      return failure();
    attr = ArrayAttr::get(
        p.getContext(),
        {IntegerAttr::get(IntegerType::get(p.getContext(), 64), value)});
    return success();
  }
  return p.parseAttribute(attr);
}

//===----------------------------------------------------------------------===//
// TestAttrNewlineAndIndent
//===----------------------------------------------------------------------===//

Attribute TestAttrNewlineAndIndentAttr::parse(::mlir::AsmParser &parser,
                                              ::mlir::Type type) {
  Type indentType;
  if (parser.parseLess() || parser.parseType(indentType) ||
      parser.parseGreater()) {
    return Attribute();
  }
  return get(parser.getContext(), indentType);
}

void TestAttrNewlineAndIndentAttr::print(::mlir::AsmPrinter &printer) const {
  printer << "<";
  printer.increaseIndent();
  printer.printNewline();
  printer << getIndentType();
  printer.decreaseIndent();
  printer.printNewline();
  printer << ">";
}

//===----------------------------------------------------------------------===//
// Tablegen Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "TestAttributesSyntax.cpp.inc"

//===----------------------------------------------------------------------===//
// SlashAttr
//===----------------------------------------------------------------------===//

Attribute SlashAttr::parse(AsmParser &parser, Type type) {
  int lhs, rhs;

  if (parser.parseLess() || parser.parseInteger(lhs) || parser.parseSlash() ||
      parser.parseInteger(rhs) || parser.parseGreater())
    return Attribute();

  return SlashAttr::get(parser.getContext(), lhs, rhs);
}

void SlashAttr::print(AsmPrinter &printer) const {
  printer << "<" << getLhs() << " / " << getRhs() << ">";
}
