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
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace test;

//===----------------------------------------------------------------------===//
// CompoundAAttr
//===----------------------------------------------------------------------===//

Attribute CompoundAAttr::parse(AsmParser &parser, Type type) {
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

void CompoundAAttr::print(AsmPrinter &printer) const {
  printer << "<" << getWidthOfSomething() << ", " << getOneType() << ", [";
  llvm::interleaveComma(getArrayOfInts(), printer);
  printer << "]>";
}

//===----------------------------------------------------------------------===//
// CompoundAAttr
//===----------------------------------------------------------------------===//

Attribute TestI64ElementsAttr::parse(AsmParser &parser, Type type) {
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
      parser.getContext(), llvm::cast<ShapedType>(type), elements);
}

void TestI64ElementsAttr::print(AsmPrinter &printer) const {
  printer << "<[";
  llvm::interleaveComma(getElements(), printer);
  printer << "]>";
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
// TestSubElementsAccessAttr
//===----------------------------------------------------------------------===//

Attribute TestSubElementsAccessAttr::parse(::mlir::AsmParser &parser,
                                           ::mlir::Type type) {
  Attribute first, second, third;
  if (parser.parseLess() || parser.parseAttribute(first) ||
      parser.parseComma() || parser.parseAttribute(second) ||
      parser.parseComma() || parser.parseAttribute(third) ||
      parser.parseGreater()) {
    return {};
  }
  return get(parser.getContext(), first, second, third);
}

void TestSubElementsAccessAttr::print(::mlir::AsmPrinter &printer) const {
  printer << "<" << getFirst() << ", " << getSecond() << ", " << getThird()
          << ">";
}

//===----------------------------------------------------------------------===//
// TestExtern1DI64ElementsAttr
//===----------------------------------------------------------------------===//

ArrayRef<uint64_t> TestExtern1DI64ElementsAttr::getElements() const {
  if (auto *blob = getHandle().getBlob())
    return blob->getDataAs<uint64_t>();
  return std::nullopt;
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
// CopyCountAttr Implementation
//===----------------------------------------------------------------------===//

CopyCount::CopyCount(const CopyCount &rhs) : value(rhs.value) {
  CopyCount::counter++;
}

CopyCount &CopyCount::operator=(const CopyCount &rhs) {
  CopyCount::counter++;
  value = rhs.value;
  return *this;
}

int CopyCount::counter;

static bool operator==(const test::CopyCount &lhs, const test::CopyCount &rhs) {
  return lhs.value == rhs.value;
}

llvm::raw_ostream &test::operator<<(llvm::raw_ostream &os,
                                    const test::CopyCount &value) {
  return os << value.value;
}

template <>
struct mlir::FieldParser<test::CopyCount> {
  static FailureOr<test::CopyCount> parse(AsmParser &parser) {
    std::string value;
    if (parser.parseKeyword(value))
      return failure();
    return test::CopyCount(value);
  }
};
namespace test {
llvm::hash_code hash_value(const test::CopyCount &copyCount) {
  return llvm::hash_value(copyCount.value);
}
} // namespace test

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
// Tablegen Generated Definitions
//===----------------------------------------------------------------------===//

#include "TestAttrInterfaces.cpp.inc"
#include "TestOpEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "TestAttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// Dynamic Attributes
//===----------------------------------------------------------------------===//

/// Define a singleton dynamic attribute.
static std::unique_ptr<DynamicAttrDefinition>
getDynamicSingletonAttr(TestDialect *testDialect) {
  return DynamicAttrDefinition::get(
      "dynamic_singleton", testDialect,
      [](function_ref<InFlightDiagnostic()> emitError,
         ArrayRef<Attribute> args) {
        if (!args.empty()) {
          emitError() << "expected 0 attribute arguments, but had "
                      << args.size();
          return failure();
        }
        return success();
      });
}

/// Define a dynamic attribute representing a pair or attributes.
static std::unique_ptr<DynamicAttrDefinition>
getDynamicPairAttr(TestDialect *testDialect) {
  return DynamicAttrDefinition::get(
      "dynamic_pair", testDialect,
      [](function_ref<InFlightDiagnostic()> emitError,
         ArrayRef<Attribute> args) {
        if (args.size() != 2) {
          emitError() << "expected 2 attribute arguments, but had "
                      << args.size();
          return failure();
        }
        return success();
      });
}

static std::unique_ptr<DynamicAttrDefinition>
getDynamicCustomAssemblyFormatAttr(TestDialect *testDialect) {
  auto verifier = [](function_ref<InFlightDiagnostic()> emitError,
                     ArrayRef<Attribute> args) {
    if (args.size() != 2) {
      emitError() << "expected 2 attribute arguments, but had " << args.size();
      return failure();
    }
    return success();
  };

  auto parser = [](AsmParser &parser,
                   llvm::SmallVectorImpl<Attribute> &parsedParams) {
    Attribute leftAttr, rightAttr;
    if (parser.parseLess() || parser.parseAttribute(leftAttr) ||
        parser.parseColon() || parser.parseAttribute(rightAttr) ||
        parser.parseGreater())
      return failure();
    parsedParams.push_back(leftAttr);
    parsedParams.push_back(rightAttr);
    return success();
  };

  auto printer = [](AsmPrinter &printer, ArrayRef<Attribute> params) {
    printer << "<" << params[0] << ":" << params[1] << ">";
  };

  return DynamicAttrDefinition::get("dynamic_custom_assembly_format",
                                    testDialect, std::move(verifier),
                                    std::move(parser), std::move(printer));
}

//===----------------------------------------------------------------------===//
// TestDialect
//===----------------------------------------------------------------------===//

void TestDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "TestAttrDefs.cpp.inc"
      >();
  registerDynamicAttr(getDynamicSingletonAttr(this));
  registerDynamicAttr(getDynamicPairAttr(this));
  registerDynamicAttr(getDynamicCustomAssemblyFormatAttr(this));
}
