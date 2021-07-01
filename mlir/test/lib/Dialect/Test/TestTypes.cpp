//===- TestTypes.cpp - MLIR Test Dialect Types ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains types defined by the TestDialect for testing various
// features of MLIR.
//
//===----------------------------------------------------------------------===//

#include "TestTypes.h"
#include "TestDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::test;

// Custom parser for SignednessSemantics.
static ParseResult
parseSignedness(DialectAsmParser &parser,
                TestIntegerType::SignednessSemantics &result) {
  StringRef signStr;
  auto loc = parser.getCurrentLocation();
  if (parser.parseKeyword(&signStr))
    return failure();
  if (signStr.compare_insensitive("u") ||
      signStr.compare_insensitive("unsigned"))
    result = TestIntegerType::SignednessSemantics::Unsigned;
  else if (signStr.compare_insensitive("s") ||
           signStr.compare_insensitive("signed"))
    result = TestIntegerType::SignednessSemantics::Signed;
  else if (signStr.compare_insensitive("n") ||
           signStr.compare_insensitive("none"))
    result = TestIntegerType::SignednessSemantics::Signless;
  else
    return parser.emitError(loc, "expected signed, unsigned, or none");
  return success();
}

// Custom printer for SignednessSemantics.
static void printSignedness(DialectAsmPrinter &printer,
                            const TestIntegerType::SignednessSemantics &ss) {
  switch (ss) {
  case TestIntegerType::SignednessSemantics::Unsigned:
    printer << "unsigned";
    break;
  case TestIntegerType::SignednessSemantics::Signed:
    printer << "signed";
    break;
  case TestIntegerType::SignednessSemantics::Signless:
    printer << "none";
    break;
  }
}

// The functions don't need to be in the header file, but need to be in the mlir
// namespace. Declare them here, then define them immediately below. Separating
// the declaration and definition adheres to the LLVM coding standards.
namespace mlir {
namespace test {
// FieldInfo is used as part of a parameter, so equality comparison is
// compulsory.
static bool operator==(const FieldInfo &a, const FieldInfo &b);
// FieldInfo is used as part of a parameter, so a hash will be computed.
static llvm::hash_code hash_value(const FieldInfo &fi); // NOLINT
} // namespace test
} // namespace mlir

// FieldInfo is used as part of a parameter, so equality comparison is
// compulsory.
static bool mlir::test::operator==(const FieldInfo &a, const FieldInfo &b) {
  return a.name == b.name && a.type == b.type;
}

// FieldInfo is used as part of a parameter, so a hash will be computed.
static llvm::hash_code mlir::test::hash_value(const FieldInfo &fi) { // NOLINT
  return llvm::hash_combine(fi.name, fi.type);
}

//===----------------------------------------------------------------------===//
// CompoundAType
//===----------------------------------------------------------------------===//

Type CompoundAType::parse(MLIRContext *ctxt, DialectAsmParser &parser) {
  int widthOfSomething;
  Type oneType;
  SmallVector<int, 4> arrayOfInts;
  if (parser.parseLess() || parser.parseInteger(widthOfSomething) ||
      parser.parseComma() || parser.parseType(oneType) || parser.parseComma() ||
      parser.parseLSquare())
    return Type();

  int i;
  while (!*parser.parseOptionalInteger(i)) {
    arrayOfInts.push_back(i);
    if (parser.parseOptionalComma())
      break;
  }

  if (parser.parseRSquare() || parser.parseGreater())
    return Type();

  return get(ctxt, widthOfSomething, oneType, arrayOfInts);
}
void CompoundAType::print(DialectAsmPrinter &printer) const {
  printer << "cmpnd_a<" << getWidthOfSomething() << ", " << getOneType()
          << ", [";
  auto intArray = getArrayOfInts();
  llvm::interleaveComma(intArray, printer);
  printer << "]>";
}

//===----------------------------------------------------------------------===//
// TestIntegerType
//===----------------------------------------------------------------------===//

// Example type validity checker.
LogicalResult
TestIntegerType::verify(function_ref<InFlightDiagnostic()> emitError,
                        unsigned width,
                        TestIntegerType::SignednessSemantics ss) {
  if (width > 8)
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// TestType
//===----------------------------------------------------------------------===//

void TestType::printTypeC(Location loc) const {
  emitRemark(loc) << *this << " - TestC";
}

//===----------------------------------------------------------------------===//
// TestTypeWithLayout
//===----------------------------------------------------------------------===//

Type TestTypeWithLayoutType::parse(MLIRContext *ctx, DialectAsmParser &parser) {
  unsigned val;
  if (parser.parseLess() || parser.parseInteger(val) || parser.parseGreater())
    return Type();
  return TestTypeWithLayoutType::get(ctx, val);
}

void TestTypeWithLayoutType::print(DialectAsmPrinter &printer) const {
  printer << "test_type_with_layout<" << getKey() << ">";
}

unsigned
TestTypeWithLayoutType::getTypeSizeInBits(const DataLayout &dataLayout,
                                          DataLayoutEntryListRef params) const {
  return extractKind(params, "size");
}

unsigned
TestTypeWithLayoutType::getABIAlignment(const DataLayout &dataLayout,
                                        DataLayoutEntryListRef params) const {
  return extractKind(params, "alignment");
}

unsigned TestTypeWithLayoutType::getPreferredAlignment(
    const DataLayout &dataLayout, DataLayoutEntryListRef params) const {
  return extractKind(params, "preferred");
}

bool TestTypeWithLayoutType::areCompatible(
    DataLayoutEntryListRef oldLayout, DataLayoutEntryListRef newLayout) const {
  unsigned old = extractKind(oldLayout, "alignment");
  return old == 1 || extractKind(newLayout, "alignment") <= old;
}

LogicalResult
TestTypeWithLayoutType::verifyEntries(DataLayoutEntryListRef params,
                                      Location loc) const {
  for (DataLayoutEntryInterface entry : params) {
    // This is for testing purposes only, so assert well-formedness.
    assert(entry.isTypeEntry() && "unexpected identifier entry");
    assert(entry.getKey().get<Type>().isa<TestTypeWithLayoutType>() &&
           "wrong type passed in");
    auto array = entry.getValue().dyn_cast<ArrayAttr>();
    assert(array && array.getValue().size() == 2 &&
           "expected array of two elements");
    auto kind = array.getValue().front().dyn_cast<StringAttr>();
    (void)kind;
    assert(kind &&
           (kind.getValue() == "size" || kind.getValue() == "alignment" ||
            kind.getValue() == "preferred") &&
           "unexpected kind");
    assert(array.getValue().back().isa<IntegerAttr>());
  }
  return success();
}

unsigned TestTypeWithLayoutType::extractKind(DataLayoutEntryListRef params,
                                             StringRef expectedKind) const {
  for (DataLayoutEntryInterface entry : params) {
    ArrayRef<Attribute> pair = entry.getValue().cast<ArrayAttr>().getValue();
    StringRef kind = pair.front().cast<StringAttr>().getValue();
    if (kind == expectedKind)
      return pair.back().cast<IntegerAttr>().getValue().getZExtValue();
  }
  return 1;
}

//===----------------------------------------------------------------------===//
// Tablegen Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "TestTypeDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// TestDialect
//===----------------------------------------------------------------------===//

void TestDialect::registerTypes() {
  addTypes<TestRecursiveType,
#define GET_TYPEDEF_LIST
#include "TestTypeDefs.cpp.inc"
           >();
}

static Type parseTestType(MLIRContext *ctxt, DialectAsmParser &parser,
                          SetVector<Type> &stack) {
  StringRef typeTag;
  if (failed(parser.parseKeyword(&typeTag)))
    return Type();

  {
    Type genType;
    auto parseResult = generatedTypeParser(ctxt, parser, typeTag, genType);
    if (parseResult.hasValue())
      return genType;
  }

  if (typeTag != "test_rec") {
    parser.emitError(parser.getNameLoc()) << "unknown type!";
    return Type();
  }

  StringRef name;
  if (parser.parseLess() || parser.parseKeyword(&name))
    return Type();
  auto rec = TestRecursiveType::get(parser.getBuilder().getContext(), name);

  // If this type already has been parsed above in the stack, expect just the
  // name.
  if (stack.contains(rec)) {
    if (failed(parser.parseGreater()))
      return Type();
    return rec;
  }

  // Otherwise, parse the body and update the type.
  if (failed(parser.parseComma()))
    return Type();
  stack.insert(rec);
  Type subtype = parseTestType(ctxt, parser, stack);
  stack.pop_back();
  if (!subtype || failed(parser.parseGreater()) || failed(rec.setBody(subtype)))
    return Type();

  return rec;
}

Type TestDialect::parseType(DialectAsmParser &parser) const {
  SetVector<Type> stack;
  return parseTestType(getContext(), parser, stack);
}

static void printTestType(Type type, DialectAsmPrinter &printer,
                          SetVector<Type> &stack) {
  if (succeeded(generatedTypePrinter(type, printer)))
    return;

  auto rec = type.cast<TestRecursiveType>();
  printer << "test_rec<" << rec.getName();
  if (!stack.contains(rec)) {
    printer << ", ";
    stack.insert(rec);
    printTestType(rec.getBody(), printer, stack);
    stack.pop_back();
  }
  printer << ">";
}

void TestDialect::printType(Type type, DialectAsmPrinter &printer) const {
  SetVector<Type> stack;
  printTestType(type, printer, stack);
}
