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
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace test;

// Custom parser for SignednessSemantics.
static ParseResult
parseSignedness(AsmParser &parser,
                TestIntegerType::SignednessSemantics &result) {
  StringRef signStr;
  auto loc = parser.getCurrentLocation();
  if (parser.parseKeyword(&signStr))
    return failure();
  if (signStr.equals_insensitive("u") || signStr.equals_insensitive("unsigned"))
    result = TestIntegerType::SignednessSemantics::Unsigned;
  else if (signStr.equals_insensitive("s") ||
           signStr.equals_insensitive("signed"))
    result = TestIntegerType::SignednessSemantics::Signed;
  else if (signStr.equals_insensitive("n") ||
           signStr.equals_insensitive("none"))
    result = TestIntegerType::SignednessSemantics::Signless;
  else
    return parser.emitError(loc, "expected signed, unsigned, or none");
  return success();
}

// Custom printer for SignednessSemantics.
static void printSignedness(AsmPrinter &printer,
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
namespace test {
// FieldInfo is used as part of a parameter, so equality comparison is
// compulsory.
static bool operator==(const FieldInfo &a, const FieldInfo &b);
// FieldInfo is used as part of a parameter, so a hash will be computed.
static llvm::hash_code hash_value(const FieldInfo &fi); // NOLINT
} // namespace test

// FieldInfo is used as part of a parameter, so equality comparison is
// compulsory.
static bool test::operator==(const FieldInfo &a, const FieldInfo &b) {
  return a.name == b.name && a.type == b.type;
}

// FieldInfo is used as part of a parameter, so a hash will be computed.
static llvm::hash_code test::hash_value(const FieldInfo &fi) { // NOLINT
  return llvm::hash_combine(fi.name, fi.type);
}

//===----------------------------------------------------------------------===//
// TestCustomType
//===----------------------------------------------------------------------===//

static LogicalResult parseCustomTypeA(AsmParser &parser,
                                      FailureOr<int> &a_result) {
  a_result.emplace();
  return parser.parseInteger(*a_result);
}

static void printCustomTypeA(AsmPrinter &printer, int a) { printer << a; }

static LogicalResult parseCustomTypeB(AsmParser &parser, int a,
                                      FailureOr<Optional<int>> &b_result) {
  if (a < 0)
    return success();
  for (int i : llvm::seq(0, a))
    if (failed(parser.parseInteger(i)))
      return failure();
  b_result.emplace(0);
  return parser.parseInteger(**b_result);
}

static void printCustomTypeB(AsmPrinter &printer, int a, Optional<int> b) {
  if (a < 0)
    return;
  printer << ' ';
  for (int i : llvm::seq(0, a))
    printer << i << ' ';
  printer << *b;
}

static LogicalResult parseFooString(AsmParser &parser,
                                    FailureOr<std::string> &foo) {
  std::string result;
  if (parser.parseString(&result))
    return failure();
  foo = std::move(result);
  return success();
}

static void printFooString(AsmPrinter &printer, StringRef foo) {
  printer << '"' << foo << '"';
}

static LogicalResult parseBarString(AsmParser &parser, StringRef foo) {
  return parser.parseKeyword(foo);
}

static void printBarString(AsmPrinter &printer, StringRef foo) {
  printer << ' ' << foo;
}
//===----------------------------------------------------------------------===//
// Tablegen Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "TestTypeDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// CompoundAType
//===----------------------------------------------------------------------===//

Type CompoundAType::parse(AsmParser &parser) {
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

  return get(parser.getContext(), widthOfSomething, oneType, arrayOfInts);
}
void CompoundAType::print(AsmPrinter &printer) const {
  printer << "<" << getWidthOfSomething() << ", " << getOneType() << ", [";
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

Type TestIntegerType::parse(AsmParser &parser) {
  SignednessSemantics signedness;
  int width;
  if (parser.parseLess() || parseSignedness(parser, signedness) ||
      parser.parseComma() || parser.parseInteger(width) ||
      parser.parseGreater())
    return Type();
  Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());
  return getChecked(loc, loc.getContext(), width, signedness);
}

void TestIntegerType::print(AsmPrinter &p) const {
  p << "<";
  printSignedness(p, getSignedness());
  p << ", " << getWidth() << ">";
}

//===----------------------------------------------------------------------===//
// TestStructType
//===----------------------------------------------------------------------===//

Type StructType::parse(AsmParser &p) {
  SmallVector<FieldInfo, 4> parameters;
  if (p.parseLess())
    return Type();
  while (succeeded(p.parseOptionalLBrace())) {
    Type type;
    StringRef name;
    if (p.parseKeyword(&name) || p.parseComma() || p.parseType(type) ||
        p.parseRBrace())
      return Type();
    parameters.push_back(FieldInfo{name, type});
    if (p.parseOptionalComma())
      break;
  }
  if (p.parseGreater())
    return Type();
  return get(p.getContext(), parameters);
}

void StructType::print(AsmPrinter &p) const {
  p << "<";
  llvm::interleaveComma(getFields(), p, [&](const FieldInfo &field) {
    p << "{" << field.name << "," << field.type << "}";
  });
  p << ">";
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

Type TestTypeWithLayoutType::parse(AsmParser &parser) {
  unsigned val;
  if (parser.parseLess() || parser.parseInteger(val) || parser.parseGreater())
    return Type();
  return TestTypeWithLayoutType::get(parser.getContext(), val);
}

void TestTypeWithLayoutType::print(AsmPrinter &printer) const {
  printer << "<" << getKey() << ">";
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
// TestDialect
//===----------------------------------------------------------------------===//

namespace {

struct PtrElementModel
    : public LLVM::PointerElementTypeInterface::ExternalModel<PtrElementModel,
                                                              SimpleAType> {};
} // namespace

void TestDialect::registerTypes() {
  addTypes<TestRecursiveType,
#define GET_TYPEDEF_LIST
#include "TestTypeDefs.cpp.inc"
           >();
  SimpleAType::attachInterface<PtrElementModel>(*getContext());
}

static Type parseTestType(AsmParser &parser, SetVector<Type> &stack) {
  StringRef typeTag;
  if (failed(parser.parseKeyword(&typeTag)))
    return Type();

  {
    Type genType;
    auto parseResult = generatedTypeParser(parser, typeTag, genType);
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
  auto rec = TestRecursiveType::get(parser.getContext(), name);

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
  Type subtype = parseTestType(parser, stack);
  stack.pop_back();
  if (!subtype || failed(parser.parseGreater()) || failed(rec.setBody(subtype)))
    return Type();

  return rec;
}

Type TestDialect::parseType(DialectAsmParser &parser) const {
  SetVector<Type> stack;
  return parseTestType(parser, stack);
}

static void printTestType(Type type, AsmPrinter &printer,
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
