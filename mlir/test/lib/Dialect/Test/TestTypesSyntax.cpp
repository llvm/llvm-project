//===- TestTypesSyntax.cpp - Test types in `Syntax` -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the types of the test dialect in partition `Syntax`, which
// exercise custom and declarative assembly formats.
//
//===----------------------------------------------------------------------===//

#include "TestTypesSyntax.h"
#include "TestDialectBase.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace test;

//===----------------------------------------------------------------------===//
// TestCustomType
//===----------------------------------------------------------------------===//

static ParseResult parseCustomTypeA(AsmParser &parser, int &aResult) {
  return parser.parseInteger(aResult);
}

static void printCustomTypeA(AsmPrinter &printer, int a) { printer << a; }

static ParseResult parseCustomTypeB(AsmParser &parser, int a,
                                    std::optional<int> &bResult) {
  if (a < 0)
    return success();
  for (int i : llvm::seq(0, a))
    if (failed(parser.parseInteger(i)))
      return failure();
  bResult.emplace(0);
  return parser.parseInteger(*bResult);
}

static void printCustomTypeB(AsmPrinter &printer, int a, std::optional<int> b) {
  if (a < 0)
    return;
  printer << ' ';
  for (int i : llvm::seq(0, a))
    printer << i << ' ';
  printer << *b;
}

static ParseResult parseFooString(AsmParser &parser, std::string &foo) {
  std::string result;
  if (parser.parseString(&result))
    return failure();
  foo = std::move(result);
  return success();
}

static void printFooString(AsmPrinter &printer, StringRef foo) {
  printer << '"' << foo << '"';
}

static ParseResult parseBarString(AsmParser &parser, StringRef foo) {
  return parser.parseKeyword(foo);
}

static void printBarString(AsmPrinter &printer, StringRef foo) {
  printer << foo;
}
//===----------------------------------------------------------------------===//
// Tablegen Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "TestTypesSyntax.cpp.inc"

//===----------------------------------------------------------------------===//
// TestTypeNewlineAndIndent
//===----------------------------------------------------------------------===//

Type TestTypeNewlineAndIndentType::parse(::mlir::AsmParser &parser) {
  if (parser.parseLess() || parser.parseKeyword("indented_content") ||
      parser.parseGreater()) {
    return Type();
  }
  return get(parser.getContext());
}

void TestTypeNewlineAndIndentType::print(::mlir::AsmPrinter &printer) const {
  printer << "<";
  printer.increaseIndent();
  printer.printNewline();
  printer << "indented_content";
  printer.decreaseIndent();
  printer.printNewline();
  printer << ">";
}
