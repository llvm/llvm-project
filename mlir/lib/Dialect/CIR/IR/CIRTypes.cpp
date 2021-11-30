//===- CIRTypes.cpp - MLIR CIR Types --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the types in the CIR dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/CIR/IR/CIRTypes.h"
#include "mlir/Dialect/CIR/IR/CIRDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/CIR/IR/CIROpsTypes.cpp.inc"

using namespace mlir;
using namespace mlir::cir;

//===----------------------------------------------------------------------===//
// General CIR parsing / printing
//===----------------------------------------------------------------------===//

Type CIRDialect::parseType(DialectAsmParser &parser) const {
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  StringRef mnemonic;
  Type genType;
  OptionalParseResult parseResult =
      generatedTypeParser(parser, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;
  parser.emitError(typeLoc, "unknown type in CIR dialect");
  return Type();
}

void CIRDialect::printType(Type type, DialectAsmPrinter &os) const {
  if (failed(generatedTypePrinter(type, os)))
    llvm_unreachable("unexpected CIR type kind");
}

Type PointerType::parse(mlir::AsmParser &parser) {
  if (parser.parseLess())
    return Type();
  Type pointeeType;
  if (parser.parseType(pointeeType))
    return Type();
  if (parser.parseGreater())
    return Type();
  return get(parser.getContext(), pointeeType);
}

void PointerType::print(mlir::AsmPrinter &printer) const {
  printer << "<";
  printer.printType(getPointee());
  printer << '>';
}

Type BoolType::parse(mlir::AsmParser &parser) {
  return get(parser.getContext());
}

void BoolType::print(mlir::AsmPrinter &printer) const {
}

//===----------------------------------------------------------------------===//
// CIR Dialect
//===----------------------------------------------------------------------===//

void CIRDialect::registerTypes() { addTypes<PointerType, BoolType>(); }
