//===- GENXDialect.cpp - GENX IR Ops and Dialect registration -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the types and operation details for the GENX IR dialect in
// MLIR, and the LLVM IR dialect.  It also registers the dialect.
//
// The GENX dialect only contains GPU specific additions on top of the general
// LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/GENXDialect.h"
#include "mlir/Dialect/LLVMIR/GENXTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::GENX;

#include "mlir/Dialect/LLVMIR/GENXOpsDialect.cpp.inc"
#include "mlir/Dialect/LLVMIR/GENXOpsEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/LLVMIR/GENXOpsAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// GENXDialect
//===----------------------------------------------------------------------===//

void GENXDialect::initialize() {
  registerTypes();

  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LLVMIR/GENXOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/LLVMIR/GENXOpsAttributes.cpp.inc"
      >();

  // Support unknown operations because not all GENX operations are registered.
  allowUnknownOperations();
}

LogicalResult GENXDialect::verifyOperationAttribute(Operation *op,
                                                    NamedAttribute attr) {
  // TODO: fill this in.
  return success();
}

//===----------------------------------------------------------------------===//
// Type Parsing
//===----------------------------------------------------------------------===//
namespace {

Type parseJointMatrixElemType(GENXDialect const &dialect,
                              DialectAsmParser &parser) {
  Type type;
  SMLoc typeLoc = parser.getCurrentLocation();
  if (parser.parseType(type))
    return Type();

  // Allow GENX dialect types.
  if (&type.getDialect() == &dialect)
    return type;

  // Intel XMX allows only certain floating point and integer types.
  if (auto t = type.dyn_cast<FloatType>()) {
    // TODO: add check for TF32 once available.
    if (!t.isF16() && !t.isBF16() && !t.isF32() /*&& !t.isTF32()*/) {
      parser.emitError(
          typeLoc,
          "only fp16, bf16, f32 floating point types allowed but found ")
          << t;
      return Type();
    }
  } else if (auto t = type.dyn_cast<IntegerType>()) {
    if (!llvm::is_contained({8u, 32u}, t.getWidth())) {
      parser.emitError(typeLoc,
                       "only 8/32-bit integer types allowed but found ")
          << t;
      return Type();
    }
  } else {
    parser.emitError(typeLoc, "cannot use ") << type << " as element type";
    return Type();
  }

  return type;
}

/// Parses the next keyword in `parser` as an enumerator.
template <typename EnumClass, typename ParserType>
ParseResult parseEnumKeywordAttr(EnumClass &value, ParserType &parser,
                                 StringRef attrName) {
  StringRef keyword;
  SmallVector<NamedAttribute, 1> attr;
  auto loc = parser.getCurrentLocation();
  if (parser.parseKeyword(&keyword))
    return failure();
  if (std::optional<EnumClass> attr = GENX::symbolizeEnum<EnumClass>(keyword)) {
    value = *attr;
    return success();
  }
  return parser.emitError(loc, "invalid ")
         << attrName << " attribute specification: " << keyword;
}

// joint-matrix-type ::= `!genx.jointmatrix` `<`rows `x` cols `x` element-type
//                                           `,` layout `>`
Type parseJointMatrixType(GENXDialect const &dialect,
                          DialectAsmParser &parser) {
  if (parser.parseLess())
    return Type();

  SmallVector<int64_t, 2> dims;
  SMLoc countLoc = parser.getCurrentLocation();
  if (parser.parseDimensionList(dims, /*allowDynamic=*/false))
    return Type();

  if (dims.size() != 2) {
    parser.emitError(countLoc, "expected rows and columns size");
    return Type();
  }

  auto elementTy = parseJointMatrixElemType(dialect, parser);
  if (!elementTy)
    return Type();

  MatrixLayout matrixLayout;
  if (parser.parseComma() ||
      parseEnumKeywordAttr(matrixLayout, parser, "matrixLayout <id>"))
    return Type();

  return parser.parseGreater()
             ? Type()
             : JointMatrixType::get(elementTy, dims[0], dims[1], matrixLayout);
}

} // namespace

Type GENXDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();

  if (keyword == "jointmatrix")
    return parseJointMatrixType(*this, parser);

  parser.emitError(parser.getNameLoc(), "unknown GENX type: ") << keyword;
  return Type();
}

//===----------------------------------------------------------------------===//
// Type Printing
//===----------------------------------------------------------------------===//

void GENXDialect::printType(Type type, DialectAsmPrinter &os) const {
  auto print = [](JointMatrixType type, DialectAsmPrinter &os) {
    os << "jointmatrix<" << type.getNumRows() << "x" << type.getNumColumns()
       << "x" << type.getElementType() << ", "
       << stringifyMatrixLayout(type.getMatrixLayout()) << ">";
  };

  TypeSwitch<Type>(type)
      .Case<JointMatrixType>([&](auto type) { print(type, os); })
      .Default([](Type) { llvm_unreachable("unhandled GENX type"); });
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/GENXOps.cpp.inc"
