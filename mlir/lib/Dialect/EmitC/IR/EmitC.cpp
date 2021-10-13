//===- EmitC.cpp - EmitC Dialect ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::emitc;

#include "mlir/Dialect/EmitC/IR/EmitCDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// EmitCDialect
//===----------------------------------------------------------------------===//

void EmitCDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/EmitC/IR/EmitC.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/EmitC/IR/EmitCTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/EmitC/IR/EmitCAttributes.cpp.inc"
      >();
}

/// Materialize a single constant operation from a given attribute value with
/// the desired resultant type.
Operation *EmitCDialect::materializeConstant(OpBuilder &builder,
                                             Attribute value, Type type,
                                             Location loc) {
  return builder.create<emitc::ConstantOp>(loc, type, value);
}

//===----------------------------------------------------------------------===//
// ApplyOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(ApplyOp op) {
  StringRef applicableOperator = op.applicableOperator();

  // Applicable operator must not be empty.
  if (applicableOperator.empty())
    return op.emitOpError("applicable operator must not be empty");

  // Only `*` and `&` are supported.
  if (applicableOperator != "&" && applicableOperator != "*")
    return op.emitOpError("applicable operator is illegal");

  return success();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(emitc::CallOp op) {
  // Callee must not be empty.
  if (op.callee().empty())
    return op.emitOpError("callee must not be empty");

  if (Optional<ArrayAttr> argsAttr = op.args()) {
    for (Attribute arg : argsAttr.getValue()) {
      if (arg.getType().isa<IndexType>()) {
        int64_t index = arg.cast<IntegerAttr>().getInt();
        // Args with elements of type index must be in range
        // [0..operands.size).
        if ((index < 0) || (index >= static_cast<int64_t>(op.getNumOperands())))
          return op.emitOpError("index argument is out of range");

        // Args with elements of type ArrayAttr must have a type.
      } else if (arg.isa<ArrayAttr>() && arg.getType().isa<NoneType>()) {
        return op.emitOpError("array argument has no type");
      }
    }
  }

  if (Optional<ArrayAttr> templateArgsAttr = op.template_args()) {
    for (Attribute tArg : templateArgsAttr.getValue()) {
      if (!tArg.isa<TypeAttr>() && !tArg.isa<IntegerAttr>() &&
          !tArg.isa<FloatAttr>() && !tArg.isa<emitc::OpaqueAttr>())
        return op.emitOpError("template argument has invalid type");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

/// The constant op requires that the attribute's type matches the return type.
static LogicalResult verify(emitc::ConstantOp &op) {
  Attribute value = op.value();
  Type type = op.getType();
  if (!value.getType().isa<NoneType>() && type != value.getType())
    return op.emitOpError() << "requires attribute's type (" << value.getType()
                            << ") to match op's return type (" << type << ")";
  return success();
}

OpFoldResult emitc::ConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return value();
}

//===----------------------------------------------------------------------===//
// IncludeOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, IncludeOp &op) {
  bool standardInclude = op.is_standard_include();

  p << " ";
  if (standardInclude)
    p << "<";
  p << "\"" << op.include() << "\"";
  if (standardInclude)
    p << ">";
}

static ParseResult parseIncludeOp(OpAsmParser &parser, OperationState &result) {
  bool standardInclude = !parser.parseOptionalLess();

  StringAttr include;
  OptionalParseResult includeParseResult =
      parser.parseOptionalAttribute(include, "include", result.attributes);
  if (!includeParseResult.hasValue())
    return parser.emitError(parser.getNameLoc()) << "expected string attribute";

  if (standardInclude && parser.parseOptionalGreater())
    return parser.emitError(parser.getNameLoc())
           << "expected trailing '>' for standard include";

  if (standardInclude)
    result.addAttribute("is_standard_include",
                        UnitAttr::get(parser.getContext()));

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/EmitC/IR/EmitC.cpp.inc"

//===----------------------------------------------------------------------===//
// EmitC Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/EmitC/IR/EmitCAttributes.cpp.inc"

Attribute emitc::OpaqueAttr::parse(DialectAsmParser &parser, Type type) {
  if (parser.parseLess())
    return Attribute();
  std::string value;
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseOptionalString(&value)) {
    parser.emitError(loc) << "expected string";
    return Attribute();
  }
  if (parser.parseGreater())
    return Attribute();
  return get(parser.getContext(), value);
}

Attribute EmitCDialect::parseAttribute(DialectAsmParser &parser,
                                       Type type) const {
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  StringRef mnemonic;
  if (parser.parseKeyword(&mnemonic))
    return Attribute();
  Attribute genAttr;
  OptionalParseResult parseResult =
      generatedAttributeParser(parser, mnemonic, type, genAttr);
  if (parseResult.hasValue())
    return genAttr;
  parser.emitError(typeLoc, "unknown attribute in EmitC dialect");
  return Attribute();
}

void EmitCDialect::printAttribute(Attribute attr, DialectAsmPrinter &os) const {
  if (failed(generatedAttributePrinter(attr, os)))
    llvm_unreachable("unexpected 'EmitC' attribute kind");
}

void emitc::OpaqueAttr::print(DialectAsmPrinter &printer) const {
  printer << "opaque<\"";
  llvm::printEscapedString(getValue(), printer.getStream());
  printer << "\">";
}

//===----------------------------------------------------------------------===//
// EmitC Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/EmitC/IR/EmitCTypes.cpp.inc"

Type emitc::OpaqueType::parse(DialectAsmParser &parser) {
  if (parser.parseLess())
    return Type();
  std::string value;
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseOptionalString(&value) || value.empty()) {
    parser.emitError(loc) << "expected non empty string";
    return Type();
  }
  if (parser.parseGreater())
    return Type();
  return get(parser.getContext(), value);
}

Type EmitCDialect::parseType(DialectAsmParser &parser) const {
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  StringRef mnemonic;
  if (parser.parseKeyword(&mnemonic))
    return Type();
  Type genType;
  OptionalParseResult parseResult =
      generatedTypeParser(parser, mnemonic, genType);
  if (parseResult.hasValue())
    return genType;
  parser.emitError(typeLoc, "unknown type in EmitC dialect");
  return Type();
}

void EmitCDialect::printType(Type type, DialectAsmPrinter &os) const {
  if (failed(generatedTypePrinter(type, os)))
    llvm_unreachable("unexpected 'EmitC' type kind");
}

void emitc::OpaqueType::print(DialectAsmPrinter &printer) const {
  printer << "opaque<\"";
  llvm::printEscapedString(getValue(), printer.getStream());
  printer << "\">";
}
