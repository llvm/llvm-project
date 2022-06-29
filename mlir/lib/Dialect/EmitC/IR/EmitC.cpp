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

LogicalResult ApplyOp::verify() {
  StringRef applicableOperatorStr = getApplicableOperator();

  // Applicable operator must not be empty.
  if (applicableOperatorStr.empty())
    return emitOpError("applicable operator must not be empty");

  // Only `*` and `&` are supported.
  if (applicableOperatorStr != "&" && applicableOperatorStr != "*")
    return emitOpError("applicable operator is illegal");

  return success();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  Type input = inputs.front(), output = outputs.front();

  return ((input.isa<IntegerType, FloatType, IndexType, emitc::OpaqueType,
                     emitc::PointerType>()) &&
          (output.isa<IntegerType, FloatType, IndexType, emitc::OpaqueType,
                      emitc::PointerType>()));
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult emitc::CallOp::verify() {
  // Callee must not be empty.
  if (getCallee().empty())
    return emitOpError("callee must not be empty");

  if (Optional<ArrayAttr> argsAttr = getArgs()) {
    for (Attribute arg : *argsAttr) {
      if (arg.getType().isa<IndexType>()) {
        int64_t index = arg.cast<IntegerAttr>().getInt();
        // Args with elements of type index must be in range
        // [0..operands.size).
        if ((index < 0) || (index >= static_cast<int64_t>(getNumOperands())))
          return emitOpError("index argument is out of range");

        // Args with elements of type ArrayAttr must have a type.
      } else if (arg.isa<ArrayAttr>() && arg.getType().isa<NoneType>()) {
        return emitOpError("array argument has no type");
      }
    }
  }

  if (Optional<ArrayAttr> templateArgsAttr = getTemplateArgs()) {
    for (Attribute tArg : *templateArgsAttr) {
      if (!tArg.isa<TypeAttr>() && !tArg.isa<IntegerAttr>() &&
          !tArg.isa<FloatAttr>() && !tArg.isa<emitc::OpaqueAttr>())
        return emitOpError("template argument has invalid type");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

/// The constant op requires that the attribute's type matches the return type.
LogicalResult emitc::ConstantOp::verify() {
  Attribute value = getValueAttr();
  Type type = getType();
  if (!value.getType().isa<NoneType>() && type != value.getType())
    return emitOpError() << "requires attribute's type (" << value.getType()
                         << ") to match op's return type (" << type << ")";
  return success();
}

OpFoldResult emitc::ConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return getValue();
}

//===----------------------------------------------------------------------===//
// IncludeOp
//===----------------------------------------------------------------------===//

void IncludeOp::print(OpAsmPrinter &p) {
  bool standardInclude = getIsStandardInclude();

  p << " ";
  if (standardInclude)
    p << "<";
  p << "\"" << getInclude() << "\"";
  if (standardInclude)
    p << ">";
}

ParseResult IncludeOp::parse(OpAsmParser &parser, OperationState &result) {
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
// VariableOp
//===----------------------------------------------------------------------===//

/// The variable op requires that the attribute's type matches the return type.
LogicalResult emitc::VariableOp::verify() {
  Attribute value = getValueAttr();
  Type type = getType();
  if (!value.getType().isa<NoneType>() && type != value.getType())
    return emitOpError() << "requires attribute's type (" << value.getType()
                         << ") to match op's return type (" << type << ")";
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

Attribute emitc::OpaqueAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess())
    return Attribute();
  std::string value;
  SMLoc loc = parser.getCurrentLocation();
  if (parser.parseOptionalString(&value)) {
    parser.emitError(loc) << "expected string";
    return Attribute();
  }
  if (parser.parseGreater())
    return Attribute();
  return get(parser.getContext(), value);
}

void emitc::OpaqueAttr::print(AsmPrinter &printer) const {
  printer << "<\"";
  llvm::printEscapedString(getValue(), printer.getStream());
  printer << "\">";
}

//===----------------------------------------------------------------------===//
// EmitC Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/EmitC/IR/EmitCTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// OpaqueType
//===----------------------------------------------------------------------===//

Type emitc::OpaqueType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();
  std::string value;
  SMLoc loc = parser.getCurrentLocation();
  if (parser.parseOptionalString(&value) || value.empty()) {
    parser.emitError(loc) << "expected non empty string in !emitc.opaque type";
    return Type();
  }
  if (value.back() == '*') {
    parser.emitError(loc) << "pointer not allowed as outer type with "
                             "!emitc.opaque, use !emitc.ptr instead";
    return Type();
  }
  if (parser.parseGreater())
    return Type();
  return get(parser.getContext(), value);
}

void emitc::OpaqueType::print(AsmPrinter &printer) const {
  printer << "<\"";
  llvm::printEscapedString(getValue(), printer.getStream());
  printer << "\">";
}
