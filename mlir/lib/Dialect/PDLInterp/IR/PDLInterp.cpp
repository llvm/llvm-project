//===- PDLInterp.cpp - PDL Interpreter Dialect ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::pdl_interp;

#include "mlir/Dialect/PDLInterp/IR/PDLInterpOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// PDLInterp Dialect
//===----------------------------------------------------------------------===//

void PDLInterpDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/PDLInterp/IR/PDLInterpOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// pdl_interp::CreateOperationOp
//===----------------------------------------------------------------------===//

static ParseResult parseCreateOperationOpAttributes(
    OpAsmParser &p, SmallVectorImpl<OpAsmParser::OperandType> &attrOperands,
    ArrayAttr &attrNamesAttr) {
  Builder &builder = p.getBuilder();
  SmallVector<Attribute, 4> attrNames;
  if (succeeded(p.parseOptionalLBrace())) {
    do {
      StringAttr nameAttr;
      OpAsmParser::OperandType operand;
      if (p.parseAttribute(nameAttr) || p.parseEqual() ||
          p.parseOperand(operand))
        return failure();
      attrNames.push_back(nameAttr);
      attrOperands.push_back(operand);
    } while (succeeded(p.parseOptionalComma()));
    if (p.parseRBrace())
      return failure();
  }
  attrNamesAttr = builder.getArrayAttr(attrNames);
  return success();
}

static void printCreateOperationOpAttributes(OpAsmPrinter &p,
                                             CreateOperationOp op,
                                             OperandRange attrArgs,
                                             ArrayAttr attrNames) {
  if (attrNames.empty())
    return;
  p << " {";
  interleaveComma(llvm::seq<int>(0, attrNames.size()), p,
                  [&](int i) { p << attrNames[i] << " = " << attrArgs[i]; });
  p << '}';
}

//===----------------------------------------------------------------------===//
// pdl_interp::GetValueTypeOp
//===----------------------------------------------------------------------===//

/// Given the result type of a `GetValueTypeOp`, return the expected input type.
static Type getGetValueTypeOpValueType(Type type) {
  Type valueTy = pdl::ValueType::get(type.getContext());
  return type.isa<pdl::RangeType>() ? pdl::RangeType::get(valueTy) : valueTy;
}

//===----------------------------------------------------------------------===//
// TableGen Auto-Generated Op and Interface Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/PDLInterp/IR/PDLInterpOps.cpp.inc"
