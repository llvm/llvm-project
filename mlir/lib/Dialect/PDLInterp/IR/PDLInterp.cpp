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

static ParseResult parseCreateOperationOp(OpAsmParser &p,
                                          OperationState &state) {
  if (p.parseOptionalAttrDict(state.attributes))
    return failure();
  Builder &builder = p.getBuilder();

  // Parse the operation name.
  StringAttr opName;
  if (p.parseAttribute(opName, "name", state.attributes))
    return failure();

  // Parse the operands.
  SmallVector<OpAsmParser::OperandType, 4> operands;
  if (p.parseLParen() || p.parseOperandList(operands) || p.parseRParen() ||
      p.resolveOperands(operands, builder.getType<pdl::ValueType>(),
                        state.operands))
    return failure();

  // Parse the attributes.
  SmallVector<Attribute, 4> attrNames;
  if (succeeded(p.parseOptionalLBrace())) {
    SmallVector<OpAsmParser::OperandType, 4> attrOps;
    do {
      StringAttr nameAttr;
      OpAsmParser::OperandType operand;
      if (p.parseAttribute(nameAttr) || p.parseEqual() ||
          p.parseOperand(operand))
        return failure();
      attrNames.push_back(nameAttr);
      attrOps.push_back(operand);
    } while (succeeded(p.parseOptionalComma()));

    if (p.parseRBrace() ||
        p.resolveOperands(attrOps, builder.getType<pdl::AttributeType>(),
                          state.operands))
      return failure();
  }
  state.addAttribute("attributeNames", builder.getArrayAttr(attrNames));
  state.addTypes(builder.getType<pdl::OperationType>());

  // Parse the result types.
  SmallVector<OpAsmParser::OperandType, 4> opResultTypes;
  if (p.parseArrow())
    return failure();
  if (succeeded(p.parseOptionalLParen())) {
    if (p.parseRParen())
      return failure();
  } else if (p.parseOperandList(opResultTypes) ||
             p.resolveOperands(opResultTypes, builder.getType<pdl::TypeType>(),
                               state.operands)) {
    return failure();
  }

  int32_t operandSegmentSizes[] = {static_cast<int32_t>(operands.size()),
                                   static_cast<int32_t>(attrNames.size()),
                                   static_cast<int32_t>(opResultTypes.size())};
  state.addAttribute("operand_segment_sizes",
                     builder.getI32VectorAttr(operandSegmentSizes));
  return success();
}

static void print(OpAsmPrinter &p, CreateOperationOp op) {
  p << "pdl_interp.create_operation ";
  p.printOptionalAttrDict(op.getAttrs(),
                          {"attributeNames", "name", "operand_segment_sizes"});
  p << '"' << op.name() << "\"(" << op.operands() << ')';

  // Emit the optional attributes.
  ArrayAttr attrNames = op.attributeNames();
  if (!attrNames.empty()) {
    Operation::operand_range attrArgs = op.attributes();
    p << " {";
    interleaveComma(llvm::seq<int>(0, attrNames.size()), p,
                    [&](int i) { p << attrNames[i] << " = " << attrArgs[i]; });
    p << '}';
  }

  // Print the result type constraints of the operation.
  auto types = op.types();
  if (types.empty())
    p << " -> ()";
  else
    p << " -> " << op.types();
}

//===----------------------------------------------------------------------===//
// TableGen Auto-Generated Op and Interface Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/PDLInterp/IR/PDLInterpOps.cpp.inc"
