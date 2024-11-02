//===- ROCDLDialect.cpp - ROCDL IR Ops and Dialect registration -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the types and operation details for the ROCDL IR dialect in
// MLIR, and the LLVM IR dialect.  It also registers the dialect.
//
// The ROCDL dialect only contains GPU specific additions on top of the general
// LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace ROCDL;

#include "mlir/Dialect/LLVMIR/ROCDLOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Parsing for ROCDL ops
//===----------------------------------------------------------------------===//

// <operation> ::=
//     `llvm.amdgcn.buffer.load.* %rsrc, %vindex, %offset, %glc, %slc :
//     result_type`
ParseResult MubufLoadOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 8> ops;
  Type type;
  if (parser.parseOperandList(ops, 5) || parser.parseColonType(type) ||
      parser.addTypeToList(type, result.types))
    return failure();

  MLIRContext *context = parser.getContext();
  auto int32Ty = IntegerType::get(context, 32);
  auto int1Ty = IntegerType::get(context, 1);
  auto i32x4Ty = LLVM::getFixedVectorType(int32Ty, 4);
  return parser.resolveOperands(ops,
                                {i32x4Ty, int32Ty, int32Ty, int1Ty, int1Ty},
                                parser.getNameLoc(), result.operands);
}

void MubufLoadOp::print(OpAsmPrinter &p) {
  p << " " << getOperands() << " : " << (*this)->getResultTypes();
}

// <operation> ::=
//     `llvm.amdgcn.buffer.store.* %vdata, %rsrc, %vindex, %offset, %glc, %slc :
//     result_type`
ParseResult MubufStoreOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 8> ops;
  Type type;
  if (parser.parseOperandList(ops, 6) || parser.parseColonType(type))
    return failure();

  MLIRContext *context = parser.getContext();
  auto int32Ty = IntegerType::get(context, 32);
  auto int1Ty = IntegerType::get(context, 1);
  auto i32x4Ty = LLVM::getFixedVectorType(int32Ty, 4);

  if (parser.resolveOperands(ops,
                             {type, i32x4Ty, int32Ty, int32Ty, int1Ty, int1Ty},
                             parser.getNameLoc(), result.operands))
    return failure();
  return success();
}

void MubufStoreOp::print(OpAsmPrinter &p) {
  p << " " << getOperands() << " : " << vdata().getType();
}

// <operation> ::=
//     `llvm.amdgcn.raw.buffer.load.* %rsrc, %offset, %soffset, %aux
//     : result_type`
ParseResult RawBufferLoadOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> ops;
  Type type;
  if (parser.parseOperandList(ops, 4) || parser.parseColonType(type) ||
      parser.addTypeToList(type, result.types))
    return failure();

  auto bldr = parser.getBuilder();
  auto int32Ty = bldr.getI32Type();
  auto i32x4Ty = VectorType::get({4}, int32Ty);
  return parser.resolveOperands(ops, {i32x4Ty, int32Ty, int32Ty, int32Ty},
                                parser.getNameLoc(), result.operands);
}

void RawBufferLoadOp::print(OpAsmPrinter &p) {
  p << " " << getOperands() << " : " << res().getType();
}

// <operation> ::=
//     `llvm.amdgcn.raw.buffer.store.* %vdata, %rsrc,  %offset,
//     %soffset, %aux : result_type`
ParseResult RawBufferStoreOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 5> ops;
  Type type;
  if (parser.parseOperandList(ops, 5) || parser.parseColonType(type))
    return failure();

  auto bldr = parser.getBuilder();
  auto int32Ty = bldr.getI32Type();
  auto i32x4Ty = VectorType::get({4}, int32Ty);

  if (parser.resolveOperands(ops, {type, i32x4Ty, int32Ty, int32Ty, int32Ty},
                             parser.getNameLoc(), result.operands))
    return failure();
  return success();
}

void RawBufferStoreOp::print(OpAsmPrinter &p) {
  p << " " << getOperands() << " : " << vdata().getType();
}

// <operation> ::=
//     `llvm.amdgcn.raw.buffer.atomic.fadd.* %vdata, %rsrc,  %offset,
//     %soffset, %aux : result_type`
ParseResult RawBufferAtomicFAddOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 5> ops;
  Type type;
  if (parser.parseOperandList(ops, 5) || parser.parseColonType(type))
    return failure();

  auto bldr = parser.getBuilder();
  auto int32Ty = bldr.getI32Type();
  auto i32x4Ty = VectorType::get({4}, int32Ty);

  if (parser.resolveOperands(ops, {type, i32x4Ty, int32Ty, int32Ty, int32Ty},
                             parser.getNameLoc(), result.operands))
    return failure();
  return success();
}

void RawBufferAtomicFAddOp::print(mlir::OpAsmPrinter &p) {
  p << " " << getOperands() << " : " << vdata().getType();
}

//===----------------------------------------------------------------------===//
// ROCDLDialect initialization, type parsing, and registration.
//===----------------------------------------------------------------------===//

// TODO: This should be the llvm.rocdl dialect once this is supported.
void ROCDLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LLVMIR/ROCDLOps.cpp.inc"
      >();

  // Support unknown operations because not all ROCDL operations are registered.
  allowUnknownOperations();
}

LogicalResult ROCDLDialect::verifyOperationAttribute(Operation *op,
                                                     NamedAttribute attr) {
  // Kernel function attribute should be attached to functions.
  if (attr.getName() == ROCDLDialect::getKernelFuncAttrName()) {
    if (!isa<LLVM::LLVMFuncOp>(op)) {
      return op->emitError() << "'" << ROCDLDialect::getKernelFuncAttrName()
                             << "' attribute attached to unexpected op";
    }
  }
  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/ROCDLOps.cpp.inc"
