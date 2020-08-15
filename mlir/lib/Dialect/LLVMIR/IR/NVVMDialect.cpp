//===- NVVMDialect.cpp - NVVM IR Ops and Dialect registration -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the types and operation details for the NVVM IR dialect in
// MLIR, and the LLVM IR dialect.  It also registers the dialect.
//
// The NVVM dialect only contains GPU specific additions on top of the general
// LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace NVVM;

//===----------------------------------------------------------------------===//
// Printing/parsing for NVVM ops
//===----------------------------------------------------------------------===//

static void printNVVMIntrinsicOp(OpAsmPrinter &p, Operation *op) {
  p << op->getName() << " " << op->getOperands();
  if (op->getNumResults() > 0)
    p << " : " << op->getResultTypes();
}

// <operation> ::=
//     `llvm.nvvm.shfl.sync.bfly %dst, %val, %offset, %clamp_and_mask`
//      ({return_value_and_is_valid})? : result_type
static ParseResult parseNVVMShflSyncBflyOp(OpAsmParser &parser,
                                           OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 8> ops;
  Type resultType;
  if (parser.parseOperandList(ops) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(resultType) ||
      parser.addTypeToList(resultType, result.types))
    return failure();

  auto type = resultType.cast<LLVM::LLVMType>();
  for (auto &attr : result.attributes) {
    if (attr.first != "return_value_and_is_valid")
      continue;
    if (type.isStructTy() && type.getStructNumElements() > 0)
      type = type.getStructElementType(0);
    break;
  }

  auto int32Ty = LLVM::LLVMType::getInt32Ty(parser.getBuilder().getContext());
  return parser.resolveOperands(ops, {int32Ty, type, int32Ty, int32Ty},
                                parser.getNameLoc(), result.operands);
}

// <operation> ::= `llvm.nvvm.vote.ballot.sync %mask, %pred` : result_type
static ParseResult parseNVVMVoteBallotOp(OpAsmParser &parser,
                                         OperationState &result) {
  MLIRContext *context = parser.getBuilder().getContext();
  auto int32Ty = LLVM::LLVMType::getInt32Ty(context);
  auto int1Ty = LLVM::LLVMType::getInt1Ty(context);

  SmallVector<OpAsmParser::OperandType, 8> ops;
  Type type;
  return failure(parser.parseOperandList(ops) ||
                 parser.parseOptionalAttrDict(result.attributes) ||
                 parser.parseColonType(type) ||
                 parser.addTypeToList(type, result.types) ||
                 parser.resolveOperands(ops, {int32Ty, int1Ty},
                                        parser.getNameLoc(), result.operands));
}

static LogicalResult verify(MmaOp op) {
  MLIRContext *context = op.getContext();
  auto f16Ty = LLVM::LLVMType::getHalfTy(context);
  auto f16x2Ty = LLVM::LLVMType::getVectorTy(f16Ty, 2);
  auto f32Ty = LLVM::LLVMType::getFloatTy(context);
  auto f16x2x4StructTy = LLVM::LLVMType::getStructTy(
      context, {f16x2Ty, f16x2Ty, f16x2Ty, f16x2Ty});
  auto f32x8StructTy = LLVM::LLVMType::getStructTy(
      context, {f32Ty, f32Ty, f32Ty, f32Ty, f32Ty, f32Ty, f32Ty, f32Ty});

  SmallVector<Type, 12> operand_types(op.getOperandTypes().begin(),
                                      op.getOperandTypes().end());
  if (operand_types != SmallVector<Type, 8>(8, f16x2Ty) &&
      operand_types != SmallVector<Type, 12>{f16x2Ty, f16x2Ty, f16x2Ty, f16x2Ty,
                                             f32Ty, f32Ty, f32Ty, f32Ty, f32Ty,
                                             f32Ty, f32Ty, f32Ty}) {
    return op.emitOpError(
        "expected operands to be 4 <halfx2>s followed by either "
        "4 <halfx2>s or 8 floats");
  }
  if (op.getType() != f32x8StructTy && op.getType() != f16x2x4StructTy) {
    return op.emitOpError("expected result type to be a struct of either 4 "
                          "<halfx2>s or 8 floats");
  }

  auto alayout = op.getAttrOfType<StringAttr>("alayout");
  auto blayout = op.getAttrOfType<StringAttr>("blayout");

  if (!(alayout && blayout) ||
      !(alayout.getValue() == "row" || alayout.getValue() == "col") ||
      !(blayout.getValue() == "row" || blayout.getValue() == "col")) {
    return op.emitOpError(
        "alayout and blayout attributes must be set to either "
        "\"row\" or \"col\"");
  }

  if (operand_types == SmallVector<Type, 12>{f16x2Ty, f16x2Ty, f16x2Ty, f16x2Ty,
                                             f32Ty, f32Ty, f32Ty, f32Ty, f32Ty,
                                             f32Ty, f32Ty, f32Ty} &&
      op.getType() == f32x8StructTy && alayout.getValue() == "row" &&
      blayout.getValue() == "col") {
    return success();
  }
  return op.emitOpError("unimplemented mma.sync variant");
}

//===----------------------------------------------------------------------===//
// NVVMDialect initialization, type parsing, and registration.
//===----------------------------------------------------------------------===//

// TODO: This should be the llvm.nvvm dialect once this is supported.
void NVVMDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LLVMIR/NVVMOps.cpp.inc"
      >();

  // Support unknown operations because not all NVVM operations are registered.
  allowUnknownOperations();
}

namespace mlir {
namespace NVVM {
#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMOps.cpp.inc"
} // namespace NVVM
} // namespace mlir

