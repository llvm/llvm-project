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
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace ROCDL;

//===----------------------------------------------------------------------===//
// Parsing for ROCDL ops
//===----------------------------------------------------------------------===//

static LLVM::LLVMDialect *getLlvmDialect(OpAsmParser &parser) {
  return parser.getBuilder()
      .getContext()
      ->getRegisteredDialect<LLVM::LLVMDialect>();
}

// <operation> ::=
//     `llvm.amdgcn.buffer.load.* %rsrc, %vindex, %offset, %glc, %slc :
//     result_type`
static ParseResult parseROCDLMubufLoadOp(OpAsmParser &parser,
                                         OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 8> ops;
  Type type;
  if (parser.parseOperandList(ops, 5) || parser.parseColonType(type) ||
      parser.addTypeToList(type, result.types))
    return failure();

  auto int32Ty = LLVM::LLVMType::getInt32Ty(getLlvmDialect(parser));
  auto int1Ty = LLVM::LLVMType::getInt1Ty(getLlvmDialect(parser));
  auto i32x4Ty = LLVM::LLVMType::getVectorTy(int32Ty, 4);
  return parser.resolveOperands(ops,
                                {i32x4Ty, int32Ty, int32Ty, int1Ty, int1Ty},
                                parser.getNameLoc(), result.operands);
}

// <operation> ::=
//     `llvm.amdgcn.buffer.store.* %vdata, %rsrc, %vindex, %offset, %glc, %slc :
//     result_type`
static ParseResult parseROCDLMubufStoreOp(OpAsmParser &parser,
                                          OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 8> ops;
  Type type;
  if (parser.parseOperandList(ops, 6) || parser.parseColonType(type))
    return failure();

  auto int32Ty = LLVM::LLVMType::getInt32Ty(getLlvmDialect(parser));
  auto int1Ty = LLVM::LLVMType::getInt1Ty(getLlvmDialect(parser));
  auto i32x4Ty = LLVM::LLVMType::getVectorTy(int32Ty, 4);

  if (parser.resolveOperands(ops,
                             {type, i32x4Ty, int32Ty, int32Ty, int1Ty, int1Ty},
                             parser.getNameLoc(), result.operands))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// ROCDLDialect initialization, type parsing, and registration.
//===----------------------------------------------------------------------===//

// TODO(herhut): This should be the llvm.rocdl dialect once this is supported.
ROCDLDialect::ROCDLDialect(MLIRContext *context) : Dialect("rocdl", context) {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LLVMIR/ROCDLOps.cpp.inc"
      >();

  // Support unknown operations because not all ROCDL operations are registered.
  allowUnknownOperations();
}

namespace mlir {
namespace ROCDL {
#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/ROCDLOps.cpp.inc"
} // namespace ROCDL
} // namespace mlir

