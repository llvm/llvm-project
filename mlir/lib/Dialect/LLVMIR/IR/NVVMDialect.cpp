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

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace NVVM;

#include "mlir/Dialect/LLVMIR/NVVMOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Printing/parsing for NVVM ops
//===----------------------------------------------------------------------===//

static void printNVVMIntrinsicOp(OpAsmPrinter &p, Operation *op) {
  p << " " << op->getOperands();
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

  for (auto &attr : result.attributes) {
    if (attr.first != "return_value_and_is_valid")
      continue;
    auto structType = resultType.dyn_cast<LLVM::LLVMStructType>();
    if (structType && !structType.getBody().empty())
      resultType = structType.getBody()[0];
    break;
  }

  auto int32Ty = IntegerType::get(parser.getContext(), 32);
  return parser.resolveOperands(ops, {int32Ty, resultType, int32Ty, int32Ty},
                                parser.getNameLoc(), result.operands);
}

// <operation> ::= `llvm.nvvm.vote.ballot.sync %mask, %pred` : result_type
static ParseResult parseNVVMVoteBallotOp(OpAsmParser &parser,
                                         OperationState &result) {
  MLIRContext *context = parser.getContext();
  auto int32Ty = IntegerType::get(context, 32);
  auto int1Ty = IntegerType::get(context, 1);

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
  auto f16Ty = Float16Type::get(context);
  auto f16x2Ty = LLVM::getFixedVectorType(f16Ty, 2);
  auto f32Ty = Float32Type::get(context);
  auto f16x2x4StructTy = LLVM::LLVMStructType::getLiteral(
      context, {f16x2Ty, f16x2Ty, f16x2Ty, f16x2Ty});
  auto f32x8StructTy = LLVM::LLVMStructType::getLiteral(
      context, {f32Ty, f32Ty, f32Ty, f32Ty, f32Ty, f32Ty, f32Ty, f32Ty});

  SmallVector<Type, 12> operandTypes(op.getOperandTypes().begin(),
                                     op.getOperandTypes().end());
  if (operandTypes != SmallVector<Type, 8>(8, f16x2Ty) &&
      operandTypes != SmallVector<Type, 12>{f16x2Ty, f16x2Ty, f16x2Ty, f16x2Ty,
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

  auto alayout = op->getAttrOfType<StringAttr>("alayout");
  auto blayout = op->getAttrOfType<StringAttr>("blayout");

  if (!(alayout && blayout) ||
      !(alayout.getValue() == "row" || alayout.getValue() == "col") ||
      !(blayout.getValue() == "row" || blayout.getValue() == "col")) {
    return op.emitOpError(
        "alayout and blayout attributes must be set to either "
        "\"row\" or \"col\"");
  }

  if (operandTypes == SmallVector<Type, 12>{f16x2Ty, f16x2Ty, f16x2Ty, f16x2Ty,
                                            f32Ty, f32Ty, f32Ty, f32Ty, f32Ty,
                                            f32Ty, f32Ty, f32Ty} &&
      op.getType() == f32x8StructTy && alayout.getValue() == "row" &&
      blayout.getValue() == "col") {
    return success();
  }
  return op.emitOpError("unimplemented mma.sync variant");
}

template <typename T>
static LogicalResult verifyWMMALoadOp(T op, StringRef operand) {
  MLIRContext *context = op.getContext();
  auto i32Ty = IntegerType::get(context, 32);
  auto i32Ptr1Ty = LLVM::LLVMPointerType::get(i32Ty, 1);
  auto i32Ptr3Ty = LLVM::LLVMPointerType::get(i32Ty, 3);
  auto i32Ptr0Ty = LLVM::LLVMPointerType::get(i32Ty, 0);
  auto f16Ty = FloatType::getF16(context);
  auto f32Ty = FloatType::getF32(context);
  auto f16x2Ty = VectorType::get(2, f16Ty);
  auto f16x2x4StructTy = LLVM::LLVMStructType::getLiteral(
      context, {f16x2Ty, f16x2Ty, f16x2Ty, f16x2Ty});
  auto f16x2x8StructTy = LLVM::LLVMStructType::getLiteral(
      context,
      {f16x2Ty, f16x2Ty, f16x2Ty, f16x2Ty, f16x2Ty, f16x2Ty, f16x2Ty, f16x2Ty});
  auto f32x8StructTy = LLVM::LLVMStructType::getLiteral(
      context, {f32Ty, f32Ty, f32Ty, f32Ty, f32Ty, f32Ty, f32Ty, f32Ty});

  SmallVector<Type, 2> operandTypes(op.getOperandTypes().begin(),
                                    op.getOperandTypes().end());
  if (operandTypes != SmallVector<Type, 2>{i32Ptr1Ty, i32Ty} &&
      operandTypes != SmallVector<Type, 2>{i32Ptr3Ty, i32Ty} &&
      operandTypes != SmallVector<Type, 2>{i32Ptr0Ty, i32Ty}) {
    return op.emitOpError("expected operands to be a source pointer in memory "
                          "space 0, 1, 3 followed by ldm of the source");
  }

  if (operand.equals("AOp") || operand.equals("BOp")) {
    if (op.getType() != f16x2x8StructTy) {
      return op.emitOpError("expected result type of loadAOp and loadBOp to be "
                            "a struct of 8 <halfx2>s");
    }
  } else if (operand.equals("COp")) {
    if (op.getType() != f16x2x4StructTy && op.getType() != f32x8StructTy) {
      return op.emitOpError("expected result type of loadCOp to be a struct of "
                            "4 <halfx2>s or 8 f32s");
    }
  }

  return success();
}

static LogicalResult verify(WMMALoadAM16N16K16Op op) {
  return verifyWMMALoadOp(op, "AOp");
}

static LogicalResult verify(WMMALoadBM16N16K16Op op) {
  return verifyWMMALoadOp(op, "BOp");
}

static LogicalResult verify(WMMALoadCF16M16N16K16Op op) {
  return verifyWMMALoadOp(op, "COp");
}

static LogicalResult verify(WMMALoadCF32M16N16K16Op op) {
  return verifyWMMALoadOp(op, "COp");
}

template <typename T>
static bool verifyWMMAStoreOp(T op, SmallVector<Type> &containedElems) {
  SmallVector<Type> operandTypes(op.getOperandTypes().begin(),
                                 op.getOperandTypes().end());
  if (operandTypes == containedElems)
    return true;

  return false;
}

static LogicalResult verify(WMMAStoreF16M16N16K16Op op) {
  MLIRContext *context = op.getContext();
  auto i32Ty = IntegerType::get(context, 32);
  auto i32Ptr1Ty = LLVM::LLVMPointerType::get(i32Ty, 1);
  auto i32Ptr3Ty = LLVM::LLVMPointerType::get(i32Ty, 3);
  auto i32Ptr0Ty = LLVM::LLVMPointerType::get(i32Ty, 0);
  auto f16Ty = FloatType::getF16(context);
  auto f16x2Ty = VectorType::get(2, f16Ty);
  SmallVector<Type> type1{i32Ptr1Ty, f16x2Ty, f16x2Ty, f16x2Ty, f16x2Ty, i32Ty};
  SmallVector<Type> type0{i32Ptr0Ty, f16x2Ty, f16x2Ty, f16x2Ty, f16x2Ty, i32Ty};
  SmallVector<Type> type3{i32Ptr3Ty, f16x2Ty, f16x2Ty, f16x2Ty, f16x2Ty, i32Ty};
  if (verifyWMMAStoreOp(op, type1) || verifyWMMAStoreOp(op, type0) ||
      verifyWMMAStoreOp(op, type3))
    return success();

  return op.emitOpError("expected operands to be a source pointer in memory"
                        "space 0, 1, 3 followed by ldm of the source");
}

static LogicalResult verify(WMMAStoreF32M16N16K16Op op) {
  MLIRContext *context = op.getContext();
  auto i32Ty = IntegerType::get(context, 32);
  auto i32Ptr1Ty = LLVM::LLVMPointerType::get(i32Ty, 1);
  auto i32Ptr3Ty = LLVM::LLVMPointerType::get(i32Ty, 3);
  auto i32Ptr0Ty = LLVM::LLVMPointerType::get(i32Ty, 0);
  auto f32Ty = FloatType::getF32(context);

  SmallVector<Type> type1{i32Ptr1Ty, f32Ty, f32Ty, f32Ty, f32Ty,
                          f32Ty,     f32Ty, f32Ty, f32Ty, i32Ty};
  SmallVector<Type> type0{i32Ptr0Ty, f32Ty, f32Ty, f32Ty, f32Ty,
                          f32Ty,     f32Ty, f32Ty, f32Ty, i32Ty};
  SmallVector<Type> type3{i32Ptr3Ty, f32Ty, f32Ty, f32Ty, f32Ty,
                          f32Ty,     f32Ty, f32Ty, f32Ty, i32Ty};
  if (verifyWMMAStoreOp(op, type0) || verifyWMMAStoreOp(op, type1) ||
      verifyWMMAStoreOp(op, type3))
    return success();

  return op.emitOpError("expected operands to be a source pointer in memory"
                        "space 0, 1, 3 followed by ldm of the source");
}

static LogicalResult verify(WMMAMmaF16F16M16N16K16Op op) {
  MLIRContext *context = op.getContext();
  auto f16Ty = FloatType::getF16(context);
  auto f16x2Ty = VectorType::get(2, f16Ty);
  auto f16x2x4StructTy = LLVM::LLVMStructType::getLiteral(
      context, {f16x2Ty, f16x2Ty, f16x2Ty, f16x2Ty});

  SmallVector<Type, 2> operandTypes(op.getOperandTypes().begin(),
                                    op.getOperandTypes().end());
  if (operandTypes != SmallVector<Type, 20>(20, f16x2Ty))
    return op.emitOpError("expected 20 <halfx2>s as operands");

  if (op.getResult().getType() != f16x2x4StructTy)
    return op.emitOpError("expected result type to be a struct of 4 <halfx2>s");

  return success();
}

static LogicalResult parseWMMAMmaF16F16M16N16K16Op(OpAsmParser &parser,
                                                   OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 4> operands;
  ::llvm::SMLoc operandsLoc;
  Type operandType;
  Type resType;

  operandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(operands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(operandType) || parser.parseArrow())
    return failure();

  unsigned numOperands = operands.size();
  SmallVector<Type> operandTypes(numOperands, operandType);
  if (parser.parseType(resType))
    return failure();
  result.addTypes(resType);
  if (parser.resolveOperands(operands, operandTypes, operandsLoc,
                             result.operands))
    return failure();
  return success();
}

static void printWMMAMmaF16F16M16N16K16Op(OpAsmPrinter &p,
                                          WMMAMmaF16F16M16N16K16Op &op) {
  p << ' ';
  p << op.args();
  p.printOptionalAttrDict(op->getAttrs(), {});
  p << " : ";
  p << op->getOperand(0).getType();
  p << ' ' << "->";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(op.res().getType());
}

static LogicalResult verify(WMMAMmaF32F32M16N16K16Op op) {
  unsigned numABOperands = 16;
  unsigned numCOperands = 8;
  MLIRContext *context = op.getContext();
  auto f16Ty = FloatType::getF16(context);
  auto f32Ty = FloatType::getF32(context);
  auto f16x2Ty = VectorType::get(2, f16Ty);
  auto f32x8StructTy = LLVM::LLVMStructType::getLiteral(
      context, {f32Ty, f32Ty, f32Ty, f32Ty, f32Ty, f32Ty, f32Ty, f32Ty});

  SmallVector<Type> abOpTypes;
  SmallVector<Type> bOpTypes;
  SmallVector<Type> cOpTypes;

  for (auto operand : op->getOperands().take_front(numABOperands)) {
    abOpTypes.push_back(operand.getType());
  }

  for (auto operand :
       op->getOperands().drop_front(numABOperands).take_front(numCOperands)) {
    cOpTypes.push_back(operand.getType());
  }

  if (abOpTypes != SmallVector<Type>(16, f16x2Ty))
    return op.emitOpError("expected 16 <halfx2>s for `a` and `b` operand");

  if (cOpTypes != SmallVector<Type>(8, f32Ty))
    return op.emitOpError("expected 8 f32s for `c` operand");

  if (op.getResult().getType() != f32x8StructTy)
    return op.emitOpError("expected result type to be a struct of 8 f32s");

  return success();
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

  // Support unknown operations because not all NVVM operations are
  // registered.
  allowUnknownOperations();
}

LogicalResult NVVMDialect::verifyOperationAttribute(Operation *op,
                                                    NamedAttribute attr) {
  // Kernel function attribute should be attached to functions.
  if (attr.first == NVVMDialect::getKernelFuncAttrName()) {
    if (!isa<LLVM::LLVMFuncOp>(op)) {
      return op->emitError() << "'" << NVVMDialect::getKernelFuncAttrName()
                             << "' attribute attached to unexpected op";
    }
  }
  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMOps.cpp.inc"
