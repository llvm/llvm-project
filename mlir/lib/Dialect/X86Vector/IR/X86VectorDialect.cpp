//===- X86VectorDialect.cpp - MLIR X86Vector ops implementation -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the X86Vector dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/X86Vector/X86VectorDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

using namespace mlir;

#include "mlir/Dialect/X86Vector/X86VectorInterfaces.cpp.inc"

#include "mlir/Dialect/X86Vector/X86VectorDialect.cpp.inc"

void x86vector::X86VectorDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/X86Vector/X86Vector.cpp.inc"
      >();
}

static SmallVector<Value>
getMemrefBuffPtr(Location loc, ::mlir::TypedValue<::mlir::MemRefType> memrefVal,
                 RewriterBase &rewriter,
                 const LLVMTypeConverter &typeConverter) {
  SmallVector<Value> operands;
  auto opType = memrefVal.getType();

  Type llvmStructType = typeConverter.convertType(opType);
  Value llvmStruct =
      rewriter
          .create<UnrealizedConversionCastOp>(loc, llvmStructType, memrefVal)
          .getResult(0);
  MemRefDescriptor memRefDescriptor(llvmStruct);

  Value ptr = memRefDescriptor.bufferPtr(rewriter, loc, typeConverter, opType);
  operands.push_back(ptr);

  return operands;
}

LogicalResult x86vector::MaskCompressOp::verify() {
  if (getSrc() && getConstantSrc())
    return emitError("cannot use both src and constant_src");

  if (getSrc() && (getSrc().getType() != getDst().getType()))
    return emitError("failed to verify that src and dst have same type");

  if (getConstantSrc() && (getConstantSrc()->getType() != getDst().getType()))
    return emitError(
        "failed to verify that constant_src and dst have same type");

  return success();
}

SmallVector<Value> x86vector::MaskCompressOp::getIntrinsicOperands(
    RewriterBase &rewriter, const LLVMTypeConverter &typeConverter) {
  auto loc = getLoc();

  auto opType = getA().getType();
  Value src;
  if (getSrc()) {
    src = getSrc();
  } else if (getConstantSrc()) {
    src = rewriter.create<LLVM::ConstantOp>(loc, opType, getConstantSrcAttr());
  } else {
    auto zeroAttr = rewriter.getZeroAttr(opType);
    src = rewriter.create<LLVM::ConstantOp>(loc, opType, zeroAttr);
  }

  return SmallVector<Value>{getA(), src, getK()};
}

SmallVector<Value>
x86vector::DotOp::getIntrinsicOperands(RewriterBase &rewriter,
                                       const LLVMTypeConverter &typeConverter) {
  SmallVector<Value> operands(getOperands());
  // Dot product of all elements, broadcasted to all elements.
  Value scale =
      rewriter.create<LLVM::ConstantOp>(getLoc(), rewriter.getI8Type(), 0xff);
  operands.push_back(scale);

  return operands;
}

SmallVector<Value> x86vector::BcstToPackedF32Op::getIntrinsicOperands(
    RewriterBase &rewriter, const LLVMTypeConverter &typeConverter) {
  return getMemrefBuffPtr(getLoc(), getA(), rewriter, typeConverter);
}

SmallVector<Value> x86vector::CvtPackedEvenIndexedToF32Op::getIntrinsicOperands(
    RewriterBase &rewriter, const LLVMTypeConverter &typeConverter) {
  return getMemrefBuffPtr(getLoc(), getA(), rewriter, typeConverter);
}

SmallVector<Value> x86vector::CvtPackedOddIndexedToF32Op::getIntrinsicOperands(
    RewriterBase &rewriter, const LLVMTypeConverter &typeConverter) {
  return getMemrefBuffPtr(getLoc(), getA(), rewriter, typeConverter);
}

#define GET_OP_CLASSES
#include "mlir/Dialect/X86Vector/X86Vector.cpp.inc"
