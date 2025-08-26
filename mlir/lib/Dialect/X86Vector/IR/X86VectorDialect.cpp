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
#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

#include "mlir/Dialect/X86Vector/X86VectorInterfaces.cpp.inc"

#include "mlir/Dialect/X86Vector/X86VectorDialect.cpp.inc"

void x86vector::X86VectorDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/X86Vector/X86Vector.cpp.inc"
      >();
}

static Value getMemrefBuffPtr(Location loc, MemRefType type, Value buffer,
                              const LLVMTypeConverter &typeConverter,
                              RewriterBase &rewriter) {
  MemRefDescriptor memRefDescriptor(buffer);
  return memRefDescriptor.bufferPtr(rewriter, loc, typeConverter, type);
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
    ArrayRef<Value> operands, const LLVMTypeConverter &typeConverter,
    RewriterBase &rewriter) {
  auto loc = getLoc();
  Adaptor adaptor(operands, *this);

  auto opType = adaptor.getA().getType();
  Value src;
  if (adaptor.getSrc()) {
    src = adaptor.getSrc();
  } else if (adaptor.getConstantSrc()) {
    src = LLVM::ConstantOp::create(rewriter, loc, opType,
                                   adaptor.getConstantSrcAttr());
  } else {
    auto zeroAttr = rewriter.getZeroAttr(opType);
    src = LLVM::ConstantOp::create(rewriter, loc, opType, zeroAttr);
  }

  return SmallVector<Value>{adaptor.getA(), src, adaptor.getK()};
}

SmallVector<Value>
x86vector::DotOp::getIntrinsicOperands(ArrayRef<Value> operands,
                                       const LLVMTypeConverter &typeConverter,
                                       RewriterBase &rewriter) {
  SmallVector<Value> intrinsicOperands(operands);
  // Dot product of all elements, broadcasted to all elements.
  Value scale =
      LLVM::ConstantOp::create(rewriter, getLoc(), rewriter.getI8Type(), 0xff);
  intrinsicOperands.push_back(scale);

  return intrinsicOperands;
}

SmallVector<Value> x86vector::DotInt8Op::getIntrinsicOperands(
    ArrayRef<Value> operands, const LLVMTypeConverter &typeConverter,
    RewriterBase &rewriter) {
  SmallVector<Value> intrinsicOprnds;
  Adaptor adaptor(operands, *this);
  intrinsicOprnds.push_back(adaptor.getW());
  // Bitcast `a` and `b` to i32
  Value bitcast_a = LLVM::BitcastOp::create(
      rewriter, getLoc(),
      VectorType::get((getA().getType().getShape()[0] / 4),
                      rewriter.getIntegerType(32)),
      adaptor.getA());
  intrinsicOprnds.push_back(bitcast_a);
  Value bitcast_b = LLVM::BitcastOp::create(
      rewriter, getLoc(),
      VectorType::get((getB().getType().getShape()[0] / 4),
                      rewriter.getIntegerType(32)),
      adaptor.getB());
  intrinsicOprnds.push_back(bitcast_b);

  return intrinsicOprnds;
}

SmallVector<Value> x86vector::BcstToPackedF32Op::getIntrinsicOperands(
    ArrayRef<Value> operands, const LLVMTypeConverter &typeConverter,
    RewriterBase &rewriter) {
  Adaptor adaptor(operands, *this);
  return {getMemrefBuffPtr(getLoc(), getA().getType(), adaptor.getA(),
                           typeConverter, rewriter)};
}

SmallVector<Value> x86vector::CvtPackedEvenIndexedToF32Op::getIntrinsicOperands(
    ArrayRef<Value> operands, const LLVMTypeConverter &typeConverter,
    RewriterBase &rewriter) {
  Adaptor adaptor(operands, *this);
  return {getMemrefBuffPtr(getLoc(), getA().getType(), adaptor.getA(),
                           typeConverter, rewriter)};
}

SmallVector<Value> x86vector::CvtPackedOddIndexedToF32Op::getIntrinsicOperands(
    ArrayRef<Value> operands, const LLVMTypeConverter &typeConverter,
    RewriterBase &rewriter) {
  Adaptor adaptor(operands, *this);
  return {getMemrefBuffPtr(getLoc(), getA().getType(), adaptor.getA(),
                           typeConverter, rewriter)};
}

#define GET_OP_CLASSES
#include "mlir/Dialect/X86Vector/X86Vector.cpp.inc"
