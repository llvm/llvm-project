//===- AVX512Ops.cpp - MLIR AVX512 ops implementation ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AVX512 dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AVX512/AVX512Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

void avx512::AVX512Dialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/AVX512/AVX512.cpp.inc"
      >();
}

static LogicalResult verify(avx512::MaskCompressOp op) {
  if (op.src() && op.constant_src())
    return emitError(op.getLoc(), "cannot use both src and constant_src");

  if (op.src() && (op.src().getType() != op.dst().getType()))
    return emitError(op.getLoc(),
                     "failed to verify that src and dst have same type");

  if (op.constant_src() && (op.constant_src()->getType() != op.dst().getType()))
    return emitError(
        op.getLoc(),
        "failed to verify that constant_src and dst have same type");

  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/AVX512/AVX512.cpp.inc"
