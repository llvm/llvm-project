//===- SPIRVTosaOps.cpp - MLIR SPIR-V Tosa operations ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Tosa operations in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir::spirv {

//===----------------------------------------------------------------------===//
// TOSA Operator Verifiers.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// spirv.TosaArgmaxOp
//===----------------------------------------------------------------------===//

LogicalResult TosaArgMaxOp::verify() {
  ShapedType inputTy = getInputType();
  ShapedType resultTy = getResultType();

  if (inputTy.hasRank() && resultTy.hasRank() &&
      resultTy.getRank() !=
          (inputTy.getRank() > 1 ? inputTy.getRank() - 1 : 1)) {
    return emitOpError(
               "result rank must be max of 1 and (input rank - 1), got ")
           << resultTy.getRank();
  }

  const uint32_t axis = getAxis();
  if (inputTy.hasRank() && axis >= inputTy.getRank()) {
    return emitOpError(
               "specified axis is greater than the rank of input, got axis = ")
           << axis << " and input rank = " << inputTy.getRank();
  }

  return success();
}

} // namespace mlir::spirv
