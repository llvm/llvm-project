//===- EmulateNarrowType.cpp - Narrow type emulation ----*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/NarrowTypeEmulationConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>

using namespace mlir;

//===----------------------------------------------------------------------===//
// Public Interface Definition
//===----------------------------------------------------------------------===//

arith::NarrowTypeEmulationConverter::NarrowTypeEmulationConverter(
    unsigned targetBitwidth)
    : loadStoreBitwidth(targetBitwidth) {
  assert(llvm::isPowerOf2_32(targetBitwidth) &&
         "Only power-of-two integers are supported");

  // Allow unknown types.
  addConversion([](Type ty) -> std::optional<Type> { return ty; });

  // Function case.
  addConversion([this](FunctionType ty) -> std::optional<Type> {
    SmallVector<Type> inputs;
    if (failed(convertTypes(ty.getInputs(), inputs)))
      return std::nullopt;

    SmallVector<Type> results;
    if (failed(convertTypes(ty.getResults(), results)))
      return std::nullopt;

    return FunctionType::get(ty.getContext(), inputs, results);
  });
}

void arith::populateArithNarrowTypeEmulationPatterns(
    NarrowTypeEmulationConverter &typeConverter, RewritePatternSet &patterns) {
  // Populate `func.*` conversion patterns.
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
}
