//===- DivisionConverter.h - Complex division conversion ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_COMPLEXCOMMON_DIVISIONCONVERTER_H
#define MLIR_CONVERSION_COMPLEXCOMMON_DIVISIONCONVERTER_H

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir {
namespace complex {
/// convert a complex division to the LLVM dialect using algebraic method
void convertDivToLLVMUsingAlgebraic(ConversionPatternRewriter &rewriter,
                                    Location loc, Value lhsRe, Value lhsIm,
                                    Value rhsRe, Value rhsIm,
                                    LLVM::FastmathFlagsAttr fmf,
                                    Value *resultRe, Value *resultIm);

/// convert a complex division to the arith/math dialects using algebraic method
void convertDivToStandardUsingAlgebraic(ConversionPatternRewriter &rewriter,
                                        Location loc, Value lhsRe, Value lhsIm,
                                        Value rhsRe, Value rhsIm,
                                        arith::FastMathFlagsAttr fmf,
                                        Value *resultRe, Value *resultIm);

/// convert a complex division to the LLVM dialect using Smith's method
void convertDivToLLVMUsingRangeReduction(ConversionPatternRewriter &rewriter,
                                         Location loc, Value lhsRe, Value lhsIm,
                                         Value rhsRe, Value rhsIm,
                                         LLVM::FastmathFlagsAttr fmf,
                                         Value *resultRe, Value *resultIm);

/// convert a complex division to the arith/math dialects using Smith's method
void convertDivToStandardUsingRangeReduction(
    ConversionPatternRewriter &rewriter, Location loc, Value lhsRe, Value lhsIm,
    Value rhsRe, Value rhsIm, arith::FastMathFlagsAttr fmf, Value *resultRe,
    Value *resultIm);

} // namespace complex
} // namespace mlir

#endif // MLIR_CONVERSION_COMPLEXCOMMON_DIVISIONCONVERTER_H
