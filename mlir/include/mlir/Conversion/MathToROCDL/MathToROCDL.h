//===- MathToROCDL.h - Utils to convert from the complex dialect --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_MATHTOROCDL_MATHTOROCDL_H_
#define MLIR_CONVERSION_MATHTOROCDL_MATHTOROCDL_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include <memory>

namespace mlir {
class Pass;

#define GEN_PASS_DECL_CONVERTMATHTOROCDL
#include "mlir/Conversion/Passes.h.inc"

/// Populate the given list with patterns that convert from Math to ROCDL calls.
void populateMathToROCDLConversionPatterns(LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns);
} // namespace mlir

#endif // MLIR_CONVERSION_MATHTOROCDL_MATHTOROCDL_H_
