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

enum class MathToROCDLConversionPatternKind { All, Scalarizations, Lowerings };

/// Populate the given list with patterns that convert from Math to ROCDL calls.
///
/// Note that the default parameter value MathToROCDLConversionPatternKind::All
/// is only for compatibility but is not recommended, because lumping together
/// multiple conversion patters in the same pattern application can result in
/// type conversion failures when one of the patterns failed.
void populateMathToROCDLConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns,
    MathToROCDLConversionPatternKind patternKind =
        MathToROCDLConversionPatternKind::All);
} // namespace mlir

#endif // MLIR_CONVERSION_MATHTOROCDL_MATHTOROCDL_H_
