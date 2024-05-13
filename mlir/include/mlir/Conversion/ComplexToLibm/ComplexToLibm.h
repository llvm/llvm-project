//===- ComplexToLibm.h - Utils to convert from the complex dialect --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_COMPLEXTOLIBM_COMPLEXTOLIBM_H_
#define MLIR_CONVERSION_COMPLEXTOLIBM_COMPLEXTOLIBM_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
template <typename T>
class OperationPass;

#define GEN_PASS_DECL_CONVERTCOMPLEXTOLIBM
#include "mlir/Conversion/Passes.h.inc"

/// Populate the given list with patterns that convert from Complex to Libm
/// calls.
void populateComplexToLibmConversionPatterns(RewritePatternSet &patterns,
                                             PatternBenefit benefit);

/// Create a pass to convert Complex operations to libm calls.
std::unique_ptr<OperationPass<ModuleOp>> createConvertComplexToLibmPass();

} // namespace mlir

#endif // MLIR_CONVERSION_COMPLEXTOLIBM_COMPLEXTOLIBM_H_
