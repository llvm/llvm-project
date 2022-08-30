//===- ComplexToStandard.h - Utils to convert from the complex dialect ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_COMPLEXTOSTANDARD_COMPLEXTOSTANDARD_H_
#define MLIR_CONVERSION_COMPLEXTOSTANDARD_COMPLEXTOSTANDARD_H_

#include <memory>

namespace mlir {
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTCOMPLEXTOSTANDARDPASS
#include "mlir/Conversion/Passes.h.inc"

/// Populate the given list with patterns that convert from Complex to Standard.
void populateComplexToStandardConversionPatterns(RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_COMPLEXTOSTANDARD_COMPLEXTOSTANDARD_H_
