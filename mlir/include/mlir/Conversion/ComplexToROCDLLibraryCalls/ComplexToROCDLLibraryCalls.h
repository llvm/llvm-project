//===- ComplexToROCDLLibraryCalls.h - convert from Complex to ROCDL calls -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_COMPLEXTOROCDLLIBRARYCALLS_COMPLEXTOROCDLLIBRARYCALLS_H_
#define MLIR_CONVERSION_COMPLEXTOROCDLLIBRARYCALLS_COMPLEXTOROCDLLIBRARYCALLS_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTCOMPLEXTOROCDLLIBRARYCALLS
#include "mlir/Conversion/Passes.h.inc"

/// Populate the given list with patterns that convert from Complex to ROCDL
/// calls.
void populateComplexToROCDLLibraryCallsConversionPatterns(
    RewritePatternSet &patterns);
} // namespace mlir

#endif // MLIR_CONVERSION_COMPLEXTOROCDLLIBRARYCALLS_COMPLEXTOROCDLLIBRARYCALLS_H_
