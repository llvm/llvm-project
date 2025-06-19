#ifndef MLIR_CONVERSION_COMPLEXTOROCDL_COMPLEXTOROCDL_H_
#define MLIR_CONVERSION_COMPLEXTOROCDL_COMPLEXTOROCDL_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTCOMPLEXTOROCDL
#include "mlir/Conversion/Passes.h.inc"

/// Populate the given list with patterns that convert from Complex to ROCDL
/// calls.
void populateComplexToROCDLConversionPatterns(RewritePatternSet &patterns,
                                              PatternBenefit benefit);
} // namespace mlir

#endif // MLIR_CONVERSION_COMPLEXTOROCDL_COMPLEXTOROCDL_H_
