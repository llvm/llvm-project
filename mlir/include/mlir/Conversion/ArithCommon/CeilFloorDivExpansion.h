//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_ARITHCOMMON_CEILFLOORDIVEXPANSION_H
#define MLIR_CONVERSION_ARITHCOMMON_CEILFLOORDIVEXPANSION_H

namespace mlir {
class RewritePatternSet;

namespace arith {
// Add patterns to expand Arith ceil/floor division ops.
void populateCeilFloorDivExpandOpsPatterns(RewritePatternSet &patterns);
} // namespace arith
} // namespace mlir

#endif // MLIR_CONVERSION_ARITHCOMMON_CEILFLOORDIVEXPANSION_H
