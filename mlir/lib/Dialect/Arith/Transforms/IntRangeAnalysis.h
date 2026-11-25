//===- IntRangeAnalysis.h - Integer range analysis for Arith passes -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_DIALECT_ARITH_TRANSFORMS_INTRANGEANALYSIS_H
#define MLIR_LIB_DIALECT_ARITH_TRANSFORMS_INTRANGEANALYSIS_H

namespace mlir {
class DataFlowSolver;
class DataLayoutAnalysis;

namespace arith {
void loadIntRangeAnalysis(DataFlowSolver &solver,
                          const DataLayoutAnalysis &layouts);
} // namespace arith
} // namespace mlir

#endif // MLIR_LIB_DIALECT_ARITH_TRANSFORMS_INTRANGEANALYSIS_H
