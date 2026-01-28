//===- VectorToAMX.h - Convert vector to AMX dialect ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_VECTORTOAMX_VECTORTOAMX_H
#define MLIR_CONVERSION_VECTORTOAMX_VECTORTOAMX_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {
class Pass;
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTVECTORTOAMX
#include "mlir/Conversion/Passes.h.inc"

/// Collect a set of patterns to convert from the vector to AMX ops.
void populateVectorToAMXConversionPatterns(RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_VECTORTOAMX_VECTORTOAMX_H
