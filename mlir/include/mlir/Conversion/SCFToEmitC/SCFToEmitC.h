//===- SCFToEmitC.h - SCF to EmitC Pass entrypoint --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SCFTOEMITC_SCFTOEMITC_H
#define MLIR_CONVERSION_SCFTOEMITC_SCFTOEMITC_H

#include <memory>

namespace mlir {
class Pass;
class RewritePatternSet;

#define GEN_PASS_DECL_SCFTOEMITC
#include "mlir/Conversion/Passes.h.inc"

/// Collect a set of patterns to convert SCF operations to the EmitC dialect.
void populateSCFToEmitCConversionPatterns(RewritePatternSet &patterns);
} // namespace mlir

#endif // MLIR_CONVERSION_SCFTOEMITC_SCFTOEMITC_H
