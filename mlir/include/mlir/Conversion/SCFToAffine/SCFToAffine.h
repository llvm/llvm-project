//===- SCFToAffine.h - SCF to Affine Pass entrypoint ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SCFTOAFFINE_SCFTOAFFINE_H_
#define MLIR_CONVERSION_SCFTOAFFINE_SCFTOAFFINE_H_

#include <memory>

namespace mlir {
class Pass;
class RewritePatternSet;

#define GEN_PASS_DECL_RAISESCFTOAFFINEPASS
#include "mlir/Conversion/Passes.h.inc"

/// Collect a set of patterns to convert SCF operations to Affine operations.
void populateSCFToAffineConversionPatterns(RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_SCFTOAFFINE_SCFTOAFFINE_H_
