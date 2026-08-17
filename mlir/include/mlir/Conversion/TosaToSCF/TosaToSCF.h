//===-- TosaToSCF.h - TOSA to SCF dialect lowerings -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the passes for the TOSA to SCF Dialect conversion.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_TOSATOSCF_TOSATOSCF_H
#define MLIR_CONVERSION_TOSATOSCF_TOSATOSCF_H

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DECL_TOSATOSCFPASS
#include "mlir/Conversion/Passes.h.inc"

namespace tosa {

std::unique_ptr<Pass> createTosaToSCFPass(bool scatterHardening = true);
void populateTosaToSCFConversionPatterns(RewritePatternSet *patterns,
                                         bool scatterHardening = true);

/// Populates passes to convert from TOSA to SCF.
void addTosaToSCFPasses(OpPassManager &pm, const TosaToSCFPassOptions &options);

} // namespace tosa
} // namespace mlir

#endif // MLIR_CONVERSION_TOSATOSCF_TOSATOSCF_H
