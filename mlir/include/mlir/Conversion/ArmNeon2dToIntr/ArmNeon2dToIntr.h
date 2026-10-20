//===- ArmNeon2dToIntr.h - convert Arm Neon 2d ops to intrinsics ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_ARMNEON2DTOINTR_ARMNEON2DTOINTR_H_
#define MLIR_CONVERSION_ARMNEON2DTOINTR_ARMNEON2DTOINTR_H_

#include <memory>

namespace mlir {
class Pass;
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTARMNEON2DTOINTRPASS
#include "mlir/Conversion/Passes.h.inc"

/// Populates patterns for the lowering of Arm NEON 2D ops to intrinsics.
/// See createConvertArmNeon2dToIntrPass.
void populateConvertArmNeon2dToIntrPatterns(RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_ARMNEON2DTOINTR_ARMNEON2DTOINTR_H_
