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

#define GEN_PASS_DECL_CONVERTARMNEON2DTOINTR
#include "mlir/Conversion/Passes.h.inc"

/// Populates patterns for the lowering of Arm NEON 2D ops to intrinsics.
/// See createConvertArmNeon2dToIntrPass.
void populateConvertArmNeon2dToIntrPatterns(RewritePatternSet &patterns);

/// Creates a pass to lower Arm NEON 2D ops to intrinsics, i.e.
/// equivalent ops operating on flattened 1D vectors and mapping more
/// directly to the corresponding Arm NEON instruction.
std::unique_ptr<Pass> createConvertArmNeon2dToIntrPass();

} // namespace mlir

#endif // MLIR_CONVERSION_ARMNEON2DTOINTR_ARMNEON2DTOINTR_H_
