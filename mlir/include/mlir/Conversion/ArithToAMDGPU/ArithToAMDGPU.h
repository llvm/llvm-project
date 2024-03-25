//===- ArithToAMDGPU.h - Arith to AMDGPU dialect conversion ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_ARITHTOAMDGPU_ARITHTOAMDGPU_H
#define MLIR_CONVERSION_ARITHTOAMDGPU_ARITHTOAMDGPU_H

#include <memory>

namespace mlir {

class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_ARITHTOAMDGPUCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"

namespace arith {
/// Add patterns for rewriting `arith.extf` and `arith.truncf` on FP8 types
/// to wrappers around AMDGPU--specific intrinsics. If `saturateFP8TruncF`
/// is set, values outside the range of the destination type are clamped
/// to the largest value of that type instead of being rewritten to Inf (aka
/// NaN).
void populateArithToAMDGPUConversionPatterns(RewritePatternSet &patterns,
                                             bool saturateFP8TruncF);
} // namespace arith
} // namespace mlir

#endif // MLIR_CONVERSION_ARITHTOAMDGPU_ARITHTOAMDGPU_H
