//===- SPIRVGLCanonicalization.cpp - SPIR-V GLSL canonicalization patterns =//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the canonicalization patterns for SPIR-V GLSL-specific ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVGLCanonicalization.h"

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"

using namespace mlir;

namespace {
#include "SPIRVCanonicalization.inc"
} // namespace

namespace mlir {
namespace spirv {
void populateSPIRVGLCanonicalizationPatterns(RewritePatternSet &results) {
  results.add<ConvertComparisonIntoClamp1_SPV_FOrdLessThanOp,
              ConvertComparisonIntoClamp1_SPV_FOrdLessThanEqualOp,
              ConvertComparisonIntoClamp1_SPV_SLessThanOp,
              ConvertComparisonIntoClamp1_SPV_SLessThanEqualOp,
              ConvertComparisonIntoClamp1_SPV_ULessThanOp,
              ConvertComparisonIntoClamp1_SPV_ULessThanEqualOp,
              ConvertComparisonIntoClamp2_SPV_FOrdLessThanOp,
              ConvertComparisonIntoClamp2_SPV_FOrdLessThanEqualOp,
              ConvertComparisonIntoClamp2_SPV_SLessThanOp,
              ConvertComparisonIntoClamp2_SPV_SLessThanEqualOp,
              ConvertComparisonIntoClamp2_SPV_ULessThanOp,
              ConvertComparisonIntoClamp2_SPV_ULessThanEqualOp>(
      results.getContext());
}
} // namespace spirv
} // namespace mlir
