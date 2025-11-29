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
  results.add<ConvertComparisonIntoClamp1_SPIRV_FOrdLessThanOp,
              ConvertComparisonIntoClamp1_SPIRV_FOrdLessThanEqualOp,
              ConvertComparisonIntoClamp1_SPIRV_SLessThanOp,
              ConvertComparisonIntoClamp1_SPIRV_SLessThanEqualOp,
              ConvertComparisonIntoClamp1_SPIRV_ULessThanOp,
              ConvertComparisonIntoClamp1_SPIRV_ULessThanEqualOp,
              ConvertComparisonIntoClamp2_SPIRV_FOrdLessThanOp,
              ConvertComparisonIntoClamp2_SPIRV_FOrdLessThanEqualOp,
              ConvertComparisonIntoClamp2_SPIRV_SLessThanOp,
              ConvertComparisonIntoClamp2_SPIRV_SLessThanEqualOp,
              ConvertComparisonIntoClamp2_SPIRV_ULessThanOp,
              ConvertComparisonIntoClamp2_SPIRV_ULessThanEqualOp>(
      results.getContext());
}
} // namespace spirv
} // namespace mlir
