//===- SPIRVGLSLCanonicalization.h - GLSL-specific patterns -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a function to register SPIR-V GLSL-specific
// canonicalization patterns.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_SPIRVGLSLCANONICALIZATION_H_
#define MLIR_DIALECT_SPIRV_IR_SPIRVGLSLCANONICALIZATION_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"

//===----------------------------------------------------------------------===//
// GLSL canonicalization patterns
//===----------------------------------------------------------------------===//

namespace mlir {
namespace spirv {
void populateSPIRVGLSLCanonicalizationPatterns(
    mlir::OwningRewritePatternList &results);
} // namespace spirv
} // namespace mlir

#endif // MLIR_DIALECT_SPIRV_IR_SPIRVGLSLCANONICALIZATION_H_
