//===- MathToEmitC.h - Math to EmitC Pass -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_MATHTOEMITC_MATHTOEMITC_H
#define MLIR_CONVERSION_MATHTOEMITC_MATHTOEMITC_H

namespace mlir {
class RewritePatternSet;

void populateConvertMathToEmitCPatterns(RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_MATHTOEMITC_MATHTOEMITC_H
