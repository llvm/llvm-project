//===- MathToEmitC.h - Math to EmitC Patterns -------------------*- C++ -*-===//
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
namespace emitc {

/// Enum to specify the language target for EmitC code generation.
enum class LanguageTarget { c99, cpp11 };

} // namespace emitc

void populateConvertMathToEmitCPatterns(RewritePatternSet &patterns,
                                        emitc::LanguageTarget languageTarget);
} // namespace mlir

#endif // MLIR_CONVERSION_MATHTOEMITC_MATHTOEMITC_H
