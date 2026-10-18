//===- AGXToAGXIR.h - Convert AGX to AGXIR dialect --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_AGXTOAGXIR_AGXTOAGXIR_H_
#define MLIR_CONVERSION_AGXTOAGXIR_AGXTOAGXIR_H_

#include <memory>

namespace mlir {

class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTAGXTOAGXIRPASS
#include "mlir/Conversion/Passes.h.inc"

/// Collect the patterns to convert from the AGX dialect to AGXIR.
void populateAGXToAGXIRConversionPatterns(const LLVMTypeConverter &converter,
                                          RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_AGXTOAGXIR_AGXTOAGXIR_H_
