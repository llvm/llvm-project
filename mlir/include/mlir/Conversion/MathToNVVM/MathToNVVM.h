//===- MathToNVVM.h - Utils to convert from the Math dialect to NVVM -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_MATHTONVVM_MATHTONVVM_H_
#define MLIR_CONVERSION_MATHTONVVM_MATHTONVVM_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
class Pass;

#define GEN_PASS_DECL_CONVERTMATHTONVVM
#include "mlir/Conversion/Passes.h.inc"

/// Populate the given list with patterns that convert from Math to NVVM
/// libdevice calls.
void populateLibDeviceConversionPatterns(const LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns,
                                         PatternBenefit benefit = 1);
} // namespace mlir

#endif // MLIR_CONVERSION_MATHTONVVM_MATHTONVVM_H_
