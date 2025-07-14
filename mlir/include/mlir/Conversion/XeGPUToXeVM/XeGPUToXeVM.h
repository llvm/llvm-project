//===-- XeGPUToXeVM.h - Convert XeGPU to XeVM dialect ---------_--*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_XEGPUTOXEVM_XEGPUTOXEVMPASS_H_
#define MLIR_CONVERSION_XEGPUTOXEVM_XEGPUTOXEVMPASS_H_

#include <memory>

namespace mlir {
class DialectRegistry;
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTXEGPUTOXEVMPASS
#include "mlir/Conversion/Passes.h.inc"

void populateXeGPUToXeVMConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::LLVMTypeConverter &typeConverter);

} // namespace mlir

#endif // MLIR_CONVERSION_XEGPUTOXEVM_XEGPUTOXEVMPASS_H_
