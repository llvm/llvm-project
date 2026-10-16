//===-- XeVMToLLVM.h - Convert XeVM to LLVM dialect -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_XEVMTOLLVM_XEVMTOLLVMPASS_H_
#define MLIR_CONVERSION_XEVMTOLLVM_XEVMTOLLVMPASS_H_

#include <memory>

namespace mlir {
class ConversionTarget;
class DialectRegistry;
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTXEVMTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"

void populateXeVMToLLVMConversionPatterns(ConversionTarget &target,
                                          RewritePatternSet &patterns);

void registerConvertXeVMToLLVMInterface(DialectRegistry &registry);
} // namespace mlir

#endif // MLIR_CONVERSION_XEVMTOLLVM_XEVMTOLLVMPASS_H_
