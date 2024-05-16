//===- ToLLVMPass.h - Conversion to LLVM pass ---*- C++ -*-===================//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_CONVERTTOLLVM_TOLLVM_PASS_H
#define MLIR_CONVERSION_CONVERTTOLLVM_TOLLVM_PASS_H

#include <memory>

#include "mlir/Pass/Pass.h"

#define GEN_PASS_DECL_CONVERTTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"

namespace mlir {

/// Create a pass that performs dialect conversion to LLVM  for all dialects
/// implementing `ConvertToLLVMPatternInterface`.
std::unique_ptr<Pass> createConvertToLLVMPass();

/// Register the extension that will load dependent dialects for LLVM
/// conversion. This is useful to implement a pass similar to "convert-to-llvm".
void registerConvertToLLVMDependentDialectLoading(DialectRegistry &registry);

} // namespace mlir

#endif // MLIR_CONVERSION_CONVERTTOLLVM_TOLLVM_PASS_H
