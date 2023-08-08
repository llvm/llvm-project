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

namespace mlir {
class Pass;

/// Create a pass that performs dialect conversion to LLVM  for all dialects
/// implementing `ConvertToLLVMPatternInterface`.
std::unique_ptr<Pass> createConvertToLLVMPass();

} // namespace mlir

#endif // MLIR_CONVERSION_CONVERTTOLLVM_TOLLVM_PASS_H
