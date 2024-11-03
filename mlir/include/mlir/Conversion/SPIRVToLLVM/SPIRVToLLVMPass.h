//===- SPIRVToLLVMPass.h - SPIR-V to LLVM Passes ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides passes to convert SPIR-V dialect to LLVM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SPIRVTOLLVM_SPIRVTOLLVMPASS_H
#define MLIR_CONVERSION_SPIRVTOLLVM_SPIRVTOLLVMPASS_H

#include <memory>

namespace mlir {
class Pass;

#define GEN_PASS_DECL_LOWERHOSTCODETOLLVMPASS
#define GEN_PASS_DECL_CONVERTSPIRVTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"

} // namespace mlir

#endif // MLIR_CONVERSION_SPIRVTOLLVM_SPIRVTOLLVMPASS_H
