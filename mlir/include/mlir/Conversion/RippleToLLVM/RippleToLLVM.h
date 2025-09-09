//==============================================================================
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==============================================================================
//
// Pass to lower the ripple dialect to the llvm dialect.
//
//==============================================================================

#ifndef MLIR_CONVERSION_RIPPLETOLLVM_RIPPLETOLLVM_H_
#define MLIR_CONVERSION_RIPPLETOLLVM_RIPPLETOLLVM_H_

#include "mlir/Pass/Pass.h" // from @llvm-project

namespace mlir {
#define GEN_PASS_DECL_CONVERTRIPPLETOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"

std::unique_ptr<OperationPass<>> createRippleToLLVMPass();
} // namespace mlir

#endif // MLIR_CONVERSION_RIPPLETOLLVM_RIPPLETOLLVM_H_
