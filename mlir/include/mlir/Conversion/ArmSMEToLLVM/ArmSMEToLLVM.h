//===- ArmSMEToLLVM.h - Convert ArmSME to LLVM dialect ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_ARMSMETOLLVM_ARMSMETOLLVM_H_
#define MLIR_CONVERSION_ARMSMETOLLVM_ARMSMETOLLVM_H_

#include <memory>

namespace mlir {
class Pass;
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTARMSMETOLLVM
#include "mlir/Conversion/Passes.h.inc"

/// Create a pass to convert a subset of ArmSME ops to SCF.
std::unique_ptr<Pass> createConvertArmSMEToLLVMPass();

} // namespace mlir

#endif // MLIR_CONVERSION_ARMSMETOLLVM_ARMSMETOLLVM_H_
