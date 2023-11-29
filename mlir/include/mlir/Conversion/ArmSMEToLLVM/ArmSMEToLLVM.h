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

#include "mlir/Dialect/ArmSME/Transforms/Passes.h"

namespace mlir {
class Pass;
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTARMSMETOLLVM
#include "mlir/Conversion/Passes.h.inc"

using arm_sme::ArmSMETypeConverter;

/// Create a pass to convert from the ArmSME dialect to LLVM intrinsics.
std::unique_ptr<Pass> createConvertArmSMEToLLVMPass();

/// Configure target to convert from the ArmSME dialect to LLVM intrinsics.
void configureArmSMEToLLVMConversionLegality(ConversionTarget &target);

/// Populate the given list with patterns that convert from the ArmSME dialect
/// to LLVM intrinsics.
void populateArmSMEToLLVMConversionPatterns(ArmSMETypeConverter &converter,
                                            RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_ARMSMETOLLVM_ARMSMETOLLVM_H_
