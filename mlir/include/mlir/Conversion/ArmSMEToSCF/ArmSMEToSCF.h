//===- ArmSMEToSCF.h - Convert ArmSME to SCF dialect ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_ARMSMETOSCF_ARMSMETOSCF_H_
#define MLIR_CONVERSION_ARMSMETOSCF_ARMSMETOSCF_H_

#include <memory>

namespace mlir {
class Pass;
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTARMSMETOSCF
#include "mlir/Conversion/Passes.h.inc"

/// Collect a set of patterns to convert from the ArmSME dialect to SCF.
void populateArmSMEToSCFConversionPatterns(RewritePatternSet &patterns);

/// Create a pass to convert a subset of ArmSME ops to SCF.
std::unique_ptr<Pass> createConvertArmSMEToSCFPass();

} // namespace mlir

#endif // MLIR_CONVERSION_ARMSMETOSCF_ARMSMETOSCF_H_
