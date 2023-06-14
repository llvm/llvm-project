//===- LegalizeForLLVMExport.cpp - Prepare ArmSME for LLVM translation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Dialect/ArmSME/IR/ArmSMEDialect.h"
#include "mlir/Dialect/ArmSME/Transforms/Transforms.h"

using namespace mlir;
using namespace mlir::arm_sme;

void mlir::configureArmSMELegalizeForExportTarget(
    LLVMConversionTarget &target) {
  target.addLegalOp<aarch64_sme_zero>();
}
