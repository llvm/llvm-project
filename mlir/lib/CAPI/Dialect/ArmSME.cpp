//===- ArmSME.cpp - C Interface for ArmSME dialect ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/ArmSME.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/ArmSME/IR/ArmSME.h"

using namespace mlir::arm_sme;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(ArmSME, arm_sme,
                                      mlir::arm_sme::ArmSMEDialect)
