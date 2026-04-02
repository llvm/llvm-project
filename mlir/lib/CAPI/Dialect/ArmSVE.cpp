//===- ArmSVE.cpp - C Interface for ArmSVE dialect ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/ArmSVE.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/ArmSVE/IR/ArmSVEDialect.h"

using namespace mlir::arm_sve;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(ArmSVE, arm_sve,
                                      mlir::arm_sve::ArmSVEDialect)
