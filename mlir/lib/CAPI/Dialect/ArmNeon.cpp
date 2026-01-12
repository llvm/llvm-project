//===- ArmNeon.cpp - C Interface for ArmNeon dialect ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/ArmNeon.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"

using namespace mlir::arm_neon;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(ArmNeon, arm_neon,
                                      mlir::arm_neon::ArmNeonDialect)
