//===- UB.cpp - C Interface for UB dialect --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/UB.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/UB/IR/UBOps.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(UB, ub, mlir::ub::UBDialect)