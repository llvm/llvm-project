//===- AMX.cpp - C Interface for AMX dialect ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/AMX.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/AMX/AMXDialect.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(AMX, amx, mlir::amx::AMXDialect)
