//===- PDLInterp.cpp - C Interface for PDLInterp dialect ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/PDLInterp.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(PDLInterp, pdl_interp,
                                      mlir::pdl_interp::PDLInterpDialect)
