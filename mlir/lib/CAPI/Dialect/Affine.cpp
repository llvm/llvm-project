//===- Affine.cpp - C Interface for Affine dialect ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Affine.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Affine, affine,
                                      mlir::affine::AffineDialect)
