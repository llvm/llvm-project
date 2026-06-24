//===- Index.cpp - C Interface for Index dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#define GET_ATTRDEF_CLASSES
#include "mlir-c/Dialect/Index.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "llvm/Support/Casting.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Index, index, mlir::index::IndexDialect)

#include "mlir/Dialect/Index/IR/IndexOpsCAPIAttrs.cpp.inc"