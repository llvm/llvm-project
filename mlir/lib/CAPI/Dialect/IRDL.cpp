//===- IRDL.cpp - C Interface for IRDL dialect ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/IRDL.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/IRDL/IRDLLoading.h"

using namespace mlir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(IRDL, irdl, irdl::IRDLDialect)

extern "C" MlirLogicalResult mlirIRDLLoadDialects(MlirModule module) {
  return wrap(irdl::loadDialects(unwrap(module)));
}
