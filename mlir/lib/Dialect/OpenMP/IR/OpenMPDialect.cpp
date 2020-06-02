//===- OpenMPDialect.cpp - MLIR Dialect for OpenMP implementation ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the OpenMP dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/StringSwitch.h"

#include "mlir/Dialect/OpenMP/OpenMPOpsEnums.cpp.inc"

using namespace mlir;
using namespace mlir::omp;

OpenMPDialect::OpenMPDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/OpenMP/OpenMPOps.cpp.inc"
      >();
}

namespace mlir {
namespace omp {
#define GET_OP_CLASSES
#include "mlir/Dialect/OpenMP/OpenMPOps.cpp.inc"
} // namespace omp
} // namespace mlir
