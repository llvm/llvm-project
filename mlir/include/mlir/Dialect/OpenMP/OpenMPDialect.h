//===- OpenMPDialect.h - MLIR Dialect for OpenMP ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the OpenMP dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENMP_OPENMPDIALECT_H_
#define MLIR_DIALECT_OPENMP_OPENMPDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "mlir/Dialect/OpenMP/OpenMPOpsEnums.h.inc"

namespace mlir {
namespace omp {

#define GET_OP_CLASSES
#include "mlir/Dialect/OpenMP/OpenMPOps.h.inc"

#include "mlir/Dialect/OpenMP/OpenMPOpsDialect.h.inc"
} // namespace omp
} // namespace mlir

#endif // MLIR_DIALECT_OPENMP_OPENMPDIALECT_H_
