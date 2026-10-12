//===- GPUDialectDecl.h - GPU dialect declaration --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_IR_GPUDIALECTDECL_H
#define MLIR_DIALECT_GPU_IR_GPUDIALECTDECL_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

// Pull in enum definitions used by the generated dialect declaration.
#include "mlir/Dialect/GPU/IR/GPUOpsEnums.h.inc"

// Pull in the generated dialect declaration.
#include "mlir/Dialect/GPU/IR/GPUOpsDialect.h.inc"

#endif // MLIR_DIALECT_GPU_IR_GPUDIALECTDECL_H
