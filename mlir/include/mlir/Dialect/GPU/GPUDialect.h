//===- GPUDialect.h - MLIR Dialect for GPU Kernels --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the GPU kernel-related operations and puts them in the
// corresponding dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_GPUDIALECT_H
#define MLIR_DIALECT_GPU_GPUDIALECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
class FuncOp;

namespace gpu {

/// Utility class for the GPU dialect to represent triples of `Value`s
/// accessible through `.x`, `.y`, and `.z` similarly to CUDA notation.
struct KernelDim3 {
  Value x;
  Value y;
  Value z;
};

class AsyncTokenType
    : public Type::TypeBase<AsyncTokenType, Type, TypeStorage> {
public:
  // Used for generic hooks in TypeBase.
  using Base::Base;
};

// Adds a `gpu.async.token` to the front of the argument list.
void addAsyncDependency(Operation *op, Value token);

} // end namespace gpu
} // end namespace mlir

#include "mlir/Dialect/GPU/GPUOpsDialect.h.inc"

#include "mlir/Dialect/GPU/GPUOpInterfaces.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/GPU/GPUOps.h.inc"

#endif // MLIR_DIALECT_GPU_GPUDIALECT_H
