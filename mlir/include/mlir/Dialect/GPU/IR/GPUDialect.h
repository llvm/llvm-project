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

#ifndef MLIR_DIALECT_GPU_IR_GPUDIALECT_H
#define MLIR_DIALECT_GPU_IR_GPUDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/DLTI/Traits.h"
#include "mlir/Dialect/GPU/IR/GPUBase.h"
#include "mlir/Dialect/GPU/IR/GPUDialectDecl.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace gpu {
/// Utility class for the GPU dialect to represent triples of `Value`s
/// accessible through `.x`, `.y`, and `.z` similarly to CUDA notation.
struct KernelDim3 {
  Value x;
  Value y;
  Value z;
};

// Adds a `gpu.async.token` to the front of the argument list.
void addAsyncDependency(Operation *op, Value token);
} // namespace gpu
} // namespace mlir

#include "mlir/Dialect/GPU/IR/GPUOpInterfaces.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/GPU/IR/GPUOps.h.inc"

namespace mlir::gpu {
/// Retrieve the constant bounds for a given dimension and dimension kind
/// from the context surrounding `op`, if known, and return them. This will
/// check the bounds on an enclosing `gpu.launch`, an enclosing `gpu.func`, and
/// any `gpu.known_*_size` on other function-like operations, in that order.
std::optional<uint32_t>
getKnownDimensionSizeAround(Operation *op, DimensionKind kind, Dimension dim);
} // namespace mlir::gpu
#endif // MLIR_DIALECT_GPU_IR_GPUDIALECT_H
