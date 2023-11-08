//===- CppEmitter.h - Helpers to create C++ emitter -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines helpers to emit C++ code using the EmitC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_KOKKOSCPP_KOKKOSCPPEMITTER_H
#define MLIR_TARGET_KOKKOSCPP_KOKKOSCPPEMITTER_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include <stack>

namespace mlir {
namespace emitc {

/// Translates the given operation to Kokkos C++ code.
LogicalResult translateToKokkosCpp(Operation *op, raw_ostream &os,
                                bool enableSparseSupport = false);

/// Translates the given operation to Kokkos C++ code, with a Python wrapper module written to py_os.
LogicalResult translateToKokkosCpp(Operation *op, raw_ostream &os, raw_ostream &py_os,
                                bool enableSparseSupport = false, bool useHierarchical = false, bool isLastKernel = true);
} // namespace emitc
} // namespace mlir

#endif // MLIR_TARGET_KOKKOSCPP_KOKKOSCPPEMITTER_H
