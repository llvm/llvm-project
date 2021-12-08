//===- MemRefUtils.h - MemRef transformation utilities ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for various transformation utilities for
// the MemRefOps dialect. These are not passes by themselves but are used
// either by passes, optimization sequences, or in turn by other transformation
// utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MEMREF_UTILS_MEMREFUTILS_H
#define MLIR_DIALECT_MEMREF_UTILS_MEMREFUTILS_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {

/// Finds a single dealloc operation for the given allocated value. If there
/// are > 1 deallocates for `allocValue`, returns None, else returns the single
/// deallocate if it exists or nullptr.
llvm::Optional<Operation *> findDealloc(Value allocValue);
} // namespace mlir

#endif // MLIR_DIALECT_MEMREF_UTILS_MEMREFUTILS_H
