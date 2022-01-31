//===- IndexingUtils.h - Helpers related to index computations --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines utilities and common canonicalization patterns for
// reshape operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_UTILS_INDEXINGUTILS_H
#define MLIR_DIALECT_UTILS_INDEXINGUTILS_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class ArrayAttr;

/// Computes and returns the linearized index of 'offsets' w.r.t. 'basis'.
int64_t linearize(ArrayRef<int64_t> offsets, ArrayRef<int64_t> basis);

/// Given the strides together with a linear index in the dimension
/// space, returns the vector-space offsets in each dimension for a
/// de-linearized index.
SmallVector<int64_t, 4> delinearize(ArrayRef<int64_t> strides,
                                    int64_t linearIndex);

/// Helper that returns a subset of `arrayAttr` as a vector of int64_t.
SmallVector<int64_t, 4> getI64SubArray(ArrayAttr arrayAttr,
                                       unsigned dropFront = 0,
                                       unsigned dropBack = 0);
} // namespace mlir

#endif // MLIR_DIALECT_UTILS_INDEXINGUTILS_H
