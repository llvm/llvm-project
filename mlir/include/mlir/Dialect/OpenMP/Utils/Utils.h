//===- Utils.h - OpenMP dialect utilities -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for various OpenMP utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENMP_UTILS_UTILS_H_
#define MLIR_DIALECT_OPENMP_UTILS_UTILS_H_

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace omp {

/// Check whether the value representing an allocation, assumed to have been
/// defined in a shared device context, is used in a manner that would require
/// device shared memory for correctness.
///
/// When a use takes place inside an omp.parallel region and it's not as a
/// private clause argument, or when it is a reduction argument passed to
/// omp.parallel or a function call argument, then the defining allocation is
/// eligible for replacement with shared memory.
///
/// \see mlir::omp::opInSharedDeviceContext().
bool allocaUsesRequireSharedMem(Value alloc);

/// Check whether the given operation is located in a context where an
/// allocation to be used by multiple threads in a parallel region would have to
/// be placed in device shared memory to be accessible.
///
/// That means that it is inside of a target device module, it is a non-SPMD
/// target region, is inside of one or it's located in a device function, and it
/// is not not inside of a parallel region.
///
/// This represents a necessary but not sufficient set of conditions to use
/// device shared memory in place of regular allocas. For some variables, the
/// associated OpenMP construct or their uses might also need to be taken into
/// account.
///
/// \see mlir::omp::allocaUsesRequireSharedMem().
bool opInSharedDeviceContext(Operation &op);

} // namespace omp
} // namespace mlir

#endif // MLIR_DIALECT_OPENMP_UTILS_UTILS_H_
