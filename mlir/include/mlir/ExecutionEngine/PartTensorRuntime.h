//===- SparseTensorRuntime.h - SparseTensor runtime support lib -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file provides the enums and functions which comprise the
// public API of the `ExecutionEngine/SparseTensorRuntime.cpp` runtime
// support library for the SparseTensor dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_PARTTENSORRUNTIME_H
#define MLIR_EXECUTIONENGINE_PARTTENSORRUNTIME_H

#include "mlir/Dialect/SparseTensor/IR/Enums.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/Float16bits.h"
#include "mlir/ExecutionEngine/PartTensor/Storage.h"
#include "mlir/ExecutionEngine/SparseTensor/ArithmeticUtils.h"
#include "mlir/ExecutionEngine/SparseTensorRuntime.h"

#include <cinttypes>
#include <complex>

#define ASSERT_NO_STRIDE(MEMREF)                                               \
  do {                                                                         \
    assert((MEMREF) && "Memref is nullptr");                                   \
    assert(((MEMREF)->strides[0] == 1) && "Memref has non-trivial stride");    \
  } while (false)

#define MEMREF_GET_USIZE(MEMREF)                                               \
  detail::checkOverflowCast<uint64_t>((MEMREF)->sizes[0])

#define ASSERT_USIZE_EQ(MEMREF, SZ)                                            \
  assert(detail::safelyEQ(MEMREF_GET_USIZE(MEMREF), (SZ)) &&                   \
         "Memref size mismatch")

#define MEMREF_GET_PAYLOAD(MEMREF) ((MEMREF)->data + (MEMREF)->offset)

using namespace mlir::part_tensor;

extern "C" {

MLIR_CRUNNERUTILS_EXPORT void *_mlir_ciface_newPartTensor( // NOLINT
    StridedMemRefType<index_type, 1> *partSizesRef,
    StridedMemRefType<index_type, 1> *dimSizesRef, PrimaryType valTp,
    Action action, void *ptr);

} // extern "C"

#endif // MLIR_EXECUTIONENGINE_PARTTENSORRUNTIME_H
