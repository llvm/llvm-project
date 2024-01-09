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

namespace mlir {
namespace part_tensor {
/// Initializes the memref with the provided size and data pointer.  This
/// is designed for functions which want to "return" a memref that aliases
/// into memory owned by some other object (e.g., `SparseTensorStorage`),
/// without doing any actual copying.  (The "return" is in scarequotes
/// because the `_mlir_ciface_` calling convention migrates any returned
/// memrefs into an out-parameter passed before all the other function
/// parameters.)
///
/// We make this a function rather than a macro mainly for type safety
/// reasons.  This function does not modify the data pointer, but it
/// cannot be marked `const` because it is stored into the (necessarily)
/// non-`const` memref.  This function is templated over the `DataSizeT`
/// to work around signedness warnings due to many data types having
/// varying signedness across different platforms.  The templating allows
/// this function to ensure that it does the right thing and never
/// introduces errors due to implicit conversions.
template <typename DataSizeT, typename T>
static inline void aliasIntoMemref(DataSizeT size, T *data,
                                   StridedMemRefType<T, 1> &ref) {
  ref.basePtr = ref.data = data;
  ref.offset = 0;
  using MemrefSizeT = typename std::remove_reference_t<decltype(ref.sizes[0])>;
  ref.sizes[0] = sparse_tensor::detail::checkOverflowCast<MemrefSizeT>(size);
  ref.strides[0] = 1;
}

} // namespace part_tensor
} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_PARTTENSORRUNTIME_H
