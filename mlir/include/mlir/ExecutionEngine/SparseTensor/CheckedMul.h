//===- CheckedMul.h - multiplication that checks for overflow ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header is not part of the public API.  It is placed in the
// includes directory only because that's required by the implementations
// of template-classes.
//
// This file is part of the lightweight runtime support library for sparse
// tensor manipulations.  The functionality of the support library is meant
// to simplify benchmarking, testing, and debugging MLIR code operating on
// sparse tensors.  However, the provided functionality is **not** part of
// core MLIR itself.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_SPARSETENSOR_CHECKEDMUL_H
#define MLIR_EXECUTIONENGINE_SPARSETENSOR_CHECKEDMUL_H

#include <cassert>
#include <cinttypes>
#include <limits>

namespace mlir {
namespace sparse_tensor {
namespace detail {

// TODO: would be better to use various architectures' intrinsics to
// detect the overflow directly, instead of doing the assertion beforehand
// (which requires an expensive division).
//
/// A version of `operator*` on `uint64_t` which guards against overflows
/// (when assertions are enabled).
inline uint64_t checkedMul(uint64_t lhs, uint64_t rhs) {
  assert((lhs == 0 || rhs <= std::numeric_limits<uint64_t>::max() / lhs) &&
         "Integer overflow");
  return lhs * rhs;
}

} // namespace detail
} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_SPARSETENSOR_CHECKEDMUL_H
