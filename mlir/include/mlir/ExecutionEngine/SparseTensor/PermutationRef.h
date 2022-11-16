//===- PermutationRef.h - Permutation reference wrapper ---------*- C++ -*-===//
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

#ifndef MLIR_EXECUTIONENGINE_SPARSETENSOR_PERMUTATIONREF_H
#define MLIR_EXECUTIONENGINE_SPARSETENSOR_PERMUTATIONREF_H

#include "mlir/ExecutionEngine/SparseTensor/Attributes.h"
#include "mlir/ExecutionEngine/SparseTensor/ErrorHandling.h"

#include <cassert>
#include <cinttypes>
#include <vector>

namespace mlir {
namespace sparse_tensor {
namespace detail {

/// Checks whether the `perm` array is a permutation of `[0 .. size)`.
MLIR_SPARSETENSOR_PURE bool isPermutation(uint64_t size, const uint64_t *perm);

/// Wrapper around `isPermutation` to ensure consistent error messages.
inline void assertIsPermutation(uint64_t size, const uint64_t *perm) {
#ifndef NDEBUG
  if (!isPermutation(size, perm))
    MLIR_SPARSETENSOR_FATAL("Not a permutation of [0..%" PRIu64 ")\n", size);
#endif
}

// TODO: To implement things like `inverse` and `compose` while preserving
// the knowledge that `isPermutation` is true, we'll need to also have
// an owning version of `PermutationRef`.  (Though ideally we'll really
// want to defunctionalize those methods so that we can avoid intermediate
// arrays/copies and only materialize the data on request.)

/// A non-owning class for capturing the knowledge that `isPermutation`
/// is true, to avoid needing to assert it repeatedly.
class MLIR_SPARSETENSOR_GSL_POINTER [[nodiscard]] PermutationRef final {
public:
  /// Asserts `isPermutation` and returns the witness to that being true.
  //
  // TODO: For now the assertive ctor is sufficient, but in principle
  // we'll want a factory that can optionally construct the object
  // (so callers can handle errors themselves).
  explicit PermutationRef(uint64_t size, const uint64_t *perm)
      : permSize(size), perm(perm) {
    assertIsPermutation(size, perm);
  }

  uint64_t size() const { return permSize; }

  const uint64_t *data() const { return perm; }

  const uint64_t &operator[](uint64_t i) const {
    assert(i < permSize && "index is out of bounds");
    return perm[i];
  }

  /// Constructs a pushforward array of values.  This method is the inverse
  /// of `permute` in the sense that for all `p` and `xs` we have:
  /// * `p.permute(p.pushforward(xs)) == xs`
  /// * `p.pushforward(p.permute(xs)) == xs`
  template <typename T>
  inline std::vector<T> pushforward(const std::vector<T> &values) const {
    return pushforward(values.size(), values.data());
  }

  template <typename T>
  inline std::vector<T> pushforward(uint64_t size, const T *values) const {
    std::vector<T> out(permSize);
    pushforward(size, values, out.data());
    return out;
  }

  // NOTE: This form of the method is required by `toMLIRSparseTensor`,
  // so it can reuse the `out` buffer for each iteration of a loop.
  template <typename T>
  inline void pushforward(uint64_t size, const T *values, T *out) const {
    assert(size == permSize && "size mismatch");
    for (uint64_t i = 0; i < permSize; ++i)
      out[perm[i]] = values[i];
  }

  // NOTE: this is only needed by `toMLIRSparseTensor`, which in
  // turn only needs it as a vector to hand off to `newSparseTensor`.
  // Otherwise we would want the result to be an owning-permutation,
  // to retain the knowledge that `isPermutation` is true.
  //
  /// Constructs the inverse permutation.  This is equivalent to calling
  /// `pushforward` with `std::iota` for the values.
  std::vector<uint64_t> inverse() const;

  /// Constructs a permuted array of values.  This method is the inverse
  /// of `pushforward` in the sense that for all `p` and `xs` we have:
  /// * `p.permute(p.pushforward(xs)) == xs`
  /// * `p.pushforward(p.permute(xs)) == xs`
  template <typename T>
  inline std::vector<T> permute(const std::vector<T> &values) const {
    return permute(values.size(), values.data());
  }

  template <typename T>
  inline std::vector<T> permute(uint64_t size, const T *values) const {
    std::vector<T> out(permSize);
    permute(size, values, out.data());
    return out;
  }

  template <typename T>
  inline void permute(uint64_t size, const T *values, T *out) const {
    assert(size == permSize && "size mismatch");
    for (uint64_t i = 0; i < permSize; ++i)
      out[i] = values[perm[i]];
    return out;
  }

private:
  const uint64_t permSize;
  const uint64_t *const perm; // non-owning pointer.
};

} // namespace detail
} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_SPARSETENSOR_PERMUTATIONREF_H
