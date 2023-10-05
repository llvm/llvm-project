//===- MapRef.h - A dim2lvl/lvl2dim map encoding ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A dim2lvl/lvl2dim map encoding class, with utility methods.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_SPARSETENSOR_MAPREF_H
#define MLIR_EXECUTIONENGINE_SPARSETENSOR_MAPREF_H

#include <cinttypes>

#include <cassert>

namespace mlir {
namespace sparse_tensor {

/// A class for capturing the sparse tensor type map with a compact encoding.
///
/// Currently, the following situations are supported:
///   (1) map is an identity
///   (2) map is a permutation
///   (3) map has affine ops (restricted set)
///
/// The pushforward/backward operations are fast for (1) and (2) but
/// incur some obvious overhead for situation (3).
///
class MapRef final {
public:
  MapRef(uint64_t d, uint64_t l, const uint64_t *d2l, const uint64_t *l2d);

  //
  // Push forward maps from dimensions to levels.
  //

  template <typename T>
  inline void pushforward(const T *in, T *out) const {
    switch (kind) {
    case MapKind::kIdentity:
      for (uint64_t i = 0; i < dimRank; ++i)
        out[i] = in[i]; // TODO: optimize with in == out ?
      break;
    case MapKind::kPermutation:
      for (uint64_t i = 0; i < dimRank; ++i)
        out[dim2lvl[i]] = in[i];
      break;
    case MapKind::kAffine:
      assert(0 && "coming soon");
      break;
    }
  }

  //
  // Push backward maps from levels to dimensions.
  //

  template <typename T>
  inline void pushbackward(const T *in, T *out) const {
    switch (kind) {
    case MapKind::kIdentity:
      for (uint64_t i = 0; i < lvlRank; ++i)
        out[i] = in[i];
      break;
    case MapKind::kPermutation:
      for (uint64_t i = 0; i < lvlRank; ++i)
        out[lvl2dim[i]] = in[i];
      break;
    case MapKind::kAffine:
      assert(0 && "coming soon");
      break;
    }
  }

  uint64_t getDimRank() const { return dimRank; }
  uint64_t getLvlRank() const { return lvlRank; }

private:
  enum class MapKind { kIdentity, kPermutation, kAffine };

  bool isIdentity() const;
  bool isPermutation() const;

  MapKind kind;
  const uint64_t dimRank;
  const uint64_t lvlRank;
  const uint64_t *const dim2lvl; // non-owning pointer
  const uint64_t *const lvl2dim; // non-owning pointer
};

} // namespace sparse_tensor
} // namespace mlir

#endif //  MLIR_EXECUTIONENGINE_SPARSETENSOR_MAPREF_H
