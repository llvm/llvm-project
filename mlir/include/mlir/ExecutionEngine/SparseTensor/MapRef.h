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
#include <vector>

namespace mlir {
namespace sparse_tensor {

/// A class for capturing the sparse tensor type map with a compact encoding.
///
/// Currently, the following situations are supported:
///   (1) map is a permutation
///   (2) map has affine ops (restricted set)
///
/// The pushforward/backward operations are fast for (1) but incur some obvious
/// overhead for situation (2).
///
class MapRef final {
public:
  MapRef(uint64_t d, uint64_t l, const uint64_t *d2l, const uint64_t *l2d);

  //
  // Push forward maps from dimensions to levels.
  //

  template <typename T>
  inline void pushforward(const T *in, T *out) const {
    if (isPermutation) {
      for (uint64_t i = 0; i < lvlRank; ++i)
        out[i] = in[lvl2dim[i]];
    } else {
      assert(0 && "coming soon");
    }
  }

  //
  // Push backward maps from levels to dimensions.
  //

  template <typename T>
  inline void pushbackward(const T *in, T *out) const {
    if (isPermutation) {
      for (uint64_t i = 0; i < dimRank; ++i)
        out[i] = in[dim2lvl[i]];
    } else {
      assert(0 && "coming soon");
    }
  }

  uint64_t getDimRank() const { return dimRank; }
  uint64_t getLvlRank() const { return lvlRank; }

private:
  bool isPermutationMap() const;

  const uint64_t dimRank;
  const uint64_t lvlRank;
  const uint64_t *const dim2lvl; // non-owning pointer
  const uint64_t *const lvl2dim; // non-owning pointer
  const bool isPermutation;
};

} // namespace sparse_tensor
} // namespace mlir

#endif //  MLIR_EXECUTIONENGINE_SPARSETENSOR_MAPREF_H
