//===- COO.h - Coordinate-scheme sparse tensor representation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is part of the lightweight runtime support library for sparse
// tensor manipulations.  The functionality of the support library is meant
// to simplify benchmarking, testing, and debugging MLIR code operating on
// sparse tensors.  However, the provided functionality is **not** part of
// core MLIR itself.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_SPARSETENSOR_COO_H
#define MLIR_EXECUTIONENGINE_SPARSETENSOR_COO_H

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <functional>
#include <vector>

namespace mlir {
namespace sparse_tensor {

/// A sparse tensor element in coordinate scheme (value and indices).
/// For example, a rank-1 vector element would look like
///   ({i}, a[i])
/// and a rank-5 tensor element like
///   ({i,j,k,l,m}, a[i,j,k,l,m])
/// We use pointer to a shared index pool rather than e.g. a direct
/// vector since that (1) reduces the per-element memory footprint, and
/// (2) centralizes the memory reservation and (re)allocation to one place.
template <typename V>
struct Element final {
  Element(const uint64_t *ind, V val) : indices(ind), value(val){};
  const uint64_t *indices; // pointer into shared index pool
  V value;
};

/// The type of callback functions which receive an element.  We avoid
/// packaging the coordinates and value together as an `Element` object
/// because this helps keep code somewhat cleaner.
template <typename V>
using ElementConsumer =
    const std::function<void(const std::vector<uint64_t> &, V)> &;

/// A memory-resident sparse tensor in coordinate scheme (collection of
/// elements). This data structure is used to read a sparse tensor from
/// any external format into memory and sort the elements lexicographically
/// by indices before passing it back to the client (most packed storage
/// formats require the elements to appear in lexicographic index order).
template <typename V>
class SparseTensorCOO final {
public:
  SparseTensorCOO(const std::vector<uint64_t> &dimSizes, uint64_t capacity)
      : dimSizes(dimSizes) {
    if (capacity) {
      elements.reserve(capacity);
      indices.reserve(capacity * getRank());
    }
  }

  /// Factory method. Permutes the original dimensions according to
  /// the given ordering and expects subsequent add() calls to honor
  /// that same ordering for the given indices. The result is a
  /// fully permuted coordinate scheme.
  ///
  /// Precondition: `dimSizes` and `perm` must be valid for `rank`.
  static SparseTensorCOO<V> *newSparseTensorCOO(uint64_t rank,
                                                const uint64_t *dimSizes,
                                                const uint64_t *perm,
                                                uint64_t capacity = 0) {
    std::vector<uint64_t> permsz(rank);
    for (uint64_t r = 0; r < rank; ++r) {
      assert(dimSizes[r] > 0 && "Dimension size zero has trivial storage");
      permsz[perm[r]] = dimSizes[r];
    }
    return new SparseTensorCOO<V>(permsz, capacity);
  }

  /// Get the rank of the tensor.
  uint64_t getRank() const { return dimSizes.size(); }

  /// Getter for the dimension-sizes array.
  const std::vector<uint64_t> &getDimSizes() const { return dimSizes; }

  /// Getter for the elements array.
  const std::vector<Element<V>> &getElements() const { return elements; }

  /// Adds element as indices and value.
  void add(const std::vector<uint64_t> &ind, V val) {
    assert(!iteratorLocked && "Attempt to add() after startIterator()");
    const uint64_t *base = indices.data();
    uint64_t size = indices.size();
    uint64_t rank = getRank();
    assert(ind.size() == rank && "Element rank mismatch");
    for (uint64_t r = 0; r < rank; ++r) {
      assert(ind[r] < dimSizes[r] && "Index is too large for the dimension");
      indices.push_back(ind[r]);
    }
    // This base only changes if indices were reallocated. In that case, we
    // need to correct all previous pointers into the vector. Note that this
    // only happens if we did not set the initial capacity right, and then only
    // for every internal vector reallocation (which with the doubling rule
    // should only incur an amortized linear overhead).
    const uint64_t *newBase = indices.data();
    if (newBase != base) {
      for (uint64_t i = 0, n = elements.size(); i < n; ++i)
        elements[i].indices = newBase + (elements[i].indices - base);
      base = newBase;
    }
    // Add element as (pointer into shared index pool, value) pair.
    elements.emplace_back(base + size, val);
  }

  /// Sorts elements lexicographically by index.
  void sort() {
    assert(!iteratorLocked && "Attempt to sort() after startIterator()");
    // TODO: we may want to cache an `isSorted` bit, to avoid
    // unnecessary/redundant sorting.
    uint64_t rank = getRank();
    std::sort(elements.begin(), elements.end(),
              [rank](const Element<V> &e1, const Element<V> &e2) {
                for (uint64_t r = 0; r < rank; ++r) {
                  if (e1.indices[r] == e2.indices[r])
                    continue;
                  return e1.indices[r] < e2.indices[r];
                }
                return false;
              });
  }

  /// Switch into iterator mode.
  void startIterator() {
    iteratorLocked = true;
    iteratorPos = 0;
  }

  /// Get the next element.
  const Element<V> *getNext() {
    assert(iteratorLocked && "Attempt to getNext() before startIterator()");
    if (iteratorPos < elements.size())
      return &(elements[iteratorPos++]);
    iteratorLocked = false;
    return nullptr;
  }

private:
  const std::vector<uint64_t> dimSizes; // per-dimension sizes
  std::vector<Element<V>> elements;     // all COO elements
  std::vector<uint64_t> indices;        // shared index pool
  bool iteratorLocked = false;
  unsigned iteratorPos = 0;
};

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_SPARSETENSOR_COO_H
