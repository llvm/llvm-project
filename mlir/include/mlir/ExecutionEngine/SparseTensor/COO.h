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

#include "mlir/ExecutionEngine/SparseTensor/PermutationRef.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <functional>
#include <vector>

namespace mlir {
namespace sparse_tensor {

/// An element of a sparse tensor in coordinate-scheme representation
/// (i.e., a pair of indices and value).  For example, a rank-1 vector
/// element would look like
///   ({i}, a[i])
/// and a rank-5 tensor element would look like
///   ({i,j,k,l,m}, a[i,j,k,l,m])
///
/// The indices are represented as a (non-owning) pointer into a shared
/// pool of indices, rather than being stored directly in this object.
/// This significantly improves performance because it: (1) reduces
/// the per-element memory footprint, and (2) centralizes the memory
/// management for indices.  The only downside is that the indices
/// themselves cannot be retrieved without knowing the rank of the
/// tensor to which this element belongs (and that rank is not stored
/// in this object).
template <typename V>
struct Element final {
  Element(const uint64_t *ind, V val) : indices(ind), value(val){};
  const uint64_t *indices; // pointer into shared index pool
  V value;
};

/// Closure object for `operator<` on `Element` with a given rank.
template <typename V>
struct ElementLT final {
  ElementLT(uint64_t rank) : rank(rank) {}

  /// Compares two elements a la `operator<`.
  ///
  /// Precondition: the elements must both be valid for `rank`.
  bool operator()(const Element<V> &e1, const Element<V> &e2) const {
    for (uint64_t d = 0; d < rank; ++d) {
      if (e1.indices[d] == e2.indices[d])
        continue;
      return e1.indices[d] < e2.indices[d];
    }
    return false;
  }

  const uint64_t rank;
};

/// The type of callback functions which receive an element.  We avoid
/// packaging the coordinates and value together as an `Element` object
/// because this helps keep code somewhat cleaner.
template <typename V>
using ElementConsumer =
    const std::function<void(const std::vector<uint64_t> &, V)> &;

/// A memory-resident sparse tensor in coordinate-scheme representation
/// (a collection of `Element`s).  This data structure is used as
/// an intermediate representation; e.g., for reading sparse tensors
/// from external formats into memory, or for certain conversions between
/// different `SparseTensorStorage` formats.
///
/// This class provides all the typedefs required by the "Container"
/// concept (<https://en.cppreference.com/w/cpp/named_req/Container>);
/// however, beware that it cannot fully implement that concept since
/// it cannot have a default ctor (because the `dimSizes` field is const).
/// Thus these typedefs are provided for familiarity reasons, rather
/// than as a proper implementation of the concept.
template <typename V>
class SparseTensorCOO final {
public:
  using value_type = const Element<V>;
  using reference = value_type &;
  using const_reference = reference;
  // The types associated with `std::vector` differ significantly between
  // C++11/17 vs C++20; so we explicitly defer to whatever `std::vector`
  // says the types should be.
  using vector_type = std::vector<Element<V>>;
  using iterator = typename vector_type::const_iterator;
  using const_iterator = iterator;
  using difference_type = typename vector_type::difference_type;
  using size_type = typename vector_type::size_type;

  /// Constructs a new coordinate-scheme sparse tensor with the given
  /// sizes and initial storage capacity.
  ///
  /// Asserts:
  /// * `dimSizes` has nonzero size.
  /// * the elements of `dimSizes` are nonzero.
  explicit SparseTensorCOO(const std::vector<uint64_t> &dimSizes,
                           uint64_t capacity = 0)
      : SparseTensorCOO(dimSizes.size(), dimSizes.data(), capacity) {}

  // TODO: make a class for capturing known-valid sizes (a la PermutationRef),
  // so that `SparseTensorStorage::toCOO` can avoid redoing these assertions.
  // Also so that we can enforce the asserts *before* copying into `dimSizes`.
  //
  /// Constructs a new coordinate-scheme sparse tensor with the given
  /// sizes and initial storage capacity.
  ///
  /// Precondition: `dimSizes` must be valid for `dimRank`.
  ///
  /// Asserts:
  /// * `dimRank` is nonzero.
  /// * the elements of `dimSizes` are nonzero.
  explicit SparseTensorCOO(uint64_t dimRank, const uint64_t *dimSizes,
                           uint64_t capacity = 0)
      : dimSizes(dimSizes, dimSizes + dimRank), isSorted(true) {
    assert(dimRank > 0 && "Trivial shape is not supported");
    for (uint64_t d = 0; d < dimRank; ++d)
      assert(dimSizes[d] > 0 && "Dimension size zero has trivial storage");
    if (capacity) {
      elements.reserve(capacity);
      indices.reserve(capacity * dimRank);
    }
  }

  /// Gets the rank of the tensor.
  uint64_t getRank() const { return dimSizes.size(); }

  /// Gets the dimension-sizes array.
  const std::vector<uint64_t> &getDimSizes() const { return dimSizes; }

  /// Gets the elements array.
  const std::vector<Element<V>> &getElements() const { return elements; }

  /// Returns the `operator<` closure object for the COO's element type.
  ElementLT<V> getElementLT() const { return ElementLT<V>(getRank()); }

  /// Adds an element to the tensor.  This method does not check whether
  /// `ind` is already associated with a value, it adds it regardless.
  /// Resolving such conflicts is left up to clients of the iterator
  /// interface.
  ///
  /// This method invalidates all iterators.
  ///
  /// Asserts:
  /// * the `ind` is valid for `rank`
  /// * the elements of `ind` are valid for `dimSizes`.
  void add(const std::vector<uint64_t> &ind, V val) {
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
    // Add the new element and update the sorted bit.
    Element<V> addedElem(base + size, val);
    if (!elements.empty() && isSorted)
      isSorted = getElementLT()(elements.back(), addedElem);
    elements.push_back(addedElem);
  }

  const_iterator begin() const { return elements.cbegin(); }
  const_iterator end() const { return elements.cend(); }

  /// Sorts elements lexicographically by index.  If an index is mapped to
  /// multiple values, then the relative order of those values is unspecified.
  ///
  /// This method invalidates all iterators.
  void sort() {
    if (isSorted)
      return;
    std::sort(elements.begin(), elements.end(), getElementLT());
    isSorted = true;
  }

private:
  const std::vector<uint64_t> dimSizes; // per-dimension sizes
  std::vector<Element<V>> elements;     // all COO elements
  std::vector<uint64_t> indices;        // shared index pool
  bool isSorted;
};

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_SPARSETENSOR_COO_H
