//===- RaggedArray.h - 2D array with different inner lengths ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
/// A 2D array where each row may have different length. Elements of each row
/// are stored contiguously, but rows don't have a fixed order in the storage.
template <typename T>
class RaggedArray {
public:
  /// Returns the number of rows in the 2D array.
  size_t size() const { return slices.size(); }

  /// Returns true if the are no rows in the 2D array. Note that an array with a
  /// non-zero number of empty rows is *NOT* empty.
  bool empty() const { return slices.empty(); }

  /// Accesses `pos`-th row.
  ArrayRef<T> operator[](size_t pos) const { return at(pos); }
  ArrayRef<T> at(size_t pos) const { return slices[pos]; }
  MutableArrayRef<T> operator[](size_t pos) { return at(pos); }
  MutableArrayRef<T> at(size_t pos) { return slices[pos]; }

  /// Iterator over rows.
  auto begin() { return slices.begin(); }
  auto begin() const { return slices.begin(); }
  auto end() { return slices.end(); }
  auto end() const { return slices.end(); }

  /// Reserve space to store `size` rows with `nestedSize` elements each.
  void reserve(size_t size, size_t nestedSize = 0) {
    slices.reserve(size);
    storage.reserve(size * nestedSize);
  }

  /// Appends the given range of elements as a new row to the 2D array. May
  /// invalidate the end iterator.
  template <typename Range>
  void push_back(Range &&elements) {
    slices.push_back(appendToStorage(std::forward<Range>(elements)));
  }

  /// Replaces the `pos`-th row in the 2D array with the given range of
  /// elements. Invalidates iterators and references to `pos`-th and all
  /// succeeding rows.
  template <typename Range>
  void replace(size_t pos, Range &&elements) {
    auto from = slices[pos].data();
    if (from != nullptr) {
      auto to = std::next(from, slices[pos].size());
      auto newFrom = storage.erase(from, to);
      // Update the array refs after the underlying storage was shifted.
      for (size_t i = pos + 1, e = size(); i < e; ++i) {
        slices[i] = MutableArrayRef<T>(newFrom, slices[i].size());
        std::advance(newFrom, slices[i].size());
      }
    }
    slices[pos] = appendToStorage(std::forward<Range>(elements));
  }

  /// Appends `num` empty rows to the array.
  void appendEmptyRows(size_t num) { slices.resize(slices.size() + num); }

private:
  /// Appends the given elements to the storage and returns an ArrayRef pointing
  /// to them in the storage.
  template <typename Range>
  MutableArrayRef<T> appendToStorage(Range &&elements) {
    size_t start = storage.size();
    llvm::append_range(storage, std::forward<Range>(elements));
    return MutableArrayRef<T>(storage).drop_front(start);
  }

  /// Outer elements of the ragged array. Each entry is a reference to a
  /// contiguous segment in the `storage` list that contains the actual
  /// elements. This allows for elements to be stored contiguously without
  /// nested vectors and for different segments to be set or replaced in any
  /// order.
  SmallVector<MutableArrayRef<T>> slices;

  /// Dense storage for ragged array elements.
  SmallVector<T> storage;
};
} // namespace mlir
