//===- ScalableVectorType.h - Scalable Vector Helpers -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_SCALABLEVECTORTYPE_H
#define MLIR_SUPPORT_SCALABLEVECTORTYPE_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// VectorDim
//===----------------------------------------------------------------------===//

/// This class represents a dimension of a vector type. Unlike other ShapedTypes
/// vector dimensions can have scalable quantities, which means the dimension
/// has a known minimum size, which is scaled by a constant that is only
/// known at runtime.
class VectorDim {
public:
  explicit constexpr VectorDim(int64_t quantity, bool scalable)
      : quantity(quantity), scalable(scalable) {};

  /// Constructs a new fixed dimension.
  constexpr static VectorDim getFixed(int64_t quantity) {
    return VectorDim(quantity, false);
  }

  /// Constructs a new scalable dimension.
  constexpr static VectorDim getScalable(int64_t quantity) {
    return VectorDim(quantity, true);
  }

  /// Returns true if this dimension is scalable;
  constexpr bool isScalable() const { return scalable; }

  /// Returns true if this dimension is fixed.
  constexpr bool isFixed() const { return !isScalable(); }

  /// Returns the minimum number of elements this dimension can contain.
  constexpr int64_t getMinSize() const { return quantity; }

  /// If this dimension is fixed returns the number of elements, otherwise
  /// aborts.
  constexpr int64_t getFixedSize() const {
    assert(isFixed());
    return quantity;
  }

  constexpr bool operator==(VectorDim const &dim) const {
    return quantity == dim.quantity && scalable == dim.scalable;
  }

  constexpr bool operator!=(VectorDim const &dim) const {
    return !(*this == dim);
  }

  /// Print the dim.
  void print(raw_ostream &os) {
    if (isScalable())
      os << '[';
    os << getMinSize();
    if (isScalable())
      os << ']';
  }

  /// Helper class for indexing into a list of sizes (and possibly empty) list
  /// of scalable dimensions, extracting VectorDim elements.
  struct Indexer {
    explicit Indexer(ArrayRef<int64_t> sizes, ArrayRef<bool> scalableDims)
        : sizes(sizes), scalableDims(scalableDims) {
      assert(
          scalableDims.empty() ||
          sizes.size() == scalableDims.size() &&
              "expected `scalableDims` to be empty or match `sizes` in length");
    }

    VectorDim operator[](size_t idx) const {
      int64_t size = sizes[idx];
      bool scalable = scalableDims.empty() ? false : scalableDims[idx];
      return VectorDim(size, scalable);
    }

    ArrayRef<int64_t> sizes;
    ArrayRef<bool> scalableDims;
  };

private:
  int64_t quantity;
  bool scalable;
};

inline raw_ostream &operator<<(raw_ostream &os, VectorDim dim) {
  dim.print(os);
  return os;
}

//===----------------------------------------------------------------------===//
// VectorDimList
//===----------------------------------------------------------------------===//

/// Represents a non-owning list of vector dimensions. The underlying dimension
/// sizes and scalability flags are stored a two seperate lists to match the
/// storage of a VectorType.
class VectorDimList : public VectorDim::Indexer {
public:
  using VectorDim::Indexer::Indexer;

  class Iterator : public llvm::iterator_facade_base<
                       Iterator, std::random_access_iterator_tag, VectorDim,
                       std::ptrdiff_t, VectorDim, VectorDim> {
  public:
    Iterator(VectorDim::Indexer indexer, size_t index)
        : indexer(indexer), index(index) {};

    // Iterator boilerplate.
    ptrdiff_t operator-(const Iterator &rhs) const { return index - rhs.index; }
    bool operator==(const Iterator &rhs) const { return index == rhs.index; }
    bool operator<(const Iterator &rhs) const { return index < rhs.index; }
    Iterator &operator+=(ptrdiff_t offset) {
      index += offset;
      return *this;
    }
    Iterator &operator-=(ptrdiff_t offset) {
      index -= offset;
      return *this;
    }
    VectorDim operator*() const { return indexer[index]; }

    VectorDim::Indexer getIndexer() const { return indexer; }
    ptrdiff_t getIndex() const { return index; }

  private:
    VectorDim::Indexer indexer;
    ptrdiff_t index;
  };

  // Generic definitions.
  using value_type = VectorDim;
  using iterator = Iterator;
  using const_iterator = Iterator;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using size_type = size_t;
  using difference_type = ptrdiff_t;

  /// Construct from iterator pair.
  VectorDimList(Iterator begin, Iterator end)
      : VectorDimList(VectorDimList(begin.getIndexer())
                          .slice(begin.getIndex(), end - begin)) {}

  VectorDimList(VectorDim::Indexer indexer) : VectorDim::Indexer(indexer) {};

  /// Construct from a VectorType.
  static VectorDimList from(VectorType vectorType) {
    if (!vectorType)
      return VectorDimList({}, {});
    return VectorDimList(vectorType.getShape(), vectorType.getScalableDims());
  }

  Iterator begin() const { return Iterator(*this, 0); }
  Iterator end() const { return Iterator(*this, size()); }

  /// Check if the dims are empty.
  bool empty() const { return sizes.empty(); }

  /// Get the number of dims.
  size_t size() const { return sizes.size(); }

  /// Return the first dim.
  VectorDim front() const { return (*this)[0]; }

  /// Return the last dim.
  VectorDim back() const { return (*this)[size() - 1]; }

  /// Chop of the first \p n dims, and keep the remaining \p m dims.
  VectorDimList slice(size_t n, size_t m) const {
    ArrayRef<int64_t> newSizes = sizes.slice(n, m);
    ArrayRef<bool> newScalableDims =
        scalableDims.empty() ? ArrayRef<bool>{} : scalableDims.slice(n, m);
    return VectorDimList(newSizes, newScalableDims);
  }

  /// Drop the first \p n dims.
  VectorDimList dropFront(size_t n = 1) const { return slice(n, size() - n); }

  /// Drop the last \p n dims.
  VectorDimList dropBack(size_t n = 1) const { return slice(0, size() - n); }

  /// Return a copy of *this with only the first \p n elements.
  VectorDimList takeFront(size_t n = 1) const {
    if (n >= size())
      return *this;
    return dropBack(size() - n);
  }

  /// Return a copy of *this with only the last \p n elements.
  VectorDimList takeBack(size_t n = 1) const {
    if (n >= size())
      return *this;
    return dropFront(size() - n);
  }

  /// Return copy of *this with the first n dims matching the predicate removed.
  template <class PredicateT>
  VectorDimList dropWhile(PredicateT predicate) const {
    return VectorDimList(llvm::find_if_not(*this, predicate), end());
  }

  /// Returns true if one or more of the dims are scalable.
  bool hasScalableDims() const {
    return llvm::is_contained(getScalableDims(), true);
  }

  /// Check for dim equality.
  bool equals(VectorDimList rhs) const {
    if (size() != rhs.size())
      return false;
    return std::equal(begin(), end(), rhs.begin());
  }

  /// Check for dim equality.
  bool equals(ArrayRef<VectorDim> rhs) const {
    if (size() != rhs.size())
      return false;
    return std::equal(begin(), end(), rhs.begin());
  }

  /// Return the underlying sizes.
  ArrayRef<int64_t> getSizes() const { return sizes; }

  /// Return the underlying scalable dims.
  ArrayRef<bool> getScalableDims() const { return scalableDims; }
};

inline bool operator==(VectorDimList lhs, VectorDimList rhs) {
  return lhs.equals(rhs);
}

inline bool operator!=(VectorDimList lhs, VectorDimList rhs) {
  return !(lhs == rhs);
}

inline bool operator==(VectorDimList lhs, ArrayRef<VectorDim> rhs) {
  return lhs.equals(rhs);
}

inline bool operator!=(VectorDimList lhs, ArrayRef<VectorDim> rhs) {
  return !(lhs == rhs);
}

//===----------------------------------------------------------------------===//
// ScalableVectorType
//===----------------------------------------------------------------------===//

/// A pseudo-type that wraps a VectorType that aims to provide safe APIs for
/// working with scalable vectors. Slightly contrary to the name this class can
/// represent both fixed and scalable vectors, however, if you are only dealing
/// with fixed vectors the plain VectorType is likely more convenient.
///
/// The main difference from the regular VectorType is that vector dimensions
/// are _not_ represented as `int64_t`, which does not allow encoding the
/// scalability into the dimension. Instead, vector dimensions are represented
/// by a VectorDim class. A VectorDim stores both the size and scalability of a
/// dimension. This makes common errors like only checking the size (but not the
/// scalability) impossible (without being explicit with your intention).
///
/// To make this convenient to work with there is VectorDimList which provides
/// ArrayRef-like helper methods along with an iterator for VectorDims.
///
/// ScalableVectorType can freely converted to VectorType (and vice versa),
/// though there are two main ways to acquire a ScalableVectorType.
///
/// Assignment:
///
/// This does not check the scalability of `myVectorType`. This is valid and the
/// helpers on ScalableVectorType will function as normal.
/// ```c++
/// VectorType myVectorType = ...;
/// ScalableVectorType scalableVector = myVectorType;
/// ```
///
/// Casting:
///
/// This checks the scalability of `myVectorType`. In this case,
/// `scalableVector` will be falsy if `myVectorType` contains no scalable dims.
/// ```c++
/// VectorType myVectorType = ...;
/// auto scalableVector = dyn_cast<ScalableVectorType>(myVectorType);
/// ```
class ScalableVectorType {
public:
  using Dim = VectorDim;
  using DimList = VectorDimList;

  ScalableVectorType(VectorType vectorType) : vectorType(vectorType) {};

  /// Construct a new ScalableVectorType.
  static ScalableVectorType get(DimList shape, Type elementType) {
    return VectorType::get(shape.getSizes(), elementType,
                           shape.getScalableDims());
  }

  /// Construct a new ScalableVectorType.
  static ScalableVectorType get(ArrayRef<Dim> shape, Type elementType) {
    SmallVector<int64_t> sizes;
    SmallVector<bool> scalableDims;
    sizes.reserve(shape.size());
    scalableDims.reserve(shape.size());
    for (Dim dim : shape) {
      sizes.push_back(dim.getMinSize());
      scalableDims.push_back(dim.isScalable());
    }
    return VectorType::get(sizes, elementType, scalableDims);
  }

  inline static bool classof(Type type) {
    auto vectorType = dyn_cast_if_present<VectorType>(type);
    return vectorType && vectorType.isScalable();
  }

  /// Returns the value of the specified dimension (including scalability).
  Dim getDim(unsigned idx) const {
    assert(idx < getRank() && "invalid dim index for vector type");
    return getDims()[idx];
  }

  /// Returns the dimensions of this vector type (including scalability).
  DimList getDims() const {
    return DimList(vectorType.getShape(), vectorType.getScalableDims());
  }

  /// Returns the rank of this vector type.
  int64_t getRank() const { return vectorType.getRank(); }

  /// Returns true if the vector contains scalable dimensions.
  bool isScalable() const { return vectorType.isScalable(); }
  bool allDimsScalable() const { return vectorType.allDimsScalable(); }

  /// Returns the element type of this vector type.
  Type getElementType() const { return vectorType.getElementType(); }

  /// Clones this vector type with a new element type.
  ScalableVectorType clone(Type elementType) {
    return vectorType.clone(elementType);
  }

  operator VectorType() const { return vectorType; }

  explicit operator bool() const { return bool(vectorType); }

private:
  VectorType vectorType;
};

} // namespace mlir

#endif
