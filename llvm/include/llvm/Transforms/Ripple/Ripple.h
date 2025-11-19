//===--------------- Ripple.h - Expand RIpple intrinsics ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass expands RIpple intrinsics.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_RIPPLE_H
#define LLVM_TRANSFORMS_VECTORIZE_RIPPLE_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Error.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <utility>

namespace llvm {

class IntegerType;
class MDNode;
class raw_ostream;
template <typename T> class ArrayRef;

/// @brief Tensor shape (shape w/ fixed number of dimensions)
/// @tparam SizeTy The type used to store dimension sizes
///
/// The dimensions are ordered from innermost (dimension 0) to outermost.
template <typename SizeTy> class TensorShapeAny {
private:
  /// @brief Base size of vector structures of this object
  static constexpr unsigned BaseTensorSize = 4;

public:
  using DimSize = SizeTy;
  using Shape = SmallVector<DimSize, BaseTensorSize>;
  using const_iterator = typename Shape::const_iterator;
  using const_reverse_iterator = typename Shape::const_reverse_iterator;

  /// @brief A scalar!
  TensorShapeAny() = default;

  /// @brief Constructor moving a shape
  TensorShapeAny(Shape &&s) : shape(std::move(s)) {}

  TensorShapeAny(unsigned Rank, DimSize Val = DimSize(1)) : shape(Rank, Val) {}

  /// @brief Constructor from a container (with begin and end methods)
  /// @param shape the shape
  TensorShapeAny(ArrayRef<DimSize> &s) : shape(s) {}

  /// @brief Constructor from iterators
  /// @tparam It iterator type
  /// @param sBegin the beginning of the shape values
  /// @param sEnd the ending of the shape values
  template <typename It, typename It2,
            typename = EnableIfConvertibleToInputIterator<It>>
  TensorShapeAny(It sBegin, It sEnd) : shape(sBegin, sEnd) {}

  /// @brief Constructor with a set rank and unique dimension set to size
  /// @param rank the rank of the tensor
  /// @param dim the dimension to set the size
  /// @param size the size
  TensorShapeAny(unsigned rank, unsigned dim, DimSize size)
      : shape(rank, DimSize(1)) {
    shape[dim] = size;
  }

  /// @brief Getter for the vector shape of this value
  /// @return the vector shape
  const Shape &getShape() const { return shape; }

  /// @brief Accessor for the shape at a given index
  /// @param index the index
  /// @return the shape value at the specified index
  /// N.B. If the index is greater than the rank, a size of 1 is returned
  inline DimSize operator[](size_t index) const {
    return index < shape.size() ? shape[index] : DimSize(1);
  }

  /// @brief Check if the value is scalar
  /// @return True if this shape is scalar, false otherwise
  bool isScalar() const {
    return !std::any_of(shape.begin(), shape.end(),
                        [](auto &val) { return val > DimSize(1); });
  }

  /// @brief Check if the value is a vector
  /// @return True if this shape is a vector, false otherwise
  bool isVector() const { return !isScalar(); }

  /// @brief Returns the number of dimensions of this shape (rank)
  /// @return the rank of this shape
  unsigned rank() const { return shape.size(); }

  /// @brief Returns a flat (1D) shape of this value
  /// @return the flat shape
  DimSize flatShape() const {
    return std::accumulate(begin(), end(), DimSize(1),
                           std::multiplies<DimSize>());
  };

  /// @brief An iterator starting w/ the innermost dimension shape
  const_iterator begin() const { return shape.begin(); }
  const_iterator end() const { return shape.end(); }

  /// @brief A reverse iterator starting w/ the outermost dimension shape
  const_reverse_iterator rbegin() const { return shape.rbegin(); }
  const_reverse_iterator rend() const { return shape.rend(); }

  /// @brief Prints this shape
  /// @param O the output stream
  void print(raw_ostream &O) const;

  /// @brief Equality operator
  /// @return true if the shapes match, false otherwise
  /// N.B.: The shapes do not need to be of the same rank, only non-empty
  /// dimensions are checked for equality.
  bool operator==(const TensorShapeAny<DimSize> &other) const;

  /// @brief Inequality operator
  /// @see operator==()
  inline bool operator!=(const TensorShapeAny<DimSize> &other) const {
    return !this->operator==(other);
  }

  /// @brief Lexicographical ordering of shapes, from higher to lower dimensions
  ///
  /// For example TensorShape[32][2] > TensorShape[32][1] >
  /// TensorShape[31][4000]
  bool operator<(const TensorShapeAny<DimSize> &other) const;
  bool operator>(const TensorShapeAny<DimSize> &other) const;
  bool operator<=(const TensorShapeAny<DimSize> &other) const;
  bool operator>=(const TensorShapeAny<DimSize> &other) const;

  /// @brief Combines the shape of other into *this* to construct a broadcast
  /// shape.
  /// @param other the other shape.
  /// @return An error string or success.
  Error combineShapeBcast(const TensorShapeAny<DimSize> &other);

  /// @brief Checks that this shape can be combined with other.
  /// @param other The shape to be broadcasted to.
  /// @return A string error message or success if it can be broadcasted.
  Error canCombineWith(const TensorShapeAny<DimSize> &other) const;

  /// @brief Checks that this shape can be broadcasted to other.
  /// @param other The shape to be broadcasted to.
  /// @return A string error message or success if it can be broadcasted.
  Error isBroadcastError(const TensorShapeAny<DimSize> &other) const;

  /// @brief Reduces *this* for the indices set in the BitVector
  /// @param bv the dimensions to be reduced
  /// @returns true if any dimension was reduced, false otherwise
  bool reduceDimensions(const BitVector &bv);

  /// @brief Keeps only the dimensions set in the BitVector
  /// @param bv the dimensions to keep
  /// @returns true if any dimension was removed, false otherwise
  bool keepDimensions(const BitVector &bv);

  /// @brief Returns true if the shape is scalar after being reduced along the
  /// dimensions specified by reduction.
  /// @param reduction The set of dimensions to be reduced
  /// @return True if scalar after reduction, false otherwise
  bool reducedToScalarBy(const BitVector &reduction) const;

  /// @brief Returns an offset, in number of elements from the start of the
  /// Tensor, for the given multi-dimensional access vector.
  ///
  /// We allow indices to be out of bound for dimensions that are empty (size
  /// 1). This is useful to compute mappings from an index from a shape that is
  /// a broadcast of self and the index coming from the shape itself to compute
  /// a shuffle mask.
  ///
  /// @param coordinates Index for which the offset is computed
  /// @return the offset
  size_t getOffsetAt(ArrayRef<size_t> coordinate) const;

  /// @brief Calls the function *f* for each valid coordinate of *this* shape
  void foreachIndex(std::function<void(ArrayRef<size_t>)> f) const;

  /// @brief bit vector representing the dimensions
  ///        whose size is greater than one.
  BitVector nonEmptyDims() const;

  /// @brief applies a test to all dimensions of this tensor shape and \p other,
  ///        and reports the result as a BitVector.
  /// @param test the test to apply to this tensor's dimensions and
  ///             \p other 's dimensions (one by one)
  BitVector
  testBothDims(const TensorShapeAny<DimSize> &other,
               const std::function<bool(DimSize, DimSize)> &test) const;

  /// @brief Returns the bitset of the dimensions that are simulatenously non-1
  /// in both this shape as well as in \p other.
  /// @param other another shape
  /// @return the dimensions that are non empty in both this and other.
  BitVector bothNonEmptyDims(const TensorShapeAny<DimSize> &other) const;

  /// @brief Returns the bitset of dimensions of this that needs to be reduced
  /// before a broadcast to other is feasible.
  /// @param other another shape
  /// @return the dimensions that needs to be reduced to enable broadcasting
  /// this to other
  BitVector reductionDimensionsBeforeBroadcast(
      const TensorShapeAny<DimSize> &other) const;

  /// @brief dimensions of this tensor shape that would need a splat/broadcast
  ///        to match up those of \p other
  BitVector requiredSplat(const TensorShapeAny<SizeTy> &other) const;

  /// Create a metadata node to store this tensor shape
  MDNode *toConstMetadata(IntegerType *Ty) const;

  /// Retrieve a tensor shape if it's rank is lower or equal to Rank, nullptr
  /// otherwise or if the metadata is not present
  static std::unique_ptr<TensorShapeAny<SizeTy>>
  fromConstMetadata(unsigned Rank, const MDNode *Node);

  /// @brief Returns the result of broadcasting all the references to
  /// TensorShapes in the provided iterator. Returns an Error if the shapes are
  /// incompatible
  /// @param AllToBcast an iterator over references to TensorShapeAny
  template <typename IteratorT>
  static Expected<TensorShapeAny<DimSize>>
  broadcastShapeFromAll(llvm::iterator_range<IteratorT> AllToBcast);
  static Expected<TensorShapeAny<DimSize>>
  broadcastShapeFromAll(ArrayRef<const TensorShapeAny<DimSize> *> AllToBcast);

private:
  /// @brief Tensor shape; the innermost dimension starts at index 0
  Shape shape;
  Error checkDims(const TensorShapeAny<DimSize> &other,
                  std::function<Error(unsigned Idx, DimSize, DimSize)>) const;
};

using TensorShape = TensorShapeAny<uint64_t>;

/// @brief Convenience arrow operator to print TensorShapeAny
inline raw_ostream &operator<<(raw_ostream &OS, const TensorShape &tshape) {
  tshape.print(OS);
  return OS;
}

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_RIPPLE_H
