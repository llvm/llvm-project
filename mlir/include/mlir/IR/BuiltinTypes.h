//===- BuiltinTypes.h - MLIR Builtin Type Classes ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_BUILTINTYPES_H
#define MLIR_IR_BUILTINTYPES_H

#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Support/ADTExtras.h"
#include "llvm/ADT/STLExtras.h"

namespace llvm {
class BitVector;
struct fltSemantics;
} // namespace llvm

//===----------------------------------------------------------------------===//
// Tablegen Interface Declarations
//===----------------------------------------------------------------------===//

namespace mlir {
class AffineExpr;
class AffineMap;
class FloatType;
class IndexType;
class IntegerType;
class MemRefType;
class RankedTensorType;
class StringAttr;
class TypeRange;

namespace detail {
struct FunctionTypeStorage;
struct IntegerTypeStorage;
struct TupleTypeStorage;
} // namespace detail

//===----------------------------------------------------------------------===//
// FloatType
//===----------------------------------------------------------------------===//

class FloatType : public Type {
public:
  using Type::Type;

  // Convenience factories.
  static FloatType getBF16(MLIRContext *ctx);
  static FloatType getF16(MLIRContext *ctx);
  static FloatType getF32(MLIRContext *ctx);
  static FloatType getTF32(MLIRContext *ctx);
  static FloatType getF64(MLIRContext *ctx);
  static FloatType getF80(MLIRContext *ctx);
  static FloatType getF128(MLIRContext *ctx);
  static FloatType getFloat8E5M2(MLIRContext *ctx);
  static FloatType getFloat8E4M3FN(MLIRContext *ctx);
  static FloatType getFloat8E5M2FNUZ(MLIRContext *ctx);
  static FloatType getFloat8E4M3FNUZ(MLIRContext *ctx);
  static FloatType getFloat8E4M3B11FNUZ(MLIRContext *ctx);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Type type);

  /// Return the bitwidth of this float type.
  unsigned getWidth();

  /// Return the width of the mantissa of this type.
  /// The width includes the integer bit.
  unsigned getFPMantissaWidth();

  /// Get or create a new FloatType with bitwidth scaled by `scale`.
  /// Return null if the scaled element type cannot be represented.
  FloatType scaleElementBitwidth(unsigned scale);

  /// Return the floating semantics of this float type.
  const llvm::fltSemantics &getFloatSemantics();
};

//===----------------------------------------------------------------------===//
// TensorType
//===----------------------------------------------------------------------===//

/// Tensor types represent multi-dimensional arrays, and have two variants:
/// RankedTensorType and UnrankedTensorType.
/// Note: This class attaches the ShapedType trait to act as a mixin to
///       provide many useful utility functions. This inheritance has no effect
///       on derived tensor types.
class TensorType : public Type, public ShapedType::Trait<TensorType> {
public:
  using Type::Type;

  /// Returns the element type of this tensor type.
  Type getElementType() const;

  /// Returns if this type is ranked, i.e. it has a known number of dimensions.
  bool hasRank() const;

  /// Returns the shape of this tensor type.
  ArrayRef<int64_t> getShape() const;

  /// Clone this type with the given shape and element type. If the
  /// provided shape is `std::nullopt`, the current shape of the type is used.
  TensorType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                       Type elementType) const;

  // Make sure that base class overloads are visible.
  using ShapedType::Trait<TensorType>::clone;

  /// Return a clone of this type with the given new shape and element type.
  /// The returned type is ranked, even if this type is unranked.
  RankedTensorType clone(ArrayRef<int64_t> shape, Type elementType) const;

  /// Return a clone of this type with the given new shape. The returned type
  /// is ranked, even if this type is unranked.
  RankedTensorType clone(ArrayRef<int64_t> shape) const;

  /// Return true if the specified element type is ok in a tensor.
  static bool isValidElementType(Type type);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Type type);

  /// Allow implicit conversion to ShapedType.
  operator ShapedType() const { return llvm::cast<ShapedType>(*this); }
};

//===----------------------------------------------------------------------===//
// BaseMemRefType
//===----------------------------------------------------------------------===//

/// This class provides a shared interface for ranked and unranked memref types.
/// Note: This class attaches the ShapedType trait to act as a mixin to
///       provide many useful utility functions. This inheritance has no effect
///       on derived memref types.
class BaseMemRefType : public Type, public ShapedType::Trait<BaseMemRefType> {
public:
  using Type::Type;

  /// Returns the element type of this memref type.
  Type getElementType() const;

  /// Returns if this type is ranked, i.e. it has a known number of dimensions.
  bool hasRank() const;

  /// Returns the shape of this memref type.
  ArrayRef<int64_t> getShape() const;

  /// Clone this type with the given shape and element type. If the
  /// provided shape is `std::nullopt`, the current shape of the type is used.
  BaseMemRefType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                           Type elementType) const;

  // Make sure that base class overloads are visible.
  using ShapedType::Trait<BaseMemRefType>::clone;

  /// Return a clone of this type with the given new shape and element type.
  /// The returned type is ranked, even if this type is unranked.
  MemRefType clone(ArrayRef<int64_t> shape, Type elementType) const;

  /// Return a clone of this type with the given new shape. The returned type
  /// is ranked, even if this type is unranked.
  MemRefType clone(ArrayRef<int64_t> shape) const;

  /// Return true if the specified element type is ok in a memref.
  static bool isValidElementType(Type type);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Type type);

  /// Returns the memory space in which data referred to by this memref resides.
  Attribute getMemorySpace() const;

  /// [deprecated] Returns the memory space in old raw integer representation.
  /// New `Attribute getMemorySpace()` method should be used instead.
  unsigned getMemorySpaceAsInt() const;

  /// Allow implicit conversion to ShapedType.
  operator ShapedType() const { return llvm::cast<ShapedType>(*this); }
};

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
      : quantity(quantity), scalable(scalable){};

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
// VectorDims
//===----------------------------------------------------------------------===//

/// Represents a non-owning list of vector dimensions. The underlying dimension
/// sizes and scalability flags are stored a two seperate lists to match the
/// storage of a VectorType.
class VectorDims : public VectorDim::Indexer {
public:
  using VectorDim::Indexer::Indexer;

  class Iterator : public llvm::iterator_facade_base<
                       Iterator, std::random_access_iterator_tag, VectorDim,
                       std::ptrdiff_t, VectorDim, VectorDim> {
  public:
    Iterator(VectorDim::Indexer indexer, size_t index)
        : indexer(indexer), index(index){};

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
  VectorDims(Iterator begin, Iterator end)
      : VectorDims(VectorDims(begin.getIndexer())
                       .slice(begin.getIndex(), end - begin)) {}

  VectorDims(VectorDim::Indexer indexer) : VectorDim::Indexer(indexer){};

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

  /// Chop of thie first \p n dims, and keep the remaining \p m
  /// dims.
  VectorDims slice(size_t n, size_t m) const {
    ArrayRef<int64_t> newSizes = sizes.slice(n, m);
    ArrayRef<bool> newScalableDims =
        scalableDims.empty() ? ArrayRef<bool>{} : scalableDims.slice(n, m);
    return VectorDims(newSizes, newScalableDims);
  }

  /// Drop the first \p n dims.
  VectorDims dropFront(size_t n = 1) const { return slice(n, size() - n); }

  /// Drop the last \p n dims.
  VectorDims dropBack(size_t n = 1) const { return slice(0, size() - n); }

  /// Return a copy of *this with only the first \p n elements.
  VectorDims takeFront(size_t n = 1) const {
    if (n >= size())
      return *this;
    return dropBack(size() - n);
  }

  /// Return a copy of *this with only the last \p n elements.
  VectorDims takeBack(size_t n = 1) const {
    if (n >= size())
      return *this;
    return dropFront(size() - n);
  }

  /// Return copy of *this with the first n dims matching the predicate removed.
  template <class PredicateT>
  VectorDims dropWhile(PredicateT predicate) const {
    return VectorDims(llvm::find_if_not(*this, predicate), end());
  }

  /// Returns true if one or more of the dims are scalable.
  bool hasScalableDims() const {
    return llvm::is_contained(getScalableDims(), true);
  }

  /// Check for dim equality.
  bool equals(VectorDims rhs) const {
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

inline bool operator==(VectorDims lhs, VectorDims rhs) {
  return lhs.equals(rhs);
}

inline bool operator!=(VectorDims lhs, VectorDims rhs) { return !(lhs == rhs); }

inline bool operator==(VectorDims lhs, ArrayRef<VectorDim> rhs) {
  return lhs.equals(rhs);
}

inline bool operator!=(VectorDims lhs, ArrayRef<VectorDim> rhs) {
  return !(lhs == rhs);
}

} // namespace mlir

//===----------------------------------------------------------------------===//
// Tablegen Type Declarations
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/IR/BuiltinTypes.h.inc"

namespace mlir {

//===----------------------------------------------------------------------===//
// MemRefType
//===----------------------------------------------------------------------===//

/// This is a builder type that keeps local references to arguments. Arguments
/// that are passed into the builder must outlive the builder.
class MemRefType::Builder {
public:
  // Build from another MemRefType.
  explicit Builder(MemRefType other)
      : shape(other.getShape()), elementType(other.getElementType()),
        layout(other.getLayout()), memorySpace(other.getMemorySpace()) {}

  // Build from scratch.
  Builder(ArrayRef<int64_t> shape, Type elementType)
      : shape(shape), elementType(elementType) {}

  Builder &setShape(ArrayRef<int64_t> newShape) {
    shape = newShape;
    return *this;
  }

  Builder &setElementType(Type newElementType) {
    elementType = newElementType;
    return *this;
  }

  Builder &setLayout(MemRefLayoutAttrInterface newLayout) {
    layout = newLayout;
    return *this;
  }

  Builder &setMemorySpace(Attribute newMemorySpace) {
    memorySpace = newMemorySpace;
    return *this;
  }

  operator MemRefType() {
    return MemRefType::get(shape, elementType, layout, memorySpace);
  }

private:
  ArrayRef<int64_t> shape;
  Type elementType;
  MemRefLayoutAttrInterface layout;
  Attribute memorySpace;
};

//===----------------------------------------------------------------------===//
// RankedTensorType
//===----------------------------------------------------------------------===//

/// This is a builder type that keeps local references to arguments. Arguments
/// that are passed into the builder must outlive the builder.
class RankedTensorType::Builder {
public:
  /// Build from another RankedTensorType.
  explicit Builder(RankedTensorType other)
      : shape(other.getShape()), elementType(other.getElementType()),
        encoding(other.getEncoding()) {}

  /// Build from scratch.
  Builder(ArrayRef<int64_t> shape, Type elementType, Attribute encoding)
      : shape(shape), elementType(elementType), encoding(encoding) {}

  Builder &setShape(ArrayRef<int64_t> newShape) {
    shape = newShape;
    return *this;
  }

  Builder &setElementType(Type newElementType) {
    elementType = newElementType;
    return *this;
  }

  Builder &setEncoding(Attribute newEncoding) {
    encoding = newEncoding;
    return *this;
  }

  /// Erase a dim from shape @pos.
  Builder &dropDim(unsigned pos) {
    assert(pos < shape.size() && "overflow");
    shape.erase(pos);
    return *this;
  }

  /// Insert a val into shape @pos.
  Builder &insertDim(int64_t val, unsigned pos) {
    assert(pos <= shape.size() && "overflow");
    shape.insert(pos, val);
    return *this;
  }

  operator RankedTensorType() {
    return RankedTensorType::get(shape, elementType, encoding);
  }

private:
  CopyOnWriteArrayRef<int64_t> shape;
  Type elementType;
  Attribute encoding;
};

//===----------------------------------------------------------------------===//
// VectorType
//===----------------------------------------------------------------------===//

/// This is a builder type that keeps local references to arguments. Arguments
/// that are passed into the builder must outlive the builder.
class VectorType::Builder {
public:
  /// Build from another VectorType.
  explicit Builder(VectorType other)
      : elementType(other.getElementType()), shape(other.getShape()),
        scalableDims(other.getScalableDims()) {}

  /// Build from scratch.
  Builder(ArrayRef<int64_t> shape, Type elementType,
          ArrayRef<bool> scalableDims = {})
      : elementType(elementType), shape(shape), scalableDims(scalableDims) {}

  Builder &setShape(ArrayRef<int64_t> newShape,
                    ArrayRef<bool> newIsScalableDim = {}) {
    shape = newShape;
    scalableDims = newIsScalableDim;
    return *this;
  }

  Builder &setElementType(Type newElementType) {
    elementType = newElementType;
    return *this;
  }

  /// Erase a dim from shape @pos.
  Builder &dropDim(unsigned pos) {
    assert(pos < shape.size() && "overflow");
    shape.erase(pos);
    if (!scalableDims.empty())
      scalableDims.erase(pos);
    return *this;
  }

  /// Set a dim in shape @pos to val.
  Builder &setDim(unsigned pos, int64_t val) {
    assert(pos < shape.size() && "overflow");
    shape.set(pos, val);
    return *this;
  }

  operator VectorType() {
    return VectorType::get(shape, elementType, scalableDims);
  }

private:
  Type elementType;
  CopyOnWriteArrayRef<int64_t> shape;
  CopyOnWriteArrayRef<bool> scalableDims;
};

/// Given an `originalShape` and a `reducedShape` assumed to be a subset of
/// `originalShape` with some `1` entries erased, return the set of indices
/// that specifies which of the entries of `originalShape` are dropped to obtain
/// `reducedShape`. The returned mask can be applied as a projection to
/// `originalShape` to obtain the `reducedShape`. This mask is useful to track
/// which dimensions must be kept when e.g. compute MemRef strides under
/// rank-reducing operations. Return std::nullopt if reducedShape cannot be
/// obtained by dropping only `1` entries in `originalShape`.
std::optional<llvm::SmallDenseSet<unsigned>>
computeRankReductionMask(ArrayRef<int64_t> originalShape,
                         ArrayRef<int64_t> reducedShape);

/// Enum that captures information related to verifier error conditions on
/// slice insert/extract type of ops.
enum class SliceVerificationResult {
  Success,
  RankTooLarge,
  SizeMismatch,
  ElemTypeMismatch,
  // Error codes to ops with a memory space and a layout annotation.
  MemSpaceMismatch,
  LayoutMismatch
};

/// Check if `originalType` can be rank reduced to `candidateReducedType` type
/// by dropping some dimensions with static size `1`.
/// Return `SliceVerificationResult::Success` on success or an appropriate error
/// code.
SliceVerificationResult isRankReducedType(ShapedType originalType,
                                          ShapedType candidateReducedType);

//===----------------------------------------------------------------------===//
// Deferred Method Definitions
//===----------------------------------------------------------------------===//

inline bool BaseMemRefType::classof(Type type) {
  return llvm::isa<MemRefType, UnrankedMemRefType>(type);
}

inline bool BaseMemRefType::isValidElementType(Type type) {
  return type.isIntOrIndexOrFloat() ||
         llvm::isa<ComplexType, MemRefType, VectorType, UnrankedMemRefType>(
             type) ||
         llvm::isa<MemRefElementTypeInterface>(type);
}

inline bool FloatType::classof(Type type) {
  return llvm::isa<Float8E5M2Type, Float8E4M3FNType, Float8E5M2FNUZType,
                   Float8E4M3FNUZType, Float8E4M3B11FNUZType, BFloat16Type,
                   Float16Type, FloatTF32Type, Float32Type, Float64Type,
                   Float80Type, Float128Type>(type);
}

inline FloatType FloatType::getFloat8E5M2(MLIRContext *ctx) {
  return Float8E5M2Type::get(ctx);
}

inline FloatType FloatType::getFloat8E4M3FN(MLIRContext *ctx) {
  return Float8E4M3FNType::get(ctx);
}

inline FloatType FloatType::getFloat8E5M2FNUZ(MLIRContext *ctx) {
  return Float8E5M2FNUZType::get(ctx);
}

inline FloatType FloatType::getFloat8E4M3FNUZ(MLIRContext *ctx) {
  return Float8E4M3FNUZType::get(ctx);
}

inline FloatType FloatType::getFloat8E4M3B11FNUZ(MLIRContext *ctx) {
  return Float8E4M3B11FNUZType::get(ctx);
}

inline FloatType FloatType::getBF16(MLIRContext *ctx) {
  return BFloat16Type::get(ctx);
}

inline FloatType FloatType::getF16(MLIRContext *ctx) {
  return Float16Type::get(ctx);
}

inline FloatType FloatType::getTF32(MLIRContext *ctx) {
  return FloatTF32Type::get(ctx);
}

inline FloatType FloatType::getF32(MLIRContext *ctx) {
  return Float32Type::get(ctx);
}

inline FloatType FloatType::getF64(MLIRContext *ctx) {
  return Float64Type::get(ctx);
}

inline FloatType FloatType::getF80(MLIRContext *ctx) {
  return Float80Type::get(ctx);
}

inline FloatType FloatType::getF128(MLIRContext *ctx) {
  return Float128Type::get(ctx);
}

inline bool TensorType::classof(Type type) {
  return llvm::isa<RankedTensorType, UnrankedTensorType>(type);
}

//===----------------------------------------------------------------------===//
// Type Utilities
//===----------------------------------------------------------------------===//

/// Returns the strides of the MemRef if the layout map is in strided form.
/// MemRefs with a layout map in strided form include:
///   1. empty or identity layout map, in which case the stride information is
///      the canonical form computed from sizes;
///   2. a StridedLayoutAttr layout;
///   3. any other layout that be converted into a single affine map layout of
///      the form `K + k0 * d0 + ... kn * dn`, where K and ki's are constants or
///      symbols.
///
/// A stride specification is a list of integer values that are either static
/// or dynamic (encoded with ShapedType::kDynamic). Strides encode
/// the distance in the number of elements between successive entries along a
/// particular dimension.
LogicalResult getStridesAndOffset(MemRefType t,
                                  SmallVectorImpl<int64_t> &strides,
                                  int64_t &offset);

/// Wrapper around getStridesAndOffset(MemRefType, SmallVectorImpl<int64_t>,
/// int64_t) that will assert if the logical result is not succeeded.
std::pair<SmallVector<int64_t>, int64_t> getStridesAndOffset(MemRefType t);

/// Return a version of `t` with identity layout if it can be determined
/// statically that the layout is the canonical contiguous strided layout.
/// Otherwise pass `t`'s layout into `simplifyAffineMap` and return a copy of
/// `t` with simplified layout.
MemRefType canonicalizeStridedLayout(MemRefType t);

/// Given MemRef `sizes` that are either static or dynamic, returns the
/// canonical "contiguous" strides AffineExpr. Strides are multiplicative and
/// once a dynamic dimension is encountered, all canonical strides become
/// dynamic and need to be encoded with a different symbol.
/// For canonical strides expressions, the offset is always 0 and the fastest
/// varying stride is always `1`.
///
/// Examples:
///   - memref<3x4x5xf32> has canonical stride expression
///         `20*exprs[0] + 5*exprs[1] + exprs[2]`.
///   - memref<3x?x5xf32> has canonical stride expression
///         `s0*exprs[0] + 5*exprs[1] + exprs[2]`.
///   - memref<3x4x?xf32> has canonical stride expression
///         `s1*exprs[0] + s0*exprs[1] + exprs[2]`.
AffineExpr makeCanonicalStridedLayoutExpr(ArrayRef<int64_t> sizes,
                                          ArrayRef<AffineExpr> exprs,
                                          MLIRContext *context);

/// Return the result of makeCanonicalStrudedLayoutExpr for the common case
/// where `exprs` is {d0, d1, .., d_(sizes.size()-1)}
AffineExpr makeCanonicalStridedLayoutExpr(ArrayRef<int64_t> sizes,
                                          MLIRContext *context);

/// Return "true" if the layout for `t` is compatible with strided semantics.
bool isStrided(MemRefType t);

/// Return "true" if the last dimension of the given type has a static unit
/// stride. Also return "true" for types with no strides.
bool isLastMemrefDimUnitStride(MemRefType type);

} // namespace mlir

#endif // MLIR_IR_BUILTINTYPES_H
