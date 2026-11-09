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

/// Type trait indicating that the type has value semantics.
template <typename ConcreteType>
class ValueSemantics
    : public TypeTrait::TraitBase<ConcreteType, ValueSemantics> {};

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
class BaseMemRefType : public Type,
                       public PtrLikeTypeInterface::Trait<BaseMemRefType>,
                       public ShapedType::Trait<BaseMemRefType> {
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

  /// Clone this type with the given memory space and element type. If the
  /// provided element type is `std::nullopt`, the current element type of the
  /// type is used.
  FailureOr<PtrLikeTypeInterface>
  clonePtrWith(Attribute memorySpace, std::optional<Type> elementType) const;

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

  /// Returns that this ptr-like object has non-empty ptr metadata.
  bool hasPtrMetadata() const { return true; }

  /// Allow implicit conversion to ShapedType.
  operator ShapedType() const { return llvm::cast<ShapedType>(*this); }

  /// Allow implicit conversion to PtrLikeTypeInterface.
  operator PtrLikeTypeInterface() const {
    return llvm::cast<PtrLikeTypeInterface>(*this);
  }
};

} // namespace mlir

//===----------------------------------------------------------------------===//
// Tablegen Type Declarations
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/IR/BuiltinTypes.h.inc"

namespace mlir {
#include "mlir/IR/BuiltinTypeConstraints.h.inc"

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
/// If `matchDynamic` is true, then dynamic dims in `originalShape` and
/// `reducedShape` will be considered matching with non-dynamic dims, unless
/// the non-dynamic dim is from `originalShape` and equal to 1. For example,
/// in ([1, 3, ?], [?, 5]), the mask would be {1, 0, 0}, since 3 and 5 will
/// match with the corresponding dynamic dims.
std::optional<llvm::SmallDenseSet<unsigned>>
computeRankReductionMask(ArrayRef<int64_t> originalShape,
                         ArrayRef<int64_t> reducedShape,
                         bool matchDynamic = false);

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
// Convenience wrappers for VectorType
//
// These are provided to allow idiomatic code like:
//  * isa<vector::ScalableVectorType>(type)
//===----------------------------------------------------------------------===//
/// A vector type containing at least one scalable dimension.
class ScalableVectorType : public VectorType {
public:
  using VectorType::VectorType;

  static bool classof(Type type) {
    auto vecTy = llvm::dyn_cast<VectorType>(type);
    if (!vecTy)
      return false;
    return vecTy.isScalable();
  }
};

/// A vector type with no scalable dimensions.
class FixedVectorType : public VectorType {
public:
  using VectorType::VectorType;

  static bool classof(Type type) {
    auto vecTy = llvm::dyn_cast<VectorType>(type);
    if (!vecTy)
      return false;
    return !vecTy.isScalable();
  }
};

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

inline bool TensorType::classof(Type type) {
  return llvm::isa<RankedTensorType, UnrankedTensorType>(type);
}

//===----------------------------------------------------------------------===//
// Type Utilities
//===----------------------------------------------------------------------===//

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
} // namespace mlir

#endif // MLIR_IR_BUILTINTYPES_H
