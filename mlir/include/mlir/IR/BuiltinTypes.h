//===- BuiltinTypes.h - MLIR Builtin Type Classes ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_BUILTINTYPES_H
#define MLIR_IR_BUILTINTYPES_H

#include "BuiltinAttributeInterfaces.h"
#include "SubElementInterfaces.h"

namespace llvm {
struct fltSemantics;
} // namespace llvm

namespace mlir {
class AffineExpr;
class AffineMap;
class FloatType;
class IndexType;
class IntegerType;
class StringAttr;
class TypeRange;

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
  static FloatType getF64(MLIRContext *ctx);
  static FloatType getF80(MLIRContext *ctx);
  static FloatType getF128(MLIRContext *ctx);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Type type);

  /// Return the bitwidth of this float type.
  unsigned getWidth();

  /// Get or create a new FloatType with bitwidth scaled by `scale`.
  /// Return null if the scaled element type cannot be represented.
  FloatType scaleElementBitwidth(unsigned scale);

  /// Return the floating semantics of this float type.
  const llvm::fltSemantics &getFloatSemantics();
};

//===----------------------------------------------------------------------===//
// ShapedType
//===----------------------------------------------------------------------===//

/// This is a common base class between Vector, UnrankedTensor, RankedTensor,
/// and MemRef types because they share behavior and semantics around shape,
/// rank, and fixed element type. Any type with these semantics should inherit
/// from ShapedType.
class ShapedType : public Type {
public:
  using Type::Type;

  // TODO: merge these two special values in a single one used everywhere.
  // Unfortunately, uses of `-1` have crept deep into the codebase now and are
  // hard to track.
  static constexpr int64_t kDynamicSize = -1;
  static constexpr int64_t kDynamicStrideOrOffset =
      std::numeric_limits<int64_t>::min();

  /// Return clone of this type with new shape and element type.
  ShapedType clone(ArrayRef<int64_t> shape, Type elementType);
  ShapedType clone(ArrayRef<int64_t> shape);
  ShapedType clone(Type elementType);

  /// Return the element type.
  Type getElementType() const;

  /// If an element type is an integer or a float, return its width. Otherwise,
  /// abort.
  unsigned getElementTypeBitWidth() const;

  /// If it has static shape, return the number of elements. Otherwise, abort.
  int64_t getNumElements() const;

  /// If this is a ranked type, return the rank. Otherwise, abort.
  int64_t getRank() const;

  /// Whether or not this is a ranked type. Memrefs, vectors and ranked tensors
  /// have a rank, while unranked tensors do not.
  bool hasRank() const;

  /// If this is a ranked type, return the shape. Otherwise, abort.
  ArrayRef<int64_t> getShape() const;

  /// If this is unranked type or any dimension has unknown size (<0), it
  /// doesn't have static shape. If all dimensions have known size (>= 0), it
  /// has static shape.
  bool hasStaticShape() const;

  /// If this has a static shape and the shape is equal to `shape` return true.
  bool hasStaticShape(ArrayRef<int64_t> shape) const;

  /// If this is a ranked type, return the number of dimensions with dynamic
  /// size. Otherwise, abort.
  int64_t getNumDynamicDims() const;

  /// If this is ranked type, return the size of the specified dimension.
  /// Otherwise, abort.
  int64_t getDimSize(unsigned idx) const;

  /// Returns true if this dimension has a dynamic size (for ranked types);
  /// aborts for unranked types.
  bool isDynamicDim(unsigned idx) const;

  /// Returns the position of the dynamic dimension relative to just the dynamic
  /// dimensions, given its `index` within the shape.
  unsigned getDynamicDimIndex(unsigned index) const;

  /// Get the total amount of bits occupied by a value of this type.  This does
  /// not take into account any memory layout or widening constraints, e.g. a
  /// vector<3xi57> is reported to occupy 3x57=171 bit, even though in practice
  /// it will likely be stored as in a 4xi64 vector register.  Fail an assertion
  /// if the size cannot be computed statically, i.e. if the type has a dynamic
  /// shape or if its elemental type does not have a known bit width.
  int64_t getSizeInBits() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Type type);

  /// Whether the given dimension size indicates a dynamic dimension.
  static constexpr bool isDynamic(int64_t dSize) {
    return dSize == kDynamicSize;
  }
  static constexpr bool isDynamicStrideOrOffset(int64_t dStrideOrOffset) {
    return dStrideOrOffset == kDynamicStrideOrOffset;
  }
};

//===----------------------------------------------------------------------===//
// TensorType
//===----------------------------------------------------------------------===//

/// Tensor types represent multi-dimensional arrays, and have two variants:
/// RankedTensorType and UnrankedTensorType.
class TensorType : public ShapedType {
public:
  using ShapedType::ShapedType;

  /// Return true if the specified element type is ok in a tensor.
  static bool isValidElementType(Type type);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Type type);
};

//===----------------------------------------------------------------------===//
// BaseMemRefType
//===----------------------------------------------------------------------===//

/// Base MemRef for Ranked and Unranked variants
class BaseMemRefType : public ShapedType {
public:
  using ShapedType::ShapedType;

  /// Return true if the specified element type is ok in a memref.
  static bool isValidElementType(Type type);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Type type);

  /// Returns the memory space in which data referred to by this memref resides.
  Attribute getMemorySpace() const;

  /// [deprecated] Returns the memory space in old raw integer representation.
  /// New `Attribute getMemorySpace()` method should be used instead.
  unsigned getMemorySpaceAsInt() const;
};

} // namespace mlir

//===----------------------------------------------------------------------===//
// Tablegen Type Declarations
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/IR/BuiltinTypes.h.inc"

//===----------------------------------------------------------------------===//
// Tablegen Interface Declarations
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypeInterfaces.h.inc"

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

  // [deprecated] `setMemorySpace(Attribute)` should be used instead.
  Builder &setMemorySpace(unsigned newMemorySpace);

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
    if (storage.empty())
      storage.append(shape.begin(), shape.end());
    storage.erase(storage.begin() + pos);
    shape = {storage.data(), storage.size()};
    return *this;
  }

  operator RankedTensorType() {
    return RankedTensorType::get(shape, elementType, encoding);
  }

private:
  ArrayRef<int64_t> shape;
  // Owning shape data for copy-on-write operations.
  SmallVector<int64_t> storage;
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
      : shape(other.getShape()), elementType(other.getElementType()),
        numScalableDims(other.getNumScalableDims()) {}

  /// Build from scratch.
  Builder(ArrayRef<int64_t> shape, Type elementType,
          unsigned numScalableDims = 0)
      : shape(shape), elementType(elementType),
        numScalableDims(numScalableDims) {}

  Builder &setShape(ArrayRef<int64_t> newShape,
                    unsigned newNumScalableDims = 0) {
    numScalableDims = newNumScalableDims;
    shape = newShape;
    return *this;
  }

  Builder &setElementType(Type newElementType) {
    elementType = newElementType;
    return *this;
  }

  /// Erase a dim from shape @pos.
  Builder &dropDim(unsigned pos) {
    assert(pos < shape.size() && "overflow");
    if (pos >= shape.size() - numScalableDims)
      numScalableDims--;
    if (storage.empty())
      storage.append(shape.begin(), shape.end());
    storage.erase(storage.begin() + pos);
    shape = {storage.data(), storage.size()};
    return *this;
  }

  /// In the particular case where the vector has a single dimension that we
  /// drop, return the scalar element type.
  // TODO: unify once we have a VectorType that supports 0-D.
  operator Type() {
    if (shape.empty())
      return elementType;
    return VectorType::get(shape, elementType, numScalableDims);
  }

private:
  ArrayRef<int64_t> shape;
  // Owning shape data for copy-on-write operations.
  SmallVector<int64_t> storage;
  Type elementType;
  unsigned numScalableDims;
};

/// Given an `originalShape` and a `reducedShape` assumed to be a subset of
/// `originalShape` with some `1` entries erased, return the set of indices
/// that specifies which of the entries of `originalShape` are dropped to obtain
/// `reducedShape`. The returned mask can be applied as a projection to
/// `originalShape` to obtain the `reducedShape`. This mask is useful to track
/// which dimensions must be kept when e.g. compute MemRef strides under
/// rank-reducing operations. Return None if reducedShape cannot be obtained
/// by dropping only `1` entries in `originalShape`.
llvm::Optional<llvm::SmallDenseSet<unsigned>>
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
  return type.isa<MemRefType, UnrankedMemRefType>();
}

inline bool BaseMemRefType::isValidElementType(Type type) {
  return type.isIntOrIndexOrFloat() ||
         type.isa<ComplexType, MemRefType, VectorType, UnrankedMemRefType>() ||
         type.isa<MemRefElementTypeInterface>();
}

inline bool FloatType::classof(Type type) {
  return type.isa<BFloat16Type, Float16Type, Float32Type, Float64Type,
                  Float80Type, Float128Type>();
}

inline FloatType FloatType::getBF16(MLIRContext *ctx) {
  return BFloat16Type::get(ctx);
}

inline FloatType FloatType::getF16(MLIRContext *ctx) {
  return Float16Type::get(ctx);
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

inline bool ShapedType::classof(Type type) {
  return type.isa<RankedTensorType, VectorType, UnrankedTensorType,
                  UnrankedMemRefType, MemRefType>();
}

inline bool TensorType::classof(Type type) {
  return type.isa<RankedTensorType, UnrankedTensorType>();
}

//===----------------------------------------------------------------------===//
// Type Utilities
//===----------------------------------------------------------------------===//

/// Returns the strides of the MemRef if the layout map is in strided form.
/// MemRefs with a layout map in strided form include:
///   1. empty or identity layout map, in which case the stride information is
///      the canonical form computed from sizes;
///   2. single affine map layout of the form `K + k0 * d0 + ... kn * dn`,
///      where K and ki's are constants or symbols.
///
/// A stride specification is a list of integer values that are either static
/// or dynamic (encoded with getDynamicStrideOrOffset()). Strides encode the
/// distance in the number of elements between successive entries along a
/// particular dimension.
///
/// For example, `memref<42x16xf32, (64 * d0 + d1)>` specifies a view into a
/// non-contiguous memory region of `42` by `16` `f32` elements in which the
/// distance between two consecutive elements along the outer dimension is `1`
/// and the distance between two consecutive elements along the inner dimension
/// is `64`.
///
/// The convention is that the strides for dimensions d0, .. dn appear in
/// order to make indexing intuitive into the result.
LogicalResult getStridesAndOffset(MemRefType t,
                                  SmallVectorImpl<int64_t> &strides,
                                  int64_t &offset);
LogicalResult getStridesAndOffset(MemRefType t,
                                  SmallVectorImpl<AffineExpr> &strides,
                                  AffineExpr &offset);

/// Given a list of strides (in which MemRefType::getDynamicStrideOrOffset()
/// represents a dynamic value), return the single result AffineMap which
/// represents the linearized strided layout map. Dimensions correspond to the
/// offset followed by the strides in order. Symbols are inserted for each
/// dynamic dimension in order. A stride cannot take value `0`.
///
/// Examples:
/// =========
///
///   1. For offset: 0 strides: ?, ?, 1 return
///         (i, j, k)[M, N]->(M * i + N * j + k)
///
///   2. For offset: 3 strides: 32, ?, 16 return
///         (i, j, k)[M]->(3 + 32 * i + M * j + 16 * k)
///
///   3. For offset: ? strides: ?, ?, ? return
///         (i, j, k)[off, M, N, P]->(off + M * i + N * j + P * k)
AffineMap makeStridedLinearLayoutMap(ArrayRef<int64_t> strides, int64_t offset,
                                     MLIRContext *context);

/// Return a version of `t` with identity layout if it can be determined
/// statically that the layout is the canonical contiguous strided layout.
/// Otherwise pass `t`'s layout into `simplifyAffineMap` and return a copy of
/// `t` with simplified layout.
MemRefType canonicalizeStridedLayout(MemRefType t);

/// Return a version of `t` with a layout that has all dynamic offset and
/// strides. This is used to erase the static layout.
MemRefType eraseStridedLayout(MemRefType t);

/// Given MemRef `sizes` that are either static or dynamic, returns the
/// canonical "contiguous" strides AffineExpr. Strides are multiplicative and
/// once a dynamic dimension is encountered, all canonical strides become
/// dynamic and need to be encoded with a different symbol.
/// For canonical strides expressions, the offset is always 0 and and fastest
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

/// Return true if the layout for `t` is compatible with strided semantics.
bool isStrided(MemRefType t);

/// Return the layout map in strided linear layout AffineMap form.
/// Return null if the layout is not compatible with a strided layout.
AffineMap getStridedLinearLayoutMap(MemRefType t);

/// Helper determining if a memref is static-shape and contiguous-row-major
/// layout, while still allowing for an arbitrary offset (any static or
/// dynamic value).
bool isStaticShapeAndContiguousRowMajor(MemRefType memrefType);

} // namespace mlir

#endif // MLIR_IR_BUILTINTYPES_H
