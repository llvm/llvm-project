//===- BuiltinTypes.h - MLIR Builtin Type Classes ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_BUILTINTYPES_H
#define MLIR_IR_BUILTINTYPES_H

#include "mlir/IR/Types.h"

namespace llvm {
struct fltSemantics;
} // namespace llvm

namespace mlir {
class AffineExpr;
class AffineMap;
class FloatType;
class Identifier;
class IndexType;
class IntegerType;
class Location;
class MLIRContext;
class TypeRange;

namespace detail {

struct BaseMemRefTypeStorage;
struct ComplexTypeStorage;
struct FunctionTypeStorage;
struct IntegerTypeStorage;
struct MemRefTypeStorage;
struct OpaqueTypeStorage;
struct RankedTensorTypeStorage;
struct ShapedTypeStorage;
struct TupleTypeStorage;
struct UnrankedMemRefTypeStorage;
struct UnrankedTensorTypeStorage;
struct VectorTypeStorage;

} // namespace detail

//===----------------------------------------------------------------------===//
// ComplexType
//===----------------------------------------------------------------------===//

/// The 'complex' type represents a complex number with a parameterized element
/// type, which is composed of a real and imaginary value of that element type.
///
/// The element must be a floating point or integer scalar type.
///
class ComplexType
    : public Type::TypeBase<ComplexType, Type, detail::ComplexTypeStorage> {
public:
  using Base::Base;

  /// Get or create a ComplexType with the provided element type.
  static ComplexType get(Type elementType);

  /// Get or create a ComplexType with the provided element type.  This emits
  /// and error at the specified location and returns null if the element type
  /// isn't supported.
  static ComplexType getChecked(Type elementType, Location location);

  /// Verify the construction of an integer type.
  static LogicalResult verifyConstructionInvariants(Location loc,
                                                    Type elementType);

  Type getElementType();
};

//===----------------------------------------------------------------------===//
// IndexType
//===----------------------------------------------------------------------===//

/// Index is a special integer-like type with unknown platform-dependent bit
/// width.
class IndexType : public Type::TypeBase<IndexType, Type, TypeStorage> {
public:
  using Base::Base;

  /// Get an instance of the IndexType.
  static IndexType get(MLIRContext *context);

  /// Storage bit width used for IndexType by internal compiler data structures.
  static constexpr unsigned kInternalStorageBitWidth = 64;
};

//===----------------------------------------------------------------------===//
// IntegerType
//===----------------------------------------------------------------------===//

/// Integer types can have arbitrary bitwidth up to a large fixed limit.
class IntegerType
    : public Type::TypeBase<IntegerType, Type, detail::IntegerTypeStorage> {
public:
  using Base::Base;

  /// Signedness semantics.
  enum SignednessSemantics : uint32_t {
    Signless, /// No signedness semantics
    Signed,   /// Signed integer
    Unsigned, /// Unsigned integer
  };

  /// Get or create a new IntegerType of the given width within the context.
  /// The created IntegerType is signless (i.e., no signedness semantics).
  /// Assume the width is within the allowed range and assert on failures. Use
  /// getChecked to handle failures gracefully.
  static IntegerType get(unsigned width, MLIRContext *context);

  /// Get or create a new IntegerType of the given width within the context.
  /// The created IntegerType has signedness semantics as indicated via
  /// `signedness`. Assume the width is within the allowed range and assert on
  /// failures. Use getChecked to handle failures gracefully.
  static IntegerType get(unsigned width, SignednessSemantics signedness,
                         MLIRContext *context);

  /// Get or create a new IntegerType of the given width within the context,
  /// defined at the given, potentially unknown, location.  The created
  /// IntegerType is signless (i.e., no signedness semantics). If the width is
  /// outside the allowed range, emit errors and return a null type.
  static IntegerType getChecked(unsigned width, Location location);

  /// Get or create a new IntegerType of the given width within the context,
  /// defined at the given, potentially unknown, location. The created
  /// IntegerType has signedness semantics as indicated via `signedness`. If the
  /// width is outside the allowed range, emit errors and return a null type.
  static IntegerType getChecked(unsigned width, SignednessSemantics signedness,
                                Location location);

  /// Verify the construction of an integer type.
  static LogicalResult
  verifyConstructionInvariants(Location loc, unsigned width,
                               SignednessSemantics signedness);

  /// Return the bitwidth of this integer type.
  unsigned getWidth() const;

  /// Return the signedness semantics of this integer type.
  SignednessSemantics getSignedness() const;

  /// Return true if this is a signless integer type.
  bool isSignless() const { return getSignedness() == Signless; }
  /// Return true if this is a signed integer type.
  bool isSigned() const { return getSignedness() == Signed; }
  /// Return true if this is an unsigned integer type.
  bool isUnsigned() const { return getSignedness() == Unsigned; }

  /// Integer representation maximal bitwidth.
  static constexpr unsigned kMaxWidth = 4096;
};

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

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Type type);

  /// Return the bitwidth of this float type.
  unsigned getWidth();

  /// Return the floating semantics of this float type.
  const llvm::fltSemantics &getFloatSemantics();
};

//===----------------------------------------------------------------------===//
// BFloat16Type

class BFloat16Type
    : public Type::TypeBase<BFloat16Type, FloatType, TypeStorage> {
public:
  using Base::Base;

  /// Return an instance of the bfloat16 type.
  static BFloat16Type get(MLIRContext *context);
};

inline FloatType FloatType::getBF16(MLIRContext *ctx) {
  return BFloat16Type::get(ctx);
}

//===----------------------------------------------------------------------===//
// Float16Type

class Float16Type : public Type::TypeBase<Float16Type, FloatType, TypeStorage> {
public:
  using Base::Base;

  /// Return an instance of the float16 type.
  static Float16Type get(MLIRContext *context);
};

inline FloatType FloatType::getF16(MLIRContext *ctx) {
  return Float16Type::get(ctx);
}

//===----------------------------------------------------------------------===//
// Float32Type

class Float32Type : public Type::TypeBase<Float32Type, FloatType, TypeStorage> {
public:
  using Base::Base;

  /// Return an instance of the float32 type.
  static Float32Type get(MLIRContext *context);
};

inline FloatType FloatType::getF32(MLIRContext *ctx) {
  return Float32Type::get(ctx);
}

//===----------------------------------------------------------------------===//
// Float64Type

class Float64Type : public Type::TypeBase<Float64Type, FloatType, TypeStorage> {
public:
  using Base::Base;

  /// Return an instance of the float64 type.
  static Float64Type get(MLIRContext *context);
};

inline FloatType FloatType::getF64(MLIRContext *ctx) {
  return Float64Type::get(ctx);
}

//===----------------------------------------------------------------------===//
// FunctionType
//===----------------------------------------------------------------------===//

/// Function types map from a list of inputs to a list of results.
class FunctionType
    : public Type::TypeBase<FunctionType, Type, detail::FunctionTypeStorage> {
public:
  using Base::Base;

  static FunctionType get(TypeRange inputs, TypeRange results,
                          MLIRContext *context);

  /// Input types.
  unsigned getNumInputs() const;
  Type getInput(unsigned i) const { return getInputs()[i]; }
  ArrayRef<Type> getInputs() const;

  /// Result types.
  unsigned getNumResults() const;
  Type getResult(unsigned i) const { return getResults()[i]; }
  ArrayRef<Type> getResults() const;

  /// Returns a new function type without the specified arguments and results.
  FunctionType getWithoutArgsAndResults(ArrayRef<unsigned> argIndices,
                                        ArrayRef<unsigned> resultIndices);
};

//===----------------------------------------------------------------------===//
// NoneType
//===----------------------------------------------------------------------===//

/// NoneType is a unit type, i.e. a type with exactly one possible value, where
/// its value does not have a defined dynamic representation.
class NoneType : public Type::TypeBase<NoneType, Type, TypeStorage> {
public:
  using Base::Base;

  /// Get an instance of the NoneType.
  static NoneType get(MLIRContext *context);
};

//===----------------------------------------------------------------------===//
// OpaqueType
//===----------------------------------------------------------------------===//

/// Opaque types represent types of non-registered dialects. These are types
/// represented in their raw string form, and can only usefully be tested for
/// type equality.
class OpaqueType
    : public Type::TypeBase<OpaqueType, Type, detail::OpaqueTypeStorage> {
public:
  using Base::Base;

  /// Get or create a new OpaqueType with the provided dialect and string data.
  static OpaqueType get(Identifier dialect, StringRef typeData,
                        MLIRContext *context);

  /// Get or create a new OpaqueType with the provided dialect and string data.
  /// If the given identifier is not a valid namespace for a dialect, then a
  /// null type is returned.
  static OpaqueType getChecked(Identifier dialect, StringRef typeData,
                               MLIRContext *context, Location location);

  /// Returns the dialect namespace of the opaque type.
  Identifier getDialectNamespace() const;

  /// Returns the raw type data of the opaque type.
  StringRef getTypeData() const;

  /// Verify the construction of an opaque type.
  static LogicalResult verifyConstructionInvariants(Location loc,
                                                    Identifier dialect,
                                                    StringRef typeData);
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
  using ImplType = detail::ShapedTypeStorage;
  using Type::Type;

  // TODO: merge these two special values in a single one used everywhere.
  // Unfortunately, uses of `-1` have crept deep into the codebase now and are
  // hard to track.
  static constexpr int64_t kDynamicSize = -1;
  static constexpr int64_t kDynamicStrideOrOffset =
      std::numeric_limits<int64_t>::min();

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
// VectorType
//===----------------------------------------------------------------------===//

/// Vector types represent multi-dimensional SIMD vectors, and have a fixed
/// known constant shape with one or more dimension.
class VectorType
    : public Type::TypeBase<VectorType, ShapedType, detail::VectorTypeStorage> {
public:
  using Base::Base;

  /// Get or create a new VectorType of the provided shape and element type.
  /// Assumes the arguments define a well-formed VectorType.
  static VectorType get(ArrayRef<int64_t> shape, Type elementType);

  /// Get or create a new VectorType of the provided shape and element type
  /// declared at the given, potentially unknown, location.  If the VectorType
  /// defined by the arguments would be ill-formed, emit errors and return
  /// nullptr-wrapping type.
  static VectorType getChecked(ArrayRef<int64_t> shape, Type elementType,
                               Location location);

  /// Verify the construction of a vector type.
  static LogicalResult verifyConstructionInvariants(Location loc,
                                                    ArrayRef<int64_t> shape,
                                                    Type elementType);

  /// Returns true of the given type can be used as an element of a vector type.
  /// In particular, vectors can consist of integer or float primitives.
  static bool isValidElementType(Type t) {
    return t.isa<IntegerType, FloatType>();
  }

  ArrayRef<int64_t> getShape() const;
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
// RankedTensorType

/// Ranked tensor types represent multi-dimensional arrays that have a shape
/// with a fixed number of dimensions. Each shape element can be a non-negative
/// integer or unknown (represented by -1).
class RankedTensorType
    : public Type::TypeBase<RankedTensorType, TensorType,
                            detail::RankedTensorTypeStorage> {
public:
  using Base::Base;

  /// Get or create a new RankedTensorType of the provided shape and element
  /// type. Assumes the arguments define a well-formed type.
  static RankedTensorType get(ArrayRef<int64_t> shape, Type elementType);

  /// Get or create a new RankedTensorType of the provided shape and element
  /// type declared at the given, potentially unknown, location.  If the
  /// RankedTensorType defined by the arguments would be ill-formed, emit errors
  /// and return a nullptr-wrapping type.
  static RankedTensorType getChecked(ArrayRef<int64_t> shape, Type elementType,
                                     Location location);

  /// Verify the construction of a ranked tensor type.
  static LogicalResult verifyConstructionInvariants(Location loc,
                                                    ArrayRef<int64_t> shape,
                                                    Type elementType);

  ArrayRef<int64_t> getShape() const;
};

//===----------------------------------------------------------------------===//
// UnrankedTensorType

/// Unranked tensor types represent multi-dimensional arrays that have an
/// unknown shape.
class UnrankedTensorType
    : public Type::TypeBase<UnrankedTensorType, TensorType,
                            detail::UnrankedTensorTypeStorage> {
public:
  using Base::Base;

  /// Get or create a new UnrankedTensorType of the provided shape and element
  /// type. Assumes the arguments define a well-formed type.
  static UnrankedTensorType get(Type elementType);

  /// Get or create a new UnrankedTensorType of the provided shape and element
  /// type declared at the given, potentially unknown, location.  If the
  /// UnrankedTensorType defined by the arguments would be ill-formed, emit
  /// errors and return a nullptr-wrapping type.
  static UnrankedTensorType getChecked(Type elementType, Location location);

  /// Verify the construction of a unranked tensor type.
  static LogicalResult verifyConstructionInvariants(Location loc,
                                                    Type elementType);

  ArrayRef<int64_t> getShape() const { return llvm::None; }
};

//===----------------------------------------------------------------------===//
// BaseMemRefType
//===----------------------------------------------------------------------===//

/// Base MemRef for Ranked and Unranked variants
class BaseMemRefType : public ShapedType {
public:
  using ImplType = detail::BaseMemRefTypeStorage;
  using ShapedType::ShapedType;

  /// Return true if the specified element type is ok in a memref.
  static bool isValidElementType(Type type) {
    return type.isIntOrIndexOrFloat() || type.isa<VectorType, ComplexType>();
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Type type);

  /// Returns the memory space in which data referred to by this memref resides.
  unsigned getMemorySpace() const;
};

//===----------------------------------------------------------------------===//
// MemRefType

/// MemRef types represent a region of memory that have a shape with a fixed
/// number of dimensions. Each shape element can be a non-negative integer or
/// unknown (represented by -1). MemRef types also have an affine map
/// composition, represented as an array AffineMap pointers.
class MemRefType : public Type::TypeBase<MemRefType, BaseMemRefType,
                                         detail::MemRefTypeStorage> {
public:
  /// This is a builder type that keeps local references to arguments. Arguments
  /// that are passed into the builder must out-live the builder.
  class Builder {
  public:
    // Build from another MemRefType.
    explicit Builder(MemRefType other)
        : shape(other.getShape()), elementType(other.getElementType()),
          affineMaps(other.getAffineMaps()),
          memorySpace(other.getMemorySpace()) {}

    // Build from scratch.
    Builder(ArrayRef<int64_t> shape, Type elementType)
        : shape(shape), elementType(elementType), affineMaps(), memorySpace(0) {
    }

    Builder &setShape(ArrayRef<int64_t> newShape) {
      shape = newShape;
      return *this;
    }

    Builder &setElementType(Type newElementType) {
      elementType = newElementType;
      return *this;
    }

    Builder &setAffineMaps(ArrayRef<AffineMap> newAffineMaps) {
      affineMaps = newAffineMaps;
      return *this;
    }

    Builder &setMemorySpace(unsigned newMemorySpace) {
      memorySpace = newMemorySpace;
      return *this;
    }

    operator MemRefType() {
      return MemRefType::get(shape, elementType, affineMaps, memorySpace);
    }

  private:
    ArrayRef<int64_t> shape;
    Type elementType;
    ArrayRef<AffineMap> affineMaps;
    unsigned memorySpace;
  };

  using Base::Base;

  /// Get or create a new MemRefType based on shape, element type, affine
  /// map composition, and memory space.  Assumes the arguments define a
  /// well-formed MemRef type.  Use getChecked to gracefully handle MemRefType
  /// construction failures.
  static MemRefType get(ArrayRef<int64_t> shape, Type elementType,
                        ArrayRef<AffineMap> affineMapComposition = {},
                        unsigned memorySpace = 0);

  /// Get or create a new MemRefType based on shape, element type, affine
  /// map composition, and memory space declared at the given location.
  /// If the location is unknown, the last argument should be an instance of
  /// UnknownLoc.  If the MemRefType defined by the arguments would be
  /// ill-formed, emits errors (to the handler registered with the context or to
  /// the error stream) and returns nullptr.
  static MemRefType getChecked(ArrayRef<int64_t> shape, Type elementType,
                               ArrayRef<AffineMap> affineMapComposition,
                               unsigned memorySpace, Location location);

  ArrayRef<int64_t> getShape() const;

  /// Returns an array of affine map pointers representing the memref affine
  /// map composition.
  ArrayRef<AffineMap> getAffineMaps() const;

  // TODO: merge these two special values in a single one used everywhere.
  // Unfortunately, uses of `-1` have crept deep into the codebase now and are
  // hard to track.
  static int64_t getDynamicStrideOrOffset() {
    return ShapedType::kDynamicStrideOrOffset;
  }

private:
  /// Get or create a new MemRefType defined by the arguments.  If the resulting
  /// type would be ill-formed, return nullptr.  If the location is provided,
  /// emit detailed error messages.
  static MemRefType getImpl(ArrayRef<int64_t> shape, Type elementType,
                            ArrayRef<AffineMap> affineMapComposition,
                            unsigned memorySpace, Optional<Location> location);
  using Base::getImpl;
};

//===----------------------------------------------------------------------===//
// UnrankedMemRefType

/// Unranked MemRef type represent multi-dimensional MemRefs that
/// have an unknown rank.
class UnrankedMemRefType
    : public Type::TypeBase<UnrankedMemRefType, BaseMemRefType,
                            detail::UnrankedMemRefTypeStorage> {
public:
  using Base::Base;

  /// Get or create a new UnrankedMemRefType of the provided element
  /// type and memory space
  static UnrankedMemRefType get(Type elementType, unsigned memorySpace);

  /// Get or create a new UnrankedMemRefType of the provided element
  /// type and memory space declared at the given, potentially unknown,
  /// location. If the UnrankedMemRefType defined by the arguments would be
  /// ill-formed, emit errors and return a nullptr-wrapping type.
  static UnrankedMemRefType getChecked(Type elementType, unsigned memorySpace,
                                       Location location);

  /// Verify the construction of a unranked memref type.
  static LogicalResult verifyConstructionInvariants(Location loc,
                                                    Type elementType,
                                                    unsigned memorySpace);

  ArrayRef<int64_t> getShape() const { return llvm::None; }
};

//===----------------------------------------------------------------------===//
// TupleType
//===----------------------------------------------------------------------===//

/// Tuple types represent a collection of other types. Note: This type merely
/// provides a common mechanism for representing tuples in MLIR. It is up to
/// dialect authors to provides operations for manipulating them, e.g.
/// extract_tuple_element. When possible, users should prefer multi-result
/// operations in the place of tuples.
class TupleType
    : public Type::TypeBase<TupleType, Type, detail::TupleTypeStorage> {
public:
  using Base::Base;

  /// Get or create a new TupleType with the provided element types. Assumes the
  /// arguments define a well-formed type.
  static TupleType get(TypeRange elementTypes, MLIRContext *context);

  /// Get or create an empty tuple type.
  static TupleType get(MLIRContext *context);

  /// Return the elements types for this tuple.
  ArrayRef<Type> getTypes() const;

  /// Accumulate the types contained in this tuple and tuples nested within it.
  /// Note that this only flattens nested tuples, not any other container type,
  /// e.g. a tuple<i32, tensor<i32>, tuple<f32, tuple<i64>>> is flattened to
  /// (i32, tensor<i32>, f32, i64)
  void getFlattenedTypes(SmallVectorImpl<Type> &types);

  /// Return the number of held types.
  size_t size() const;

  /// Iterate over the held elements.
  using iterator = ArrayRef<Type>::iterator;
  iterator begin() const { return getTypes().begin(); }
  iterator end() const { return getTypes().end(); }

  /// Return the element type at index 'index'.
  Type getType(size_t index) const {
    assert(index < size() && "invalid index for tuple type");
    return getTypes()[index];
  }
};

//===----------------------------------------------------------------------===//
// Deferred Method Definitions
//===----------------------------------------------------------------------===//

inline bool BaseMemRefType::classof(Type type) {
  return type.isa<MemRefType, UnrankedMemRefType>();
}

inline bool FloatType::classof(Type type) {
  return type.isa<BFloat16Type, Float16Type, Float32Type, Float64Type>();
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
/// MemRefs with layout maps in strided form include:
///   1. empty or identity layout map, in which case the stride information is
///      the canonical form computed from sizes;
///   2. single affine map layout of the form `K + k0 * d0 + ... kn * dn`,
///      where K and ki's are constants or symbols.
///
/// A stride specification is a list of integer values that are either static
/// or dynamic (encoded with getDynamicStrideOrOffset()). Strides encode the
/// distance in the number of elements between successive entries along a
/// particular dimension. For example, `memref<42x16xf32, (64 * d0 + d1)>`
/// specifies a view into a non-contiguous memory region of `42` by `16` `f32`
/// elements in which the distance between two consecutive elements along the
/// outer dimension is `1` and the distance between two consecutive elements
/// along the inner dimension is `64`.
///
/// Returns whether a simple strided form can be extracted from the composition
/// of the layout map.
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

} // end namespace mlir

#endif // MLIR_IR_BUILTINTYPES_H
