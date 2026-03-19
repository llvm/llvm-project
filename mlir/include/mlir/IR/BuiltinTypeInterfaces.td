//===- BuiltinTypeInterfaces.td - Builtin type interfaces --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for type interfaces that closely interact with
// attributes, types, and operations in the builtin dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_BUILTINTYPEINTERFACES_TD_
#define MLIR_IR_BUILTINTYPEINTERFACES_TD_

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// VectorElementTypeInterface
//===----------------------------------------------------------------------===//

def VectorElementTypeInterface : TypeInterface<"VectorElementTypeInterface"> {
  let cppNamespace = "::mlir";
  let description = [{
    Implementing this interface establishes a contract between this type and the
    vector type, indicating that this type can be used as element of vectors.

    Vector element types are treated as a bag of bits without any assumed
    structure. The size of an element type must be a compile-time constant.
    However, the bit-width may remain opaque or unavailable during
    transformations that do not depend on the element type.

    Note: This type interface is still evolving. It currently has no methods
    and is just used as marker to allow types to opt into being vector elements.
    This may change in the future, for example, to require types to provide
    their size or alignment given a data layout. Please post an RFC before
    adding this interface to additional types. Implementing this interface on
    downstream types is discouraged, until we specified the exact properties of
    a vector element type in more detail.
  }];
}

//===----------------------------------------------------------------------===//
// DenseElementTypeInterface
//===----------------------------------------------------------------------===//

def DenseElementTypeInterface : TypeInterface<"DenseElementType"> {
  let cppNamespace = "::mlir";
  let description = [{
    This interface allows custom types to be used as element types in
    DenseElementsAttr. Types implementing this interface define:

    1. The bit size for element storage.
    2. Helper methods for converting from/to Attribute. This assumes that there
       is a corresponding attribute for each type that implements this
       interface.

    The helper methods for converting from/to Attribute are utilized when
    parsing/printing IR or iterating over the elements via Attribute.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Return the number of bits required to store one element in dense
        storage.
        
        Note: The DenseElementsAttr infrastructure will automatically align
        every element to a full byte in storage. This limitation could be lifted
        in the future to support dense packing of non-byte-sized elements.
      }],
      /*retTy=*/"size_t",
      /*methodName=*/"getDenseElementBitSize",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Attribute deserialization / attribute factory: Convert raw storage bytes
        into an MLIR attribute. The size of `rawData` is
        "ceilDiv(getDenseElementBitSize(), 8)".
      }],
      /*retTy=*/"::mlir::Attribute",
      /*methodName=*/"convertToAttribute",
      /*args=*/(ins "::llvm::ArrayRef<char>":$rawData)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Attribute serialization: Convert an MLIR attribute into raw bytes.
        Implementations must append "getDenseElementBitSize() / 8" values to
        `result`. Return "failure" if the attribute is incompatible with this
        element type.
      }],
      /*retTy=*/"::llvm::LogicalResult",
      /*methodName=*/"convertFromAttribute",
      /*args=*/(ins "::mlir::Attribute":$attr,
                    "::llvm::SmallVectorImpl<char>&":$result)
    >,
  ];
}

//===----------------------------------------------------------------------===//
// FloatTypeInterface
//===----------------------------------------------------------------------===//

def FloatTypeInterface : TypeInterface<"FloatType",
    [DenseElementTypeInterface, VectorElementTypeInterface]> {
  let cppNamespace = "::mlir";
  let description = [{
    This type interface should be implemented by all floating-point types. It
    defines the LLVM APFloat semantics and provides a few helper functions.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the APFloat semantics for this floating-point type.
      }],
      /*retTy=*/"const ::llvm::fltSemantics &",
      /*methodName=*/"getFloatSemantics",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns a float type with bitwidth scaled by `scale`. Returns a "null"
        float type if the scaled element type cannot be represented.
      }],
      /*retTy=*/"::mlir::FloatType",
      /*methodName=*/"scaleElementBitwidth",
      /*args=*/(ins "unsigned":$scale),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return ::mlir::FloatType();"
    >
  ];

  let extraClassDeclaration = [{
    /// Return the bitwidth of this float type.
    unsigned getWidth();

    /// Return the width of the mantissa of this type.
    /// The width includes the integer bit.
    unsigned getFPMantissaWidth();
  }];

  let extraTraitClassDeclaration = [{
    /// DenseElementTypeInterface implementations for float types.
    size_t getDenseElementBitSize() const {
      return ::mlir::detail::getFloatTypeDenseElementBitSize($_type);
    }
    ::mlir::Attribute convertToAttribute(::llvm::ArrayRef<char> rawData) const {
      return ::mlir::detail::convertFloatTypeToAttribute($_type, rawData);
    }
    ::llvm::LogicalResult
    convertFromAttribute(::mlir::Attribute attr,
                         ::llvm::SmallVectorImpl<char> &result) const {
      return ::mlir::detail::convertFloatTypeFromAttribute($_type, attr, result);
    }
  }];
}

//===----------------------------------------------------------------------===//
// MemRefElementTypeInterface
//===----------------------------------------------------------------------===//

def MemRefElementTypeInterface : TypeInterface<"MemRefElementTypeInterface"> {
  let cppNamespace = "::mlir";
  let description = [{
    Indication that this type can be used as element in memref types.

    Implementing this interface establishes a contract between this type and the
    memref type indicating that this type can be used as element of ranked or
    unranked memrefs. The type is expected to:

      - model an entity stored in memory;
      - have non-zero size.

    For example, scalar values such as integers can implement this interface,
    but indicator types such as `void` or `unit` should not.

    The interface currently has no methods and is used by types to opt into
    being memref elements. This may change in the future, in particular to
    require types to provide their size or alignment given a data layout.
  }];
}

//===----------------------------------------------------------------------===//
// PtrLikeTypeInterface
//===----------------------------------------------------------------------===//

def PtrLikeTypeInterface : TypeInterface<"PtrLikeTypeInterface"> {
  let cppNamespace = "::mlir";
  let description = [{
    A ptr-like type represents an object storing a memory address. This object
    is constituted by:
    - A memory address called the base pointer. This pointer is treated as a
      bag of bits without any assumed structure. The bit-width of the base
      pointer must be a compile-time constant. However, the bit-width may remain
      opaque or unavailable during transformations that do not depend on the
      base pointer. Finally, it is considered indivisible in the sense that as
      a `PtrLikeTypeInterface` value, it has no metadata.
    - Optional metadata about the pointer. For example, the size of the  memory
      region associated with the pointer.

    Furthermore, all ptr-like types have two properties:
    - The memory space associated with the address held by the pointer.
    - An optional element type. If the element type is not specified, the
      pointer is considered opaque.
  }];
  let methods = [
    InterfaceMethod<[{
      Returns the memory space of this ptr-like type.
    }],
    "::mlir::Attribute", "getMemorySpace">,
    InterfaceMethod<[{
      Returns the element type of this ptr-like type. Note: this method can
      return `::mlir::Type()`, in which case the pointer is considered opaque.
    }],
    "::mlir::Type", "getElementType">,
    InterfaceMethod<[{
      Returns whether this ptr-like type has non-empty metadata.
    }],
    "bool", "hasPtrMetadata">,
    InterfaceMethod<[{
      Returns a clone of this type with the given memory space and element type,
      or `failure` if the type cannot be cloned with the specified arguments.
      If the pointer is opaque and `elementType` is not `std::nullopt` the
      method will return `failure`.

      If no `elementType` is provided and ptr is not opaque, the `elementType`
      of this type is used.
    }],
    "::llvm::FailureOr<::mlir::PtrLikeTypeInterface>", "clonePtrWith", (ins
      "::mlir::Attribute":$memorySpace,
      "::std::optional<::mlir::Type>":$elementType
    )>
  ];
}

//===----------------------------------------------------------------------===//
// ShapedType
//===----------------------------------------------------------------------===//

def ShapedTypeInterface : TypeInterface<"ShapedType"> {
  let cppNamespace = "::mlir";
  let description = [{
    This interface provides a common API for interacting with multi-dimensional
    container types. These types contain a shape and an element type.

    A shape is a list of sizes corresponding to the dimensions of the container.
    If the number of dimensions in the shape is unknown, the shape is "unranked".
    If the number of dimensions is known, the shape "ranked". The sizes of the
    dimensions of the shape must be positive, or kDynamic (in which case the
    size of the dimension is dynamic, or not statically known).
  }];
  let methods = [
    InterfaceMethod<[{
      Returns a clone of this type with the given shape and element type.

      If no shape is provided, the shape of this type is used. In that case, if
      this type is unranked, so is the resulting type.

      If a shape is provided, the resulting type is always ranked, even if this
      type is unranked.
    }],
    "::mlir::ShapedType", "cloneWith", (ins
      "::std::optional<::llvm::ArrayRef<int64_t>>":$shape,
      "::mlir::Type":$elementType
    )>,

    InterfaceMethod<[{
      Returns the element type of this shaped type.
    }],
    "::mlir::Type", "getElementType">,

    InterfaceMethod<[{
      Returns if this type is ranked, i.e. it has a known number of dimensions.
    }],
    "bool", "hasRank">,

    InterfaceMethod<[{
      Returns the shape of this type if it is ranked, otherwise asserts.
    }],
    "::llvm::ArrayRef<int64_t>", "getShape">,
  ];

  let extraClassDeclaration = [{
    static constexpr int64_t kDynamic =
        std::numeric_limits<int64_t>::min();

    /// Whether the given dimension size indicates a dynamic dimension.
    static constexpr bool isDynamic(int64_t dValue) {
      return dValue == kDynamic;
    }

    /// Whether the given shape has any size that indicates a dynamic dimension.
    static bool isDynamicShape(ArrayRef<int64_t> dSizes) {
      return llvm::any_of(dSizes, isDynamic);
    }

    /// Whether the given dimension size indicates a statically-sized dimension.
    static constexpr bool isStatic(int64_t dValue) {
      return dValue != kDynamic;
    }

    /// Whether the given shape has static dimensions only.
    static bool isStaticShape(ArrayRef<int64_t> dSizes) {
      return llvm::all_of(dSizes, isStatic);
    }

    /// Return the number of elements present in the given shape.
    static int64_t getNumElements(ArrayRef<int64_t> shape);

    /// Return the number of elements present in the given shape, or
    /// std::nullopt if the computation would overflow.
    static std::optional<int64_t> tryGetNumElements(ArrayRef<int64_t> shape);

    /// Return a clone of this type with the given new shape and element type.
    /// The returned type is ranked, even if this type is unranked.
    auto clone(::llvm::ArrayRef<int64_t> shape, Type elementType) {
      return cloneWith(shape, elementType);
    }

    /// Return a clone of this type with the given new shape. The returned type
    /// is ranked, even if this type is unranked.
    auto clone(::llvm::ArrayRef<int64_t> shape) {
      return cloneWith(shape, getElementType());
    }
  }];

  let extraSharedClassDeclaration = [{
    /// Return a clone of this type with the given new element type. The
    /// returned type is ranked if and only if this type is ranked. In that
    /// case, the returned type has the same shape as this type.
    auto clone(::mlir::Type elementType) {
      return $_type.cloneWith(/*shape=*/std::nullopt, elementType);
    }

    /// If an element type is an integer or a float, return its width. Otherwise,
    /// abort.
    unsigned getElementTypeBitWidth() const {
      return $_type.getElementType().getIntOrFloatBitWidth();
    }

    /// If this is a ranked type, return the rank. Otherwise, abort.
    int64_t getRank() const {
      assert($_type.hasRank() && "cannot query rank of unranked shaped type");
      return $_type.getShape().size();
    }

    /// If it has static shape, return the number of elements. Otherwise, abort.
    int64_t getNumElements() const {
      assert(hasStaticShape() && "cannot get element count of dynamic shaped type");
      return ::mlir::ShapedType::getNumElements($_type.getShape());
    }

    /// Return the number of elements, or std::nullopt if the computation would overflow.
    /// Precondition: `hasStaticShape()`, otherwise abort.
    std::optional<int64_t> tryGetNumElements() const {
      assert(hasStaticShape() && "cannot get element count of dynamic shaped type");
      return ::mlir::ShapedType::tryGetNumElements($_type.getShape());
    }

    /// Returns true if this dimension has a dynamic size (for ranked types);
    /// aborts for unranked types.
    bool isDynamicDim(unsigned idx) const {
      assert(idx < getRank() && "invalid index for shaped type");
      return ::mlir::ShapedType::isDynamic($_type.getShape()[idx]);
    }

    /// Returns true if this dimension has a static size (for ranked types);
    /// aborts for unranked types.
    bool isStaticDim(unsigned idx) const {
      assert(idx < getRank() && "invalid index for shaped type");
      return ::mlir::ShapedType::isStatic($_type.getShape()[idx]);
    }

    /// Returns if this type has a static shape, i.e. if the type is ranked and
    /// all dimensions have known size (>= 0).
    bool hasStaticShape() const {
      return $_type.hasRank() &&
             ::mlir::ShapedType::isStaticShape($_type.getShape());
    }

    /// Returns if this type has a static shape and the shape is equal to
    /// `shape` return true.
    bool hasStaticShape(::llvm::ArrayRef<int64_t> shape) const {
      return hasStaticShape() && $_type.getShape() == shape;
    }

    /// If this is a ranked type, return the number of dimensions with dynamic
    /// size. Otherwise, abort.
    size_t getNumDynamicDims() const {
      return llvm::count_if($_type.getShape(), ::mlir::ShapedType::isDynamic);
    }

    /// If this is ranked type, return the size of the specified dimension.
    /// Otherwise, abort.
    int64_t getDimSize(unsigned idx) const {
      assert(idx < getRank() && "invalid index for shaped type");
      return $_type.getShape()[idx];
    }

    /// Returns the position of the dynamic dimension relative to just the dynamic
    /// dimensions, given its `index` within the shape.
    unsigned getDynamicDimIndex(unsigned index) const {
      assert(index < getRank() && "invalid index");
      assert(::mlir::ShapedType::isDynamic(getDimSize(index)) && "invalid index");
      return llvm::count_if($_type.getShape().take_front(index),
                            ::mlir::ShapedType::isDynamic);
    }
  }];
}

#endif // MLIR_IR_BUILTINTYPEINTERFACES_TD_
