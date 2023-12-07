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
      return any_of(dSizes, [](int64_t dSize) { return isDynamic(dSize); });
    }

    /// Return the number of elements present in the given shape.
    static int64_t getNumElements(ArrayRef<int64_t> shape);

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

    /// Returns true if this dimension has a dynamic size (for ranked types);
    /// aborts for unranked types.
    bool isDynamicDim(unsigned idx) const {
      assert(idx < getRank() && "invalid index for shaped type");
      return ::mlir::ShapedType::isDynamic($_type.getShape()[idx]);
    }

    /// Returns if this type has a static shape, i.e. if the type is ranked and
    /// all dimensions have known size (>= 0).
    bool hasStaticShape() const {
      return $_type.hasRank() &&
             !::mlir::ShapedType::isDynamicShape($_type.getShape());
    }

    /// Returns if this type has a static shape and the shape is equal to
    /// `shape` return true.
    bool hasStaticShape(::llvm::ArrayRef<int64_t> shape) const {
      return hasStaticShape() && $_type.getShape() == shape;
    }

    /// If this is a ranked type, return the number of dimensions with dynamic
    /// size. Otherwise, abort.
    int64_t getNumDynamicDims() const {
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
