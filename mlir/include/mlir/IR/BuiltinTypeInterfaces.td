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
    fixed dimensions are positive, scalable dimensions are negative, and unknown
    dimensions are kDynamic.
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
    static constexpr int64_t kDynamic = ShapeDim::kDynamic();

    /// Whether the given dimension size indicates a dynamic dimension.
    static constexpr bool isDynamic(int64_t dValue) {
      return ShapeDim(dValue).isDynamic();
    }

    /// Whether the given dimension is scalable (i.e. will be scaled a runtime
    /// multiplier).
    static constexpr bool isScalable(int64_t dValue) {
      return ShapeDim(dValue).isScalable();
    }

    /// Whether the given shape has any size that indicates a dynamic dimension.
    static bool isDynamicShape(ArrayRef<int64_t> dSizes) {
      return any_of(dSizes, [](ShapeDim dSize) { return dSize.isDynamic(); });
    }

    /// Whether the given shape has any size that indicates a scalable dimension.
    /// If any size is dynamic, the overall shape is considered dynamic, not
    /// scalable.
    static bool isScalableShape(ArrayRef<int64_t> dSizes) {
      bool hasScalableDims = false;
      for (ShapeDim dim: dSizes) {
        if (dim.isDynamic()) return false;
        hasScalableDims |= dim.isScalable();
      }
      return hasScalableDims;
    }

    /// Whether all dimensions of the shape have a static size (i.e.
    /// not dynamic or scalable).
    static bool isStaticShape(ArrayRef<int64_t> dSizes) {
       return all_of(dSizes, [](ShapeDim dSize) { return dSize.isFixed(); });
    }

    /// Return the minimum number of elements a shape could hold. For dynamic
    /// shapes this is always zero. For scalable shapes this is the number of
    /// elements when the runtime multiplier is one.
    static int64_t getMinNumElements(ArrayRef<int64_t> shape);

    /// Return the number of elements present in the given shape.
    /// Requires: isStaticShape.
    static int64_t getNumElements(ArrayRef<int64_t> shape) {
      assert(isStaticShape(shape));
      return getMinNumElements(shape);
    }

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

    /// Return the minimum number of elements this type can hold. If this has
    /// a dynamic shape that will be zero. If this has a scalable this is the
    /// number of elements when the runtime multiplier is one.
    int64_t getMinNumElements() const {
      return ::mlir::ShapedType::getMinNumElements($_type.getShape());
    }

    /// If it has static shape, return the number of elements. Otherwise, abort.
    int64_t getNumElements() const {
      assert(hasStaticShape()
        && "cannot get number of elements for scalable/dynamic size");
      return getMinNumElements();
    }

    /// Returns true if this dimension has a dynamic size (for ranked types);
    /// aborts for unranked types.
    bool isDynamicDim(unsigned idx) const {
      assert(idx < getRank() && "invalid index for shaped type");
      return ::mlir::ShapedType::isDynamic($_type.getShape()[idx]);
    }

    /// Returns true if this dimension has a scalable size (for ranked types);
    /// aborts for unranked types.
    bool isScalableDim(unsigned idx) const {
      assert(idx < getRank() && "invalid index for shaped type");
      return ::mlir::ShapedType::isScalable($_type.getShape()[idx]);
    }

    /// Returns if this type has a static shape, i.e. if the type is ranked and
    /// all dimensions have known size at runtime.
    bool hasStaticShape() const {
      return $_type.hasRank() &&
        ::mlir::ShapedType::isStaticShape($_type.getShape());
    }

    /// Returns if this type has a scalable shape, i.e. if the type is ranked
    /// and any dimension is scalable (but not dynamic).
    bool hasScalableShape() const {
      return $_type.hasRank() &&
        ::mlir::ShapedType::isScalableShape($_type.getShape());
    }

    /// Returns if this type has a dynamic shape, i.e. if the type is ranked
    /// and any dimension is dynamic.
    bool hasDynamicShape() const {
      return $_type.hasRank() &&
        ::mlir::ShapedType::isDynamicShape($_type.getShape());
    }

    /// If this is a ranked type, return the number of dimensions with dynamic
    /// size. Otherwise, abort.
    int64_t getNumDynamicDims() const {
      return llvm::count_if($_type.getShape(), ::mlir::ShapedType::isDynamic);
    }

    /// If this is a ranked type, return the number of dimensions with scalable
    /// size. Otherwise, abort.
    int64_t getNumScalableDims() const {
      return llvm::count_if($_type.getShape(), ::mlir::ShapedType::isScalable);
    }

    /// Deprecated. Using getDim() and explicitly querying for dim.fixedSize(),
    /// dim.scalableSize(), dim.minSize(), or comparing with
    /// ShapeDim::kDynamic() is preferred.
    /// Gets the size of a dimension. For dynamic dimensions this returns
    /// kDynamic, for scalable dimensions the min size, or the size for fixed
    /// dimensions.
    int64_t getDimSize(unsigned idx) const {
      ::mlir::ShapeDim dim = getDim(idx);
      if (dim.isDynamic())
        return dim;
      return dim.minSize();
    }

    /// If this is a ranked type, return the specified dimension.
    /// Otherwise, abort.
    ShapeDim getDim(unsigned idx) const {
      assert(idx < getRank() && "invalid index for shaped type");
      return $_type.getShape()[idx];
    }

    /// Returns the position of the dynamic dimension relative to just the dynamic
    /// dimensions, given its `index` within the shape.
    unsigned getDynamicDimIndex(unsigned index) const {
      assert(index < getRank() && "invalid index");
      assert(::mlir::ShapedType::isDynamic(getDim(index)) && "invalid index");
      return llvm::count_if($_type.getShape().take_front(index),
                            ::mlir::ShapedType::isDynamic);
    }
  }];
}

#endif // MLIR_IR_BUILTINTYPEINTERFACES_TD_
