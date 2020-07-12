//===- InferTypeOpInterface.h - Infer Type Interfaces -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the infer op interfaces defined in
// `InferTypeOpInterface.td`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_INFERTYPEOPINTERFACE_H_
#define MLIR_INTERFACES_INFERTYPEOPINTERFACE_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

/// ShapedTypeComponents that represents the components of a ShapedType.
/// The components consist of
///  - A ranked or unranked shape with the dimension specification match those
///    of ShapeType's getShape() (e.g., dynamic dimension represented using
///    ShapedType::kDynamicSize)
///  - A element type, may be unset (nullptr)
///  - A attribute, may be unset (nullptr)
/// Used by ShapedType type inferences.
class ShapedTypeComponents {
  /// Internal storage type for shape.
  using ShapeStorageT = SmallVector<int64_t, 3>;

public:
  /// Default construction is an unranked shape.
  ShapedTypeComponents() : ranked(false), elementType(nullptr), attr(nullptr){};
  ShapedTypeComponents(Type elementType)
      : ranked(false), elementType(elementType), attr(nullptr) {}
  template <typename Arg, typename = typename std::enable_if_t<
                              std::is_constructible<ShapeStorageT, Arg>::value>>
  ShapedTypeComponents(Arg &&arg, Type elementType = nullptr,
                       Attribute attr = nullptr)
      : dims(std::forward<Arg>(arg)), ranked(true), elementType(elementType),
        attr(attr) {}
  ShapedTypeComponents(ArrayRef<int64_t> vec, Type elementType = nullptr,
                       Attribute attr = nullptr)
      : dims(vec.begin(), vec.end()), ranked(true), elementType(elementType),
        attr(attr) {}

  /// Return the dimensions of the shape.
  /// Requires: shape is ranked.
  ArrayRef<int64_t> getDims() const {
    assert(ranked && "requires ranked shape");
    return dims;
  }

  /// Return whether the shape has a rank.
  bool hasRank() const { return ranked; };

  /// Return the element type component.
  Type getElementType() const { return elementType; };

  /// Return the raw attribute component.
  Attribute getAttribute() const { return attr; };

private:
  ShapeStorageT dims;
  bool ranked;
  Type elementType;
  Attribute attr;
};

namespace detail {
// Helper function to infer return tensor returns types given element and shape
// inference function.
//
// TODO: Consider generating typedefs for trait member functions if this usage
// becomes more common.
LogicalResult inferReturnTensorTypes(
    function_ref<LogicalResult(
        MLIRContext *, Optional<Location> location, ValueRange operands,
        DictionaryAttr attributes, RegionRange regions,
        SmallVectorImpl<ShapedTypeComponents> &retComponents)>
        componentTypeFn,
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes);

/// Verifies that the inferred result types match the actual result types for
/// the op. Precondition: op implements InferTypeOpInterface.
LogicalResult verifyInferredResultTypes(Operation *op);
} // namespace detail

namespace OpTrait {

/// Tensor type inference trait that constructs a tensor from the inferred
/// shape and elemental types.
/// Requires: Op implements functions of InferShapedTypeOpInterface.
template <typename ConcreteType>
class InferTensorType : public TraitBase<ConcreteType, InferTensorType> {
public:
  static LogicalResult
  inferReturnTypes(MLIRContext *context, Optional<Location> location,
                   ValueRange operands, DictionaryAttr attributes,
                   RegionRange regions,
                   SmallVectorImpl<Type> &inferredReturnTypes) {
    return ::mlir::detail::inferReturnTensorTypes(
        ConcreteType::inferReturnTypeComponents, context, location, operands,
        attributes, regions, inferredReturnTypes);
  }
};

} // namespace OpTrait
} // namespace mlir

/// Include the generated interface declarations.
#include "mlir/Interfaces/InferTypeOpInterface.h.inc"

#endif // MLIR_INTERFACES_INFERTYPEOPINTERFACE_H_
