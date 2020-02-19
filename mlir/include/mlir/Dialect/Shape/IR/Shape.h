//===- Shape.h - MLIR Shape dialect -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the shape dialect that is used to describe and solve shape
// relations of MLIR operations using ShapedType.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SHAPE_IR_SHAPE_H
#define MLIR_SHAPE_IR_SHAPE_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace shape {

/// This dialect contains shape inference related operations and facilities.
class ShapeDialect : public Dialect {
public:
  /// Create the dialect in the given `context`.
  explicit ShapeDialect(MLIRContext *context);
};

namespace ShapeTypes {
enum Kind {
  Component = Type::FIRST_SHAPE_TYPE,
  Element,
  Shape,
  Size,
  ValueShape
};
} // namespace ShapeTypes

/// The component type corresponding to shape, element type and attribute.
class ComponentType : public Type::TypeBase<ComponentType, Type> {
public:
  using Base::Base;

  static ComponentType get(MLIRContext *context) {
    return Base::get(context, ShapeTypes::Kind::Component);
  }

  /// Support method to enable LLVM-style type casting.
  static bool kindof(unsigned kind) {
    return kind == ShapeTypes::Kind::Component;
  }
};

/// The element type of the shaped type.
class ElementType : public Type::TypeBase<ElementType, Type> {
public:
  using Base::Base;

  static ElementType get(MLIRContext *context) {
    return Base::get(context, ShapeTypes::Kind::Element);
  }

  /// Support method to enable LLVM-style type casting.
  static bool kindof(unsigned kind) {
    return kind == ShapeTypes::Kind::Element;
  }
};

/// The shape descriptor type represents rank and dimension sizes.
class ShapeType : public Type::TypeBase<ShapeType, Type> {
public:
  using Base::Base;

  static ShapeType get(MLIRContext *context) {
    return Base::get(context, ShapeTypes::Kind::Shape);
  }

  /// Support method to enable LLVM-style type casting.
  static bool kindof(unsigned kind) { return kind == ShapeTypes::Kind::Shape; }
};

/// The type of a single dimension.
class SizeType : public Type::TypeBase<SizeType, Type> {
public:
  using Base::Base;

  static SizeType get(MLIRContext *context) {
    return Base::get(context, ShapeTypes::Kind::Size);
  }

  /// Support method to enable LLVM-style type casting.
  static bool kindof(unsigned kind) { return kind == ShapeTypes::Kind::Size; }
};

/// The ValueShape represents a (potentially unknown) runtime value and shape.
class ValueShapeType : public Type::TypeBase<ValueShapeType, Type> {
public:
  using Base::Base;

  static ValueShapeType get(MLIRContext *context) {
    return Base::get(context, ShapeTypes::Kind::ValueShape);
  }

  /// Support method to enable LLVM-style type casting.
  static bool kindof(unsigned kind) {
    return kind == ShapeTypes::Kind::ValueShape;
  }
};

#define GET_OP_CLASSES
#include "mlir/Dialect/Shape/IR/ShapeOps.h.inc"

} // namespace shape
} // namespace mlir

#endif // MLIR_SHAPE_IR_SHAPE_H
