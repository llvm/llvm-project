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
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace shape {

namespace ShapeTypes {
enum Kind {
  Component = Type::FIRST_SHAPE_TYPE,
  Element,
  Shape,
  Size,
  ValueShape,
  Witness,
  LAST_SHAPE_TYPE = Witness
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

/// The Witness represents a runtime constraint, to be used as shape related
/// preconditions on code execution.
class WitnessType : public Type::TypeBase<WitnessType, Type> {
public:
  using Base::Base;

  static WitnessType get(MLIRContext *context) {
    return Base::get(context, ShapeTypes::Kind::Witness);
  }

  /// Support method to enable LLVM-style type casting.
  static bool kindof(unsigned kind) {
    return kind == ShapeTypes::Kind::Witness;
  }
};

#define GET_OP_CLASSES
#include "mlir/Dialect/Shape/IR/ShapeOps.h.inc"

#include "mlir/Dialect/Shape/IR/ShapeOpsDialect.h.inc"

} // namespace shape
} // namespace mlir

#endif // MLIR_SHAPE_IR_SHAPE_H
