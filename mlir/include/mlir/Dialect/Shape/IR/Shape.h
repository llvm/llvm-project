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

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
class PatternRewriter;

namespace shape {

/// Alias type for extent tensors.
RankedTensorType getExtentTensorType(MLIRContext *ctx,
                                     int64_t rank = ShapedType::kDynamicSize);

// Check if a type is an extent tensor, e.g., tensor<?xindex>.
bool isExtentTensorType(Type);

// Given an input shape Value, try to obtain the shape's values.
LogicalResult getShapeVec(Value input, SmallVectorImpl<int64_t> &shapeValues);

/// The shape descriptor type represents rank and dimension sizes.
class ShapeType : public Type::TypeBase<ShapeType, Type, TypeStorage> {
public:
  using Base::Base;
};

/// The type of a single dimension.
class SizeType : public Type::TypeBase<SizeType, Type, TypeStorage> {
public:
  using Base::Base;
};

/// The ValueShape represents a (potentially unknown) runtime value and shape.
class ValueShapeType
    : public Type::TypeBase<ValueShapeType, Type, TypeStorage> {
public:
  using Base::Base;
};

/// The Witness represents a runtime constraint, to be used as shape related
/// preconditions on code execution.
class WitnessType : public Type::TypeBase<WitnessType, Type, TypeStorage> {
public:
  using Base::Base;
};

} // namespace shape
} // namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/Shape/IR/ShapeOps.h.inc"

#include "mlir/Dialect/Shape/IR/ShapeOpsDialect.h.inc"

#endif // MLIR_SHAPE_IR_SHAPE_H
