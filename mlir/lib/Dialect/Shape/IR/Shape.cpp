//===- Shape.cpp - MLIR Shape Operations ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Shape/IR/Shape.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::shape;

ShapeDialect::ShapeDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Shape/IR/ShapeOps.cpp.inc"
      >();
  addTypes<ComponentType, ElementType, ShapeType, SizeType, ValueShapeType>();
  // Allow unknown operations during prototyping and testing. As the dialect is
  // still evolving it makes it simple to start with an unregistered ops and
  // try different variants before actually defining the op.
  allowUnknownOperations();
}

/// Parse a type registered to this dialect.
Type ShapeDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();

  if (keyword == "component")
    return ComponentType::get(getContext());
  if (keyword == "element")
    return ElementType::get(getContext());
  if (keyword == "shape")
    return ShapeType::get(getContext());
  if (keyword == "size")
    return SizeType::get(getContext());
  if (keyword == "value_shape")
    return ValueShapeType::get(getContext());

  parser.emitError(parser.getNameLoc(), "unknown shape type: ") << keyword;
  return Type();
}

/// Print a type registered to this dialect.
void ShapeDialect::printType(Type type, DialectAsmPrinter &os) const {
  switch (type.getKind()) {
  case ShapeTypes::Component:
    os << "component";
    return;
  case ShapeTypes::Element:
    os << "element";
    return;
  case ShapeTypes::Size:
    os << "size";
    return;
  case ShapeTypes::Shape:
    os << "shape";
    return;
  case ShapeTypes::ValueShape:
    os << "value_shape";
    return;
  default:
    llvm_unreachable("unexpected 'shape' type kind");
  }
}

//===----------------------------------------------------------------------===//
// Constant*Op
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, ConstantOp &op) {
  p << "shape.constant ";
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"value"});

  if (op.getAttrs().size() > 1)
    p << ' ';
  p.printAttributeWithoutType(op.value());
  p << " : " << op.getType();
}

static ParseResult parseConstantOp(OpAsmParser &parser,
                                   OperationState &result) {
  Attribute valueAttr;
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  Type i64Type = parser.getBuilder().getIntegerType(64);
  if (parser.parseAttribute(valueAttr, i64Type, "value", result.attributes))
    return failure();

  Type type;
  if (parser.parseColonType(type))
    return failure();

  // Add the attribute type to the list.
  return parser.addTypeToList(type, result.types);
}

static LogicalResult verify(ConstantOp &op) { return success(); }

//===----------------------------------------------------------------------===//
// SplitAtOp
//===----------------------------------------------------------------------===//

LogicalResult SplitAtOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    ArrayRef<NamedAttribute> attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto shapeType = ShapeType::get(context);
  inferredReturnTypes.push_back(shapeType);
  inferredReturnTypes.push_back(shapeType);
  return success();
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

LogicalResult ConcatOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    ArrayRef<NamedAttribute> attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto shapeType = ShapeType::get(context);
  inferredReturnTypes.push_back(shapeType);
  return success();
}

namespace mlir {
namespace shape {

#define GET_OP_CLASSES
#include "mlir/Dialect/Shape/IR/ShapeOps.cpp.inc"

} // namespace shape
} // namespace mlir
