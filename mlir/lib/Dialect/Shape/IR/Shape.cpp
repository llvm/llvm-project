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
#include "mlir/IR/PatternMatch.h"
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

Operation *ShapeDialect::materializeConstant(OpBuilder &builder,
                                             Attribute value, Type type,
                                             Location loc) {
  if (auto shapeType = type.dyn_cast<ShapeType>()) {
    return builder.create<ConstShapeOp>(loc, type,
                                        value.cast<DenseIntElementsAttr>());
  }
  if (auto sizeType = type.dyn_cast<SizeType>()) {
    return builder.create<ConstSizeOp>(loc, type, value.cast<IntegerAttr>());
  }
  return nullptr;
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
// ConstShapeOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, ConstShapeOp &op) {
  p << "shape.const_shape ";
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"shape"});
  p << "[";
  interleaveComma(op.shape().getValues<int64_t>(), p,
                  [&](int64_t i) { p << i; });
  p << "]";
}

static ParseResult parseConstShapeOp(OpAsmParser &parser,
                                     OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  // We piggy-back on ArrayAttr parsing, though we don't internally store the
  // shape as an ArrayAttr.
  // TODO: Implement custom parser and maybe make syntax a bit more concise.
  Attribute extentsRaw;
  SmallVector<NamedAttribute, 6> dummy;
  if (parser.parseAttribute(extentsRaw, "dummy", dummy))
    return failure();
  auto extentsArray = extentsRaw.dyn_cast<ArrayAttr>();
  if (!extentsArray)
    return failure();
  SmallVector<int64_t, 6> ints;
  for (Attribute extent : extentsArray) {
    IntegerAttr attr = extent.dyn_cast<IntegerAttr>();
    if (!attr)
      return failure();
    ints.push_back(attr.getInt());
  }
  Builder &builder = parser.getBuilder();
  result.addAttribute("shape", builder.getI64TensorAttr(ints));

  result.types.push_back(ShapeType::get(builder.getContext()));
  return success();
}

OpFoldResult ConstShapeOp::fold(ArrayRef<Attribute>) { return shape(); }

LogicalResult ConstShapeOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    ArrayRef<NamedAttribute> attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(ShapeType::get(context));
  return success();
}

//===----------------------------------------------------------------------===//
// ConstSizeOp
//===----------------------------------------------------------------------===//

LogicalResult ConstSizeOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    ArrayRef<NamedAttribute> attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(SizeType::get(context));
  return success();
}

//===----------------------------------------------------------------------===//
// ShapeOfOp
//===----------------------------------------------------------------------===//

OpFoldResult ShapeOfOp::fold(ArrayRef<Attribute>) {
  auto type = getOperand().getType().dyn_cast<ShapedType>();
  if (!type || !type.hasStaticShape())
    return nullptr;
  Builder builder(getContext());
  return builder.getI64TensorAttr(type.getShape());
}

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
