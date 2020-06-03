//===- Shape.cpp - MLIR Shape Operations ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Shape/IR/Shape.h"

#include "mlir/Dialect/Traits.h"
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
  addTypes<ComponentType, ElementType, ShapeType, SizeType, ValueShapeType,
           WitnessType>();
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
  if (keyword == "witness")
    return WitnessType::get(getContext());

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
  case ShapeTypes::Witness:
    os << "witness";
    return;
  default:
    llvm_unreachable("unexpected 'shape' type kind");
  }
}

//===----------------------------------------------------------------------===//
// AnyOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AssumingOp
//===----------------------------------------------------------------------===//

static ParseResult parseAssumingOp(OpAsmParser &parser,
                                   OperationState &result) {
  result.regions.reserve(1);
  Region *doRegion = result.addRegion();

  auto &builder = parser.getBuilder();
  OpAsmParser::OperandType cond;
  if (parser.parseOperand(cond) ||
      parser.resolveOperand(cond, builder.getType<WitnessType>(),
                            result.operands))
    return failure();

  // Parse optional results type list.
  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();

  // Parse the region and add a terminator if elided.
  if (parser.parseRegion(*doRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();
  AssumingOp::ensureTerminator(*doRegion, parser.getBuilder(), result.location);

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

static void print(OpAsmPrinter &p, AssumingOp op) {
  bool yieldsResults = !op.results().empty();

  p << AssumingOp::getOperationName() << " " << op.witness();
  if (yieldsResults) {
    p << " -> (" << op.getResultTypes() << ")";
  }
  p.printRegion(op.doRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/yieldsResults);
  p.printOptionalAttrDict(op.getAttrs());
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

OpFoldResult BroadcastOp::fold(ArrayRef<Attribute> operands) {
  if (!operands[0] || !operands[1])
    return nullptr;
  auto lhsShape = llvm::to_vector<6>(
      operands[0].cast<DenseIntElementsAttr>().getValues<int64_t>());
  auto rhsShape = llvm::to_vector<6>(
      operands[1].cast<DenseIntElementsAttr>().getValues<int64_t>());
  SmallVector<int64_t, 6> resultShape;
  // If the shapes are not compatible, we can't fold it.
  // TODO: Fold to an "error".
  if (!OpTrait::util::getBroadcastedShape(lhsShape, rhsShape, resultShape))
    return nullptr;
  Builder builder(getContext());
  return builder.getIndexTensorAttr(resultShape);
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

OpFoldResult ConcatOp::fold(ArrayRef<Attribute> operands) {
  if (!operands[0] || !operands[1])
    return nullptr;
  auto lhsShape = llvm::to_vector<6>(
      operands[0].cast<DenseIntElementsAttr>().getValues<int64_t>());
  auto rhsShape = llvm::to_vector<6>(
      operands[1].cast<DenseIntElementsAttr>().getValues<int64_t>());
  SmallVector<int64_t, 6> resultShape;
  resultShape.append(lhsShape.begin(), lhsShape.end());
  resultShape.append(rhsShape.begin(), rhsShape.end());
  Builder builder(getContext());
  return builder.getIndexTensorAttr(resultShape);
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
  NamedAttrList dummy;
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
  result.addAttribute("shape", builder.getIndexTensorAttr(ints));

  result.types.push_back(ShapeType::get(builder.getContext()));
  return success();
}

OpFoldResult ConstShapeOp::fold(ArrayRef<Attribute>) { return shapeAttr(); }

//===----------------------------------------------------------------------===//
// ConstSizeOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstSizeOp::fold(ArrayRef<Attribute>) { return valueAttr(); }

//===----------------------------------------------------------------------===//
// IndexToSizeOp
//===----------------------------------------------------------------------===//

OpFoldResult IndexToSizeOp::fold(ArrayRef<Attribute> operands) {
  // Constant values of both types, `shape.size` and `index`, are represented as
  // `IntegerAttr`s which makes constant folding simple.
  if (Attribute arg = operands[0])
    return arg;
  return {};
}

//===----------------------------------------------------------------------===//
// FromExtentsOp
//===----------------------------------------------------------------------===//

OpFoldResult FromExtentsOp::fold(ArrayRef<Attribute> operands) {
  if (llvm::any_of(operands, [](Attribute a) { return !a; }))
    return nullptr;
  SmallVector<int64_t, 6> extents;
  for (auto attr : operands)
    extents.push_back(attr.cast<IntegerAttr>().getInt());
  Builder builder(getContext());
  return builder.getIndexTensorAttr(extents);
}

//===----------------------------------------------------------------------===//
// GetExtentOp
//===----------------------------------------------------------------------===//

OpFoldResult GetExtentOp::fold(ArrayRef<Attribute> operands) {
  auto elements = operands[0].dyn_cast_or_null<DenseIntElementsAttr>();
  if (!elements)
    return nullptr;
  uint64_t dimToGet = dim().getLimitedValue();
  // TODO: Constant fold this to some kind of constant error.
  if (dimToGet >= (uint64_t)elements.getNumElements())
    return nullptr;
  return elements.getValue({dimToGet});
}

//===----------------------------------------------------------------------===//
// NumElementsOp
//===----------------------------------------------------------------------===//

OpFoldResult NumElementsOp::fold(ArrayRef<Attribute> operands) {

  // Fold only when argument constant.
  Attribute shape = operands[0];
  if (!shape)
    return {};

  APInt product(64, 1);
  for (auto value : shape.cast<DenseIntElementsAttr>())
    product *= value;
  Builder builder(getContext());
  return builder.getIndexAttr(product.getLimitedValue());
}

//===----------------------------------------------------------------------===//
// ShapeOfOp
//===----------------------------------------------------------------------===//

OpFoldResult ShapeOfOp::fold(ArrayRef<Attribute>) {
  auto type = getOperand().getType().dyn_cast<ShapedType>();
  if (!type || !type.hasStaticShape())
    return nullptr;
  Builder builder(getContext());
  return builder.getIndexTensorAttr(type.getShape());
}

//===----------------------------------------------------------------------===//
// SizeToIndexOp
//===----------------------------------------------------------------------===//

OpFoldResult SizeToIndexOp::fold(ArrayRef<Attribute> operands) {
  // Constant values of both types, `shape.size` and `index`, are represented as
  // `IntegerAttr`s which makes constant folding simple.
  if (Attribute arg = operands[0])
    return arg;
  return {};
}

//===----------------------------------------------------------------------===//
// SplitAtOp
//===----------------------------------------------------------------------===//

LogicalResult SplitAtOp::fold(ArrayRef<Attribute> operands,
                              SmallVectorImpl<OpFoldResult> &results) {
  if (!operands[0] || !operands[1])
    return failure();
  auto shapeVec = llvm::to_vector<6>(
      operands[0].cast<DenseIntElementsAttr>().getValues<int64_t>());
  auto shape = llvm::makeArrayRef(shapeVec);
  auto splitPoint = operands[1].cast<IntegerAttr>().getInt();
  // Verify that the split point is in the correct range.
  // TODO: Constant fold to an "error".
  int64_t rank = shape.size();
  if (!(-rank <= splitPoint && splitPoint <= rank))
    return failure();
  if (splitPoint < 0)
    splitPoint += shape.size();
  Builder builder(operands[0].getContext());
  results.push_back(builder.getIndexTensorAttr(shape.take_front(splitPoint)));
  results.push_back(builder.getIndexTensorAttr(shape.drop_front(splitPoint)));
  return success();
}

//===----------------------------------------------------------------------===//
// ToExtentTensorOp
//===----------------------------------------------------------------------===//

OpFoldResult ToExtentTensorOp::fold(ArrayRef<Attribute> operands) {
  if (!operands[0])
    return nullptr;
  Builder builder(getContext());
  auto shape = llvm::to_vector<6>(
      operands[0].cast<DenseIntElementsAttr>().getValues<int64_t>());
  auto type = RankedTensorType::get({static_cast<int64_t>(shape.size())},
                                    builder.getIndexType());
  return DenseIntElementsAttr::get(type, shape);
}

namespace mlir {
namespace shape {

#define GET_OP_CLASSES
#include "mlir/Dialect/Shape/IR/ShapeOps.cpp.inc"

} // namespace shape
} // namespace mlir
