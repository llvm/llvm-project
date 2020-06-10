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
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::shape;

namespace {
#include "ShapeCanonicalization.inc"
}

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
  if (auto shapeType = type.dyn_cast<ShapeType>())
    return builder.create<ConstShapeOp>(loc, type,
                                        value.cast<DenseIntElementsAttr>());
  if (auto sizeType = type.dyn_cast<SizeType>())
    return builder.create<ConstSizeOp>(loc, type, value.cast<IntegerAttr>());
  if (auto witnessType = type.dyn_cast<WitnessType>())
    return builder.create<ConstWitnessOp>(loc, type, value.cast<BoolAttr>());
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

// TODO: Canonicalization should be implemented for shapes that can be
// determined through mixtures of the known dimensions of the inputs.
OpFoldResult AnyOp::fold(ArrayRef<Attribute> operands) {
  // Only the last operand is checked because AnyOp is commutative.
  if (operands.back())
    return operands.back();

  return nullptr;
}

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

namespace {
// Removes AssumingOp with a passing witness and inlines the region.
struct AssumingWithTrue : public OpRewritePattern<AssumingOp> {
  using OpRewritePattern<AssumingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AssumingOp op,
                                PatternRewriter &rewriter) const override {
    auto witness = op.witness().getDefiningOp<ConstWitnessOp>();
    if (!witness || !witness.passingAttr())
      return failure();

    auto *blockBeforeAssuming = rewriter.getInsertionBlock();
    auto *assumingBlock = op.getBody();
    auto initPosition = rewriter.getInsertionPoint();
    auto *blockAfterAssuming =
        rewriter.splitBlock(blockBeforeAssuming, initPosition);

    // Remove the AssumingOp and AssumingYieldOp.
    auto &yieldOp = assumingBlock->back();
    rewriter.inlineRegionBefore(op.doRegion(), blockAfterAssuming);
    rewriter.replaceOp(op, yieldOp.getOperands());
    rewriter.eraseOp(&yieldOp);

    // Merge blocks together as there was no branching behavior from the
    // AssumingOp.
    rewriter.mergeBlocks(assumingBlock, blockBeforeAssuming);
    rewriter.mergeBlocks(blockAfterAssuming, blockBeforeAssuming);
    return success();
  }
};
}; // namespace

void AssumingOp::getCanonicalizationPatterns(OwningRewritePatternList &patterns,
                                             MLIRContext *context) {
  // If taking a passing witness, inline region
  patterns.insert<AssumingWithTrue>(context);
}

//===----------------------------------------------------------------------===//
// AssumingAllOp
//===----------------------------------------------------------------------===//
OpFoldResult AssumingAllOp::fold(ArrayRef<Attribute> operands) {
  // Iterate in reverse to first handle all constant operands. They are
  // guaranteed to be the tail of the inputs because this is commutative.
  for (int idx = operands.size() - 1; idx >= 0; idx--) {
    Attribute a = operands[idx];
    // Cannot fold if any inputs are not constant;
    if (!a)
      return nullptr;

    // We do not need to keep statically known values after handling them in
    // this method.
    getOperation()->eraseOperand(idx);

    // Always false if any input is statically known false
    if (!a.cast<BoolAttr>().getValue())
      return a;
  }
  // If this is reached, all inputs were statically known passing.
  return BoolAttr::get(true, getContext());
}

static LogicalResult verify(AssumingAllOp op) {
  // Ensure that AssumingAllOp contains at least one operand
  if (op.getNumOperands() == 0)
    return op.emitOpError("no operands specified");

  return success();
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
// CstrBroadcastableOp
//===----------------------------------------------------------------------===//

void CstrBroadcastableOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  // If inputs are equal, return passing witness
  patterns.insert<CstrBroadcastableEqOps>(context);
}

OpFoldResult CstrBroadcastableOp::fold(ArrayRef<Attribute> operands) {
  if (!operands[0] || !operands[1])
    return nullptr;
  auto lhsShape = llvm::to_vector<6>(
      operands[0].cast<DenseIntElementsAttr>().getValues<int64_t>());
  auto rhsShape = llvm::to_vector<6>(
      operands[1].cast<DenseIntElementsAttr>().getValues<int64_t>());
  SmallVector<int64_t, 6> resultShape;
  if (OpTrait::util::getBroadcastedShape(lhsShape, rhsShape, resultShape))
    return BoolAttr::get(true, getContext());

  // Because a failing witness result here represents an eventual assertion
  // failure, we do not replace it with a constant witness.
  return nullptr;
}

//===----------------------------------------------------------------------===//
// CstrEqOp
//===----------------------------------------------------------------------===//

void CstrEqOp::getCanonicalizationPatterns(OwningRewritePatternList &patterns,
                                           MLIRContext *context) {
  // If inputs are equal, return passing witness
  patterns.insert<CstrEqEqOps>(context);
}

OpFoldResult CstrEqOp::fold(ArrayRef<Attribute> operands) {
  if (llvm::all_of(operands,
                   [&](Attribute a) { return a && a == operands[0]; }))
    return BoolAttr::get(true, getContext());

  // Because a failing witness result here represents an eventual assertion
  // failure, we do not try to replace it with a constant witness. Similarly, we
  // cannot if there are any non-const inputs.
  return nullptr;
}

//===----------------------------------------------------------------------===//
// ConstSizeOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstSizeOp::fold(ArrayRef<Attribute>) { return valueAttr(); }

void ConstSizeOp::getAsmResultNames(
    llvm::function_ref<void(Value, StringRef)> setNameFn) {
  SmallString<4> buffer;
  llvm::raw_svector_ostream os(buffer);
  os << "c" << value();
  setNameFn(getResult(), os.str());
}

//===----------------------------------------------------------------------===//
// ConstWitnessOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstWitnessOp::fold(ArrayRef<Attribute>) { return passingAttr(); }

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

Optional<int64_t> GetExtentOp::getConstantDim() {
  if (auto constSizeOp = dim().getDefiningOp<ConstSizeOp>()) {
    return constSizeOp.value().getLimitedValue();
  }
  return llvm::None;
}

OpFoldResult GetExtentOp::fold(ArrayRef<Attribute> operands) {
  auto elements = operands[0].dyn_cast_or_null<DenseIntElementsAttr>();
  if (!elements)
    return nullptr;
  Optional<int64_t> dim = getConstantDim();
  if (!dim.hasValue())
    return nullptr;
  if (dim.getValue() >= elements.getNumElements())
    return nullptr;
  return elements.getValue({(uint64_t)dim.getValue()});
}

void GetExtentOp::build(OpBuilder &builder, OperationState &result, Value shape,
                        int64_t dim) {
  auto loc = result.location;
  auto dimAttr = builder.getIndexAttr(dim);
  Value dimValue = builder.create<ConstSizeOp>(loc, dimAttr);
  build(builder, result, shape, dimValue);
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
// YieldOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(YieldOp op) {
  auto *parentOp = op.getParentOp();
  auto results = parentOp->getResults();
  auto operands = op.getOperands();

  if (parentOp->getNumResults() != op.getNumOperands())
    return op.emitOpError() << "number of operands does not match number of "
                               "results of its parent";
  for (auto e : llvm::zip(results, operands))
    if (std::get<0>(e).getType() != std::get<1>(e).getType())
      return op.emitOpError()
             << "types mismatch between yield op and its parent";

  return success();
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

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

void ReduceOp::build(OpBuilder &builder, OperationState &result, Value shape,
                     ValueRange initVals) {
  result.addOperands(shape);
  result.addOperands(initVals);

  Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  Block &bodyBlock = bodyRegion->front();
  bodyBlock.addArgument(builder.getIndexType());
  bodyBlock.addArgument(SizeType::get(builder.getContext()));

  for (Type initValType : initVals.getTypes()) {
    bodyBlock.addArgument(initValType);
    result.addTypes(initValType);
  }
}

static LogicalResult verify(ReduceOp op) {
  // Verify block arg types.
  Block &block = op.region().front();

  auto blockArgsCount = op.initVals().size() + 2;
  if (block.getNumArguments() != blockArgsCount)
    return op.emitOpError() << "ReduceOp body is expected to have "
                            << blockArgsCount << " arguments";

  if (block.getArgument(0).getType() != IndexType::get(op.getContext()))
    return op.emitOpError(
        "argument 0 of ReduceOp body is expected to be of IndexType");

  if (block.getArgument(1).getType() != SizeType::get(op.getContext()))
    return op.emitOpError(
        "argument 1 of ReduceOp body is expected to be of SizeType");

  for (auto type : llvm::enumerate(op.initVals()))
    if (block.getArgument(type.index() + 2).getType() != type.value().getType())
      return op.emitOpError()
             << "type mismatch between argument " << type.index() + 2
             << " of ReduceOp body and initial value " << type.index();
  return success();
}

static ParseResult parseReduceOp(OpAsmParser &parser, OperationState &result) {
  auto *ctx = parser.getBuilder().getContext();
  // Parse operands.
  SmallVector<OpAsmParser::OperandType, 3> operands;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/-1,
                              OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalArrowTypeList(result.types))
    return failure();

  // Resolve operands.
  auto initVals = llvm::makeArrayRef(operands).drop_front();
  if (parser.resolveOperand(operands.front(), ShapeType::get(ctx),
                            result.operands) ||
      parser.resolveOperands(initVals, result.types, parser.getNameLoc(),
                             result.operands))
    return failure();

  // Parse the body.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*args=*/{}, /*argTypes=*/{}))
    return failure();

  // Parse attributes.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

static void print(OpAsmPrinter &p, ReduceOp op) {
  p << op.getOperationName() << '(' << op.shape() << ", " << op.initVals()
    << ") ";
  p.printOptionalArrowTypeList(op.getResultTypes());
  p.printRegion(op.region());
  p.printOptionalAttrDict(op.getAttrs());
}

namespace mlir {
namespace shape {

#define GET_OP_CLASSES
#include "mlir/Dialect/Shape/IR/ShapeOps.cpp.inc"

} // namespace shape
} // namespace mlir
