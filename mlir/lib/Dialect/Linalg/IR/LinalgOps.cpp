//===- LinalgOps.cpp - Implementation of the linalg operations ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Linalg operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::linalg;

//===----------------------------------------------------------------------===//
// Support for named Linalg ops defined in ods-gen.
//===----------------------------------------------------------------------===//

using RegionBuilderFn = llvm::function_ref<void(ImplicitLocOpBuilder &, Block &,
                                                ArrayRef<NamedAttribute>)>;

/// Fills the region of a structured operation using the provided
/// `regionBuilder`. The method is used by both named structured ops created by
/// ods-gen and by manually defined C++ ops. It is called by both builders and
/// parsers and creates a block with arguments corresponding to the elemental
/// types of `inputTypes` and `outputTypes`. All output types are asserted to be
/// ShapedType.
static void fillStructuredOpRegion(OpBuilder &opBuilder, Region &region,
                                   TypeRange inputTypes, TypeRange outputTypes,
                                   ArrayRef<NamedAttribute> attrs,
                                   RegionBuilderFn regionBuilder) {
  assert(llvm::all_of(outputTypes, [](Type t) { return t.isa<ShapedType>(); }));

  // TODO: atm all operands go through getElementTypeOrSelf,
  // reconsider when we have evidence we need to.
  SmallVector<Type, 8> argTypes;
  SmallVector<Location, 8> argLocs;
  for (auto containers : {inputTypes, outputTypes}) {
    for (auto t : containers) {
      argTypes.push_back(getElementTypeOrSelf(t));

      // TODO: Pass in a proper location here.
      argLocs.push_back(opBuilder.getUnknownLoc());
    }
  }

  // RAII.
  OpBuilder::InsertionGuard guard(opBuilder);
  Block *body =
      opBuilder.createBlock(&region, /*insertPt=*/{}, argTypes, argLocs);

  opBuilder.setInsertionPointToStart(body);
  ImplicitLocOpBuilder b(opBuilder.getUnknownLoc(), opBuilder);
  regionBuilder(b, *body, attrs);

  // indexing_maps is an auto-generated method.

  // iterator_types is an auto-generated method.
}

/// Creates a structured operation given `inputs`, `outputs`, and `attributes`.
/// The result types are derived automatically if `resultTensorTypes` is none.
/// The body of the operation is filled using `regionBuilder`. All ods-gen
/// created structured operations use the method to implement their builders.
static void buildStructuredOp(OpBuilder &b, OperationState &state,
                              llvm::Optional<TypeRange> resultTensorTypes,
                              ValueRange inputs, ValueRange outputs,
                              ArrayRef<NamedAttribute> attributes,
                              RegionBuilderFn regionBuilder) {
  // Derive the result types if needed.
  SmallVector<Type> derivedResultTypes =
      resultTensorTypes.value_or(TypeRange());
  if (!resultTensorTypes)
    copy_if(outputs.getTypes(), std::back_inserter(derivedResultTypes),
            [](Type type) { return type.isa<RankedTensorType>(); });

  state.addOperands(inputs);
  state.addOperands(outputs);
  state.addTypes(derivedResultTypes);
  state.addAttributes(attributes);
  state.addAttribute(
      "operand_segment_sizes",
      b.getDenseI32ArrayAttr({static_cast<int32_t>(inputs.size()),
                              static_cast<int32_t>(outputs.size())}));

  // Create and fill the region of the structured operation.
  Region &region = *state.addRegion();
  fillStructuredOpRegion(b, region, TypeRange(inputs), TypeRange(outputs),
                         state.attributes.getAttrs(), regionBuilder);
}

/// Common parsing used for both named structured ops created by ods-gen and by
/// manually defined C++ ops. Does not handle regions.
static ParseResult
parseCommonStructuredOpParts(OpAsmParser &parser, OperationState &result,
                             SmallVectorImpl<Type> &inputTypes,
                             SmallVectorImpl<Type> &outputTypes,
                             bool addOperandSegmentSizes = true) {
  SMLoc inputsOperandsLoc, outputsOperandsLoc;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> inputsOperands,
      outputsOperands;

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (succeeded(parser.parseOptionalKeyword("ins"))) {
    if (parser.parseLParen())
      return failure();

    inputsOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(inputsOperands) ||
        parser.parseColonTypeList(inputTypes) || parser.parseRParen())
      return failure();
  }

  if (succeeded(parser.parseOptionalKeyword("outs"))) {
    outputsOperandsLoc = parser.getCurrentLocation();
    if (parser.parseLParen() || parser.parseOperandList(outputsOperands) ||
        parser.parseColonTypeList(outputTypes) || parser.parseRParen())
      return failure();
  }

  if (parser.resolveOperands(inputsOperands, inputTypes, inputsOperandsLoc,
                             result.operands) ||
      parser.resolveOperands(outputsOperands, outputTypes, outputsOperandsLoc,
                             result.operands))
    return failure();

  if (addOperandSegmentSizes) {
    result.addAttribute("operand_segment_sizes",
                        parser.getBuilder().getDenseI32ArrayAttr(
                            {static_cast<int32_t>(inputsOperands.size()),
                             static_cast<int32_t>(outputsOperands.size())}));
  }
  return success();
}

static void printCommonStructuredOpParts(OpAsmPrinter &p, ValueRange inputs,
                                         ValueRange outputs) {
  if (!inputs.empty())
    p << " ins(" << inputs << " : " << inputs.getTypes() << ")";
  if (!outputs.empty())
    p << " outs(" << outputs << " : " << outputs.getTypes() << ")";
}

static void printCommonStructuredOpPartsWithNewLine(OpAsmPrinter &p,
                                                    ValueRange inputs,
                                                    ValueRange outputs) {
  p.printNewline();
  if (!inputs.empty())
    p << "ins(" << inputs << " : " << inputs.getTypes() << ")";
  p.printNewline();
  if (!outputs.empty())
    p << "outs(" << outputs << " : " << outputs.getTypes() << ")";
}
//===----------------------------------------------------------------------===//
// Specific parsing and printing for named structured ops created by ods-gen.
//===----------------------------------------------------------------------===//

static ParseResult parseNamedStructuredOpRegion(
    OpAsmParser &parser, Region &region, unsigned numRegionArgs,
    TypeRange inputTypes, TypeRange outputTypes, ArrayRef<NamedAttribute> attrs,
    RegionBuilderFn regionBuilder) {
  if (numRegionArgs != inputTypes.size() + outputTypes.size()) {
    return parser.emitError(
        parser.getCurrentLocation(),
        llvm::formatv("[parseNamedStructuredOpRegion] ods-gen generated "
                      "region expects {0} args, got {1}",
                      numRegionArgs, inputTypes.size() + outputTypes.size()));
  }

  OpBuilder opBuilder(parser.getContext());
  fillStructuredOpRegion(opBuilder, region, inputTypes, outputTypes, attrs,
                         regionBuilder);
  return success();
}

static ParseResult
parseNamedStructuredOpResults(OpAsmParser &parser,
                              SmallVectorImpl<Type> &resultTypes) {
  if (parser.parseOptionalArrowTypeList(resultTypes))
    return failure();
  return success();
}

static ParseResult parseNamedStructuredOp(OpAsmParser &parser,
                                          OperationState &result,
                                          unsigned numRegionArgs,
                                          RegionBuilderFn regionBuilder) {
  // TODO: Enable when ods-gen supports captures.
  SmallVector<Type, 1> inputTypes, outputTypes;
  if (parseCommonStructuredOpParts(parser, result, inputTypes, outputTypes))
    return failure();

  // TODO: consider merging results parsing into region parsing.
  // Need to wait for declarative assembly resolution to decide.
  SmallVector<Type, 1> outputTensorsTypes;
  if (parseNamedStructuredOpResults(parser, outputTensorsTypes))
    return failure();
  result.addTypes(outputTensorsTypes);

  std::unique_ptr<Region> region = std::make_unique<Region>();
  if (parseNamedStructuredOpRegion(parser, *region, numRegionArgs, inputTypes,
                                   outputTypes, result.attributes.getAttrs(),
                                   regionBuilder))
    return failure();
  result.addRegion(std::move(region));

  return success();
}

static void printNamedStructuredOpResults(OpAsmPrinter &p,
                                          TypeRange resultTypes) {
  if (resultTypes.empty())
    return;
  p.printOptionalArrowTypeList(resultTypes);
}

static void printNamedStructuredOp(OpAsmPrinter &p, Operation *op,
                                   ValueRange inputs, ValueRange outputs) {
  p.printOptionalAttrDict(
      op->getAttrs(),
      /*elidedAttrs=*/{"operand_segment_sizes",
                       // See generated code in mlir-linalg-yaml-gen.cpp
                       "linalg.memoized_indexing_maps"});

  // Printing is shared with generic ops, except for the region and
  // attributes.
  printCommonStructuredOpParts(p, inputs, outputs);

  // Results printing.
  printNamedStructuredOpResults(p, op->getResultTypes());

  // Region is elided.
}

//===----------------------------------------------------------------------===//
// Region builder helper.
// TODO: Move this to a utility library.
// The public methods on this class are referenced directly from generated code.
// Helper build the unary, binary, and type conversion functions defined by the
// DSL. See mlir-linalg-ods-yaml-gen.cpp for the code that uses this class.
//
// Implementations of the math functions must be polymorphic over numeric types,
// internally performing necessary casts. If the function application makes no
// sense, then the only recourse is to assert and return nullptr. This can be
// extended later if it becomes possible to fail construction of the region. The
// invariant should be enforced at a higher level.
//
// TODO: These helpers are currently type polymorphic over the class of integer
// and floating point types, but they will not internally cast within bit
// widths of a class (mixed precision such as i8->i32) or across classes
// (i.e. mixed float and integer). Many such combinations are ambiguous or need
// to be handled with care and work is being considered to extend the op
// language to make such cases explicit. In the mean-time, violating this will
// fail verification, which is deemed acceptable.
//===----------------------------------------------------------------------===//

namespace {

class RegionBuilderHelper {
public:
  RegionBuilderHelper(MLIRContext *context, Block &block)
      : context(context), block(block) {}

  // Build the unary functions defined by OpDSL.
  Value buildUnaryFn(UnaryFn unaryFn, Value arg) {
    if (!isFloatingPoint(arg))
      llvm_unreachable("unsupported non numeric type");
    OpBuilder builder = getBuilder();
    switch (unaryFn) {
    case UnaryFn::exp:
      return builder.create<math::ExpOp>(arg.getLoc(), arg);
    case UnaryFn::log:
      return builder.create<math::LogOp>(arg.getLoc(), arg);
    case UnaryFn::abs:
      return builder.create<math::AbsFOp>(arg.getLoc(), arg);
    case UnaryFn::ceil:
      return builder.create<math::CeilOp>(arg.getLoc(), arg);
    case UnaryFn::floor:
      return builder.create<math::FloorOp>(arg.getLoc(), arg);
    case UnaryFn::negf:
      return builder.create<arith::NegFOp>(arg.getLoc(), arg);
    }
    llvm_unreachable("unsupported unary function");
  }

  // Build the binary functions defined by OpDSL.
  Value buildBinaryFn(BinaryFn binaryFn, Value arg0, Value arg1) {
    bool allComplex = isComplex(arg0) && isComplex(arg1);
    bool allFloatingPoint = isFloatingPoint(arg0) && isFloatingPoint(arg1);
    bool allInteger = isInteger(arg0) && isInteger(arg1);
    bool allBool = allInteger && arg0.getType().getIntOrFloatBitWidth() == 1 &&
                   arg1.getType().getIntOrFloatBitWidth() == 1;
    if (!allComplex && !allFloatingPoint && !allInteger)
      llvm_unreachable("unsupported non numeric type");
    OpBuilder builder = getBuilder();
    switch (binaryFn) {
    case BinaryFn::add:
      if (allComplex)
        return builder.create<complex::AddOp>(arg0.getLoc(), arg0, arg1);
      if (allFloatingPoint)
        return builder.create<arith::AddFOp>(arg0.getLoc(), arg0, arg1);
      if (allBool)
        return builder.create<arith::OrIOp>(arg0.getLoc(), arg0, arg1);
      return builder.create<arith::AddIOp>(arg0.getLoc(), arg0, arg1);
    case BinaryFn::sub:
      if (allComplex)
        return builder.create<complex::SubOp>(arg0.getLoc(), arg0, arg1);
      if (allFloatingPoint)
        return builder.create<arith::SubFOp>(arg0.getLoc(), arg0, arg1);
      if (allBool)
        llvm_unreachable("unsupported operation: sub with bools");
      return builder.create<arith::SubIOp>(arg0.getLoc(), arg0, arg1);
    case BinaryFn::mul:
      if (allComplex)
        return builder.create<complex::MulOp>(arg0.getLoc(), arg0, arg1);
      if (allFloatingPoint)
        return builder.create<arith::MulFOp>(arg0.getLoc(), arg0, arg1);
      if (allBool)
        return builder.create<arith::AndIOp>(arg0.getLoc(), arg0, arg1);
      return builder.create<arith::MulIOp>(arg0.getLoc(), arg0, arg1);
    case BinaryFn::max_signed:
      assert(!allComplex);
      if (allFloatingPoint)
        return builder.create<arith::MaxFOp>(arg0.getLoc(), arg0, arg1);
      return builder.create<arith::MaxSIOp>(arg0.getLoc(), arg0, arg1);
    case BinaryFn::min_signed:
      assert(!allComplex);
      if (allFloatingPoint)
        return builder.create<arith::MinFOp>(arg0.getLoc(), arg0, arg1);
      return builder.create<arith::MinSIOp>(arg0.getLoc(), arg0, arg1);
    case BinaryFn::max_unsigned:
      assert(!allComplex);
      if (allFloatingPoint)
        return builder.create<arith::MaxFOp>(arg0.getLoc(), arg0, arg1);
      return builder.create<arith::MaxUIOp>(arg0.getLoc(), arg0, arg1);
    case BinaryFn::min_unsigned:
      assert(!allComplex);
      if (allFloatingPoint)
        return builder.create<arith::MinFOp>(arg0.getLoc(), arg0, arg1);
      return builder.create<arith::MinUIOp>(arg0.getLoc(), arg0, arg1);
    }
    llvm_unreachable("unsupported binary function");
  }

  // Build the type functions defined by OpDSL.
  Value buildTypeFn(TypeFn typeFn, Type toType, Value operand) {
    switch (typeFn) {
    case TypeFn::cast_signed:
      return cast(toType, operand, false);
    case TypeFn::cast_unsigned:
      return cast(toType, operand, true);
    }
    llvm_unreachable("unsupported type conversion function");
  }

  void yieldOutputs(ValueRange values) {
    OpBuilder builder = getBuilder();
    Location loc = builder.getUnknownLoc();
    builder.create<YieldOp>(loc, values);
  }

  Value constant(const std::string &value) {
    OpBuilder builder = getBuilder();
    Location loc = builder.getUnknownLoc();
    Attribute valueAttr = parseAttribute(value, builder.getContext());
    Type type = NoneType::get(builder.getContext());
    if (auto typedAttr = valueAttr.dyn_cast<TypedAttr>())
      type = typedAttr.getType();
    return builder.create<arith::ConstantOp>(loc, type, valueAttr);
  }

  Value index(int64_t dim) {
    OpBuilder builder = getBuilder();
    return builder.create<IndexOp>(builder.getUnknownLoc(), dim);
  }

  Type getIntegerType(unsigned width) {
    return IntegerType::get(context, width);
  }

  Type getFloat32Type() { return Float32Type::get(context); }
  Type getFloat64Type() { return Float64Type::get(context); }

private:
  // Generates operations to cast the given operand to a specified type.
  // If the cast cannot be performed, a warning will be issued and the
  // operand returned as-is (which will presumably yield a verification
  // issue downstream).
  Value cast(Type toType, Value operand, bool isUnsignedCast) {
    OpBuilder builder = getBuilder();
    auto loc = operand.getLoc();

    if (operand.getType() == toType)
      return operand;
    if (auto toIntType = toType.dyn_cast<IntegerType>()) {
      // If operand is floating point, cast directly to the int type.
      if (operand.getType().isa<FloatType>()) {
        if (isUnsignedCast)
          return builder.create<arith::FPToUIOp>(loc, toType, operand);
        return builder.create<arith::FPToSIOp>(loc, toType, operand);
      }
      // Cast index operands directly to the int type.
      if (operand.getType().isIndex())
        return builder.create<arith::IndexCastOp>(loc, toType, operand);
      if (auto fromIntType = operand.getType().dyn_cast<IntegerType>()) {
        // Either extend or truncate.
        if (toIntType.getWidth() > fromIntType.getWidth()) {
          if (isUnsignedCast)
            return builder.create<arith::ExtUIOp>(loc, toType, operand);
          return builder.create<arith::ExtSIOp>(loc, toType, operand);
        }
        if (toIntType.getWidth() < fromIntType.getWidth())
          return builder.create<arith::TruncIOp>(loc, toType, operand);
      }
    } else if (auto toFloatType = toType.dyn_cast<FloatType>()) {
      // If operand is integer, cast directly to the float type.
      // Note that it is unclear how to cast from BF16<->FP16.
      if (operand.getType().isa<IntegerType>()) {
        if (isUnsignedCast)
          return builder.create<arith::UIToFPOp>(loc, toFloatType, operand);
        return builder.create<arith::SIToFPOp>(loc, toFloatType, operand);
      }
      if (auto fromFloatType = operand.getType().dyn_cast<FloatType>()) {
        if (toFloatType.getWidth() > fromFloatType.getWidth())
          return builder.create<arith::ExtFOp>(loc, toFloatType, operand);
        if (toFloatType.getWidth() < fromFloatType.getWidth())
          return builder.create<arith::TruncFOp>(loc, toFloatType, operand);
      }
    }

    emitWarning(operand.getLoc()) << "could not cast operand of type "
                                  << operand.getType() << " to " << toType;
    return operand;
  }

  bool isComplex(Value value) { return value.getType().isa<ComplexType>(); }
  bool isFloatingPoint(Value value) { return value.getType().isa<FloatType>(); }
  bool isInteger(Value value) { return value.getType().isa<IntegerType>(); }

  OpBuilder getBuilder() {
    OpBuilder builder(context);
    builder.setInsertionPointToEnd(&block);
    return builder;
  }

  MLIRContext *context;
  Block &block;
};

} // namespace

//===----------------------------------------------------------------------===//
// FillOp
//===----------------------------------------------------------------------===//

namespace {

/// Fold linalg.fill -> tensor.expand/collapse_shape chain.
///
/// For such op chains, we can create new linalg.fill ops with the result
/// type of the tensor.expand/collapse_shape op.
template <typename TensorReshapeOp>
struct FoldFillWithTensorReshape : OpRewritePattern<TensorReshapeOp> {
  using OpRewritePattern<TensorReshapeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TensorReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    auto oldFill = reshapeOp.getSrc().template getDefiningOp<FillOp>();
    if (!oldFill)
      return failure();

    Location loc = oldFill.getLoc();
    auto newInit = rewriter.create<TensorReshapeOp>(
        loc, reshapeOp.getResultType(), oldFill.output(),
        reshapeOp.getReassociation());
    rewriter.replaceOpWithNewOp<FillOp>(reshapeOp, ValueRange{oldFill.value()},
                                        ValueRange{newInit});

    return success();
  }
};

/// Fold tensor.pad(linalg.fill) into linalg.fill if the padding value and the
/// filling value are the same.
struct FoldFillWithPad final : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    auto fillOp = padOp.getSource().getDefiningOp<linalg::FillOp>();
    if (!fillOp)
      return failure();

    // We can only fold if the padding value is the same as the original
    // filling value.
    Value padValue = padOp.getConstantPaddingValue();
    if (!padValue || fillOp.value() != padValue)
      return failure();

    ReifiedRankedShapedTypeDims reifiedShape;
    ReifyRankedShapedTypeOpInterface interface =
        cast<ReifyRankedShapedTypeOpInterface>(padOp.getOperation());
    if (failed(interface.reifyResultShapes(rewriter, reifiedShape)))
      return rewriter.notifyMatchFailure(
          padOp, "failed to reify tensor.pad op result shape");

    auto oldResultType = padOp.getResultType();
    SmallVector<int64_t, 4> staticShape(oldResultType.getRank(),
                                        ShapedType::kDynamicSize);
    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        padOp.getLoc(), staticShape, oldResultType.getElementType(),
        reifiedShape.front());
    auto newFillOp = rewriter.create<FillOp>(
        fillOp.getLoc(), ValueRange{padValue}, ValueRange{emptyTensor});
    rewriter.replaceOpWithNewOp<tensor::CastOp>(padOp, oldResultType,
                                                newFillOp.result());

    return success();
  }
};

/// Fold tensor.insert_slice(tensor.pad(<input>), linalg.fill) into
/// tensor.insert_slice(<input>, linalg.fill) if the padding value and the
/// filling value are the same.
struct FoldInsertPadIntoFill : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    auto srcPadOp = insertOp.getSource().getDefiningOp<tensor::PadOp>();
    if (!srcPadOp)
      return failure();

    if (insertOp.getType().getRank() != insertOp.getSourceType().getRank())
      return failure();

    // Walk back the tensor.insert_slice chain and find the first destination
    // value at the start of the chain.
    Value firstDest = insertOp.getDest();
    while (auto prevOp = firstDest.getDefiningOp<tensor::InsertSliceOp>()) {
      if (prevOp.getType().getRank() != prevOp.getSourceType().getRank())
        return failure();

      // Make sure the range of values accessed are disjoint. Without this, we
      // cannot fold tensor.pad away.
      bool disjoint = false;
      for (int i = 0, e = prevOp.getType().getRank(); i < e; ++i) {
        // If the dimension has dynamic offset/size, we cannot guarantee
        // disjoint. So just skip it.
        if (insertOp.isDynamicOffset(i) || insertOp.isDynamicSize(i) ||
            insertOp.isDynamicStride(i) || prevOp.isDynamicOffset(i) ||
            prevOp.isDynamicSize(i) || prevOp.isDynamicStride(i))
          continue;

        // Get the range start and end, inclusively for both.
        int64_t prevStart = prevOp.getStaticOffset(i);
        int64_t prevEnd = prevStart + (prevOp.getStaticSize(i) - 1) *
                                          prevOp.getStaticStride(i);
        int64_t nextStart = insertOp.getStaticOffset(i);
        int64_t nextEnd = nextStart + (insertOp.getStaticSize(i) - 1) *
                                          insertOp.getStaticStride(i);
        if (prevEnd < nextStart || nextEnd < prevStart) {
          disjoint = true;
          break;
        }
      }

      if (!disjoint)
        break;
      firstDest = prevOp.getDest();
    }

    // Check whether the first destination is a fill op. For overlapped cases,
    // this also cannot be true.
    auto dstFillOp = firstDest.getDefiningOp<linalg::FillOp>();
    if (!dstFillOp)
      return failure();

    // We can only fold if the padding value is the same as the original
    // filling value.
    Value padValue = srcPadOp.getConstantPaddingValue();
    if (!padValue || dstFillOp.value() != padValue)
      return failure();

    SmallVector<OpFoldResult> lowPads = srcPadOp.getMixedLowPad();
    SmallVector<OpFoldResult> oldOffsets = insertOp.getMixedOffsets();

    Location loc = insertOp.getLoc();
    MLIRContext *context = getContext();

    AffineExpr sym0, sym1;
    bindSymbols(context, sym0, sym1);
    auto addMap = AffineMap::get(0, 2, {sym0 + sym1}, context);

    // Calculate the new offsets for the insert. It should be the old offsets
    // plus low padding sizes.
    SmallVector<OpFoldResult, 4> newOffsets;
    for (const auto &p : llvm::zip(lowPads, oldOffsets)) {
      newOffsets.push_back(makeComposedFoldedAffineApply(
          rewriter, loc, addMap, {std::get<0>(p), std::get<1>(p)}));
    }

    SmallVector<OpFoldResult, 4> newSizes;
    for (int i = 0, e = srcPadOp.getSourceType().getRank(); i < e; ++i) {
      newSizes.push_back(
          rewriter.create<tensor::DimOp>(loc, srcPadOp.getSource(), i)
              .getResult());
    }

    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        insertOp, srcPadOp.getSource(), insertOp.getDest(), newOffsets,
        newSizes, insertOp.getMixedStrides());
    return success();
  }
};

} // namespace

void FillOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results
      .add<FoldFillWithPad, FoldFillWithTensorReshape<tensor::CollapseShapeOp>,
           FoldFillWithTensorReshape<tensor::ExpandShapeOp>,
           FoldInsertPadIntoFill>(context);
}

//===----------------------------------------------------------------------===//
// GenericOp
//===----------------------------------------------------------------------===//

void GenericOp::getAsmBlockArgumentNames(Region &region,
                                         OpAsmSetValueNameFn setNameFn) {
  for (Value v : getRegionInputArgs())
    setNameFn(v, "in");
  for (Value v : getRegionOutputArgs())
    setNameFn(v, "out");
}

void GenericOp::build(
    OpBuilder &builder, OperationState &result, TypeRange resultTensorTypes,
    ValueRange inputs, ValueRange outputs, ArrayAttr indexingMaps,
    ArrayAttr iteratorTypes, StringAttr doc, StringAttr libraryCall,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild,
    ArrayRef<NamedAttribute> attributes) {
  build(builder, result, resultTensorTypes, inputs, outputs, indexingMaps,
        iteratorTypes, doc, libraryCall);
  result.addAttributes(attributes);
  if (!bodyBuild)
    return;

  SmallVector<Type, 4> blockArgTypes;
  SmallVector<Location, 4> blockArgLocs;
  for (ValueRange container : {inputs, outputs}) {
    for (Value v : container) {
      blockArgTypes.push_back(getElementTypeOrSelf(v));
      blockArgLocs.push_back(v.getLoc());
    }
  }

  OpBuilder::InsertionGuard guard(builder);
  auto &region = *result.regions.front();
  Block *bodyBlock =
      builder.createBlock(&region, region.end(), blockArgTypes, blockArgLocs);
  bodyBuild(builder, result.location, bodyBlock->getArguments());
}

void GenericOp::build(
    OpBuilder &builder, OperationState &result, TypeRange resultTensorTypes,
    ValueRange inputs, ValueRange outputs, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<StringRef> iteratorTypes, StringRef doc, StringRef libraryCall,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild,
    ArrayRef<NamedAttribute> attributes) {
  build(builder, result, resultTensorTypes, inputs, outputs,
        builder.getAffineMapArrayAttr(indexingMaps),
        builder.getStrArrayAttr(iteratorTypes),
        doc.empty() ? StringAttr() : builder.getStringAttr(doc),
        libraryCall.empty() ? StringAttr() : builder.getStringAttr(libraryCall),
        bodyBuild, attributes);
}

void GenericOp::build(
    OpBuilder &builder, OperationState &result, ValueRange inputs,
    ValueRange outputs, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<StringRef> iteratorTypes, StringRef doc, StringRef libraryCall,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild,
    ArrayRef<NamedAttribute> attributes) {
  build(builder, result, TypeRange{}, inputs, outputs, indexingMaps,
        iteratorTypes, doc, libraryCall, bodyBuild, attributes);
}

void GenericOp::build(
    OpBuilder &builder, OperationState &result, ValueRange inputs,
    ValueRange outputs, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<StringRef> iteratorTypes,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild,
    ArrayRef<NamedAttribute> attributes) {
  build(builder, result, inputs, outputs, indexingMaps, iteratorTypes,
        /*doc=*/"",
        /*libraryCall=*/"", bodyBuild, attributes);
}

void GenericOp::build(
    OpBuilder &builder, OperationState &result, TypeRange resultTensorTypes,
    ValueRange inputs, ValueRange outputs, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<StringRef> iteratorTypes,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild,
    ArrayRef<NamedAttribute> attributes) {
  build(builder, result, resultTensorTypes, inputs, outputs, indexingMaps,
        iteratorTypes,
        /*doc=*/"",
        /*libraryCall=*/"", bodyBuild, attributes);
}

void GenericOp::print(OpAsmPrinter &p) {
  p << " ";

  // Print extra attributes.
  auto genericAttrNames = linalgTraitAttrNames();

  llvm::StringSet<> genericAttrNamesSet;
  genericAttrNamesSet.insert(genericAttrNames.begin(), genericAttrNames.end());
  SmallVector<NamedAttribute, 8> genericAttrs;
  for (auto attr : (*this)->getAttrs())
    if (genericAttrNamesSet.count(attr.getName().strref()) > 0)
      genericAttrs.push_back(attr);
  if (!genericAttrs.empty()) {
    auto genericDictAttr = DictionaryAttr::get(getContext(), genericAttrs);
    p << genericDictAttr;
  }

  // Printing is shared with named ops, except for the region and attributes
  printCommonStructuredOpParts(p, SmallVector<Value>(getInputOperands()),
                               SmallVector<Value>(getOutputOperands()));

  genericAttrNames.push_back("operand_segment_sizes");
  genericAttrNamesSet.insert(genericAttrNames.back());

  bool hasExtraAttrs = false;
  for (NamedAttribute n : (*this)->getAttrs()) {
    if ((hasExtraAttrs = !genericAttrNamesSet.contains(n.getName().strref())))
      break;
  }
  if (hasExtraAttrs) {
    p << " attrs = ";
    p.printOptionalAttrDict((*this)->getAttrs(),
                            /*elidedAttrs=*/genericAttrNames);
  }

  // Print region.
  if (!getRegion().empty()) {
    p << ' ';
    p.printRegion(getRegion());
  }

  // Print results.
  printNamedStructuredOpResults(p, getResultTensors().getTypes());
}

ParseResult GenericOp::parse(OpAsmParser &parser, OperationState &result) {
  DictionaryAttr dictAttr;
  // Parse the core linalg traits that must check into a dictAttr.
  // The name is unimportant as we will overwrite result.attributes.
  // The core linalg traits must contain the information necessary to pass the
  // verifier.
  if (parser.parseAttribute(dictAttr, "_", result.attributes))
    return failure();
  result.attributes.assign(dictAttr.getValue().begin(),
                           dictAttr.getValue().end());

  // Parsing is shared with named ops, except for the region.
  SmallVector<Type, 1> inputTypes, outputTypes;
  if (parseCommonStructuredOpParts(parser, result, inputTypes, outputTypes))
    return failure();

  // Optional attributes may be added.
  if (succeeded(parser.parseOptionalKeyword("attrs")))
    if (failed(parser.parseEqual()) ||
        failed(parser.parseOptionalAttrDict(result.attributes)))
      return failure();

  std::unique_ptr<Region> region = std::make_unique<Region>();
  if (parser.parseRegion(*region, {}))
    return failure();
  result.addRegion(std::move(region));

  // Generic ops may specify that a subset of its outputs are tensors. Such
  // outputs are specified in the result type.
  // TODO: may need to move output parsing before region parsing.
  // Need to wait for declarative assembly resolution to decide.
  SmallVector<Type, 1> outputTensorsTypes;
  if (parseNamedStructuredOpResults(parser, outputTensorsTypes))
    return failure();
  result.addTypes(outputTensorsTypes);

  return success();
}

static void getGenericEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    ValueRange results, OpOperandVector inputOperands,
    OpOperandVector outputOperands) {
  for (auto *operand : inputOperands) {
    if (!operand->get().getType().isa<MemRefType>())
      continue;
    effects.emplace_back(MemoryEffects::Read::get(), operand->get(),
                         SideEffects::DefaultResource::get());
  }
  for (auto *operand : outputOperands) {
    if (!operand->get().getType().isa<MemRefType>())
      continue;
    effects.emplace_back(MemoryEffects::Read::get(), operand->get(),
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), operand->get(),
                         SideEffects::DefaultResource::get());
  }
}

void GenericOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getGenericEffectsImpl(effects, getOperation()->getResults(),
                        getInputOperands(), getOutputOperands());
}

static bool isResultValueDead(linalg::GenericOp genericOp, OpResult result) {
  if (!result.use_empty())
    return false;
  // If out operand not used in payload, we can drop it.
  OpOperand *outputOpOperand =
      genericOp.getOutputOperand(result.getResultNumber());
  if (!genericOp.payloadUsesValueFromOperand(outputOpOperand))
    return true;

  // The out operand that is part of a payload can be dropped if
  // these conditions are met:
  // - Result from out operand is dead.
  // - User of arg is yield.
  // - outArg data is not being used by other outArgs.

  // Check block arg and cycle from out operand has a single use.
  BlockArgument outputArg =
      genericOp.getRegionOutputArgs()[result.getResultNumber()];
  if (!outputArg.hasOneUse())
    return false;
  Operation *argUserOp = *outputArg.user_begin();

  // Check argUser has no other use.
  if (!argUserOp->use_empty())
    return false;

  // Check that argUser is a yield.
  auto yieldOp = dyn_cast<linalg::YieldOp>(argUserOp);
  if (!yieldOp)
    return false;

  // Check outArg data is not being used by other outArgs.
  if (yieldOp.getOperand(result.getResultNumber()) != outputArg)
    return false;

  return true;
}

LogicalResult GenericOp::verify() { return success(); }

namespace {

struct DeduplicateAndRemoveDeadOperandsAndResults
    : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    // Create a map from argument position in the original op to the argument
    // position in the new op. If the argument is dropped it wont have an entry.
    SmallVector<OpOperand *> droppedOpOperands;

    // Information needed to build the new op.
    SmallVector<Value> newInputOperands, newOutputOperands;
    SmallVector<AffineMap> newIndexingMaps;

    // Gather information about duplicate input operands.
    llvm::SmallDenseMap<unsigned, unsigned> origInsToNewInsPos =
        deduplicateInputOperands(genericOp, droppedOpOperands, newInputOperands,
                                 newIndexingMaps);

    // Gather information about the dropped outputs.
    llvm::SmallDenseMap<unsigned, unsigned> origOutsToNewOutsPos =
        deduplicateOutputOperands(genericOp, droppedOpOperands,
                                  newOutputOperands, newIndexingMaps);

    // Check if there is any change to operands.
    if (newInputOperands.size() + newOutputOperands.size() ==
        genericOp->getNumOperands())
      return failure();

    // Create the new op with the body being empty.
    Location loc = genericOp.getLoc();
    SmallVector<Type> newResultTypes;
    if (genericOp.hasTensorSemantics()) {
      newResultTypes = llvm::to_vector(llvm::map_range(
          newOutputOperands, [](Value v) { return v.getType(); }));
    }
    auto newOp = rewriter.create<GenericOp>(
        loc, newResultTypes, newInputOperands, newOutputOperands,
        rewriter.getAffineMapArrayAttr(newIndexingMaps),
        genericOp.getIteratorTypes(), genericOp.getDocAttr(),
        genericOp.getLibraryCallAttr(),
        [](OpBuilder & /*builder*/, Location /*loc*/, ValueRange /*args*/) {
          return;
        });
    // Copy over unknown attributes. They might be load bearing for some flow.
    ArrayRef<StringRef> odsAttrs = genericOp.getAttributeNames();
    for (NamedAttribute kv : genericOp->getAttrs())
      if (!llvm::is_contained(odsAttrs, kv.getName().getValue()))
        newOp->setAttr(kv.getName(), kv.getValue());

    // Fix up the payload of the canonicalized operation.
    populateOpPayload(genericOp, newOp, origInsToNewInsPos,
                      origOutsToNewOutsPos, rewriter);

    // Replace all live uses of the op.
    SmallVector<Value> replacementsVals(genericOp->getNumResults(), nullptr);
    for (const auto &result : llvm::enumerate(genericOp.getResults())) {
      auto it = origOutsToNewOutsPos.find(result.index());
      if (it == origOutsToNewOutsPos.end())
        continue;
      replacementsVals[result.index()] = newOp.getResult(it->second);
    }
    rewriter.replaceOp(genericOp, replacementsVals);
    return success();
  }

private:
  // Deduplicate input operands, and return the
  // - Mapping from operand position in the original op, to operand position in
  // the canonicalized op.
  // - The preserved input operands list (by reference).
  llvm::SmallDenseMap<unsigned, unsigned>
  deduplicateInputOperands(GenericOp genericOp,
                           SmallVector<OpOperand *> &droppedOpOperands,
                           SmallVector<Value> &newInputOperands,
                           SmallVector<AffineMap> &newIndexingMaps) const {
    llvm::SmallDenseMap<unsigned, unsigned> origToNewPos;
    llvm::SmallDenseMap<std::pair<Value, AffineMap>, unsigned> dedupedInputs;
    for (const auto &en : llvm::enumerate(genericOp.getInputOperands())) {
      OpOperand *inputOpOperand = en.value();
      // Check if operand is dead and if dropping the indexing map makes the
      // loops to shape computation invalid.
      if (!genericOp.payloadUsesValueFromOperand(inputOpOperand)) {
        // Add the current operands to the list of potentially droppable
        // operands. If it cannot be dropped, this needs to be popped back.
        droppedOpOperands.push_back(inputOpOperand);
        if (genericOp.canOpOperandsBeDropped(droppedOpOperands))
          continue;
        droppedOpOperands.pop_back();
      }

      // Check if this operand is a duplicate.
      AffineMap indexingMap = genericOp.getMatchingIndexingMap(inputOpOperand);
      auto it = dedupedInputs.find(
          std::make_pair(inputOpOperand->get(), indexingMap));
      if (it != dedupedInputs.end()) {
        origToNewPos[en.index()] = it->second;
        droppedOpOperands.push_back(inputOpOperand);
        continue;
      }

      // This is a preserved argument.
      origToNewPos[en.index()] = newInputOperands.size();
      dedupedInputs[{inputOpOperand->get(), indexingMap}] =
          newInputOperands.size();
      newInputOperands.push_back(inputOpOperand->get());
      newIndexingMaps.push_back(indexingMap);
    }
    return origToNewPos;
  }

  // Deduplicate output operands, and return the
  // - Mapping from operand position in the original op, to operand position in
  // the canonicalized op.
  // - The preserved output operands list (by reference).
  llvm::SmallDenseMap<unsigned, unsigned>
  deduplicateOutputOperands(GenericOp genericOp,
                            SmallVector<OpOperand *> &droppedOpOperands,
                            SmallVector<Value> &newOutputOperands,
                            SmallVector<AffineMap> &newIndexingMaps) const {
    llvm::SmallDenseMap<unsigned, unsigned> origToNewPos;
    llvm::SmallDenseMap<std::tuple<Value, AffineMap, Value>, unsigned>
        dedupedOutpts;
    // If the op doesnt have tensor semantics, keep all the outputs as
    // preserved.
    if (!genericOp.hasTensorSemantics()) {
      for (const auto &en : llvm::enumerate(genericOp.getOutputOperands())) {
        origToNewPos[en.index()] = newOutputOperands.size();
        newOutputOperands.push_back(en.value()->get());
        newIndexingMaps.push_back(genericOp.getMatchingIndexingMap(en.value()));
      }
      return origToNewPos;
    }
    // Output argument can be dropped if the result has
    // - no users, and
    // - it is not used in the payload, and
    // - the corresponding indexing maps are not needed for loop bound
    //   computation.
    auto yieldOp = cast<YieldOp>(genericOp.getBody()->getTerminator());
    for (const auto &outputOpOperand :
         llvm::enumerate(genericOp.getOutputOperands())) {
      OpResult result = genericOp.getTiedOpResult(outputOpOperand.value());
      AffineMap indexingMap =
          genericOp.getMatchingIndexingMap(outputOpOperand.value());
      auto key = std::make_tuple(outputOpOperand.value()->get(), indexingMap,
                                 yieldOp->getOperand(outputOpOperand.index()));
      if (isResultValueDead(genericOp, result)) {
        // Check if the opoperand can be dropped without affecting loop
        // bound computation. Add the operand to the list of dropped op
        // operand for checking. If it cannot be dropped, need to pop the
        // value back.
        droppedOpOperands.push_back(outputOpOperand.value());
        if (genericOp.canOpOperandsBeDropped(droppedOpOperands)) {
          continue;
        }
        droppedOpOperands.pop_back();
      }

      if (!genericOp.payloadUsesValueFromOperand(outputOpOperand.value())) {
        // The out operand can also be dropped if it is computed redundantly
        // by another result, the conditions for that are
        // - The same operand is used as the out operand
        // - The same indexing map is used
        // - The same yield value is used.
        auto it = dedupedOutpts.find(key);
        if (it != dedupedOutpts.end()) {
          origToNewPos[outputOpOperand.index()] = it->second;
          droppedOpOperands.push_back(outputOpOperand.value());
          continue;
        }
      }

      origToNewPos[outputOpOperand.index()] = newOutputOperands.size();
      dedupedOutpts[key] = newOutputOperands.size();
      newOutputOperands.push_back(outputOpOperand.value()->get());
      newIndexingMaps.push_back(
          genericOp.getMatchingIndexingMap(outputOpOperand.value()));
    }
    return origToNewPos;
  }

  // Populate the body of the canonicalized operation.
  void populateOpPayload(
      GenericOp genericOp, GenericOp newOp,
      const llvm::SmallDenseMap<unsigned, unsigned> &origInsToNewInsPos,
      const llvm::SmallDenseMap<unsigned, unsigned> &origOutsToNewOutsPos,
      PatternRewriter &rewriter) const {
    // Merge the body of the original op with the new op.
    Block *newOpBlock = &newOp.getRegion().front();
    assert(newOpBlock->empty() && "expected new op to have an empty payload");
    Block *origOpBlock = &genericOp.getRegion().front();
    SmallVector<Value> replacements(origOpBlock->getNumArguments(), nullptr);

    // Replace all arguments in the original op, with arguments from the
    // canonicalized op.
    auto updateReplacements =
        [&](OpOperandVector &origOperands, OpOperandVector &newOperands,
            const llvm::SmallDenseMap<unsigned, unsigned> &map) {
          for (const auto &origOperand : llvm::enumerate(origOperands)) {
            auto it = map.find(origOperand.index());
            if (it == map.end())
              continue;
            OpOperand *newOperand = newOperands[it->second];
            replacements[origOperand.value()->getOperandNumber()] =
                newOpBlock->getArgument(newOperand->getOperandNumber());
          }
        };

    OpOperandVector origInputOperands = genericOp.getInputOperands();
    OpOperandVector newInputOperands = newOp.getInputOperands();
    updateReplacements(origInputOperands, newInputOperands, origInsToNewInsPos);

    OpOperandVector origOutputOperands = genericOp.getOutputOperands();
    OpOperandVector newOutputOperands = newOp.getOutputOperands();
    updateReplacements(origOutputOperands, newOutputOperands,
                       origOutsToNewOutsPos);

    // Drop the unused yield args.
    if (newOp.getNumOutputs() != genericOp.getNumOutputs()) {
      OpBuilder::InsertionGuard g(rewriter);
      YieldOp origYieldOp = cast<YieldOp>(origOpBlock->getTerminator());
      rewriter.setInsertionPoint(origYieldOp);

      SmallVector<Value> newYieldVals(newOp.getNumOutputs(), nullptr);
      for (const auto &yieldOpOperands :
           llvm::enumerate(origYieldOp.getValues())) {
        auto it = origOutsToNewOutsPos.find(yieldOpOperands.index());
        if (it == origOutsToNewOutsPos.end())
          continue;
        newYieldVals[it->second] = yieldOpOperands.value();
      }
      rewriter.replaceOpWithNewOp<YieldOp>(origYieldOp, newYieldVals);
    }

    rewriter.mergeBlocks(origOpBlock, newOpBlock, replacements);
  }
};

/// Remove generic operations (on tensors) that are just copying
/// the values from inputs to the results. Requirements are
/// 1) All iterator types are parallel
/// 2) The body contains just a yield operation with the yielded values being
///    the arguments corresponding to the operands.
struct EraseIdentityGenericOp : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    // Check all indexing maps are identity.
    if (llvm::any_of(genericOp.getIndexingMapsArray(),
                     [](AffineMap map) { return !map.isIdentity(); }))
      return failure();

    // Check that the body of the linalg operation is just a linalg.yield
    // operation.
    Block &body = genericOp.getRegion().front();
    if (!llvm::hasSingleElement(body))
      return failure();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yieldOp)
      return failure();

    // In the buffer case, we need to check exact buffer equality.
    if (genericOp.hasBufferSemantics()) {
      if (genericOp.getNumInputs() == 1 && genericOp.getNumOutputs() == 1 &&
          genericOp.getInputOperand(0)->get() ==
              genericOp.getOutputOperand(0)->get()) {
        rewriter.eraseOp(genericOp);
        return success();
      }
      return failure();
    }

    // Mixed semantics is not supported yet.
    if (!genericOp.hasTensorSemantics())
      return failure();

    // Get the argument number of the returned values. That is the operand
    // number to use for replacing uses of this operation.
    SmallVector<Value> returnedArgs;
    for (const auto &yieldVal : llvm::enumerate(yieldOp.getValues())) {
      auto yieldArg = yieldVal.value().dyn_cast<BlockArgument>();
      if (!yieldArg || yieldArg.getOwner() != &body)
        return failure();
      unsigned argumentNumber = yieldArg.getArgNumber();
      Value returnedArg = genericOp->getOperand(argumentNumber);
      Type resultType = genericOp->getResult(yieldVal.index()).getType();
      // The input can have a different type than the result, e.g. a dynamic
      // input dimension can be turned into a static output dimension.
      Type returnType = returnedArg.getType();
      if (returnType != resultType) {
        // Distinguish between sparse conversion or dense tensor casting.
        // TODO: unify the two ops?
        if (sparse_tensor::getSparseTensorEncoding(returnType) ||
            sparse_tensor::getSparseTensorEncoding(resultType))
          returnedArg = rewriter.create<sparse_tensor::ConvertOp>(
              genericOp.getLoc(), resultType, returnedArg);
        else {
          if (!tensor::CastOp::areCastCompatible(returnedArg.getType(),
                                                 resultType))
            return failure();
          returnedArg = rewriter.create<tensor::CastOp>(
              genericOp.getLoc(), resultType, returnedArg);
        }
      }
      returnedArgs.push_back(returnedArg);
    }

    if (returnedArgs.size() != genericOp->getNumResults())
      return failure();
    rewriter.replaceOp(genericOp, returnedArgs);
    return success();
  }
};

/// Remove unused cycles.
/// We can remove unused cycle within a payload of generic region
/// if these conditions are met:
/// - Result from out operand is dead.
/// - Block arg from out operand has a single use in the %cycle
/// instruction.
/// - Cycle has a single use and it is in yield.
struct RemoveUnusedCycleInGenericOp : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {

    // If the op doesnt have tensor semantics, preserve the outputs as is.
    if (!genericOp.hasTensorSemantics())
      return failure();

    bool hasRemovedCycles = false;
    // Iterate over output operands and remove any unused cycles.
    for (const auto &outputOpOperand :
         llvm::enumerate(genericOp.getOutputOperands())) {

      // Check that result from out operand is dead.
      Value result = genericOp.getResult(outputOpOperand.index());
      if (!result.use_empty())
        continue;

      // Check that outputArg has one use in cycle.
      BlockArgument outputArg =
          genericOp.getRegionOutputArgs()[outputOpOperand.index()];
      if (!outputArg.hasOneUse())
        continue;

      // Check cycle has at most one use.
      Operation *cycleOp = *outputArg.user_begin();
      if (!cycleOp->hasOneUse())
        continue;

      // Check that the cycleUser is a yield.
      Operation *cycleUserOp = *cycleOp->user_begin();
      if (!isa<linalg::YieldOp>(cycleUserOp))
        continue;

      // Check that argIndex matches yieldIndex, else data is being used.
      if (cycleUserOp->getOperand(outputOpOperand.index()) !=
          cycleOp->getResult(0))
        continue;

      // Directly replace the cycle with the blockArg such that
      // Deduplicate pattern can eliminate it along with unused yield.
      rewriter.replaceOp(cycleOp, outputArg);
      rewriter.updateRootInPlace(genericOp, [] {});
      hasRemovedCycles = true;
    }

    if (hasRemovedCycles) {
      return success();
    }

    return failure();
  }
};
} // namespace

void GenericOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<DeduplicateAndRemoveDeadOperandsAndResults,
              EraseIdentityGenericOp, RemoveUnusedCycleInGenericOp>(context);
}

LogicalResult GenericOp::fold(ArrayRef<Attribute>,
                              SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

//===----------------------------------------------------------------------===//
// MapOp
//===----------------------------------------------------------------------===//

static ParseResult parseDstStyleOp(
    OpAsmParser &parser, OperationState &result,
    function_ref<ParseResult(OpAsmParser &, NamedAttrList &)> parseAttrsFn =
        nullptr) {
  // Parse `ins` and `outs`.
  SmallVector<Type, 4> inputTypes, outputTypes;
  if (parseCommonStructuredOpParts(parser, result, inputTypes, outputTypes,
                                   /*addOperandSegmentSizes=*/false))
    return failure();

  // Add result types.
  for (Type outputType : outputTypes) {
    if (outputType.isa<RankedTensorType>())
      result.addTypes(outputType);
  }

  // Parse required attributes.
  if (parseAttrsFn && failed(parseAttrsFn(parser, result.attributes)))
    return failure();

  // Parse optional attributes.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void MapOp::getAsmBlockArgumentNames(Region &region,
                                     OpAsmSetValueNameFn setNameFn) {
  for (Value v : getRegionInputArgs())
    setNameFn(v, "in");
}

void MapOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
  if (!getResults().empty())
    setNameFn(getResults().front(), "mapped");
}

ParseResult MapOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parseDstStyleOp(parser, result))
    return failure();

  SmallVector<OpAsmParser::Argument> regionArgs;
  if (parser.parseArgumentList(regionArgs, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true, /*allowAttrs=*/true)) {
    return failure();
  }

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return failure();

  return success();
}

void MapOp::print(OpAsmPrinter &p) {
  p.increaseIndent();
  printCommonStructuredOpPartsWithNewLine(
      p, SmallVector<Value>(getInputOperands()),
      SmallVector<Value>(getOutputOperands()));
  p.printOptionalAttrDict((*this)->getAttrs());

  p.printNewline();
  p << "(";
  llvm::interleaveComma(getMapper().getArguments(), p,
                        [&](auto arg) { p.printRegionArgument(arg); });
  p << ") ";

  p.printRegion(getMapper(), /*printEntryBlockArgs=*/false);
  p.decreaseIndent();
}

LogicalResult MapOp::verify() {
  auto *bodyBlock = getBody();
  auto blockArgs = bodyBlock->getArguments();

  // Checks if the number of `inputs` match the arity of the `mapper` region.
  if (getInputs().size() != blockArgs.size())
    return emitOpError() << "expects number of operands to match the arity of "
                            "mapper, but got: "
                         << getInputs().size() << " and " << blockArgs.size();

  // The parameters of mapper should all match the element type // of inputs.
  for (const auto &[bbArgType, inputArg] :
       llvm::zip(bodyBlock->getArgumentTypes(), getInputs())) {
    auto inputElemType = inputArg.getType().cast<ShapedType>().getElementType();
    if (bbArgType != inputElemType) {
      return emitOpError() << "expected element type of input " << inputElemType
                           << " to match bbArg type " << bbArgType;
    }
  }

  // The shape of each input must match the shape of the output.
  auto outputShape =
      getOutputOperand(0)->get().getType().cast<ShapedType>().getShape();
  for (Type inputArgType : TypeRange{getInputs()}) {
    auto inputElemShape = inputArgType.cast<ShapedType>().getShape();
    if (inputElemShape != outputShape) {
      return emitOpError() << "expected shape of input (" << inputElemShape
                           << ") to match shape of output (" << outputShape
                           << ")";
    }
  }

  return success();
}

SmallVector<StringRef> MapOp::getIteratorTypesArray() {
  int64_t rank = getInit().getType().getRank();
  return SmallVector<StringRef>(rank, getParallelIteratorTypeName());
}

ArrayAttr MapOp::getIndexingMaps() {
  Builder builder(getContext());
  int64_t rank = getInit().getType().getRank();
  int64_t numIndexingMaps = getOperands().size();
  return builder.getAffineMapArrayAttr(SmallVector<AffineMap>(
      numIndexingMaps, builder.getMultiDimIdentityMap(rank)));
}

void MapOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getGenericEffectsImpl(effects, getOperation()->getResults(),
                        getInputOperands(), getOutputOperands());
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

void ReduceOp::getAsmBlockArgumentNames(Region &region,
                                        OpAsmSetValueNameFn setNameFn) {
  for (Value v : getRegionInputArgs())
    setNameFn(v, "in");
  for (Value v : getRegionOutputArgs())
    setNameFn(v, "init");
}

void ReduceOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  if (!getResults().empty())
    setNameFn(getResults().front(), "reduced");
}

SmallVector<StringRef> ReduceOp::getIteratorTypesArray() {
  int64_t inputRank = getInputs()[0].getType().cast<ShapedType>().getRank();
  SmallVector<StringRef> iteratorTypes(inputRank,
                                       getParallelIteratorTypeName());
  for (int64_t reductionDim : getDimensions())
    iteratorTypes[reductionDim] = getReductionIteratorTypeName();
  return iteratorTypes;
}

ArrayAttr ReduceOp::getIndexingMaps() {
  int64_t inputRank = getInputs()[0].getType().cast<ShapedType>().getRank();
  SmallVector<AffineMap> affineMaps(
      getNumInputs(),
      AffineMap::getMultiDimIdentityMap(inputRank, getContext()));
  AffineMap resultMap =
      AffineMap::getMultiDimIdentityMap(inputRank, getContext())
          .dropResults(getDimensions());
  for (int64_t i = 0, e = getNumOutputs(); i < e; ++i)
    affineMaps.push_back(resultMap);
  return Builder(getContext()).getAffineMapArrayAttr(affineMaps);
}

void ReduceOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getGenericEffectsImpl(effects, getOperation()->getResults(),
                        getInputOperands(), getOutputOperands());
}

static ParseResult parseDenseI64ArrayAttr(OpAsmParser &parser,
                                          NamedAttrList &attributes,
                                          StringRef attributeName) {
  if (parser.parseKeyword(attributeName) || parser.parseEqual())
    return failure();

  attributes.set(attributeName, DenseI64ArrayAttr::parse(parser, Type{}));
  return success();
}

ParseResult ReduceOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parseDstStyleOp(
          parser, result, [&](OpAsmParser &parser, NamedAttrList &attributes) {
            return parseDenseI64ArrayAttr(parser, attributes, "dimensions");
          }))
    return failure();

  SmallVector<OpAsmParser::Argument> regionArgs;
  if (parser.parseArgumentList(regionArgs, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true, /*allowAttrs=*/true)) {
    return failure();
  }

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return failure();

  return success();
}

static void printDenseI64ArrayAttr(OpAsmPrinter &p, StringRef attributeName,
                                   ArrayRef<int64_t> attributeValue) {
  p << attributeName << " = [" << attributeValue << "] ";
}

void ReduceOp::print(OpAsmPrinter &p) {
  p.increaseIndent();
  printCommonStructuredOpPartsWithNewLine(
      p, SmallVector<Value>(getInputOperands()),
      SmallVector<Value>(getOutputOperands()));
  p.printNewline();
  printDenseI64ArrayAttr(p, getDimensionsAttrName(), getDimensions());
  p.printOptionalAttrDict((*this)->getAttrs(), {getDimensionsAttrName()});

  p.printNewline();
  p << "(";
  llvm::interleaveComma(getCombiner().getArguments(), p,
                        [&](auto arg) { p.printRegionArgument(arg); });
  p << ") ";

  p.printRegion(getCombiner(), /*printEntryBlockArgs=*/false);
  p.decreaseIndent();
}

LogicalResult ReduceOp::verify() {
  ArrayRef<int64_t> dimensionsRef = getDimensions();

  for (int64_t i = 1; i < getNumInputs(); ++i) {
    if (getInputs()[i].getType().cast<ShapedType>().getShape() !=
        getInputs()[0].getType().cast<ShapedType>().getShape()) {
      return emitOpError() << "expects all inputs to have the same shapes. "
                              "Shape at input-index "
                           << i
                           << " is not equal to the shape at input-index 0.";
    }
  }
  for (int64_t i = 1; i < getNumOutputs(); ++i) {
    if (getInits()[i].getType().cast<ShapedType>().getShape() !=
        getInits()[0].getType().cast<ShapedType>().getShape()) {
      return emitOpError() << "expects all outputs to have the same shapes. "
                              "Shape at output-index "
                           << i
                           << " is not equal to the shape at output-index 0.";
    }
  }
  auto inputType = getInputs()[0].getType().cast<ShapedType>();
  auto initType = getInits()[0].getType().cast<ShapedType>();

  DenseSet<int64_t> dimensionsToReduce;
  for (int64_t dimension : dimensionsRef) {
    if (dimension < 0 || dimension >= inputType.getRank()) {
      return emitOpError()
             << "dimensions for reduction should be in the range [0, "
             << inputType.getRank() - 1 << "].";
    }
    dimensionsToReduce.insert(dimension);
  }

  auto inputDims = inputType.getShape();
  auto initDims = initType.getShape();

  // Input dimensions that will be left after the reduction.
  SmallVector<int64_t> reducedInputDims;
  for (const auto &en : llvm::enumerate(inputDims)) {
    if (!dimensionsToReduce.count(en.index()))
      reducedInputDims.push_back(en.value());
  }

  if (reducedInputDims.size() != static_cast<size_t>(initType.getRank())) {
    return emitOpError() << "number of dimensions after reduction "
                         << reducedInputDims.size()
                         << " doesn't match the init rank "
                         << initType.getRank();
  }

  if (reducedInputDims != initDims)
    return emitOpError() << "init dimensions [" << initDims
                         << "] doesn't match input dimensions after reduction ["
                         << reducedInputDims << "]";

  Block *block = getBody();
  if (block->getNumArguments() != this->getNumOperands())
    return emitOpError()
           << "mismatching number of operands and block arguments";

  // Check that the first block arguments match the element type of the inputs.
  for (auto [input, bbArg] : llvm::zip(getInputs(), block->getArguments())) {
    Type inputElementType = input.getType().cast<ShapedType>().getElementType();
    if (inputElementType != bbArg.getType())
      return emitOpError()
             << "input element type " << inputElementType
             << " does not match corresponding block argument type "
             << bbArg.getType();
  }

  // Check that the last block arguments match the element type of the outputs.
  for (auto [output, bbArg] :
       llvm::zip(getOutputOperands(),
                 block->getArguments().take_back(getNumOutputs()))) {
    auto outputElementType =
        output->get().getType().cast<ShapedType>().getElementType();
    if (outputElementType != bbArg.getType())
      return emitOpError()
             << "output element type " << outputElementType
             << " does not match corresponding block argument type "
             << bbArg.getType();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
                   mlir::ArrayRef<mlir::NamedAttribute>)>
TransposeOp::getRegionBuilder() {
  return [](mlir::ImplicitLocOpBuilder &b, mlir::Block &block,
            mlir::ArrayRef<mlir::NamedAttribute>) {
    b.create<linalg::YieldOp>(block.getArguments().back());
  };
}

void TransposeOp::createRegion(::mlir::OpBuilder &opBuilder,
                               ::mlir::OperationState &odsState) {
  Region *region = odsState.addRegion();

  SmallVector<Type> argTypes;
  SmallVector<Location> argLocs;
  for (auto t : odsState.operands) {
    argTypes.push_back(getElementTypeOrSelf(t));
    argLocs.push_back(opBuilder.getUnknownLoc());
  }

  // RAII.
  OpBuilder::InsertionGuard guard(opBuilder);
  Block *body =
      opBuilder.createBlock(region, /*insertPt=*/{}, argTypes, argLocs);

  ImplicitLocOpBuilder b(opBuilder.getUnknownLoc(), opBuilder);
  getRegionBuilder()(b, *body, odsState.attributes.getAttrs());
}

void TransposeOp::build(::mlir::OpBuilder &odsBuilder,
                        ::mlir::OperationState &odsState, Value input,
                        Value init, DenseI64ArrayAttr permutation,
                        ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(input);
  odsState.addOperands(init);
  odsState.addAttribute(getPermutationAttrName(odsState.name), permutation);
  odsState.addAttributes(attributes);
  odsState.addTypes(init.getType());

  createRegion(odsBuilder, odsState);
}

void TransposeOp::build(::mlir::OpBuilder &odsBuilder,
                        ::mlir::OperationState &odsState, Value input,
                        Value init, ArrayRef<int64_t> permutation,
                        ArrayRef<NamedAttribute> attributes) {
  build(odsBuilder, odsState, input, init,
        odsBuilder.getDenseI64ArrayAttr(permutation), attributes);
}

ParseResult TransposeOp::parse(OpAsmParser &parser, OperationState &result) {
  if (failed(parseDstStyleOp(
          parser, result, [&](OpAsmParser &parser, NamedAttrList &attributes) {
            return parseDenseI64ArrayAttr(parser, attributes, "permutation");
          })))
    return failure();

  OpBuilder opBuilder(parser.getContext());
  createRegion(opBuilder, result);
  return success();
}

void TransposeOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  if (!getResults().empty())
    setNameFn(getResults().front(), "transposed");
}

void TransposeOp::print(OpAsmPrinter &p) {
  p.increaseIndent();
  printCommonStructuredOpPartsWithNewLine(
      p, SmallVector<Value>(getInputOperands()),
      SmallVector<Value>(getOutputOperands()));
  p.printNewline();
  printDenseI64ArrayAttr(p, getPermutationAttrName(), getPermutation());
  p.printOptionalAttrDict((*this)->getAttrs(), {getPermutationAttrName()});
  p.decreaseIndent();
}

LogicalResult TransposeOp::verify() {
  ArrayRef<int64_t> permutationRef = getPermutation();

  if (!isPermutation(permutationRef))
    return emitOpError("permutation is not valid");

  auto inputType = getInput().getType();
  auto initType = getInit().getType();

  int64_t rank = inputType.getRank();

  if (rank != initType.getRank())
    return emitOpError() << "input rank " << rank
                         << " does not match init rank " << initType.getRank();

  if (rank != static_cast<int64_t>(permutationRef.size()))
    return emitOpError() << "size of permutation " << permutationRef.size()
                         << " does not match the argument rank " << rank;

  auto inputDims = inputType.getShape();
  auto initDims = initType.getShape();

  for (int64_t i = 0; i < rank; ++i) {
    int64_t inputDim = inputDims[permutationRef[i]];
    int64_t initDim = initDims[i];

    if (inputDim != initDim) {
      return emitOpError() << "dim(result, " << i << ") = " << initDim
                           << " doesn't match dim(input, permutation[" << i
                           << "]) = " << inputDim;
    }
  }

  return success();
}

SmallVector<StringRef> TransposeOp::getIteratorTypesArray() {
  int64_t rank = getInit().getType().getRank();
  return SmallVector<StringRef>(rank, getParallelIteratorTypeName());
}

ArrayAttr TransposeOp::getIndexingMaps() {
  Builder builder(getContext());
  int64_t rank = getInit().getType().getRank();
  return builder.getAffineMapArrayAttr(
      {builder.getMultiDimIdentityMap(rank),
       AffineMap::getPermutationMap(
           llvm::to_vector_of<unsigned>(getPermutation()), getContext())});
}

void TransposeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getGenericEffectsImpl(effects, getOperation()->getResults(),
                        getInputOperands(), getOutputOperands());
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

void linalg::YieldOp::print(OpAsmPrinter &p) {
  if (getNumOperands() > 0)
    p << ' ' << getOperands();
  p.printOptionalAttrDict((*this)->getAttrs());
  if (getNumOperands() > 0)
    p << " : " << getOperandTypes();
}

ParseResult YieldOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 2> opInfo;
  SmallVector<Type, 2> types;
  SMLoc loc = parser.getCurrentLocation();
  return failure(parser.parseOperandList(opInfo) ||
                 parser.parseOptionalAttrDict(result.attributes) ||
                 (!opInfo.empty() && parser.parseColonTypeList(types)) ||
                 parser.resolveOperands(opInfo, types, loc, result.operands));
}

// Check the operand number and types must match the element types of the
// LinalgOp interface's shaped operands.
static LogicalResult verifyYield(linalg::YieldOp op, LinalgOp linalgOp) {
  if (op.getNumOperands() != linalgOp.getNumOutputs())
    return op.emitOpError("expected number of yield values (")
           << linalgOp.getNumOutputs()
           << ") to match the number of operands of the enclosing "
           << "LinalgOp (" << op.getNumOperands() << ")";

  for (OpOperand &opOperand : op->getOpOperands()) {
    OpOperand *outputOperand =
        linalgOp.getOutputOperand(opOperand.getOperandNumber());
    Type elementType = getElementTypeOrSelf(outputOperand->get().getType());
    if (opOperand.get().getType() != elementType)
      return op.emitOpError("type of yield operand ")
             << (opOperand.getOperandNumber() + 1) << " ("
             << opOperand.get().getType() << ") doesn't match "
             << "the element type of the enclosing linalg.generic op ("
             << elementType << ")";
  }
  return success();
}

LogicalResult linalg::YieldOp::verify() {
  auto *parentOp = (*this)->getParentOp();
  if (parentOp->getNumRegions() != 1 || parentOp->getRegion(0).empty())
    return emitOpError("expected single non-empty parent region");

  if (auto linalgOp = dyn_cast<LinalgOp>(parentOp))
    return verifyYield(*this, linalgOp);

  return emitOpError("expected parent op with LinalgOp interface");
}

//===----------------------------------------------------------------------===//
// IndexOp
//===----------------------------------------------------------------------===//

LogicalResult IndexOp::verify() {
  auto linalgOp = dyn_cast<LinalgOp>((*this)->getParentOp());
  if (!linalgOp)
    return emitOpError("expected parent op with LinalgOp interface");
  if (linalgOp.getNumLoops() <= getDim())
    return emitOpError("expected dim (")
           << getDim() << ") to be lower than the number of loops ("
           << linalgOp.getNumLoops() << ") of the enclosing LinalgOp";
  return success();
}

/////// Operations corresponding to library calls defined with Tablegen ////////

#include "mlir/Dialect/Linalg/IR/LinalgNamedStructuredOps.yamlgen.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgOps.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"

AffineMap mlir::linalg::extractOrIdentityMap(Optional<AffineMap> maybeMap,
                                             unsigned rank,
                                             MLIRContext *context) {
  if (maybeMap)
    return *maybeMap;
  if (rank == 0)
    return AffineMap::get(context);
  return AffineMap::getMultiDimIdentityMap(rank, context);
}

SmallVector<AffineExpr, 4>
mlir::linalg::makeAffineDimExprs(unsigned num, unsigned &startIdx,
                                 MLIRContext *context) {
  SmallVector<AffineExpr, 4> res;
  res.reserve(num);
  for (unsigned i = 0; i < num; ++i)
    res.push_back(getAffineDimExpr(startIdx++, context));
  return res;
}

SmallVector<AffineExpr, 4> mlir::linalg::concat(ArrayRef<AffineExpr> a,
                                                ArrayRef<AffineExpr> b) {
  auto rangeA = llvm::make_range(a.begin(), a.end());
  auto rangeB = llvm::make_range(b.begin(), b.end());
  auto concatRanges = llvm::concat<const AffineExpr>(rangeA, rangeB);
  return llvm::to_vector<4>(concatRanges);
}

bool mlir::linalg::isPermutation(ArrayRef<int64_t> permutation) {
  // Count the number of appearances for all indices.
  SmallVector<int64_t> indexCounts(permutation.size(), 0);
  for (auto index : permutation) {
    // Exit if the index is out-of-range.
    if (index < 0 || index >= static_cast<int64_t>(permutation.size()))
      return false;
    ++indexCounts[index];
  }
  // Return true if all indices appear once.
  return count(indexCounts, 1) == static_cast<int64_t>(permutation.size());
}

static void appendMangledType(llvm::raw_string_ostream &ss, Type t) {
  if (auto memref = t.dyn_cast<MemRefType>()) {
    ss << "view";
    for (auto size : memref.getShape())
      if (size < 0)
        ss << "sx";
      else
        ss << size << "x";
    appendMangledType(ss, memref.getElementType());
  } else if (auto vec = t.dyn_cast<VectorType>()) {
    ss << "vector";
    llvm::interleave(
        vec.getShape(), [&](int64_t i) { ss << i; }, [&]() { ss << "x"; });
    appendMangledType(ss, vec.getElementType());
  } else if (t.isSignlessIntOrIndexOrFloat()) {
    ss << t;
  } else {
    llvm_unreachable("Invalid type for linalg library name mangling");
  }
}

std::string mlir::linalg::generateLibraryCallName(Operation *op) {
  assert(isa<LinalgOp>(op));
  std::string name(op->getName().getStringRef().str());
  name.reserve(128);
  std::replace(name.begin(), name.end(), '.', '_');
  llvm::raw_string_ostream ss(name);
  ss << "_";
  auto types = op->getOperandTypes();
  llvm::interleave(
      types.begin(), types.end(), [&](Type t) { appendMangledType(ss, t); },
      [&]() { ss << "_"; });
  return ss.str();
}

//===----------------------------------------------------------------------===//
// Canonicalizers and Folders.
//===----------------------------------------------------------------------===//

namespace {
struct EraseDeadLinalgOp : public OpInterfaceRewritePattern<LinalgOp> {
  using OpInterfaceRewritePattern<LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LinalgOp op,
                                PatternRewriter &rewriter) const override {
    for (OpOperand &opOperand : op->getOpOperands()) {
      // Linalg "inputs" may be either tensor or memref type.
      // tensor<0xelt_type> is a convention that may not always mean
      // "0 iterations". Only erase in cases we see memref<...x0x...>.
      auto mt = opOperand.get().getType().dyn_cast<MemRefType>();
      if (!mt)
        continue;
      if (llvm::is_contained(op.getShape(&opOperand), 0)) {
        rewriter.eraseOp(op);
        return success();
      }
    }
    return failure();
  }
};

struct FoldTensorCastProducerOp : public OpInterfaceRewritePattern<LinalgOp> {
  using OpInterfaceRewritePattern<LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LinalgOp op,
                                PatternRewriter &rewriter) const override {
    // If no operand comes from a tensor::CastOp and can be folded then fail.
    bool hasTensorCastOperand =
        llvm::any_of(op->getOpOperands(), [&](OpOperand &opOperand) {
          if (opOperand.get().isa<BlockArgument>())
            return false;
          auto castOp = opOperand.get().getDefiningOp<tensor::CastOp>();
          return castOp && canFoldIntoConsumerOp(castOp);
        });
    if (!hasTensorCastOperand)
      return failure();

    SmallVector<Type, 4> newResultTypes;
    newResultTypes.reserve(op->getNumResults());
    SmallVector<Value, 4> newOperands;
    newOperands.reserve(op->getNumOperands());
    // Inputs may fold.
    for (auto *input : op.getInputOperands()) {
      auto tensorCastOp = input->get().getDefiningOp<tensor::CastOp>();
      newOperands.push_back(canFoldIntoConsumerOp(tensorCastOp)
                                ? tensorCastOp.getSource()
                                : input->get());
    }
    // Init tensors may fold, in which case the resultType must also change.
    for (auto *output : op.getOutputOperands()) {
      auto tensorCastOp = output->get().getDefiningOp<tensor::CastOp>();
      bool fold = canFoldIntoConsumerOp(tensorCastOp);
      newOperands.push_back(fold ? tensorCastOp.getOperand() : output->get());
      newResultTypes.push_back(newOperands.back().getType());
    }
    // Clone op.
    Operation *newOp =
        op.clone(rewriter, op->getLoc(), newResultTypes, newOperands);
    SmallVector<Value, 4> replacements;
    replacements.reserve(newOp->getNumResults());
    for (auto result : llvm::zip(op->getResults(), newOp->getResults())) {
      Value oldResult = std::get<0>(result);
      Value newResult = std::get<1>(result);
      if (newResult.getType() != oldResult.getType()) {
        replacements.push_back(rewriter.create<tensor::CastOp>(
            op->getLoc(), oldResult.getType(), newResult));
      } else {
        replacements.push_back(newResult);
      }
    }
    rewriter.replaceOp(op, replacements);

    return success();
  }
};

/// Fold LinalgOps with `tensor.cast` consumer if the `tensor.cast` has
/// result that is more static than the linalg op.
struct FoldTensorCastConsumerOp : public OpRewritePattern<tensor::CastOp> {
  using OpRewritePattern<tensor::CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::CastOp castOp,
                                PatternRewriter &rewriter) const override {
    if (!tensor::canFoldIntoProducerOp(castOp))
      return failure();

    auto linalgOp = castOp.getSource().getDefiningOp<LinalgOp>();
    if (!linalgOp)
      return failure();

    // Cast can be in conditionally reachable region, if which case folding will
    // generate invalid code. Only conservatively fold ops in same block for
    // now.
    if (castOp->getBlock() != linalgOp->getBlock())
      return failure();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(linalgOp);

    Location loc = linalgOp.getLoc();
    OpResult resultValue = castOp.getSource().cast<OpResult>();
    unsigned resultNumber = resultValue.getResultNumber();
    auto resultType = castOp->getResult(0).getType().cast<RankedTensorType>();
    // Replace the `outs` for the result with a `tensor.cast`. This cast is now
    // going from a more dynamic shape to a less dynamic shape. If the producer
    // for this cast, i.e. producer of the out operand, is also an operation
    // that folds with tensor.cast consumer (like this pattern), the cast will
    // continue to propagate as far up the stack as it can go.
    OpOperand *outOperand = linalgOp.getOutputOperand(resultNumber);
    Value newOperand =
        rewriter.create<tensor::CastOp>(loc, resultType, outOperand->get());
    SmallVector<Value> newOperands{linalgOp.getInputOperands()};
    SmallVector<Value> outputOperands{linalgOp.getOutputOperands()};
    outputOperands[resultNumber] = newOperand;
    newOperands.append(outputOperands.begin(), outputOperands.end());

    SmallVector<Type> resultTypes(linalgOp->result_type_begin(),
                                  linalgOp->result_type_end());
    resultTypes[resultNumber] = resultType;
    Operation *newOp = linalgOp.clone(rewriter, loc, resultTypes, newOperands);

    // Create a tensor.cast operation back to the original type.
    Value castBack = rewriter.create<tensor::CastOp>(
        loc, resultValue.getType(), newOp->getResult(resultNumber));

    SmallVector<Value> results(newOp->result_begin(), newOp->result_end());
    results[resultNumber] = castBack;
    rewriter.replaceOp(linalgOp, results);
    rewriter.replaceOp(castOp, newOp->getResult(resultNumber));
    return success();
  }
};

/// For each of the operand in `operands` this function maps the static sizes of
/// dimensions to their affine dim expressions.
static void populateMap(LinalgOp linalgOp, MutableArrayRef<OpOperand> operands,
                        llvm::DenseMap<AffineExpr, int64_t> &affineExprToSize) {
  for (OpOperand &opOperand : operands) {
    if (linalgOp.isScalar(&opOperand))
      continue;
    Value src = opOperand.get();
    auto sourceType = src.getType().cast<RankedTensorType>();
    auto sourceMap = linalgOp.getMatchingIndexingMap(&opOperand);

    // Get the `sourceShape` of the `sourceType`. If the operand is a result of
    // `tensor.cast` operation and source of the cast operation has a static
    // shape, then assign it to the `sourceShape`.
    auto *parentOp = src.getDefiningOp();
    ArrayRef<int64_t> sourceShape = sourceType.getShape();
    if (parentOp) {
      if (auto castOp = dyn_cast<tensor::CastOp>(parentOp)) {
        Value castSource = castOp.getSource();
        auto castSourceType = castSource.getType().cast<RankedTensorType>();
        if (castSourceType.hasStaticShape())
          sourceShape = castSourceType.getShape();
      }
    }

    // If the source shape's dimension has a static shape, map the affine dim
    // expression to the known static size.
    for (unsigned i = 0; i < sourceShape.size(); i++) {
      if (sourceType.isDynamicDim(i))
        continue;
      if (auto affineDimExpr = sourceMap.getResult(i).dyn_cast<AffineDimExpr>())
        affineExprToSize.try_emplace(affineDimExpr, sourceShape[i]);
    }
  }
}

/// Creates new operand w.r.t 'opOperand' of `linalgOp` with static sizes
/// mapped in `affineExprToSize`. New operands are created in `newOperands` and
/// their result types is stored in `resultTypes`. If `opOperand` requires no
/// change then `changeNeeded` is false and same operand is added in the
/// `newOperands` list.
static void createNewOperandWithStaticSizes(
    Location loc, PatternRewriter &rewriter, OpOperand *opOperand,
    llvm::DenseMap<AffineExpr, int64_t> &affineExprToSize, LinalgOp linalgOp,
    SmallVector<Value> &newOperands, SmallVector<Type> &resultTypes,
    bool &changeNeeded) {
  Value src = opOperand->get();
  newOperands.push_back(src);
  if (linalgOp.isScalar(opOperand))
    return;
  auto sourceType = src.getType().cast<RankedTensorType>();
  Type resultType = sourceType;
  if (sourceType.hasStaticShape() && linalgOp.isOutput(opOperand)) {
    resultTypes.push_back(resultType);
    return;
  }
  ArrayRef<int64_t> sourceShape = sourceType.getShape();
  AffineMap sourceMap = linalgOp.getMatchingIndexingMap(opOperand);
  SmallVector<int64_t> newShape;
  // If operand is updated with new shape, `newOperandNeeded` will be
  // true.
  bool newOperandNeeded = false;
  for (unsigned i = 0; i < sourceShape.size(); i++) {
    int64_t dimShape = sourceShape[i];
    AffineExpr dimExpr = sourceMap.getResult(i);
    if (affineExprToSize.find(dimExpr) == affineExprToSize.end() ||
        !sourceType.isDynamicDim(i)) {
      newShape.push_back(dimShape);
      continue;
    }
    // Dimension has a dynamic shape and corresponding affine dim
    // expression is present in the map. So assign the size for the
    // given affine dim expression to the dimension.
    newShape.push_back(affineExprToSize[dimExpr]);
    newOperandNeeded = true;
  }
  resultType = RankedTensorType::get(newShape, sourceType.getElementType());
  if (newOperandNeeded) {
    changeNeeded = true;
    // Get the new operand value given its size and element type by
    // casting it.
    Value newOperand = rewriter.create<tensor::CastOp>(loc, resultType, src);
    unsigned index = opOperand->getOperandNumber();
    newOperands[index] = newOperand;
  }
  if (linalgOp.isOutput(opOperand))
    resultTypes.push_back(resultType);
}

/// Static shapes for the operands can be inferred if any one of the operands
/// have a static shape. This can be done by referring to the affine dim
/// expressions for the operand.
struct InferStaticShapeOfOperands : public OpInterfaceRewritePattern<LinalgOp> {
  using OpInterfaceRewritePattern<LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!linalgOp.hasTensorSemantics())
      return failure();

    // Maps must be projected permutations.
    if (llvm::any_of(linalgOp.getIndexingMapsArray(), [](AffineMap map) {
          return !map.isProjectedPermutation();
        }))
      return failure();

    // Maps affine dim expressions to the static size of that dimension.
    llvm::DenseMap<AffineExpr, int64_t> affineExprToSize;
    Location loc = linalgOp.getLoc();

    // For each of the affine dim expression, check if the size is known. If
    // known add that in the map.
    populateMap(linalgOp, linalgOp->getOpOperands(), affineExprToSize);

    SmallVector<Value> newOperands;
    SmallVector<Type> resultTypes;

    // `changeNeeded` is `false` if the operands of `linalgOp` require no
    // change in their types.
    bool changeNeeded = false;
    newOperands.reserve(linalgOp->getNumOperands());
    resultTypes.reserve(linalgOp.getNumOutputs());

    // Iterate over all the operands and update the static sizes.
    for (OpOperand &opOperand : linalgOp->getOpOperands()) {
      createNewOperandWithStaticSizes(loc, rewriter, &opOperand,
                                      affineExprToSize, linalgOp, newOperands,
                                      resultTypes, changeNeeded);
    }

    // If the generic op has all the required static information, no
    // canonicalization needed.
    if (!changeNeeded)
      return failure();

    // Clone op.
    Operation *newOp =
        linalgOp.clone(rewriter, linalgOp->getLoc(), resultTypes, newOperands);
    SmallVector<Value> replacements;
    replacements.reserve(newOp->getNumResults());
    for (auto it : llvm::zip(linalgOp->getResults(), newOp->getResults())) {
      Value newResult = std::get<1>(it);
      Value oldResult = std::get<0>(it);
      Type newType = newResult.getType();
      Type oldType = oldResult.getType();
      replacements.push_back(
          (newType != oldType)
              ? rewriter.create<tensor::CastOp>(loc, oldType, newResult)
              : newResult);
    }
    rewriter.replaceOp(linalgOp, replacements);
    return success();
  }
};

} // namespace

// All named ops canonicalizers and folders are auto-generated in the
// .cpp.inc.

//===----------------------------------------------------------------------===//
// LinalgDialect
//===----------------------------------------------------------------------===//

void LinalgDialect::getCanonicalizationPatterns(
    RewritePatternSet &results) const {
  results.add<EraseDeadLinalgOp, FoldTensorCastConsumerOp,
              FoldTensorCastProducerOp, InferStaticShapeOfOperands>(
      getContext());
}

Operation *LinalgDialect::materializeConstant(OpBuilder &builder,
                                              Attribute value, Type type,
                                              Location loc) {
  return builder.create<arith::ConstantOp>(loc, type, value);
}
