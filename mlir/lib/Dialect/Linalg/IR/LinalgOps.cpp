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

#include "mlir/Dialect/Arithmetic/Utils/Utils.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Parser.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::linalg;

#include "mlir/Dialect/Linalg/IR/LinalgOpsDialect.cpp.inc"

/// Forward declarations.

/// Generic entry point to create the block for the region of a LinalgOp.
/// This is used by both named structured ops created by ods-gen and by manually
/// defined C++ ops.
/// This is used by both builders and parsers.
/// This function creates the block in the region with arguments corresponding
/// to the elemental types of `inputTypes` and `outputTypes`. The latter are
/// asserted to be of ShapedType.
template <typename NamedStructuredOpType>
static void fillStructuredOpRegion(
    OpBuilder &opBuilder, Region &region, TypeRange inputTypes,
    TypeRange outputTypes, ArrayRef<NamedAttribute> attrs,
    llvm::function_ref<void(unsigned, unsigned)> errorHandler = nullptr);

/// Generic entry point to create both the region and the block of a LinalgOp.
template <typename NamedStructuredOpType>
static void
createAndFillStructuredOpRegion(OpBuilder &opBuilder, OperationState &result,
                                TypeRange inputTypes, TypeRange outputTypes);

/// Common parsing and printing used for both named structured ops created by
/// ods-gen and by manually defined C++ ops. Does not handle regions.
static ParseResult
parseCommonStructuredOpParts(OpAsmParser &parser, OperationState &result,
                             SmallVectorImpl<Type> &inputTypes,
                             SmallVectorImpl<Type> &outputTypes);
template <typename NamedStructuredOpType>
static void printCommonStructuredOpParts(OpAsmPrinter &p,
                                         NamedStructuredOpType op);

/// Specific parsing and printing for named structured ops created by ods-gen.
template <typename NamedStructuredOpType>
static ParseResult
parseNamedStructuredOpRegion(OpAsmParser &parser, Region &region,
                             TypeRange inputTypes, TypeRange outputTypes,
                             ArrayRef<NamedAttribute> attrs);

static ParseResult
parseNamedStructuredOpResults(OpAsmParser &parser,
                              SmallVectorImpl<Type> &resultTypes);

template <typename NamedStructuredOpType>
static ParseResult parseNamedStructuredOp(OpAsmParser &parser,
                                          OperationState &result);

static void printNamedStructuredOpResults(OpAsmPrinter &p,
                                          TypeRange resultTypes);

template <typename NamedStructuredOpType>
static void printNamedStructuredOp(OpAsmPrinter &p, NamedStructuredOpType op);

/// This is a common class used for patterns of the form
/// ```
///    someop(memrefcast(%src)) -> someop(%src)
/// ```
/// It folds the source of the memref.cast into the root operation directly.
static LogicalResult foldMemRefCast(Operation *op) {
  bool folded = false;
  for (OpOperand &operand : op->getOpOperands()) {
    auto castOp = operand.get().getDefiningOp<memref::CastOp>();
    if (castOp && memref::CastOp::canFoldIntoConsumerOp(castOp)) {
      operand.set(castOp.getOperand());
      folded = true;
    }
  }
  return success(folded);
}

/// This is a specialization of `foldMemRefCast` used for patterns of the form
/// ```
///    tiled_loop(memrefcast(%src)) -> tiled_loop(%src)
/// ```
/// It folds the source of the memref.cast into the root operation directly.
static LogicalResult foldMemRefCastInTiledLoopOp(TiledLoopOp op) {
  bool folded = false;
  Location loc = op->getLoc();

  Block *body = op.getBody();
  OpBuilder b = OpBuilder::atBlockBegin(body);

  // Update `input` and `output` operands and block arguments if necessary.
  // Operands list: [lbs, ubs, steps, inputs, outputs].
  // Block args list: [ivs, inputs, outputs].
  for (size_t operandIndex = op.getNumControlOperands(),
              bbArgIndex = op.getNumLoops(), e = op.getNumOperands();
       operandIndex < e; ++operandIndex, ++bbArgIndex) {
    OpOperand &operand = op->getOpOperand(operandIndex);

    auto castOp = operand.get().getDefiningOp<memref::CastOp>();
    if (castOp && memref::CastOp::canFoldIntoConsumerOp(castOp)) {
      operand.set(castOp.getOperand());
      BlockArgument newBbArg = body->insertArgument(
          bbArgIndex, castOp.getOperand().getType(), op.getLoc());
      BlockArgument oldBbArg = body->getArgument(newBbArg.getArgNumber() + 1);

      // Insert memref.cast back to the original type.
      oldBbArg.replaceAllUsesWith(
          b.create<memref::CastOp>(loc, oldBbArg.getType(), newBbArg));
      body->eraseArgument(oldBbArg.getArgNumber());

      folded = true;
    }
  }
  return success(folded);
}

//===----------------------------------------------------------------------===//
// Region builder helper.
// TODO: Move this to a utility library.
// The public methods on this class are referenced directly from generated code
// and bind by name to math and type conversion functions in the DSL as:
//   `arithfn__{fnName}`
//   `typefn__{fnName}`
// Examples:
//   `arithfn__add`
//   `arithfn__mul`
//   `typefn__cast`
// The naming convention is intentional in order to match snake-cased DSL names.
// See mlir-linalg-ods-yaml-gen.cpp for the code that mates to this class.
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

  // NOLINTNEXTLINE(*-identifier-naming): externally called.
  Value typefn__cast(Type toType, Value operand) {
    return cast(toType, operand, false);
  }

  // NOLINTNEXTLINE(*-identifier-naming): externally called.
  Value typefn__cast_unsigned(Type toType, Value operand) {
    return cast(toType, operand, true);
  }

  // NOLINTNEXTLINE(*-identifier-naming): externally called.
  Value arithfn__add(Value lhs, Value rhs) {
    OpBuilder builder = getBuilder();
    if (isFloatingPoint(lhs))
      return builder.create<arith::AddFOp>(lhs.getLoc(), lhs, rhs);
    if (isInteger(lhs))
      return builder.create<arith::AddIOp>(lhs.getLoc(), lhs, rhs);
    llvm_unreachable("unsupported non numeric type");
  }

  // NOLINTNEXTLINE(*-identifier-naming): externally called.
  Value arithfn__exp(Value x) {
    OpBuilder builder = getBuilder();
    if (isFloatingPoint(x))
      return builder.create<math::ExpOp>(x.getLoc(), x);
    llvm_unreachable("unsupported non numeric type");
  }

  // NOLINTNEXTLINE(*-identifier-naming): externally called.
  Value arithfn__log(Value x) {
    OpBuilder builder = getBuilder();
    if (isFloatingPoint(x))
      return builder.create<math::LogOp>(x.getLoc(), x);
    llvm_unreachable("unsupported non numeric type");
  }

  // NOLINTNEXTLINE(*-identifier-naming): externally called.
  Value arithfn__sub(Value lhs, Value rhs) {
    OpBuilder builder = getBuilder();
    if (isFloatingPoint(lhs))
      return builder.create<arith::SubFOp>(lhs.getLoc(), lhs, rhs);
    if (isInteger(lhs))
      return builder.create<arith::SubIOp>(lhs.getLoc(), lhs, rhs);
    llvm_unreachable("unsupported non numeric type");
  }

  // NOLINTNEXTLINE(*-identifier-naming): externally called.
  Value arithfn__mul(Value lhs, Value rhs) {
    OpBuilder builder = getBuilder();
    if (isFloatingPoint(lhs))
      return builder.create<arith::MulFOp>(lhs.getLoc(), lhs, rhs);
    if (isInteger(lhs))
      return builder.create<arith::MulIOp>(lhs.getLoc(), lhs, rhs);
    llvm_unreachable("unsupported non numeric type");
  }

  // NOLINTNEXTLINE(*-identifier-naming): externally called.
  Value arithfn__max(Value lhs, Value rhs) {
    OpBuilder builder = getBuilder();
    if (isFloatingPoint(lhs))
      return builder.create<arith::MaxFOp>(lhs.getLoc(), lhs, rhs);
    if (isInteger(lhs))
      return builder.create<arith::MaxSIOp>(lhs.getLoc(), lhs, rhs);
    llvm_unreachable("unsupported non numeric type");
  }

  // NOLINTNEXTLINE(*-identifier-naming): externally called.
  Value arithfn__max_unsigned(Value lhs, Value rhs) {
    OpBuilder builder = getBuilder();
    if (isFloatingPoint(lhs))
      return builder.create<arith::MaxFOp>(lhs.getLoc(), lhs, rhs);
    if (isInteger(lhs))
      return builder.create<arith::MaxUIOp>(lhs.getLoc(), lhs, rhs);
    llvm_unreachable("unsupported non numeric type");
  }

  // NOLINTNEXTLINE(*-identifier-naming): externally called.
  Value arithfn__min(Value lhs, Value rhs) {
    OpBuilder builder = getBuilder();
    if (isFloatingPoint(lhs))
      return builder.create<arith::MinFOp>(lhs.getLoc(), lhs, rhs);
    if (isInteger(lhs))
      return builder.create<arith::MinSIOp>(lhs.getLoc(), lhs, rhs);
    llvm_unreachable("unsupported non numeric type");
  }

  // NOLINTNEXTLINE(*-identifier-naming): externally called.
  Value arithfn__min_unsigned(Value lhs, Value rhs) {
    OpBuilder builder = getBuilder();
    if (isFloatingPoint(lhs))
      return builder.create<arith::MinFOp>(lhs.getLoc(), lhs, rhs);
    if (isInteger(lhs))
      return builder.create<arith::MinUIOp>(lhs.getLoc(), lhs, rhs);
    llvm_unreachable("unsupported non numeric type");
  }

  void yieldOutputs(ValueRange values) {
    assert(!values.empty() && "linalg ops must yield outputs");
    if (values.empty())
      return;
    Value first = values.front();
    OpBuilder builder = getBuilder();
    builder.create<YieldOp>(first.getLoc(), values);
  }

  Value constant(const std::string &value) {
    OpBuilder builder = getBuilder();
    Location loc = builder.getUnknownLoc();
    Attribute valueAttr = parseAttribute(value, builder.getContext());
    return builder.create<arith::ConstantOp>(loc, valueAttr.getType(),
                                             valueAttr);
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
  MLIRContext *context;
  Block &block;

  bool isFloatingPoint(Value value) { return value.getType().isa<FloatType>(); }
  bool isInteger(Value value) { return value.getType().isa<IntegerType>(); }

  OpBuilder getBuilder() {
    OpBuilder builder(context);
    builder.setInsertionPointToEnd(&block);
    return builder;
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// FillOp
//===----------------------------------------------------------------------===//
void FillOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block,
                           ArrayRef<NamedAttribute> attrs) {
  assert(block.getNumArguments() == 2 && "FillOp regionBuilder expects 2 args");
  b.create<linalg::YieldOp>(block.getArgument(0));
}

void FillOp::build(OpBuilder &builder, OperationState &result, Value value,
                   Value output) {
  build(builder, result, output.getType().dyn_cast<RankedTensorType>(), value,
        output);
  fillStructuredOpRegion<FillOp>(
      builder, *result.regions.front(), TypeRange{value.getType()},
      TypeRange{output.getType()}, result.attributes.getAttrs(), {});
}

ParseResult parseFillOpRegion(OpAsmParser &parser, Region &r, Type valueType,
                              Type outputType) {
  OpBuilder opBuilder(parser.getContext());
  fillStructuredOpRegion<FillOp>(opBuilder, r, TypeRange{valueType},
                                 TypeRange{outputType}, {});
  return success();
}

/// FillOp region is elided when printing.
void printFillOpRegion(OpAsmPrinter &, Operation *, Region &, Type, Type) {}

LogicalResult FillOp::verify() {
  OpOperand *output = getOutputOperand(0);
  Type fillType = value().getType();
  if (getElementTypeOrSelf(output->get()) != fillType)
    return emitOpError("expects fill type to match view elemental type");
  return success();
}

void FillOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (output().getType().isa<MemRefType>())
    effects.emplace_back(MemoryEffects::Write::get(), output(),
                         SideEffects::DefaultResource::get());
}

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
    auto oldFill = reshapeOp.src().template getDefiningOp<FillOp>();
    if (!oldFill)
      return failure();

    Location loc = oldFill.getLoc();
    auto newInit = rewriter.create<TensorReshapeOp>(
        loc, reshapeOp.getResultType(), oldFill.output(),
        reshapeOp.reassociation());
    rewriter.replaceOpWithNewOp<FillOp>(reshapeOp, oldFill.value(), newInit);

    return success();
  }
};

/// Fold tensor.pad(linalg.fill) into linalg.fill if the padding value and the
/// filling value are the same.
struct FoldFillWithPad final : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    auto fillOp = padOp.source().getDefiningOp<linalg::FillOp>();
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
    auto newInitOp = rewriter.create<InitTensorOp>(
        padOp.getLoc(), reifiedShape.front(), staticShape,
        oldResultType.getElementType());
    auto newFillOp =
        rewriter.create<FillOp>(fillOp.getLoc(), padValue, newInitOp);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(padOp, oldResultType,
                                                newFillOp.result());

    return success();
  }
};

} // namespace

void FillOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results
      .add<FoldFillWithPad, FoldFillWithTensorReshape<tensor::CollapseShapeOp>,
           FoldFillWithTensorReshape<tensor::ExpandShapeOp>>(context);
}

//===----------------------------------------------------------------------===//
// GenericOps
//===----------------------------------------------------------------------===//
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
        libraryCall.empty() ? StringAttr()
                            : builder.getStringAttr(libraryCall));
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
  printCommonStructuredOpParts(p, *this);

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
  if (!region().empty()) {
    p << ' ';
    p.printRegion(region());
  }

  // Print results.
  printNamedStructuredOpResults(p, result_tensors().getTypes());
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

  SmallVector<OpAsmParser::OperandType, 8> regionOperands;
  std::unique_ptr<Region> region = std::make_unique<Region>();
  SmallVector<Type, 8> operandTypes, regionTypes;
  if (parser.parseRegion(*region, regionOperands, regionTypes))
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
    ValueRange results, ValueRange inputBuffers, ValueRange outputs) {
  for (Value value : results) {
    effects.emplace_back(MemoryEffects::Allocate::get(), value,
                         SideEffects::DefaultResource::get());
  }
  for (Value value : inputBuffers) {
    effects.emplace_back(MemoryEffects::Read::get(), value,
                         SideEffects::DefaultResource::get());
  }
  for (Value value : outputs) {
    effects.emplace_back(MemoryEffects::Read::get(), value,
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), value,
                         SideEffects::DefaultResource::get());
  }
}

void GenericOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  SmallVector<Value> inputBuffers = getInputBufferOperands();
  SmallVector<Value> outputBuffers = getOutputBufferOperands();
  getGenericEffectsImpl(effects, getOperation()->getResults(), inputBuffers,
                        outputBuffers);
}

template <typename GenericOpType>
static LogicalResult verifyGenericOp(GenericOpType op) {
  return success();
}

LogicalResult GenericOp::verify() { return verifyGenericOp(*this); }

namespace {
// Deduplicate redundant args of a linalg generic op.
// An arg is redundant if it has the same Value and indexing map as another.
struct DeduplicateGenericOpInputs : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    // Associate each input to an equivalent "canonical" input that has the same
    // Value and indexing map.
    //
    // In the non-duplicate case, input `i` will have canonical input `i`. But
    // in the case of duplicated inputs, the canonical input could be some other
    // input `< i`. That is, a later input will have some earlier input as its
    // canonical input.
    llvm::SmallDenseMap<std::pair<Value, AffineMap>, unsigned> canonicalInput;
    // For later remapping tasks like deduplicating payload block arguments,
    // having a simple "inputIndex -> canonicalInputIndex" integer mapping is
    // convenient.
    SmallVector<unsigned> canonicalInputIndices;
    for (OpOperand *opOperand : genericOp.getInputOperands()) {
      AffineMap indexingMap = genericOp.getTiedIndexingMap(opOperand);
      // STL-like maps have a convenient behavior for our use case here. In the
      // case of duplicate keys, the insertion is rejected, and the returned
      // iterator gives access to the value already in the map.
      auto pair = canonicalInput.insert(
          {{opOperand->get(), indexingMap}, opOperand->getOperandNumber()});
      canonicalInputIndices.push_back(pair.first->second);
    }

    // If there are no duplicate args, then bail out.
    if (canonicalInput.size() == genericOp.getNumInputs())
      return failure();

    // The operands for the newly canonicalized op.
    SmallVector<Value> newInputOperands;
    for (OpOperand *opOperand : genericOp.getInputOperands())
      if (canonicalInputIndices[opOperand->getOperandNumber()] ==
          opOperand->getOperandNumber())
        newInputOperands.push_back(opOperand->get());

    // Repair the indexing maps by filtering out the ones that have been
    // eliminated.
    SmallVector<AffineMap> newIndexingMaps;
    for (OpOperand *opOperand : genericOp.getInputOperands())
      if (canonicalInputIndices[opOperand->getOperandNumber()] ==
          opOperand->getOperandNumber())
        newIndexingMaps.push_back(genericOp.getTiedIndexingMap(opOperand));
    for (OpOperand *opOperand : genericOp.getOutputOperands())
      newIndexingMaps.push_back(genericOp.getTiedIndexingMap(opOperand));

    // Clone the old op with new operands.
    SmallVector<Value> outputOperands = genericOp.getOutputOperands();
    auto newOp = rewriter.create<GenericOp>(
        genericOp.getLoc(), genericOp->getResultTypes(), newInputOperands,
        outputOperands, rewriter.getAffineMapArrayAttr(newIndexingMaps),
        genericOp.iterator_types(), genericOp.docAttr(),
        genericOp.library_callAttr());

    // Copy over unknown attributes. They might be load bearing for some flow.
    ArrayRef<StringRef> odsAttrs = genericOp.getAttributeNames();
    for (NamedAttribute kv : genericOp->getAttrs()) {
      if (!llvm::is_contained(odsAttrs, kv.getName().getValue())) {
        newOp->setAttr(kv.getName(), kv.getValue());
      }
    }

    rewriter.inlineRegionBefore(genericOp.region(), newOp.region(),
                                newOp.region().begin());

    // Repair the payload entry block by RAUW'ing redundant arguments and
    // erasing them.
    Block &payload = newOp.region().front();
    SmallVector<OpOperand *> inputOperands = genericOp.getInputOperands();
    for (OpOperand *opOperand : llvm::reverse(inputOperands)) {
      // Iterate in reverse, so that we erase later args first, preventing the
      // argument list from shifting unexpectedly and invalidating all our
      // indices.
      unsigned operandNumber = opOperand->getOperandNumber();
      if (canonicalInputIndices[operandNumber] == operandNumber)
        continue;
      payload.getArgument(operandNumber)
          .replaceAllUsesWith(
              payload.getArgument(canonicalInputIndices[operandNumber]));
      payload.eraseArgument(operandNumber);
    }

    rewriter.replaceOp(genericOp, newOp->getResults());
    return success();
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
    if (llvm::any_of(genericOp.getIndexingMaps(),
                     [](AffineMap map) { return !map.isIdentity(); }))
      return failure();

    // Check that the body of the linalg operation is just a linalg.yield
    // operation.
    Block &body = genericOp.region().front();
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

    // Get the argument number of the returned values. That is the operand
    // number to use for replacing uses of this operation.
    SmallVector<Value> returnedArgs;
    for (const auto &yieldVal : llvm::enumerate(yieldOp.values())) {
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
        else
          returnedArg = rewriter.create<tensor::CastOp>(
              genericOp.getLoc(), resultType, returnedArg);
      }
      returnedArgs.push_back(returnedArg);
    }

    if (returnedArgs.size() != genericOp->getNumResults())
      return failure();
    rewriter.replaceOp(genericOp, returnedArgs);
    return success();
  }
};
} // namespace

void GenericOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<DeduplicateGenericOpInputs, EraseIdentityGenericOp>(context);
}

//===----------------------------------------------------------------------===//
// InitTensorOp
//===----------------------------------------------------------------------===//

void InitTensorOp::build(OpBuilder &b, OperationState &result,
                         ArrayRef<OpFoldResult> sizes, Type elementType,
                         ArrayRef<NamedAttribute> attrs) {
  SmallVector<Value, 4> dynamicSizes;
  SmallVector<int64_t, 4> staticSizes;
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes,
                             ShapedType::kDynamicSize);
  auto resultType = RankedTensorType ::get(staticSizes, elementType);
  build(b, result, resultType, dynamicSizes, b.getI64ArrayAttr(staticSizes));
  result.addAttributes(attrs);
}

LogicalResult InitTensorOp::verify() {
  RankedTensorType resultType = getType();
  SmallVector<int64_t, 4> staticSizes = llvm::to_vector<4>(llvm::map_range(
      static_sizes().cast<ArrayAttr>(),
      [](Attribute a) -> int64_t { return a.cast<IntegerAttr>().getInt(); }));

  if (failed(verifyListOfOperandsOrIntegers(
          *this, "sizes", resultType.getRank(), static_sizes(), sizes(),
          ShapedType::isDynamic)))
    return failure();

  if (static_sizes().size() != static_cast<unsigned>(resultType.getRank()))
    return emitError("expected ") << resultType.getRank() << " sizes values";

  Type expectedType = InitTensorOp::inferResultType(
      staticSizes, resultType.getElementType(), resultType.getEncoding());
  if (resultType != expectedType) {
    return emitError("specified type ")
           << resultType << " does not match the inferred type "
           << expectedType;
  }
  return success();
}

Type InitTensorOp::inferResultType(ArrayRef<int64_t> staticSizes,
                                   Type elementType, Attribute encoding) {
  return RankedTensorType::get(staticSizes, elementType, encoding);
}

namespace {
/// Change the type of the result of a `linalg.init_tensor` by making the result
/// type statically sized along dimension that in the original operation where
/// defined as dynamic, but the size was defined using a `constant` op. For
/// example
///
///  %c5 = arith.constant 5: index
///  %0 = linalg.init_tensor [%arg0, %c5] : tensor<?x?xf32>
///
///  to
///
///  %0 = linalg.init_tensor [%arg0, 5] : tensor<?x5xf32>
struct ReplaceStaticShapeDims : OpRewritePattern<InitTensorOp> {
  using OpRewritePattern<InitTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(InitTensorOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value, 4> dynamicSizes;
    SmallVector<int64_t, 4> staticSizes;
    for (unsigned i = 0, e = op.getType().getRank(); i != e; ++i) {
      // If the size is already static, nothing to do.
      if (!op.isDynamicSize(i)) {
        staticSizes.push_back(op.getStaticSize(i));
        continue;
      }

      // If the size is dynamic but defined using a `constant` op, get the
      // constant value to find the static size to use.
      unsigned operandNum = op.getIndexOfDynamicSize(i);
      Value sizeOperand = op.getOperand(operandNum);
      if (auto constantIndexOp =
              sizeOperand.getDefiningOp<arith::ConstantIndexOp>()) {
        staticSizes.push_back(constantIndexOp.value());
        continue;
      }

      // Fallback case. Keep the size dynamic.
      dynamicSizes.push_back(sizeOperand);
      staticSizes.push_back(ShapedType::kDynamicSize);
    }
    RankedTensorType newType =
        RankedTensorType::get(staticSizes, op.getType().getElementType());
    if (newType == op.getType())
      return failure();
    auto newOp =
        rewriter.create<InitTensorOp>(op.getLoc(), newType, dynamicSizes,
                                      rewriter.getI64ArrayAttr(staticSizes));
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, op.getType(), newOp);
    return success();
  }
};
} // namespace

namespace {
/// Since `init_tensor` operation creates a tensor needed only for its shape, a
/// slice of this is also needed only for its shape. The result can be
/// replaced by a new init_tensor operation of the same size as the extract
/// slice op.
struct FoldInitTensorWithExtractSliceOp
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    if (!sliceOp.source().getDefiningOp<linalg::InitTensorOp>())
      return failure();
    // ExtractSliceOp may be rank-reducing; its dynamic sizes must be preserved
    // as well as its result type.
    rewriter.replaceOpWithNewOp<linalg::InitTensorOp>(
        sliceOp, sliceOp.sizes(),
        sliceOp.result().getType().cast<RankedTensorType>().getShape(),
        sliceOp.getSourceType().getElementType());
    return success();
  }
};

template <typename TensorReshapeOp>
struct FoldInitTensorWithTensorReshapeOp
    : public OpRewritePattern<TensorReshapeOp> {
  using OpRewritePattern<TensorReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    if (!reshapeOp.src().template getDefiningOp<InitTensorOp>())
      return failure();
    Location loc = reshapeOp.getLoc();
    ReifiedRankedShapedTypeDims resultShapes;
    ReifyRankedShapedTypeOpInterface reifyShapedTypeInterface =
        cast<ReifyRankedShapedTypeOpInterface>(reshapeOp.getOperation());
    if (failed(reifyShapedTypeInterface.reifyResultShapes(rewriter,
                                                          resultShapes)) ||
        !llvm::hasSingleElement(resultShapes))
      return failure();
    Value initTensor = rewriter.create<InitTensorOp>(
        loc, getAsOpFoldResult(resultShapes[0]),
        reshapeOp.getResultType().getElementType());
    if (initTensor.getType() != reshapeOp.getResultType()) {
      rewriter.replaceOpWithNewOp<tensor::CastOp>(
          reshapeOp, reshapeOp.getResultType(), initTensor);
    } else {
      rewriter.replaceOp(reshapeOp, initTensor);
    }
    return success();
  }
};

struct FoldInitTensorWithDimOp : public OpRewritePattern<tensor::DimOp> {
  using OpRewritePattern<tensor::DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::DimOp dimOp,
                                PatternRewriter &rewriter) const override {
    Optional<int64_t> maybeConstantIndex = dimOp.getConstantIndex();
    auto initTensorOp = dimOp.source().getDefiningOp<linalg::InitTensorOp>();
    if (!initTensorOp || !maybeConstantIndex)
      return failure();
    if (!initTensorOp.isDynamicSize(*maybeConstantIndex))
      return failure();
    rewriter.replaceOp(dimOp, initTensorOp.getDynamicSize(*maybeConstantIndex));
    return success();
  }
};
} // namespace

void InitTensorOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.add<FoldInitTensorWithDimOp, FoldInitTensorWithExtractSliceOp,
              FoldInitTensorWithTensorReshapeOp<tensor::ExpandShapeOp>,
              FoldInitTensorWithTensorReshapeOp<tensor::CollapseShapeOp>,
              ReplaceStaticShapeDims>(context);
}

LogicalResult InitTensorOp::reifyResultShapes(
    OpBuilder &builder, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  auto shapes = llvm::to_vector<4>(llvm::map_range(
      llvm::seq<int64_t>(0, getType().getRank()), [&](int64_t dim) -> Value {
        if (isDynamicSize(dim))
          return getDynamicSize(dim);
        return builder.create<arith::ConstantIndexOp>(getLoc(),
                                                      getStaticSize(dim));
      }));
  reifiedReturnShapes.emplace_back(std::move(shapes));
  return success();
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
  SmallVector<OpAsmParser::OperandType, 2> opInfo;
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
    return verifyYield(*this, cast<LinalgOp>(parentOp));

  if (auto tiledLoopOp = dyn_cast<linalg::TiledLoopOp>(parentOp)) {
    // Check if output args with tensor types match results types.
    SmallVector<Value, 2> tensorOuts;
    llvm::copy_if(
        tiledLoopOp.outputs(), std::back_inserter(tensorOuts),
        [&](Value out) { return out.getType().isa<RankedTensorType>(); });
    if (tensorOuts.size() != values().size())
      return emitOpError("expected number of tensor output args = ")
             << tensorOuts.size()
             << " to match the number of yield operands = " << values().size();

    TypeRange tensorTypes(llvm::makeArrayRef(tensorOuts));
    for (auto &item :
         llvm::enumerate(llvm::zip(tensorTypes, getOperandTypes()))) {
      Type outType, resultType;
      unsigned index = item.index();
      std::tie(outType, resultType) = item.value();
      if (outType != resultType)
        return emitOpError("expected yield operand ")
               << index << " with type = " << resultType
               << " to match output arg type = " << outType;
    }
    return success();
  }
  return emitOpError("expected parent op with LinalgOp interface");
}

//===----------------------------------------------------------------------===//
// TiledLoopOp
//===----------------------------------------------------------------------===//

void TiledLoopOp::build(OpBuilder &builder, OperationState &result,
                        ValueRange lowerBounds, ValueRange upperBounds,
                        ValueRange steps, ValueRange inputs, ValueRange outputs,
                        ArrayAttr iteratorTypes,
                        function_ref<void(OpBuilder &, Location, ValueRange,
                                          ValueRange, ValueRange)>
                            bodyBuilderFn) {
  build(builder, result, lowerBounds, upperBounds, steps, inputs, outputs,
        iteratorTypes, llvm::None, bodyBuilderFn);
}

void TiledLoopOp::build(OpBuilder &builder, OperationState &result,
                        ValueRange lowerBounds, ValueRange upperBounds,
                        ValueRange steps, ValueRange inputs, ValueRange outputs,
                        ArrayAttr iteratorTypes,
                        Optional<ArrayAttr> distributionTypes,
                        function_ref<void(OpBuilder &, Location, ValueRange,
                                          ValueRange, ValueRange)>
                            bodyBuilderFn) {
  result.addOperands(lowerBounds);
  result.addOperands(upperBounds);
  result.addOperands(steps);
  result.addOperands(inputs);
  result.addOperands(outputs);
  result.addAttribute(
      TiledLoopOp::getOperandSegmentSizeAttr(),
      builder.getI32VectorAttr({static_cast<int32_t>(lowerBounds.size()),
                                static_cast<int32_t>(upperBounds.size()),
                                static_cast<int32_t>(steps.size()),
                                static_cast<int32_t>(inputs.size()),
                                static_cast<int32_t>(outputs.size())}));
  result.addAttribute(getIteratorTypesAttrName(), iteratorTypes);

  if (distributionTypes.hasValue())
    result.addAttribute(getDistributionTypesAttrName(),
                        distributionTypes.getValue());

  // Add output types for `RankedTensorType` output arguments.
  for (Value output : outputs) {
    Type outputType = output.getType();
    if (outputType.isa<RankedTensorType>())
      result.addTypes(outputType);
  }

  OpBuilder::InsertionGuard guard(builder);
  unsigned numIVs = steps.size();
  SmallVector<Type, 8> argTypes(numIVs, builder.getIndexType());
  SmallVector<Location, 8> argLocs(numIVs, result.location);
  for (Value input : inputs) {
    argTypes.push_back(input.getType());
    argLocs.push_back(input.getLoc());
  }
  for (Value output : outputs) {
    argTypes.push_back(output.getType());
    argLocs.push_back(output.getLoc());
  }
  Region *bodyRegion = result.addRegion();
  Block *bodyBlock = builder.createBlock(bodyRegion, {}, argTypes, argLocs);

  if (bodyBuilderFn) {
    builder.setInsertionPointToStart(bodyBlock);
    bodyBuilderFn(builder, result.location,
                  bodyBlock->getArguments().take_front(numIVs),
                  bodyBlock->getArguments().slice(numIVs, inputs.size()),
                  bodyBlock->getArguments().take_back(outputs.size()));
    TiledLoopOp::ensureTerminator(*bodyRegion, builder, result.location);
  }
}

void TiledLoopOp::print(OpAsmPrinter &p) {
  p << " (" << getInductionVars() << ") = (" << lowerBound() << ") to ("
    << upperBound() << ") step (" << step() << ")";

  if (!inputs().empty()) {
    p << " ins (";
    llvm::interleaveComma(llvm::zip(getRegionInputArgs(), inputs()), p,
                          [&](auto it) {
                            p << std::get<0>(it) << " = " << std::get<1>(it)
                              << ": " << std::get<1>(it).getType();
                          });
    p << ")";
  }
  if (!outputs().empty()) {
    p << " outs (";
    llvm::interleaveComma(llvm::zip(getRegionOutputArgs(), outputs()), p,
                          [&](auto it) {
                            p << std::get<0>(it) << " = " << std::get<1>(it)
                              << ": " << std::get<1>(it).getType();
                          });
    p << ")";
  }

  if (llvm::any_of(iterator_types(), [](Attribute attr) {
        return attr.cast<StringAttr>().getValue() !=
               getParallelIteratorTypeName();
      }))
    p << " iterators" << iterator_types();

  if (distribution_types().hasValue())
    p << " distribution" << distribution_types().getValue();

  p << ' ';
  p.printRegion(region(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{
                              TiledLoopOp::getOperandSegmentSizeAttr(),
                              getIteratorTypesAttrName(),
                              getDistributionTypesAttrName()});
}

ParseResult TiledLoopOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  // Parse an opening `(` followed by induction variables followed by `)`
  SmallVector<OpAsmParser::OperandType, 4> ivs;
  if (parser.parseRegionArgumentList(ivs, /*requiredOperandCount=*/-1,
                                     OpAsmParser::Delimiter::Paren))
    return failure();

  // Parse loop bounds.
  SmallVector<OpAsmParser::OperandType, 4> lower;
  if (parser.parseEqual() ||
      parser.parseOperandList(lower, ivs.size(),
                              OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(lower, builder.getIndexType(), result.operands))
    return failure();

  SmallVector<OpAsmParser::OperandType, 4> upper;
  if (parser.parseKeyword("to") ||
      parser.parseOperandList(upper, ivs.size(),
                              OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(upper, builder.getIndexType(), result.operands))
    return failure();

  // Parse step values.
  SmallVector<OpAsmParser::OperandType, 4> steps;
  if (parser.parseKeyword("step") ||
      parser.parseOperandList(steps, ivs.size(),
                              OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(steps, builder.getIndexType(), result.operands))
    return failure();

  // Parse input tensors.
  SmallVector<OpAsmParser::OperandType, 4> inputs, inputRegionArgs;
  SmallVector<Type, 4> inputTypes;
  if (succeeded(parser.parseOptionalKeyword("ins"))) {
    SMLoc inputsOperandsLoc = parser.getCurrentLocation();

    if (parser.parseAssignmentListWithTypes(inputRegionArgs, inputs,
                                            inputTypes))
      return failure();

    if (parser.resolveOperands(inputs, inputTypes, inputsOperandsLoc,
                               result.operands))
      return failure();
  }

  // Parse output tensors.
  SmallVector<OpAsmParser::OperandType, 4> outputs, outputRegionArgs;
  SmallVector<Type, 4> outputTypes;
  if (succeeded(parser.parseOptionalKeyword("outs"))) {
    SMLoc outputsOperandsLoc = parser.getCurrentLocation();

    if (parser.parseAssignmentListWithTypes(outputRegionArgs, outputs,
                                            outputTypes))
      return failure();

    if (parser.resolveOperands(outputs, outputTypes, outputsOperandsLoc,
                               result.operands))
      return failure();
    for (Type outputType : outputTypes)
      if (outputType.isa<RankedTensorType>())
        result.addTypes(outputType);
  }

  // Parse attributes.
  SmallVector<Attribute, 4> iterTypes, distributionTypes;
  auto parseAttr = [&](StringRef keyword, SmallVector<Attribute, 4> *attrs) {
    if (succeeded(parser.parseOptionalKeyword(keyword))) {
      StringAttr attr;

      if (parser.parseLSquare() || parser.parseAttribute(attr))
        return failure();
      attrs->push_back(attr);
      for (int i = 1, e = ivs.size(); i < e; ++i) {
        if (parser.parseComma() || parser.parseAttribute(attr))
          return failure();
        attrs->push_back(attr);
      }
      if (parser.parseRSquare())
        return failure();
    }
    return success();
  };
  if (failed(parseAttr("iterators", &iterTypes)) ||
      failed(parseAttr("distribution", &distributionTypes)))
    return failure();

  // Set all loop iterator types to "parallel" if they are not printed in IR.
  if (iterTypes.empty()) {
    auto parallelIter = builder.getStringAttr(getParallelIteratorTypeName());
    iterTypes = SmallVector<Attribute, 4>(ivs.size(), parallelIter);
  }
  result.addAttribute(getIteratorTypesAttrName(),
                      builder.getArrayAttr(iterTypes));
  if (!distributionTypes.empty())
    result.addAttribute(getDistributionTypesAttrName(),
                        builder.getArrayAttr(distributionTypes));
  result.addAttribute(
      TiledLoopOp::getOperandSegmentSizeAttr(),
      builder.getI32VectorAttr({static_cast<int32_t>(lower.size()),
                                static_cast<int32_t>(upper.size()),
                                static_cast<int32_t>(steps.size()),
                                static_cast<int32_t>(inputs.size()),
                                static_cast<int32_t>(outputs.size())}));

  // Parse the body.
  Region *body = result.addRegion();

  SmallVector<Type, 4> regionTypes(ivs.size(), builder.getIndexType());
  regionTypes.append(inputTypes);
  regionTypes.append(outputTypes);

  SmallVector<OpAsmParser::OperandType, 4> regionArgs(ivs);
  regionArgs.append(inputRegionArgs);
  regionArgs.append(outputRegionArgs);

  if (parser.parseRegion(*body, regionArgs, regionTypes))
    return failure();

  // Parse optional attributes.
  parser.parseOptionalAttrDict(result.attributes);

  return success();
}

Region &TiledLoopOp::getLoopBody() { return region(); }

LogicalResult TiledLoopOp::moveOutOfLoop(ArrayRef<Operation *> ops) {
  for (auto *op : ops)
    op->moveBefore(*this);
  return success();
}

bool TiledLoopOp::isDefinedOutsideOfLoop(Value value) {
  return !region().isAncestor(value.getParentRegion());
}

LogicalResult TiledLoopOp::verify() {
  // Check if iterator types are provided for every loop dimension.
  if (iterator_types().size() != getNumLoops())
    return emitOpError("expected iterator types array attribute size = ")
           << iterator_types().size()
           << " to match the number of loops = " << getNumLoops();

  // Check if types of input arguments match region args types.
  for (auto &item :
       llvm::enumerate(llvm::zip(inputs(), getRegionInputArgs()))) {
    Value input, inputRegionArg;
    unsigned index = item.index();
    std::tie(input, inputRegionArg) = item.value();
    if (input.getType() != inputRegionArg.getType())
      return emitOpError("expected input arg ")
             << index << " with type = " << input.getType()
             << " to match region arg " << index + getNumLoops()
             << " type = " << inputRegionArg.getType();
  }

  // Check if types of input arguments match region args types.
  for (auto &item :
       llvm::enumerate(llvm::zip(outputs(), getRegionOutputArgs()))) {
    Value output, outputRegionArg;
    unsigned index = item.index();
    std::tie(output, outputRegionArg) = item.value();
    if (output.getType() != outputRegionArg.getType())
      return emitOpError("expected output arg ")
             << index << " with type = " << output.getType()
             << " to match region arg "
             << index + getNumLoops() + inputs().size()
             << " type = " << outputRegionArg.getType();
  }
  return success();
}

namespace {

static constexpr int64_t kNoMatch = -1;

// Folds away TiledLoopOp inputs if they have no uses within the body.
//
// Example:
//
// %0 = linalg.tiled_loop ...  ins (%in_ = %in: tensor<...>,
//                                  %in_buf_ = %in_buf: memref<...>) {...}
// Becomes
//
// linalg.tiled_loop ...  ins (%in_buf_ = %in_buf: memref<...>) {...}
struct TiledLoopInputsFolder : public OpRewritePattern<linalg::TiledLoopOp> {
  using OpRewritePattern<linalg::TiledLoopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TiledLoopOp tiledLoop,
                                PatternRewriter &rewriter) const final {
    SmallVector<Value, 2> newInputs, regionInputTensorArgs;
    // Store ids of the corresponding old and new input operands.
    SmallVector<int64_t, 2> oldInputIdToNew(tiledLoop.inputs().size(),
                                            kNoMatch);
    for (const auto &en : llvm::enumerate(
             llvm::zip(tiledLoop.inputs(), tiledLoop.getRegionInputArgs()))) {
      Value in, bbArg;
      size_t index = en.index();
      std::tie(in, bbArg) = en.value();
      if (!bbArg.use_empty()) {
        oldInputIdToNew[index] = newInputs.size();
        newInputs.push_back(in);
      }
    }
    if (newInputs.size() == tiledLoop.inputs().size())
      return failure();
    Location loc = tiledLoop.getLoc();
    auto newTiledLoop = rewriter.create<TiledLoopOp>(
        loc, tiledLoop.lowerBound(), tiledLoop.upperBound(), tiledLoop.step(),
        newInputs, tiledLoop.outputs(), tiledLoop.iterator_types(),
        tiledLoop.distribution_types());

    // Clone the region.
    BlockAndValueMapping bvm;
    bvm.map(tiledLoop.getInductionVars(), newTiledLoop.getInductionVars());
    bvm.map(tiledLoop.getRegionOutputArgs(),
            newTiledLoop.getRegionOutputArgs());
    for (const auto &en : llvm::enumerate(oldInputIdToNew))
      if (en.value() != kNoMatch)
        bvm.map(tiledLoop.getRegionInputArgs()[en.index()],
                newTiledLoop.getRegionInputArgs()[en.value()]);
    OpBuilder innerBuilder =
        OpBuilder::atBlockEnd(newTiledLoop.getBody(), rewriter.getListener());
    for (auto &op : *tiledLoop.getBody())
      innerBuilder.clone(op, bvm);
    rewriter.replaceOp(tiledLoop, newTiledLoop.getResults());

    return success();
  }
};

} // namespace

/// A simple, conservative analysis to determine if the loop is shape
/// conserving. I.e., the type of the arg-th yielded value is the same as the
/// type of the corresponding basic block argument of the loop.
/// Note: This function handles only simple cases. Expand as needed.
static bool isShapePreserving(TiledLoopOp loopOp, int64_t arg) {
  auto yieldOp = cast<YieldOp>(loopOp.getLoopBody().front().getTerminator());
  if (yieldOp.values().empty())
    // Tiled loop either has no outputs or is a "memref-based version". In
    // either case, the loop is shape conserving.
    return true;
  assert(arg < static_cast<int64_t>(yieldOp.values().size()) &&
         "arg is out of bounds");
  Value value = yieldOp.values()[arg];
  while (value) {
    if (value == loopOp.getRegionOutputArgs()[arg])
      return true;
    OpResult opResult = value.dyn_cast<OpResult>();
    if (!opResult)
      return false;

    using tensor::InsertSliceOp;
    value = llvm::TypeSwitch<Operation *, Value>(opResult.getOwner())
                .template Case<InsertSliceOp>(
                    [&](InsertSliceOp op) { return op.dest(); })
                .template Case<TiledLoopOp>([&](TiledLoopOp loopOp) {
                  return isShapePreserving(loopOp, opResult.getResultNumber())
                             ? loopOp.outputs()[opResult.getResultNumber()]
                             : Value();
                })
                .Default([&](auto op) { return Value(); });
  }
  return false;
}

namespace {

/// Fold dim(x) where `x` is an input/output argument of a TiledLoopOp block
/// to dim(y) where `y` is the initial input/output value of the argument.
///
/// E.g.:
/// %y = ... : tensor<...>
/// linalg.tiled_loop ... ins(%x = %y : tensor<...>) {
///   tensor.dim %x, %c0 : tensor<...>
/// }
///
/// is folded to:
/// %y = ... : tensor<...>
/// linalg.tiled_loop ... ins(%x = %y : tensor<...>) {
///   tensor.dim %y, %c0 : tensor<...>
/// }
///
/// Note: Dim ops are folded only if it can be proven that the runtime type of
/// the yielded value (in case of outputs) does not change with loop iterations.
template <typename OpTy>
struct DimOfTiledLoopInsOutsFolder : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy dimOp,
                                PatternRewriter &rewriter) const final {
    auto src = dimOp.source().template dyn_cast<BlockArgument>();
    if (!src)
      return failure();
    auto loopOp =
        dyn_cast<TiledLoopOp>(src.getOwner()->getParent()->getParentOp());
    if (!loopOp)
      return failure();
    unsigned numLoops = loopOp.getNumLoops();
    unsigned numInputArgs = loopOp.getRegionInputArgs().size();
    if (src.getArgNumber() >= numInputArgs + numLoops &&
        !isShapePreserving(loopOp,
                           src.getArgNumber() - numInputArgs - numLoops))
      return failure();

    auto inputArgs = loopOp.getRegionInputArgs();
    auto it1 = llvm::find(inputArgs, src);
    if (it1 != inputArgs.end()) {
      rewriter.updateRootInPlace(dimOp, [&] {
        dimOp.sourceMutable().assign(loopOp.inputs()[it1 - inputArgs.begin()]);
      });
      return success();
    }

    auto outputArgs = loopOp.getRegionOutputArgs();
    auto it2 = llvm::find(outputArgs, src);
    if (it2 != outputArgs.end()) {
      rewriter.updateRootInPlace(dimOp, [&] {
        dimOp.sourceMutable().assign(
            loopOp.outputs()[it2 - outputArgs.begin()]);
      });
      return success();
    }

    return failure();
  }
};

/// Fold dim(r) where `r` is the result of a TiledLoopOp to dim(y) where `y`
/// is the initial output value of the loop.
///
/// E.g.:
/// %y = ... : tensor<...>
/// %r = linalg.tiled_loop ... outs(%i = %y : tensor<...>) {
///   ...
/// }
/// %0 = tensor.dim %r, %c0 : tensor<...>
///
/// is folded to:
/// %y = ... : tensor<...>
/// linalg.tiled_loop ... outs(%i = %y : tensor<...>) {
///   ...
/// }
/// %0 = tensor.dim %y, %c0 : tensor<...>
///
/// Note: Dim ops are folded only if it can be proven that the runtime type of
/// the yielded value (in case of outputs) does not change with loop iterations.
template <typename OpTy>
struct DimOfTiledLoopResultFolder : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy dimOp,
                                PatternRewriter &rewriter) const final {
    auto loopOp = dimOp.source().template getDefiningOp<TiledLoopOp>();
    if (!loopOp)
      return failure();
    auto opResult = dimOp.source().template cast<OpResult>();
    unsigned resultNumber = opResult.getResultNumber();
    if (!isShapePreserving(loopOp, resultNumber))
      return failure();
    rewriter.updateRootInPlace(dimOp, [&]() {
      dimOp.sourceMutable().assign(loopOp.outputs()[resultNumber]);
    });
    return success();
  }
};

// Folds away TiledLoopOp output tensors when the following conditions are met:
// * result of `linalg.tiled_loop` has no uses
// * output tensor is the argument of `linalg.yield`
//
// Example:
//
// %0 = linalg.tiled_loop ...  outs (%o_ = %out: tensor<...>,
//                                   %obuf_ = %out_buf: memref<...>) {
//   ...
//   linalg.yield %o_ : tensor ...
// }
//
// Becomes
//
// linalg.tiled_loop ...  outs (%obuf_ = %out_buf: memref<...>) {
//   ...
//   linalg.yield
// }
struct TiledLoopResultsFolder : public OpRewritePattern<linalg::TiledLoopOp> {
  using OpRewritePattern<linalg::TiledLoopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TiledLoopOp tiledLoop,
                                PatternRewriter &rewriter) const final {
    if (tiledLoop.getNumResults() == 0)
      return failure();

    Block *block = tiledLoop.getBody();
    auto yieldOp = cast<linalg::YieldOp>(block->getTerminator());

    // Match the pattern and collect output buffers that will replace the output
    // tensors and also the ops that will be ignored when cloning the body.
    SmallVector<Value, 2> newOutputOperands, newYieldArgs;
    int resultId = 0;
    // Store ids of the corresponding old and new output operands.
    SmallVector<int64_t, 2> oldOutputIdToNew(tiledLoop.outputs().size(),
                                             kNoMatch);
    // Store ids of the corresponding old and new results.
    SmallVector<int64_t, 2> oldResultIdToNew(tiledLoop.getNumResults(),
                                             kNoMatch);
    SmallVector<Value, 2> resultReplacement(tiledLoop.getNumResults());
    for (const auto &en : llvm::enumerate(
             llvm::zip(tiledLoop.outputs(), tiledLoop.getRegionOutputArgs()))) {
      size_t index = en.index();
      Value out = std::get<0>(en.value());
      Value outRegionArg = std::get<1>(en.value());

      if (!out.getType().isa<RankedTensorType>()) {
        oldOutputIdToNew[index] = newOutputOperands.size();
        newOutputOperands.push_back(out);
        continue;
      }
      Value result = tiledLoop.getResult(resultId);
      Value yieldArg = yieldOp.getOperand(resultId);
      if (yieldArg != outRegionArg || !result.use_empty()) {
        oldOutputIdToNew[index] = newOutputOperands.size();
        oldResultIdToNew[resultId] = newYieldArgs.size();
        resultReplacement[resultId] = out;
        newOutputOperands.push_back(out);
        newYieldArgs.push_back(yieldArg);
      }
      ++resultId;
    }
    if (newOutputOperands.size() == tiledLoop.outputs().size())
      return failure();

    Location loc = tiledLoop.getLoc();
    auto newTiledLoop = rewriter.create<TiledLoopOp>(
        loc, tiledLoop.lowerBound(), tiledLoop.upperBound(), tiledLoop.step(),
        tiledLoop.inputs(), newOutputOperands, tiledLoop.iterator_types(),
        tiledLoop.distribution_types());

    // Clone the region.
    BlockAndValueMapping bvm;
    bvm.map(tiledLoop.getInductionVars(), newTiledLoop.getInductionVars());
    bvm.map(tiledLoop.getRegionInputArgs(), newTiledLoop.getRegionInputArgs());
    for (const auto &en : llvm::enumerate(oldOutputIdToNew)) {
      if (en.value() != kNoMatch)
        bvm.map(tiledLoop.getRegionOutputArgs()[en.index()],
                newTiledLoop.getRegionOutputArgs()[en.value()]);
      else
        bvm.map(tiledLoop.getRegionOutputArgs()[en.index()],
                tiledLoop.outputs()[en.index()]);
    }
    OpBuilder innerBuilder =
        OpBuilder::atBlockEnd(newTiledLoop.getBody(), rewriter.getListener());
    for (auto &op : tiledLoop.getBody()->without_terminator())
      innerBuilder.clone(op, bvm);
    innerBuilder.create<linalg::YieldOp>(
        loc, llvm::to_vector<2>(llvm::map_range(
                 newYieldArgs, [&](Value arg) { return bvm.lookup(arg); })));

    for (const auto &en : llvm::enumerate(oldResultIdToNew))
      if (en.value() != kNoMatch)
        resultReplacement[en.index()] = newTiledLoop.getResult(en.value());
    rewriter.replaceOp(tiledLoop, resultReplacement);

    return success();
  }
};
} // namespace

void TiledLoopOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<TiledLoopInputsFolder, TiledLoopResultsFolder,
              DimOfTiledLoopInsOutsFolder<tensor::DimOp>,
              DimOfTiledLoopInsOutsFolder<memref::DimOp>,
              DimOfTiledLoopResultFolder<tensor::DimOp>,
              DimOfTiledLoopResultFolder<memref::DimOp>>(context);
}

LogicalResult TiledLoopOp::fold(ArrayRef<Attribute>,
                                SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCastInTiledLoopOp(*this);
}

//===----------------------------------------------------------------------===//
// IndexOp
//===----------------------------------------------------------------------===//

LogicalResult IndexOp::verify() {
  auto linalgOp = dyn_cast<LinalgOp>((*this)->getParentOp());
  if (!linalgOp)
    return emitOpError("expected parent op with LinalgOp interface");
  if (linalgOp.getNumLoops() <= dim())
    return emitOpError("expected dim (")
           << dim() << ") to be lower than the number of loops ("
           << linalgOp.getNumLoops() << ") of the enclosing LinalgOp";
  return success();
}

/////// Operations corresponding to library calls defined with Tablegen ////////

#include "mlir/Dialect/Linalg/IR/LinalgNamedStructuredOps.yamlgen.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgOps.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"

/// Return the dims that are `iteratorTypeName` loops in the LinalgOp `op`.
/// Assumes `op` is a LinalgOp.
void mlir::linalg::getDimsOfType(Operation *op, StringRef iteratorTypeName,
                                 SmallVectorImpl<unsigned> &res) {
  if (!cast<LinalgOp>(op).iterator_types())
    return;

  unsigned dim = 0;
  for (auto tn :
       cast<LinalgOp>(op).iterator_types().getAsValueRange<StringAttr>()) {
    if (tn == iteratorTypeName)
      res.push_back(dim);
    ++dim;
  }
}

AffineMap mlir::linalg::extractOrIdentityMap(Optional<AffineMap> maybeMap,
                                             unsigned rank,
                                             MLIRContext *context) {
  if (maybeMap)
    return maybeMap.getValue();
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
// Support for named Linalg ops defined in ods-gen.
//===----------------------------------------------------------------------===//

/// Generic entry point to create the block for the region of a LinalgOp.
/// This is used by both named structured ops created by ods-gen and by manually
/// defined C++ ops.
/// This is used by both builders and parsers.
/// This function creates the block in the region with arguments corresponding
/// to the elemental types of `inputTypes` and `outputTypes`, which are asserted
/// to be ShapedType.
template <typename NamedStructuredOpType>
static void fillStructuredOpRegion(
    OpBuilder &opBuilder, Region &region, TypeRange inputTypes,
    TypeRange outputTypes, ArrayRef<NamedAttribute> attrs,
    llvm::function_ref<void(unsigned, unsigned)> errorHandler) {
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
  unsigned actual = body->getNumArguments();
  unsigned expected = NamedStructuredOpType::getNumRegionArgs();
  if (expected != actual) {
    if (errorHandler)
      errorHandler(expected, actual);
    return;
  }

  opBuilder.setInsertionPointToStart(body);
  ImplicitLocOpBuilder b(opBuilder.getUnknownLoc(), opBuilder);
  NamedStructuredOpType::regionBuilder(b, *body, attrs);

  // indexing_maps is an auto-generated method.

  // iterator_types is an auto-generated method.
}

/// Generic entry point to create both the region and the block of a LinalgOp.
template <typename NamedStructuredOpType>
void createAndFillStructuredOpRegion(OpBuilder &opBuilder,
                                     OperationState &result,
                                     TypeRange inputTypes,
                                     TypeRange outputTypes) {
  Region &region = *result.addRegion();
  fillStructuredOpRegion<NamedStructuredOpType>(
      opBuilder, region, inputTypes, outputTypes, result.attributes.getAttrs(),
      [&](unsigned expected, unsigned actual) {
        assert(expected != actual && "incorrect number of arguments");
      });
}

/// Common parsing used for both named structured ops created by ods-gen and by
/// manually defined C++ ops. Does not handle regions.
static ParseResult
parseCommonStructuredOpParts(OpAsmParser &parser, OperationState &result,
                             SmallVectorImpl<Type> &inputTypes,
                             SmallVectorImpl<Type> &outputTypes) {
  SMLoc inputsOperandsLoc, outputsOperandsLoc;
  SmallVector<OpAsmParser::OperandType, 4> inputsOperands, outputsOperands;

  parser.parseOptionalAttrDict(result.attributes);

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

  result.addAttribute("operand_segment_sizes",
                      parser.getBuilder().getI32VectorAttr(
                          {static_cast<int32_t>(inputsOperands.size()),
                           static_cast<int32_t>(outputsOperands.size())}));
  return success();
}

template <typename NamedStructuredOpType>
static void printCommonStructuredOpParts(OpAsmPrinter &p,
                                         NamedStructuredOpType op) {
  if (!op.inputs().empty())
    p << " ins(" << op.inputs() << " : " << op.inputs().getTypes() << ")";
  if (!op.outputs().empty())
    p << " outs(" << op.outputs() << " : " << op.outputs().getTypes() << ")";
}

//===----------------------------------------------------------------------===//
// Specific parsing and printing for named structured ops created by ods-gen.
//===----------------------------------------------------------------------===//

template <typename NamedStructuredOpType>
static ParseResult
parseNamedStructuredOpRegion(OpAsmParser &parser, Region &region,
                             TypeRange inputTypes, TypeRange outputTypes,
                             ArrayRef<NamedAttribute> attrs) {
  ParseResult res = success();
  OpBuilder opBuilder(parser.getContext());
  // Resolve `captures` into `capturedValues` at parse time so we can build the
  // region with captures.
  SmallVector<Value> capturedValues;
  fillStructuredOpRegion<NamedStructuredOpType>(
      opBuilder, region, inputTypes, outputTypes, attrs,
      [&](unsigned expected, unsigned actual) {
        res = parser.emitError(
            parser.getCurrentLocation(),
            llvm::formatv("[parseNamedStructuredOpRegion] ods-gen generated "
                          "region expects {0} args, got {1}",
                          expected, actual));
        region.front().dump();
      });
  return res;
}

static ParseResult
parseNamedStructuredOpResults(OpAsmParser &parser,
                              SmallVectorImpl<Type> &resultTypes) {
  if (parser.parseOptionalArrowTypeList(resultTypes))
    return failure();
  return success();
}

template <typename NamedStructuredOpType>
static ParseResult parseNamedStructuredOp(OpAsmParser &parser,
                                          OperationState &result) {
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
  if (parseNamedStructuredOpRegion<NamedStructuredOpType>(
          parser, *region, inputTypes, outputTypes,
          result.attributes.getAttrs()))
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

template <typename NamedStructuredOpType>
static void printNamedStructuredOp(OpAsmPrinter &p, NamedStructuredOpType op) {
  p.printOptionalAttrDict(
      op->getAttrs(),
      /*elidedAttrs=*/{"operand_segment_sizes",
                       // See generated code in mlir-linalg-yaml-gen.cpp
                       "linalg.memoized_indexing_maps"});

  // Printing is shared with generic ops, except for the region and
  // attributes.
  printCommonStructuredOpParts(p, op);

  // Results printing.
  printNamedStructuredOpResults(p, op.result_tensors().getTypes());

  // Region is elided.
}

template <typename NamedStructuredOpType>
static LogicalResult verifyNamedStructuredOp(NamedStructuredOpType op) {
  return verifyGenericOp<NamedStructuredOpType>(op);
}

//===----------------------------------------------------------------------===//
// Canonicalizers and Folders.
//===----------------------------------------------------------------------===//

namespace {
struct EraseDeadLinalgOp : public OpInterfaceRewritePattern<LinalgOp> {
  using OpInterfaceRewritePattern<LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LinalgOp op,
                                PatternRewriter &rewriter) const override {
    for (OpOperand *opOperand : op.getInputAndOutputOperands()) {
      // Linalg "inputs" may be either tensor or memref type.
      // tensor<0xelt_type> is a convention that may not always mean
      // "0 iterations". Only erase in cases we see memref<...x0x...>.
      auto mt = opOperand->get().getType().dyn_cast<MemRefType>();
      if (!mt)
        continue;
      if (llvm::is_contained(op.getShape(opOperand), 0)) {
        rewriter.eraseOp(op);
        return success();
      }
    }
    return failure();
  }
};

struct FoldTensorCastOp : public OpInterfaceRewritePattern<LinalgOp> {
  using OpInterfaceRewritePattern<LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LinalgOp op,
                                PatternRewriter &rewriter) const override {
    // If no operand comes from a tensor::CastOp and can be folded then fail.
    bool hasTensorCastOperand =
        llvm::any_of(op.getInputAndOutputOperands(), [&](OpOperand *opOperand) {
          if (opOperand->get().isa<BlockArgument>())
            return false;
          auto castOp = opOperand->get().getDefiningOp<tensor::CastOp>();
          return castOp && canFoldIntoConsumerOp(castOp);
        });
    if (!hasTensorCastOperand)
      return failure();

    SmallVector<Type, 4> newResultTypes;
    newResultTypes.reserve(op->getNumResults());
    SmallVector<Value, 4> newOperands;
    newOperands.reserve(op->getNumOperands());
    // Inputs may fold.
    for (OpOperand *opOperand : op.getInputOperands()) {
      auto tensorCastOp = opOperand->get().getDefiningOp<tensor::CastOp>();
      newOperands.push_back(canFoldIntoConsumerOp(tensorCastOp)
                                ? tensorCastOp.source()
                                : opOperand->get());
    }
    // Init tensors may fold, in which case the resultType must also change.
    for (OpOperand *opOperand : op.getOutputOperands()) {
      auto tensorCastOp = opOperand->get().getDefiningOp<tensor::CastOp>();
      bool fold = canFoldIntoConsumerOp(tensorCastOp);
      newOperands.push_back(fold ? tensorCastOp.getOperand()
                                 : opOperand->get());
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

} // namespace

#define LINALGOP_FOLDERS(XXX)                                                  \
  LogicalResult XXX::fold(ArrayRef<Attribute>,                                 \
                          SmallVectorImpl<OpFoldResult> &) {                   \
    return foldMemRefCast(*this);                                              \
  }

LINALGOP_FOLDERS(FillOp)
LINALGOP_FOLDERS(GenericOp)

// All named ops canonicalizers and folders are auto-generated in the
// .cpp.inc.

//===----------------------------------------------------------------------===//
// LinalgDialect
//===----------------------------------------------------------------------===//

void LinalgDialect::getCanonicalizationPatterns(
    RewritePatternSet &results) const {
  results.add<EraseDeadLinalgOp, FoldTensorCastOp>(getContext());
}

Operation *LinalgDialect::materializeConstant(OpBuilder &builder,
                                              Attribute value, Type type,
                                              Location loc) {
  return builder.create<arith::ConstantOp>(loc, type, value);
}
