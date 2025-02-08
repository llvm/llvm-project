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
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <optional>

using namespace mlir;
using namespace mlir::linalg;

/// Return a `memref.dim` or `tensor.dim` for the shape of `v` at `dim`.
static OpFoldResult getDimValue(OpBuilder &builder, Location loc, Value v,
                                int64_t dim) {
  auto type = cast<ShapedType>(v.getType());
  if (!type.isDynamicDim(dim))
    return builder.getIndexAttr(type.getDimSize(dim));

  return getAsOpFoldResult(
      TypeSwitch<Type, Value>(v.getType())
          .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
            return builder.create<tensor::DimOp>(loc, v, dim);
          })
          .Case<MemRefType>([&](MemRefType t) -> Value {
            return builder.create<memref::DimOp>(loc, v, dim);
          }));
}

/// Returns a memref.subview or a tensor.extract_slice based on the type of the
/// `source`.
static Operation *getSlice(OpBuilder &b, Location loc, Value source,
                           ArrayRef<OpFoldResult> offsets,
                           ArrayRef<OpFoldResult> sizes,
                           ArrayRef<OpFoldResult> strides) {
  return TypeSwitch<Type, Operation *>(source.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Operation * {
        return b.create<tensor::ExtractSliceOp>(loc, source, offsets, sizes,
                                                strides);
      })
      .Case<MemRefType>([&](MemRefType type) -> Operation * {
        return b.create<memref::SubViewOp>(loc, source, offsets, sizes,
                                           strides);
      })
      .Default([&](Type t) -> Operation * { return nullptr; });
}

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

Value linalg::createOrFoldDimOp(OpBuilder &b, Location loc, Value source,
                                int64_t dim) {
  if (llvm::isa<UnrankedMemRefType, MemRefType>(source.getType()))
    return b.createOrFold<memref::DimOp>(loc, source, dim);
  if (llvm::isa<UnrankedTensorType, RankedTensorType>(source.getType()))
    return b.createOrFold<tensor::DimOp>(loc, source, dim);
  llvm_unreachable("Expected MemRefType or TensorType");
}

OpFoldResult linalg::createFoldedDimOp(OpBuilder &b, Location loc, Value source,
                                       int64_t dim) {
  auto shapedType = llvm::cast<ShapedType>(source.getType());
  if (!shapedType.hasRank() || shapedType.isDynamicDim(dim))
    return createOrFoldDimOp(b, loc, source, dim);
  return b.getIndexAttr(shapedType.getDimSize(dim));
}

//===----------------------------------------------------------------------===//
// Support for named Linalg ops defined in ods-gen.
//===----------------------------------------------------------------------===//

using RegionBuilderFn = llvm::function_ref<void(ImplicitLocOpBuilder &, Block &,
                                                ArrayRef<NamedAttribute>)>;

/// Fills the region of a structured operation using the provided
/// `regionBuilder`. The method is used by both named structured ops created by
/// ods-gen and by manually defined C++ ops. It is called by both builders and
/// parsers and creates a block with arguments corresponding to the elemental
/// types of `inputTypes` and `outputTypes`.
static void fillStructuredOpRegion(OpBuilder &opBuilder, Region &region,
                                   TypeRange inputTypes, TypeRange outputTypes,
                                   ArrayRef<NamedAttribute> attrs,
                                   RegionBuilderFn regionBuilder) {
  SmallVector<Type, 8> argTypes;
  SmallVector<Location, 8> argLocs;
  for (auto containers : {inputTypes, outputTypes}) {
    for (auto t : containers) {
      argTypes.push_back(
          isa<MemRefType, RankedTensorType>(t) ? getElementTypeOrSelf(t) : t);

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
                              std::optional<TypeRange> resultTensorTypes,
                              ValueRange inputs, ValueRange outputs,
                              ArrayRef<NamedAttribute> attributes,
                              RegionBuilderFn regionBuilder) {
  // Derive the result types if needed.
  SmallVector<Type> derivedResultTypes =
      resultTensorTypes.value_or(TypeRange());
  if (!resultTensorTypes)
    copy_if(outputs.getTypes(), std::back_inserter(derivedResultTypes),
            llvm::IsaPred<RankedTensorType>);

  state.addOperands(inputs);
  state.addOperands(outputs);
  state.addTypes(derivedResultTypes);

  state.addAttributes(attributes);
  state.addAttribute(
      "operandSegmentSizes",
      b.getDenseI32ArrayAttr({static_cast<int32_t>(inputs.size()),
                              static_cast<int32_t>(outputs.size())}));

  // Create and fill the region of the structured operation.
  Region &region = *state.addRegion();
  fillStructuredOpRegion(b, region, TypeRange(inputs), TypeRange(outputs),
                         state.attributes.getAttrs(), regionBuilder);
}

static void buildMatmulOp(OpBuilder &b, OperationState &state,
                          std::optional<TypeRange> resultTensorTypes,
                          ValueRange inputs, ValueRange outputs,
                          ArrayRef<NamedAttribute> attributes,
                          RegionBuilderFn regionBuilder,
                          ArrayRef<AffineMap> indexingMaps) {
  // Initialize indexingMaps attribute, for MatmulOp.
  SmallVector<Attribute, 3> indexingMapsAttrVal;
  indexingMapsAttrVal = llvm::map_to_vector(
      MatmulOp::getDefaultIndexingMaps(b.getContext()),
      [](AffineMap map) -> Attribute { return AffineMapAttr::get(map); });
  state.addAttribute("indexing_maps", b.getArrayAttr(indexingMapsAttrVal));
  return buildStructuredOp(b, state, resultTensorTypes, inputs, outputs,
                           attributes, regionBuilder);
}

/// Common parsing used for both named structured ops created by ods-gen and by
/// manually defined C++ ops. Does not handle regions.
static ParseResult
parseCommonStructuredOpParts(OpAsmParser &parser, OperationState &result,
                             SmallVectorImpl<Type> &inputTypes,
                             SmallVectorImpl<Type> &outputTypes,
                             bool addOperandSegmentSizes = true) {
  SMLoc attrsLoc, inputsOperandsLoc, outputsOperandsLoc;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> inputsOperands,
      outputsOperands;

  if (succeeded(parser.parseOptionalLess())) {
    if (parser.parseAttribute(result.propertiesAttr) || parser.parseGreater())
      return failure();
  }
  attrsLoc = parser.getCurrentLocation();
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
    // This is a bit complex because we're trying to be backward compatible with
    // operation syntax that mix the inherent attributes and the discardable
    // ones in the same dictionary. If the properties are used, we append the
    // operandSegmentSizes there directly. Otherwise we append it to the
    // discardable attributes dictionary where it is handled by the generic
    // Operation::create(...) method.
    if (result.propertiesAttr) {
      NamedAttrList attrs = llvm::cast<DictionaryAttr>(result.propertiesAttr);
      attrs.append("operandSegmentSizes",
                   parser.getBuilder().getDenseI32ArrayAttr(
                       {static_cast<int32_t>(inputsOperands.size()),
                        static_cast<int32_t>(outputsOperands.size())}));
      result.propertiesAttr = attrs.getDictionary(parser.getContext());
    } else {
      result.addAttribute("operandSegmentSizes",
                          parser.getBuilder().getDenseI32ArrayAttr(
                              {static_cast<int32_t>(inputsOperands.size()),
                               static_cast<int32_t>(outputsOperands.size())}));
    }
  }
  if (!result.propertiesAttr) {
    std::optional<RegisteredOperationName> info =
        result.name.getRegisteredInfo();
    if (info) {
      if (failed(info->verifyInherentAttrs(result.attributes, [&]() {
            return parser.emitError(attrsLoc)
                   << "'" << result.name.getStringRef() << "' op ";
          })))
        return failure();
    }
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

  // Parse optional attributes.
  if (parser.parseOptionalAttrDict(result.attributes))
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
                                   ValueRange inputs, ValueRange outputs,
                                   ArrayRef<StringRef> elidedAttrs = {}) {
  p.printOptionalAttrDict(op->getAttrs(), elidedAttrs);

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
// DSL. See LinalgNamedStructuredOps.yamlgen.cpp.inc for the code that uses this
// class.
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
  RegionBuilderHelper(OpBuilder &builder, Block &block)
      : builder(builder), block(block) {}

  // Build the unary functions defined by OpDSL.
  Value buildUnaryFn(UnaryFn unaryFn, Value arg) {
    if (!isFloatingPoint(arg))
      llvm_unreachable("unsupported non numeric type");
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToEnd(&block);
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
    case UnaryFn::reciprocal: {
      Attribute oneAttr = builder.getOneAttr(arg.getType());
      auto one = builder.create<arith::ConstantOp>(arg.getLoc(),
                                                   ::cast<TypedAttr>(oneAttr));
      return builder.create<arith::DivFOp>(arg.getLoc(), one, arg);
    }
    case UnaryFn::round:
      return builder.create<math::RoundOp>(arg.getLoc(), arg);
    case UnaryFn::sqrt:
      return builder.create<math::SqrtOp>(arg.getLoc(), arg);
    case UnaryFn::rsqrt:
      return builder.create<math::RsqrtOp>(arg.getLoc(), arg);
    case UnaryFn::square:
      return builder.create<arith::MulFOp>(arg.getLoc(), arg, arg);
    case UnaryFn::tanh:
      return builder.create<math::TanhOp>(arg.getLoc(), arg);
    case UnaryFn::erf:
      return builder.create<math::ErfOp>(arg.getLoc(), arg);
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
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToEnd(&block);
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
    case BinaryFn::div:
      if (allComplex)
        return builder.create<complex::DivOp>(arg0.getLoc(), arg0, arg1);
      if (allFloatingPoint)
        return builder.create<arith::DivFOp>(arg0.getLoc(), arg0, arg1);
      if (allBool)
        llvm_unreachable("unsupported operation: div with bools");
      return builder.create<arith::DivSIOp>(arg0.getLoc(), arg0, arg1);
    case BinaryFn::div_unsigned:
      if (!allInteger || allBool)
        llvm_unreachable("unsupported operation: unsigned div not on uint");
      return builder.create<arith::DivUIOp>(arg0.getLoc(), arg0, arg1);
    case BinaryFn::max_signed:
      assert(!allComplex);
      if (allFloatingPoint)
        return builder.create<arith::MaximumFOp>(arg0.getLoc(), arg0, arg1);
      return builder.create<arith::MaxSIOp>(arg0.getLoc(), arg0, arg1);
    case BinaryFn::min_signed:
      assert(!allComplex);
      if (allFloatingPoint)
        return builder.create<arith::MinimumFOp>(arg0.getLoc(), arg0, arg1);
      return builder.create<arith::MinSIOp>(arg0.getLoc(), arg0, arg1);
    case BinaryFn::max_unsigned:
      assert(!allComplex);
      if (allFloatingPoint)
        return builder.create<arith::MaximumFOp>(arg0.getLoc(), arg0, arg1);
      return builder.create<arith::MaxUIOp>(arg0.getLoc(), arg0, arg1);
    case BinaryFn::min_unsigned:
      assert(!allComplex);
      if (allFloatingPoint)
        return builder.create<arith::MinimumFOp>(arg0.getLoc(), arg0, arg1);
      return builder.create<arith::MinUIOp>(arg0.getLoc(), arg0, arg1);
    case BinaryFn::powf:
      assert(allFloatingPoint);
      return builder.create<math::PowFOp>(arg0.getLoc(), arg0, arg1);
    }
    llvm_unreachable("unsupported binary function");
  }

  // Build the ternary functions defined by OpDSL.
  Value buildTernaryFn(TernaryFn ternaryFn, Value arg0, Value arg1,
                       Value arg2) {
    bool headBool =
        isInteger(arg0) && arg0.getType().getIntOrFloatBitWidth() == 1;
    bool tailFloatingPoint =
        isFloatingPoint(arg0) && isFloatingPoint(arg1) && isFloatingPoint(arg2);
    bool tailInteger = isInteger(arg0) && isInteger(arg1) && isInteger(arg2);
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToEnd(&block);
    switch (ternaryFn) {
    case TernaryFn::select:
      if (!headBool && !(tailFloatingPoint || tailInteger))
        llvm_unreachable("unsupported non numeric type");
      return builder.create<arith::SelectOp>(arg0.getLoc(), arg0, arg1, arg2);
    }
    llvm_unreachable("unsupported ternary function");
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
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToEnd(&block);
    Location loc = builder.getUnknownLoc();
    builder.create<YieldOp>(loc, values);
  }

  Value constant(const std::string &value) {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToEnd(&block);
    Location loc = builder.getUnknownLoc();
    Attribute valueAttr = parseAttribute(value, builder.getContext());
    return builder.create<arith::ConstantOp>(loc, ::cast<TypedAttr>(valueAttr));
  }

  Value index(int64_t dim) {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToEnd(&block);
    return builder.create<IndexOp>(builder.getUnknownLoc(), dim);
  }

  Type getIntegerType(unsigned width) {
    return IntegerType::get(builder.getContext(), width);
  }

  Type getFloat32Type() { return Float32Type::get(builder.getContext()); }
  Type getFloat64Type() { return Float64Type::get(builder.getContext()); }

private:
  // Generates operations to cast the given operand to a specified type.
  // If the cast cannot be performed, a warning will be issued and the
  // operand returned as-is (which will presumably yield a verification
  // issue downstream).
  Value cast(Type toType, Value operand, bool isUnsignedCast) {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToEnd(&block);
    auto loc = operand.getLoc();
    return convertScalarToDtype(builder, loc, operand, toType, isUnsignedCast);
  }

  bool isComplex(Value value) {
    return llvm::isa<ComplexType>(value.getType());
  }
  bool isFloatingPoint(Value value) {
    return llvm::isa<FloatType>(value.getType());
  }
  bool isInteger(Value value) {
    return llvm::isa<IntegerType>(value.getType());
  }

  OpBuilder &builder;
  Block &block;
};

} // namespace

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

namespace {

struct EraseSelfCopy : OpRewritePattern<CopyOp> {
  using OpRewritePattern<CopyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    if (copyOp.getInputs() != copyOp.getOutputs())
      return rewriter.notifyMatchFailure(copyOp, "not a self copy");
    if (copyOp.hasPureBufferSemantics())
      rewriter.eraseOp(copyOp);
    else
      rewriter.replaceOp(copyOp, copyOp.getInputs());

    return success();
  }
};

} // namespace

void CopyOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<EraseSelfCopy>(context);
}

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
    TensorReshapeOp newInit;
    if constexpr (std::is_same<TensorReshapeOp, tensor::ExpandShapeOp>::value) {

      newInit = rewriter.create<TensorReshapeOp>(
          loc, reshapeOp.getResultType(), oldFill.output(),
          reshapeOp.getReassociation(), reshapeOp.getOutputShape(),
          reshapeOp.getStaticOutputShape());
    } else {
      newInit = rewriter.create<TensorReshapeOp>(loc, reshapeOp.getResultType(),
                                                 oldFill.output(),
                                                 reshapeOp.getReassociation());
    }
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
    if (failed(reifyResultShapes(rewriter, padOp, reifiedShape)))
      return rewriter.notifyMatchFailure(
          padOp, "failed to reify tensor.pad op result shape");

    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        padOp.getLoc(), reifiedShape.front(),
        padOp.getResultType().getElementType());
    Value replacement =
        rewriter
            .create<FillOp>(fillOp.getLoc(), ValueRange{padValue},
                            ValueRange{emptyTensor})
            .getResult(0);
    if (replacement.getType() != padOp.getResultType()) {
      replacement = rewriter.create<tensor::CastOp>(
          fillOp.getLoc(), padOp.getResultType(), replacement);
    }
    rewriter.replaceOp(padOp, replacement);
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
      newOffsets.push_back(affine::makeComposedFoldedAffineApply(
          rewriter, loc, addMap, {std::get<0>(p), std::get<1>(p)}));
    }

    RankedTensorType srcPadType = srcPadOp.getSourceType();
    SmallVector<OpFoldResult, 4> newSizes;
    for (int i = 0, e = srcPadType.getRank(); i < e; ++i) {
      if (srcPadType.isDynamicDim(i)) {
        newSizes.push_back(
            rewriter.create<tensor::DimOp>(loc, srcPadOp.getSource(), i)
                .getResult());
      } else {
        newSizes.push_back(rewriter.getIndexAttr(srcPadType.getDimSize(i)));
      }
    }

    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        insertOp, srcPadOp.getSource(), insertOp.getDest(), newOffsets,
        newSizes, insertOp.getMixedStrides());
    return success();
  }
};

/// Fold tensor.extract(linalg.fill(<input>)) into <input>
struct FoldFillWithTensorExtract : public OpRewritePattern<tensor::ExtractOp> {
public:
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    // See if tensor input of tensor.extract op is the result of a linalg.fill
    // op.
    auto fillOp = extractOp.getTensor().getDefiningOp<linalg::FillOp>();
    if (!fillOp)
      return failure();

    // Get scalar input operand of linalg.fill op.
    Value extractedScalar = fillOp.getInputs()[0];

    // Replace tensor.extract op with scalar value used to fill the tensor.
    rewriter.replaceOp(extractOp, extractedScalar);
    return success();
  }
};

/// Folds pack(fill) into a single fill op if
///   1. The pack op does not have padding value, or
///   2. The filled value and padding value are the same.
static FailureOr<FillOp> foldFillPackIntoFillOp(RewriterBase &rewriter,
                                                tensor::PackOp packOp) {
  auto fillOp = packOp.getSource().getDefiningOp<FillOp>();
  if (!fillOp)
    return failure();

  if (auto paddingValue = packOp.getPaddingValue())
    if (!isEqualConstantIntOrValue(paddingValue, fillOp.value()))
      return failure();

  Value packOpDest = packOp.getDest();
  if (!packOpDest.hasOneUse())
    return failure();

  return rewriter.create<linalg::FillOp>(packOp.getLoc(), fillOp.getInputs(),
                                         packOp.getDest());
}

/// Wrapper pattern that applies foldFillPackIntoFillOp method.
struct FoldFillWithPack : public OpRewritePattern<tensor::PackOp> {
public:
  FoldFillWithPack(MLIRContext *context)
      : OpRewritePattern<tensor::PackOp>(context) {}

  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    auto fillOp = foldFillPackIntoFillOp(rewriter, packOp);
    if (failed(fillOp))
      return failure();
    rewriter.replaceOp(packOp, fillOp.value().result());
    return success();
  }
};

/// Fold fill with copy.
struct FoldFillWithCopy : OpRewritePattern<linalg::CopyOp> {
  using OpRewritePattern<linalg::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    if (auto fillOp = copyOp.getInputs().front().getDefiningOp<FillOp>()) {
      rewriter.replaceOpWithNewOp<FillOp>(copyOp, copyOp.getResultTypes(),
                                          fillOp.getInputs(),
                                          copyOp.getOutputs());
      return success();
    }
    if (auto fillOp = copyOp.getOutputs().front().getDefiningOp<FillOp>()) {
      rewriter.replaceOpWithNewOp<linalg::CopyOp>(copyOp, copyOp.getInputs(),
                                                  fillOp.getOutputs());
      return success();
    }
    return failure();
  }
};

/// Fold fill with transpose.
struct FoldFillWithTranspose : OpRewritePattern<linalg::TransposeOp> {
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    if (auto fillOp = transposeOp.getInput().getDefiningOp<FillOp>()) {
      rewriter.replaceOpWithNewOp<FillOp>(
          transposeOp, transposeOp.getResultTypes(), fillOp.getInputs(),
          transposeOp.getDpsInitOperand(0)->get());
      return success();
    }
    return failure();
  }
};

/// Fold a concat with all elements being fills of the same value
/// into a fill of the concat result shape.
struct FoldConcatsOfFill : public OpRewritePattern<tensor::ConcatOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ConcatOp concatOp,
                                PatternRewriter &rewriter) const override {
    auto concatOperands = concatOp.getInputs();
    if (concatOperands.empty()) {
      return failure();
    }

    auto firstFillOp = concatOperands.front().getDefiningOp<linalg::FillOp>();
    if (!firstFillOp) {
      return failure();
    }
    // Prefetch the fill value.
    OpFoldResult firstFillVal =
        getAsOpFoldResult(firstFillOp.getDpsInputOperand(0)->get());
    // Collect all the outs values for the fill operations.
    SmallVector<Value> allOuts;
    allOuts.push_back(firstFillOp.getDpsInitOperand(0)->get());

    auto isDefinedByCompatibleFillOp = [&](Value v) -> bool {
      auto fillOp = v.getDefiningOp<linalg::FillOp>();
      if (!fillOp) {
        return false;
      }

      OpFoldResult fillVal =
          getAsOpFoldResult(fillOp.getDpsInputOperand(0)->get());
      if (fillVal != firstFillVal)
        return false;

      allOuts.push_back(fillOp.getDpsInitOperand(0)->get());
      return true;
    };
    if (!llvm::all_of(concatOperands.drop_front(),
                      isDefinedByCompatibleFillOp)) {
      return rewriter.notifyMatchFailure(
          concatOp, "not all operands are defined by a compatible fill op");
    }

    Value outsConcat = rewriter.create<tensor::ConcatOp>(
        concatOp.getLoc(), concatOp.getDim(), allOuts);
    rewriter.replaceOpWithNewOp<linalg::FillOp>(
        concatOp, firstFillOp.getDpsInputOperand(0)->get(), outsConcat);
    return success();
  }
};

} // namespace

void FillOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<FoldConcatsOfFill, FoldFillWithCopy, FoldFillWithTensorExtract,
              FoldFillWithPack, FoldFillWithPad,
              FoldFillWithTensorReshape<tensor::CollapseShapeOp>,
              FoldFillWithTensorReshape<tensor::ExpandShapeOp>,
              FoldInsertPadIntoFill, FoldFillWithTranspose>(context);
}

//===----------------------------------------------------------------------===//
// GenericOp
//===----------------------------------------------------------------------===//

static void buildGenericRegion(
    OpBuilder &builder, Location loc, Region &region, ValueRange inputs,
    ValueRange outputs,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild) {
  SmallVector<Type, 4> blockArgTypes;
  SmallVector<Location, 4> blockArgLocs;
  for (ValueRange container : {inputs, outputs}) {
    for (Value v : container) {
      Type t = v.getType();
      blockArgTypes.push_back(
          isa<MemRefType, RankedTensorType>(t) ? getElementTypeOrSelf(t) : t);
      blockArgLocs.push_back(v.getLoc());
    }
  }

  OpBuilder::InsertionGuard guard(builder);
  Block *bodyBlock =
      builder.createBlock(&region, region.end(), blockArgTypes, blockArgLocs);
  bodyBuild(builder, loc, bodyBlock->getArguments());
}

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
  if (bodyBuild)
    buildGenericRegion(builder, result.location, *result.regions.front(),
                       inputs, outputs, bodyBuild);
}

void GenericOp::build(
    OpBuilder &builder, OperationState &result, TypeRange resultTensorTypes,
    ValueRange inputs, ValueRange outputs, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<utils::IteratorType> iteratorTypes, StringRef doc,
    StringRef libraryCall,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild,
    ArrayRef<NamedAttribute> attributes) {
  build(builder, result, resultTensorTypes, inputs, outputs,
        builder.getAffineMapArrayAttr(indexingMaps),
        builder.getArrayAttr(llvm::to_vector(llvm::map_range(
            iteratorTypes,
            [&](utils::IteratorType iter) -> mlir::Attribute {
              return IteratorTypeAttr::get(builder.getContext(), iter);
            }))),
        doc.empty() ? StringAttr() : builder.getStringAttr(doc),
        libraryCall.empty() ? StringAttr() : builder.getStringAttr(libraryCall),
        bodyBuild, attributes);
}

void GenericOp::build(
    OpBuilder &builder, OperationState &result, ValueRange inputs,
    ValueRange outputs, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<utils::IteratorType> iteratorTypes, StringRef doc,
    StringRef libraryCall,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild,
    ArrayRef<NamedAttribute> attributes) {
  build(builder, result, TypeRange{}, inputs, outputs, indexingMaps,
        iteratorTypes, doc, libraryCall, bodyBuild, attributes);
}

void GenericOp::build(
    OpBuilder &builder, OperationState &result, ValueRange inputs,
    ValueRange outputs, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<utils::IteratorType> iteratorTypes,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild,
    ArrayRef<NamedAttribute> attributes) {
  build(builder, result, inputs, outputs, indexingMaps, iteratorTypes,
        /*doc=*/"",
        /*libraryCall=*/"", bodyBuild, attributes);
}

void GenericOp::build(
    OpBuilder &builder, OperationState &result, TypeRange resultTensorTypes,
    ValueRange inputs, ValueRange outputs, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<utils::IteratorType> iteratorTypes,
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
  for (auto attr : (*this)->getAttrs()) {
    if (attr.getName() == getIteratorTypesAttrName()) {
      auto iteratorTypes =
          llvm::cast<ArrayAttr>(attr.getValue())
              .getAsValueRange<IteratorTypeAttr, utils::IteratorType>();
      // Convert IteratorType enums into the string representation. This is
      // needed, because tests still use the old format when 'iterator_types'
      // attribute is represented as an array of strings.
      // TODO: Remove this conversion once tests are fixed.
      SmallVector<Attribute> iteratorTypeNames =
          llvm::to_vector(llvm::map_range(
              iteratorTypes, [&](utils::IteratorType t) -> Attribute {
                return StringAttr::get(getContext(), stringifyIteratorType(t));
              }));

      genericAttrs.emplace_back(
          getIteratorTypesAttrName(),
          ArrayAttr::get(getContext(), iteratorTypeNames));
    } else if (genericAttrNamesSet.count(attr.getName().strref()) > 0) {
      genericAttrs.push_back(attr);
    }
  }
  if (!genericAttrs.empty()) {
    auto genericDictAttr = DictionaryAttr::get(getContext(), genericAttrs);
    p << genericDictAttr;
  }

  // Printing is shared with named ops, except for the region and attributes
  printCommonStructuredOpParts(p, getDpsInputs(), getDpsInits());

  genericAttrNames.push_back("operandSegmentSizes");
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
  llvm::SMLoc attributeLocation = parser.getCurrentLocation();
  if (parser.parseAttribute(dictAttr, "_", result.attributes))
    return failure();
  result.attributes.assign(dictAttr.getValue().begin(),
                           dictAttr.getValue().end());

  // Convert array of string into an array of IteratorType enums. This is
  // needed, because tests still use the old format when 'iterator_types'
  // attribute is represented as an array of strings.
  // TODO: Remove this conversion once tests are fixed.
  auto iteratorTypes = dyn_cast_or_null<ArrayAttr>(
      result.attributes.get(getIteratorTypesAttrName(result.name)));
  if (!iteratorTypes) {
    return parser.emitError(attributeLocation)
           << "expected " << getIteratorTypesAttrName(result.name)
           << " array attribute";
  }

  SmallVector<Attribute> iteratorTypeAttrs;

  for (StringRef s : iteratorTypes.getAsValueRange<StringAttr>()) {
    auto maybeIteratorType = utils::symbolizeIteratorType(s);
    if (!maybeIteratorType.has_value())
      return parser.emitError(parser.getCurrentLocation())
             << "unexpected iterator_type (" << s << ")";

    iteratorTypeAttrs.push_back(
        IteratorTypeAttr::get(parser.getContext(), maybeIteratorType.value()));
  }
  result.attributes.set(getIteratorTypesAttrName(result.name),
                        parser.getBuilder().getArrayAttr(iteratorTypeAttrs));

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
    LinalgOp linalgOp) {
  for (auto [index, operand] : llvm::enumerate(linalgOp.getDpsInputs())) {
    if (!llvm::isa<MemRefType>(operand.getType()))
      continue;
    effects.emplace_back(
        MemoryEffects::Read::get(), &linalgOp->getOpOperand(index), /*stage=*/0,
        /*effectOnFullRegion=*/true, SideEffects::DefaultResource::get());
  }

  for (OpOperand &operand : linalgOp.getDpsInitsMutable()) {
    if (!llvm::isa<MemRefType>(operand.get().getType()))
      continue;
    if (linalgOp.payloadUsesValueFromOperand(&operand)) {
      effects.emplace_back(MemoryEffects::Read::get(), &operand, /*stage=*/0,
                           /*effectOnFullRegion=*/true,
                           SideEffects::DefaultResource::get());
    }
    effects.emplace_back(MemoryEffects::Write::get(), &operand, /*stage=*/0,
                         /*effectOnFullRegion=*/true,
                         SideEffects::DefaultResource::get());
  }
}

void GenericOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getGenericEffectsImpl(effects, cast<LinalgOp>(getOperation()));
}

static Speculation::Speculatability
getGenericSpeculatabilityImpl(LinalgOp linalgOp) {
  // Operands with value semantics are speculatable, while operands with memory
  // semantics are not.
  if (!linalgOp.hasPureTensorSemantics())
    return Speculation::NotSpeculatable;
  // The body of the op can still have speculation in its region.
  return Speculation::RecursivelySpeculatable;
}

Speculation::Speculatability GenericOp::getSpeculatability() {
  return getGenericSpeculatabilityImpl(cast<LinalgOp>(getOperation()));
}

LogicalResult GenericOp::verify() { return success(); }

namespace {

/// Remove any linalg operation (on tensors) that are just copying
/// the values from inputs to the results. Requirements are
/// 1) All iterator types are parallel
/// 2) The body contains just a yield operation with the yielded values being
///    the arguments corresponding to the operands.
template <typename OpTy>
struct EraseIdentityLinalgOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy linalgOp,
                                PatternRewriter &rewriter) const override {
    // All indexing maps must be equal. It follows that they are permutations.
    if (!llvm::all_equal(linalgOp.getIndexingMapsArray()))
      return failure();

    // Check that the body of the linalg operation is just a linalg.yield
    // operation.
    Block &body = linalgOp->getRegion(0).front();
    if (!llvm::hasSingleElement(body))
      return failure();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yieldOp)
      return failure();

    // In the buffer case, we need to check exact buffer equality.
    if (linalgOp.hasPureBufferSemantics()) {
      if (linalgOp.getNumDpsInputs() == 1 && linalgOp.getNumDpsInits() == 1 &&
          linalgOp.getDpsInputOperand(0)->get() ==
              linalgOp.getDpsInitOperand(0)->get()) {
        rewriter.eraseOp(linalgOp);
        return success();
      }
      return failure();
    }

    // Mixed semantics is not supported yet.
    if (!linalgOp.hasPureTensorSemantics())
      return failure();

    // Get the argument number of the returned values. That is the operand
    // number to use for replacing uses of this operation.
    SmallVector<Value> returnedArgs;
    for (const auto &yieldVal : llvm::enumerate(yieldOp.getValues())) {
      auto yieldArg = llvm::dyn_cast<BlockArgument>(yieldVal.value());
      if (!yieldArg || yieldArg.getOwner() != &body)
        return failure();
      unsigned argumentNumber = yieldArg.getArgNumber();
      Value returnedArg = linalgOp->getOperand(argumentNumber);
      Type resultType = linalgOp->getResult(yieldVal.index()).getType();
      // The input can have a different type than the result, e.g. a dynamic
      // input dimension can be turned into a static output dimension.
      Type returnType = returnedArg.getType();
      if (returnType != resultType) {
        // Distinguish between sparse conversion or dense tensor casting.
        // TODO: unify the two ops?
        if (sparse_tensor::getSparseTensorEncoding(returnType) ||
            sparse_tensor::getSparseTensorEncoding(resultType))
          returnedArg = rewriter.create<sparse_tensor::ConvertOp>(
              linalgOp.getLoc(), resultType, returnedArg);
        else {
          if (!tensor::CastOp::areCastCompatible(returnedArg.getType(),
                                                 resultType))
            return failure();
          returnedArg = rewriter.create<tensor::CastOp>(
              linalgOp.getLoc(), resultType, returnedArg);
        }
      }
      returnedArgs.push_back(returnedArg);
    }

    if (returnedArgs.size() != linalgOp->getNumResults())
      return failure();
    rewriter.replaceOp(linalgOp, returnedArgs);
    return success();
  }
};

} // namespace

void GenericOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<EraseIdentityLinalgOp<GenericOp>>(context);
}

LogicalResult GenericOp::fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
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
    if (llvm::isa<RankedTensorType>(outputType))
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

void MapOp::build(
    OpBuilder &builder, OperationState &result, ValueRange inputs, Value init,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild,
    ArrayRef<NamedAttribute> attributes) {
  build(builder, result, TypeRange{}, inputs, init);
  result.addAttributes(attributes);

  // Add output types for `RankedTensorType` output arguments.
  Type initType = init.getType();
  if (llvm::isa<RankedTensorType>(initType))
    result.addTypes(initType);

  if (bodyBuild)
    buildGenericRegion(builder, result.location, *result.regions.front(),
                       inputs, /*outputs=*/{}, bodyBuild);
}

static void addBodyWithPayloadOp(OpAsmParser &parser, OperationState &result,
                                 const OperationName &payloadOpName,
                                 const NamedAttrList &payloadOpAttrs,
                                 ArrayRef<Value> operands,
                                 bool initFirst = false) {
  OpBuilder b(parser.getContext());
  Region *body = result.addRegion();
  Block &block = body->emplaceBlock();
  b.setInsertionPointToStart(&block);
  SmallVector<Value> bbArgs;
  for (auto &operand : operands) {
    block.addArgument(
        llvm::cast<ShapedType>(operand.getType()).getElementType(),
        b.getUnknownLoc());
  }
  SmallVector<Value> payloadOpOperands;
  // If initFirst flag is enabled, we consider init as the first position of
  // payload operands.
  if (initFirst) {
    payloadOpOperands.push_back(block.getArguments().back());
    for (const auto &arg : block.getArguments().drop_back())
      payloadOpOperands.push_back(arg);
  } else {
    payloadOpOperands = {block.getArguments().begin(),
                         block.getArguments().end()};
  }

  Operation *payloadOp = b.create(
      result.location, b.getStringAttr(payloadOpName.getStringRef()),
      payloadOpOperands,
      TypeRange{llvm::cast<ShapedType>(result.operands.back().getType())
                    .getElementType()},
      payloadOpAttrs);
  b.create<YieldOp>(result.location, payloadOp->getResults());
}

ParseResult MapOp::parse(OpAsmParser &parser, OperationState &result) {
  std::optional<OperationName> payloadOpName;
  NamedAttrList payloadOpAttrs;
  if (succeeded(parser.parseOptionalLBrace())) {
    FailureOr<OperationName> operationName = parser.parseCustomOperationName();
    if (failed(operationName))
      return failure();
    if (parser.parseOptionalAttrDict(payloadOpAttrs))
      return failure();
    payloadOpName = operationName.value();
    if (parser.parseRBrace())
      return failure();
  }

  if (parseDstStyleOp(parser, result))
    return failure();

  if (payloadOpName.has_value()) {
    if (!result.operands.empty())
      addBodyWithPayloadOp(parser, result, payloadOpName.value(),
                           payloadOpAttrs,
                           ArrayRef(result.operands).drop_back());
    else
      result.addRegion();
  } else {
    SmallVector<OpAsmParser::Argument> regionArgs;
    if (parser.parseArgumentList(regionArgs, OpAsmParser::Delimiter::Paren,
                                 /*allowType=*/true, /*allowAttrs=*/true)) {
      return failure();
    }
    Region *body = result.addRegion();
    if (parser.parseRegion(*body, regionArgs))
      return failure();
  }
  return success();
}

// Retrieve the operation from the body, if it is the only one (except
// yield) and if it gets the same amount of arguments as the body does.
// If initFirst flag is enabled, we check that init takes the first position in
// operands of payload.
static Operation *findPayloadOp(Block *body, bool initFirst = false) {
  if (body->getOperations().size() != 2)
    return nullptr;
  Operation &payload = body->getOperations().front();
  assert(isa<YieldOp>(body->getOperations().back()));

  if (payload.getNumOperands() == 0 ||
      payload.getNumOperands() != body->getNumArguments())
    return nullptr;
  if (initFirst) {
    // check init
    if (payload.getOperands().back() != body->getArgument(0))
      return nullptr;
    // check rest
    for (const auto &[operand, bbArg] :
         llvm::zip(payload.getOperands(), body->getArguments().drop_front())) {
      if (bbArg != operand)
        return nullptr;
    }
  } else {
    for (const auto &[operand, bbArg] :
         llvm::zip(payload.getOperands(), body->getArguments())) {
      if (bbArg != operand)
        return nullptr;
    }
  }
  return &payload;
}

void printShortForm(OpAsmPrinter &p, Operation *payloadOp) {
  SmallVector<StringRef> elidedAttrs;
  std::string attrToElide;
  p << " { " << payloadOp->getName().getStringRef();
  for (const auto &attr : payloadOp->getAttrs()) {
    auto fastAttr =
        llvm::dyn_cast<mlir::arith::FastMathFlagsAttr>(attr.getValue());
    if (fastAttr && fastAttr.getValue() == mlir::arith::FastMathFlags::none) {
      attrToElide = attr.getName().str();
      elidedAttrs.push_back(attrToElide);
      break;
    }
  }
  p.printOptionalAttrDict(payloadOp->getAttrs(), elidedAttrs);
  p << " }";
}

void MapOp::print(OpAsmPrinter &p) {
  Block *mapper = getBody();
  Operation *payloadOp = findPayloadOp(mapper);
  if (payloadOp) {
    printShortForm(p, payloadOp);
  }

  printCommonStructuredOpParts(p, getDpsInputs(), getDpsInits());
  p.printOptionalAttrDict((*this)->getAttrs());

  if (!payloadOp) {
    // Print region if the payload op was not detected.
    p.increaseIndent();
    p.printNewline();
    p << "(";
    llvm::interleaveComma(mapper->getArguments(), p,
                          [&](auto arg) { p.printRegionArgument(arg); });
    p << ") ";

    p.printRegion(getMapper(), /*printEntryBlockArgs=*/false);
    p.decreaseIndent();
  }
}

LogicalResult MapOp::verify() {
  auto *bodyBlock = getBody();
  auto blockArgs = bodyBlock->getArguments();

  // Checks if the number of `inputs` match the arity of the `mapper` region.
  if (getInputs().size() != blockArgs.size())
    return emitOpError() << "expects number of operands to match the arity of "
                            "mapper, but got: "
                         << getInputs().size() << " and " << blockArgs.size();

  // The parameters of mapper should all match the element type of inputs.
  for (const auto &[bbArgType, inputArg] :
       llvm::zip(bodyBlock->getArgumentTypes(), getInputs())) {
    auto inputElemType =
        llvm::cast<ShapedType>(inputArg.getType()).getElementType();
    if (bbArgType != inputElemType) {
      return emitOpError() << "expected element type of input " << inputElemType
                           << " to match bbArg type " << bbArgType;
    }
  }

  // The shape of each input must match the shape of the output.
  auto outputShape = getInit().getType().getShape();
  for (Type inputArgType : TypeRange{getInputs()}) {
    auto inputElemShape = llvm::cast<ShapedType>(inputArgType).getShape();
    if (inputElemShape != outputShape) {
      return emitOpError() << "expected shape of input (" << inputElemShape
                           << ") to match shape of output (" << outputShape
                           << ")";
    }
  }

  return success();
}

SmallVector<utils::IteratorType> MapOp::getIteratorTypesArray() {
  int64_t rank = getInit().getType().getRank();
  return SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel);
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
  getGenericEffectsImpl(effects, cast<LinalgOp>(getOperation()));
}

Speculation::Speculatability MapOp::getSpeculatability() {
  return getGenericSpeculatabilityImpl(cast<LinalgOp>(getOperation()));
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

void ReduceOp::build(
    OpBuilder &builder, OperationState &result, ValueRange inputs,
    ValueRange inits, ArrayRef<int64_t> dimensions,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild,
    ArrayRef<NamedAttribute> attributes) {
  build(builder, result, TypeRange{}, inputs, inits, dimensions);
  result.addAttributes(attributes);

  // Add output types for `RankedTensorType` output arguments.
  for (Value init : inits) {
    Type initType = init.getType();
    if (llvm::isa<RankedTensorType>(initType))
      result.addTypes(initType);
  }

  if (bodyBuild)
    buildGenericRegion(builder, result.location, *result.regions.front(),
                       inputs, inits, bodyBuild);
}

SmallVector<utils::IteratorType> ReduceOp::getIteratorTypesArray() {
  int64_t inputRank =
      llvm::cast<ShapedType>(getInputs()[0].getType()).getRank();
  SmallVector<utils::IteratorType> iteratorTypes(inputRank,
                                                 utils::IteratorType::parallel);
  for (int64_t reductionDim : getDimensions())
    iteratorTypes[reductionDim] = utils::IteratorType::reduction;
  return iteratorTypes;
}

ArrayAttr ReduceOp::getIndexingMaps() {
  int64_t inputRank =
      llvm::cast<ShapedType>(getInputs()[0].getType()).getRank();
  SmallVector<AffineMap> affineMaps(
      getNumDpsInputs(),
      AffineMap::getMultiDimIdentityMap(inputRank, getContext()));
  AffineMap resultMap =
      AffineMap::getMultiDimIdentityMap(inputRank, getContext())
          .dropResults(getDimensions());
  for (int64_t i = 0, e = getNumDpsInits(); i < e; ++i)
    affineMaps.push_back(resultMap);
  return Builder(getContext()).getAffineMapArrayAttr(affineMaps);
}

void ReduceOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getGenericEffectsImpl(effects, cast<LinalgOp>(getOperation()));
}

Speculation::Speculatability ReduceOp::getSpeculatability() {
  return getGenericSpeculatabilityImpl(cast<LinalgOp>(getOperation()));
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
  std::optional<OperationName> payloadOpName;
  NamedAttrList payloadOpAttrs;
  if (succeeded(parser.parseOptionalLBrace())) {
    FailureOr<OperationName> operationName = parser.parseCustomOperationName();
    if (failed(operationName))
      return failure();
    if (parser.parseOptionalAttrDict(payloadOpAttrs))
      return failure();
    payloadOpName = operationName.value();
    if (parser.parseRBrace())
      return failure();
  }

  if (parseDstStyleOp(
          parser, result, [&](OpAsmParser &parser, NamedAttrList &attributes) {
            return parseDenseI64ArrayAttr(parser, attributes, "dimensions");
          }))
    return failure();

  if (payloadOpName.has_value()) {
    addBodyWithPayloadOp(parser, result, payloadOpName.value(), payloadOpAttrs,
                         ArrayRef(result.operands), /*initFirst=*/true);
  } else {
    SmallVector<OpAsmParser::Argument> regionArgs;
    if (parser.parseArgumentList(regionArgs, OpAsmParser::Delimiter::Paren,
                                 /*allowType=*/true, /*allowAttrs=*/true)) {
      return failure();
    }

    Region *body = result.addRegion();
    if (parser.parseRegion(*body, regionArgs))
      return failure();
  }

  return success();
}

static void printDenseI64ArrayAttr(OpAsmPrinter &p, StringRef attributeName,
                                   ArrayRef<int64_t> attributeValue) {
  p << ' ' << attributeName << " = [" << attributeValue << "] ";
}

void ReduceOp::print(OpAsmPrinter &p) {
  Block *mapper = getBody();
  Operation *payloadOp = findPayloadOp(mapper, /*initFirst=*/true);
  if (payloadOp) {
    printShortForm(p, payloadOp);
  }

  printCommonStructuredOpParts(p, getDpsInputs(), getDpsInits());
  printDenseI64ArrayAttr(p, getDimensionsAttrName(), getDimensions());
  p.printOptionalAttrDict((*this)->getAttrs(), {getDimensionsAttrName()});
  if (!payloadOp) {
    // Print region if the payload op was not detected.
    p.increaseIndent();
    p.printNewline();
    p << "(";
    llvm::interleaveComma(mapper->getArguments(), p,
                          [&](auto arg) { p.printRegionArgument(arg); });
    p << ") ";

    p.printRegion(getCombiner(), /*printEntryBlockArgs=*/false);
    p.decreaseIndent();
  }
}

LogicalResult ReduceOp::verify() {
  ArrayRef<int64_t> dimensionsRef = getDimensions();

  for (int64_t i = 1; i < getNumDpsInputs(); ++i) {
    if (llvm::cast<ShapedType>(getInputs()[i].getType()).getShape() !=
        llvm::cast<ShapedType>(getInputs()[0].getType()).getShape()) {
      return emitOpError() << "expects all inputs to have the same shapes. "
                              "Shape at input-index "
                           << i
                           << " is not equal to the shape at input-index 0.";
    }
  }
  for (int64_t i = 1; i < getNumDpsInits(); ++i) {
    if (llvm::cast<ShapedType>(getInits()[i].getType()).getShape() !=
        llvm::cast<ShapedType>(getInits()[0].getType()).getShape()) {
      return emitOpError() << "expects all outputs to have the same shapes. "
                              "Shape at output-index "
                           << i
                           << " is not equal to the shape at output-index 0.";
    }
  }
  auto inputType = llvm::cast<ShapedType>(getInputs()[0].getType());
  auto initType = llvm::cast<ShapedType>(getInits()[0].getType());

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
    Type inputElementType =
        llvm::cast<ShapedType>(input.getType()).getElementType();
    if (inputElementType != bbArg.getType())
      return emitOpError()
             << "input element type " << inputElementType
             << " does not match corresponding block argument type "
             << bbArg.getType();
  }

  // Check that the last block arguments match the element type of the outputs.
  for (auto [output, bbArg] : llvm::zip(
           getDpsInits(), block->getArguments().take_back(getNumDpsInits()))) {
    auto outputElementType =
        llvm::cast<ShapedType>(output.getType()).getElementType();
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

static void buildIdentityRegion(OpBuilder &builder, Location loc,
                                Region &region, ValueRange inputs,
                                ValueRange outputs) {
  buildGenericRegion(builder, loc, region, inputs, outputs,
                     [](OpBuilder &b, Location loc, ValueRange args) {
                       if (!args.empty())
                         b.create<linalg::YieldOp>(loc, args[0]);
                     });
}

void TransposeOp::build(::mlir::OpBuilder &builder,
                        ::mlir::OperationState &result, Value input, Value init,
                        DenseI64ArrayAttr permutation,
                        ArrayRef<NamedAttribute> attributes) {
  result.addOperands(input);
  result.addOperands(init);
  result.addAttribute(getPermutationAttrName(result.name), permutation);
  result.addAttributes(attributes);

  // Add output types for `RankedTensorType` output arguments.
  Type initType = init.getType();
  if (llvm::isa<RankedTensorType>(initType))
    result.addTypes(initType);

  buildIdentityRegion(builder, result.location, *result.addRegion(), input,
                      init);
}

void TransposeOp::build(::mlir::OpBuilder &builder,
                        ::mlir::OperationState &result, Value input, Value init,
                        ArrayRef<int64_t> permutation,
                        ArrayRef<NamedAttribute> attributes) {
  build(builder, result, input, init, builder.getDenseI64ArrayAttr(permutation),
        attributes);
}

ParseResult TransposeOp::parse(OpAsmParser &parser, OperationState &result) {
  if (failed(parseDstStyleOp(
          parser, result, [&](OpAsmParser &parser, NamedAttrList &attributes) {
            return parseDenseI64ArrayAttr(parser, attributes, "permutation");
          })))
    return failure();

  OpBuilder builder(parser.getContext());
  buildIdentityRegion(builder, result.location, *result.addRegion(),
                      /*inputs=*/result.operands,
                      /*outputs=*/{});
  return success();
}

void TransposeOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  if (!getResults().empty())
    setNameFn(getResults().front(), "transposed");
}

void TransposeOp::print(OpAsmPrinter &p) {
  printCommonStructuredOpParts(p, getDpsInputs(), getDpsInits());
  printDenseI64ArrayAttr(p, getPermutationAttrName(), getPermutation());
  p.printOptionalAttrDict((*this)->getAttrs(), {getPermutationAttrName()});
}

LogicalResult TransposeOp::verify() {
  ArrayRef<int64_t> permutationRef = getPermutation();

  if (!isPermutationVector(permutationRef))
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

SmallVector<utils::IteratorType> TransposeOp::getIteratorTypesArray() {
  int64_t rank = getInit().getType().getRank();
  return SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel);
}

ArrayAttr TransposeOp::getIndexingMaps() {
  Builder builder(getContext());
  int64_t rank = getInit().getType().getRank();
  return builder.getAffineMapArrayAttr(
      {inversePermutation(AffineMap::getPermutationMap(
           llvm::to_vector_of<unsigned>(getPermutation()), getContext())),
       builder.getMultiDimIdentityMap(rank)});
}

void TransposeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getGenericEffectsImpl(effects, cast<LinalgOp>(getOperation()));
}

Speculation::Speculatability TransposeOp::getSpeculatability() {
  return getGenericSpeculatabilityImpl(cast<LinalgOp>(getOperation()));
}

LogicalResult TransposeOp::fold(FoldAdaptor adaptor,
                                SmallVectorImpl<OpFoldResult> &result) {
  // Only the tensor type is supported.
  if (!isa<TensorType>(getInput().getType()))
    return failure();

  // Single dimension transpose.
  if (getPermutation().size() == 0) {
    result.push_back(getInput());
    return success();
  }
  // Identity permutation.
  if (isIdentityPermutation(getPermutation())) {
    result.push_back(getInput());
    return success();
  }

  return failure();
}

/// Fold transpose with transpose.
struct FoldTransposeWithTranspose : OpRewritePattern<linalg::TransposeOp> {
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    auto defTransposeOp = transposeOp.getInput().getDefiningOp<TransposeOp>();
    if (!defTransposeOp)
      return failure();
    ArrayRef<int64_t> defPerms = defTransposeOp.getPermutation();
    ArrayRef<int64_t> perms = transposeOp.getPermutation();
    SmallVector<int64_t> foldedPerms;
    foldedPerms.reserve(perms.size());
    for (int64_t perm : perms)
      foldedPerms.push_back(defPerms[perm]);

    rewriter.replaceOpWithNewOp<TransposeOp>(
        transposeOp, defTransposeOp.getInput(), transposeOp.getInit(),
        foldedPerms);
    return success();
  }
};

/// This pattern canonicalize transpose by swapping the order of
/// broadcast and transpose:
///   transpose(broadcast(input)) -> broadcast(transpose(input))
struct SwapTransposeWithBroadcast : OpRewritePattern<linalg::TransposeOp> {
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    Value input = transposeOp.getInput();
    BroadcastOp broadcastOp = input.getDefiningOp<BroadcastOp>();
    if (!input.hasOneUse() || !broadcastOp)
      return failure();

    ArrayRef<int64_t> dimensions = broadcastOp.getDimensions();
    ArrayRef<int64_t> perms = transposeOp.getPermutation();

    // Get new perms and new dimensions.
    SmallVector<int64_t> resultPerms = dropDims(perms, dimensions);
    SmallVector<int64_t> invertPerm = invertPermutationVector(perms);
    SmallVector<int64_t> resultDimensions;
    unsigned dimensionSize = dimensions.size();
    for (unsigned i = 0; i < dimensionSize; ++i)
      resultDimensions.push_back(invertPerm[dimensions[i]]);

    // Create transpose result.
    Value broadcastInput = broadcastOp.getInput();
    Location loc = transposeOp.getLoc();
    MLIRContext *ctx = transposeOp.getContext();
    SmallVector<OpFoldResult> dims;
    auto broadcastInputTy =
        mlir::cast<RankedTensorType>(broadcastInput.getType());
    unsigned inputRank = broadcastInputTy.getRank();
    for (unsigned i = 0; i < inputRank; ++i) {
      if (broadcastInputTy.isDynamicDim(i)) {
        dims.push_back(rewriter.create<tensor::DimOp>(loc, broadcastInput, i)
                           ->getResult(0));
      } else {
        dims.push_back(IntegerAttr::get(IndexType::get(ctx),
                                        broadcastInputTy.getDimSize(i)));
      }
    }
    SmallVector<OpFoldResult> transposeResultShapes =
        applyPermutation(dims, resultPerms);
    Value transposeInit = rewriter.create<tensor::EmptyOp>(
        transposeOp.getLoc(), transposeResultShapes,
        broadcastInputTy.getElementType());

    // Create broadcast(transpose(input)).
    Value transposeResult =
        rewriter
            .create<TransposeOp>(loc, broadcastOp.getInput(), transposeInit,
                                 resultPerms)
            ->getResult(0);
    rewriter.replaceOpWithNewOp<BroadcastOp>(
        transposeOp, transposeResult, transposeOp.getInit(), resultDimensions);
    return success();
  }
};

void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<FoldTransposeWithTranspose, SwapTransposeWithBroadcast>(context);
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

void BroadcastOp::build(::mlir::OpBuilder &builder,
                        ::mlir::OperationState &result, Value input, Value init,
                        DenseI64ArrayAttr dimensions,
                        ArrayRef<NamedAttribute> attributes) {
  result.addOperands(input);
  result.addOperands(init);
  result.addAttribute(getDimensionsAttrName(result.name), dimensions);
  result.addAttributes(attributes);

  // Add output types for `RankedTensorType` output arguments.
  Type initType = init.getType();
  if (llvm::isa<RankedTensorType>(initType))
    result.addTypes(initType);

  buildIdentityRegion(builder, result.location, *result.addRegion(), input,
                      init);
}

void BroadcastOp::build(::mlir::OpBuilder &builder,
                        ::mlir::OperationState &result, Value input, Value init,
                        ArrayRef<int64_t> dimensions,
                        ArrayRef<NamedAttribute> attributes) {
  build(builder, result, input, init, builder.getDenseI64ArrayAttr(dimensions),
        attributes);
}

ParseResult BroadcastOp::parse(OpAsmParser &parser, OperationState &result) {
  if (failed(parseDstStyleOp(
          parser, result, [&](OpAsmParser &parser, NamedAttrList &attributes) {
            return parseDenseI64ArrayAttr(parser, attributes, "dimensions");
          })))
    return failure();

  OpBuilder builder(parser.getContext());
  buildIdentityRegion(builder, result.location, *result.addRegion(),
                      /*inputs=*/result.operands,
                      /*outputs=*/{});
  return success();
}

void BroadcastOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  if (!getResults().empty())
    setNameFn(getResults().front(), "broadcasted");
}

void BroadcastOp::print(OpAsmPrinter &p) {
  printCommonStructuredOpParts(p, getDpsInputs(), getDpsInits());
  printDenseI64ArrayAttr(p, getDimensionsAttrName(), getDimensions());
  p.printOptionalAttrDict((*this)->getAttrs(), {getDimensionsAttrName()});
}

LogicalResult BroadcastOp::verify() {
  ArrayRef<int64_t> dimensionsRef = getDimensions();

  auto inputType = getInput().getType();
  auto initType = getInit().getType();

  int64_t inputRank = inputType.getRank();
  int64_t initRank = initType.getRank();

  auto inputShape = inputType.getShape();
  auto initShape = initType.getShape();

  if ((size_t)inputRank + dimensionsRef.size() != (size_t)initRank)
    return emitOpError() << "input rank plus added dimensions does not "
                            "match init rank. input rank: "
                         << inputRank
                         << ", dimensions size: " << dimensionsRef.size()
                         << ", init rank: " << initRank;

  for (const auto &[idx, dim] : llvm::enumerate(dimensionsRef)) {
    if (dim < 0 || dim >= initRank)
      return emitOpError() << "dimension " << idx
                           << " is out of range. expected range: [0, "
                           << initRank - 1 << "], got: " << dim;
  }

  // Mapping from input dims to init dims.
  SmallVector<int64_t> dimMap;
  for (auto dim : llvm::seq<int64_t>(0, initRank)) {
    if (!llvm::is_contained(dimensionsRef, dim))
      dimMap.push_back(dim);
  }

  for (const auto &[inputDimIdx, initDimIdx] : llvm::enumerate(dimMap)) {
    // This dimensions is mapped from the input. Init and input dims should
    // match.
    if (inputShape[inputDimIdx] != initShape[initDimIdx])
      return emitOpError() << "input dim " << inputDimIdx
                           << " should match init dim " << initDimIdx
                           << ". input: " << inputShape[inputDimIdx]
                           << ", init: " << initShape[initDimIdx];
  }

  return success();
}

SmallVector<utils::IteratorType> BroadcastOp::getIteratorTypesArray() {
  int64_t rank = getInit().getType().getRank();
  return SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel);
}

ArrayAttr BroadcastOp::getIndexingMaps() {
  Builder builder(getContext());
  int64_t rank = getInit().getType().getRank();
  return builder.getAffineMapArrayAttr(
      {builder.getMultiDimIdentityMap(rank).dropResults(getDimensions()),
       builder.getMultiDimIdentityMap(rank)});
}

void BroadcastOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getGenericEffectsImpl(effects, cast<LinalgOp>(getOperation()));
}

Speculation::Speculatability BroadcastOp::getSpeculatability() {
  return getGenericSpeculatabilityImpl(cast<LinalgOp>(getOperation()));
}

void BroadcastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<EraseIdentityLinalgOp<BroadcastOp>>(context);
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
  if (op.getNumOperands() != linalgOp.getNumDpsInits())
    return op.emitOpError("expected number of yield values (")
           << op.getNumOperands()
           << ") to match the number of inits / outs operands of the enclosing "
           << "LinalgOp (" << linalgOp.getNumDpsInits() << ")";

  for (OpOperand &opOperand : op->getOpOperands()) {
    OpOperand *outputOperand =
        linalgOp.getDpsInitOperand(opOperand.getOperandNumber());
    Type elementType = outputOperand->get().getType();
    if (isa<MemRefType, RankedTensorType>(elementType))
      elementType = getElementTypeOrSelf(outputOperand->get().getType());
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

AffineMap mlir::linalg::extractOrIdentityMap(std::optional<AffineMap> maybeMap,
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

static LogicalResult appendMangledType(llvm::raw_string_ostream &ss, Type t) {
  if (auto memref = llvm::dyn_cast<MemRefType>(t)) {
    ss << "view";
    for (auto size : memref.getShape())
      if (size < 0)
        ss << "sx";
      else
        ss << size << "x";
    if (failed(appendMangledType(ss, memref.getElementType())))
      return failure();
    if (auto as = memref.getMemorySpace()) {
      if (auto attr = llvm::dyn_cast<IntegerAttr>(as))
        ss << "as" << attr.getInt();
      else
        return failure();
    }
    return success();
  }
  if (auto vec = llvm::dyn_cast<VectorType>(t)) {
    ss << "vector";
    llvm::interleave(
        vec.getShape(), [&](int64_t i) { ss << i; }, [&]() { ss << "x"; });
    if (failed(appendMangledType(ss, vec.getElementType())))
      return failure();
    return success();
  }
  if (t.isSignlessIntOrIndexOrFloat()) {
    ss << t;
    return success();
  }
  return failure();
}

std::string mlir::linalg::generateLibraryCallName(Operation *op) {
  assert(isa<LinalgOp>(op));
  std::string name(op->getName().getStringRef().str());
  std::string fun = "";
  for (NamedAttribute kv : op->getAttrs()) {
    if (UnaryFnAttr ufa = llvm::dyn_cast<UnaryFnAttr>(kv.getValue())) {
      fun = stringifyEnum(ufa.getValue()).str() + "_";
    } else if (BinaryFnAttr bfa = llvm::dyn_cast<BinaryFnAttr>(kv.getValue())) {
      fun = stringifyEnum(bfa.getValue()).str() + "_";
    }
  }
  name.reserve(128);
  std::replace(name.begin(), name.end(), '.', '_');
  llvm::raw_string_ostream ss(name);
  ss << "_" << fun;
  for (Type t : op->getOperandTypes()) {
    if (failed(appendMangledType(ss, t)))
      return std::string();
    ss << "_";
  }
  name.pop_back();
  return name;
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
      auto mt = llvm::dyn_cast<MemRefType>(opOperand.get().getType());
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
    OpResult resultValue = llvm::cast<OpResult>(castOp.getSource());
    unsigned resultNumber = resultValue.getResultNumber();
    auto resultType =
        llvm::cast<RankedTensorType>(castOp->getResult(0).getType());
    // Replace the `outs` for the result with a `tensor.cast`. This cast is now
    // going from a more dynamic shape to a less dynamic shape. If the producer
    // for this cast, i.e. producer of the out operand, is also an operation
    // that folds with tensor.cast consumer (like this pattern), the cast will
    // continue to propagate as far up the stack as it can go.
    OpOperand *outOperand = linalgOp.getDpsInitOperand(resultNumber);
    Value newOperand =
        rewriter.create<tensor::CastOp>(loc, resultType, outOperand->get());
    SmallVector<Value> newOperands = linalgOp.getDpsInputs();
    SmallVector<Value> outputOperands(linalgOp.getDpsInits().begin(),
                                      linalgOp.getDpsInits().end());
    outputOperands[resultNumber] = newOperand;
    newOperands.append(outputOperands.begin(), outputOperands.end());

    SmallVector<Type> resultTypes(linalgOp->result_type_begin(),
                                  linalgOp->result_type_end());
    resultTypes[resultNumber] = resultType;
    Operation *newOp = clone(rewriter, linalgOp, resultTypes, newOperands);

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
    auto sourceType = llvm::cast<RankedTensorType>(src.getType());
    auto sourceMap = linalgOp.getMatchingIndexingMap(&opOperand);

    // Get the `sourceShape` of the `sourceType`. If the operand is a result of
    // `tensor.cast` operation and source of the cast operation has a static
    // shape, then assign it to the `sourceShape`.
    auto *parentOp = src.getDefiningOp();
    ArrayRef<int64_t> sourceShape = sourceType.getShape();
    if (parentOp) {
      if (auto castOp = dyn_cast<tensor::CastOp>(parentOp)) {
        Value castSource = castOp.getSource();
        auto castSourceType =
            llvm::dyn_cast<RankedTensorType>(castSource.getType());
        if (castSourceType && castSourceType.hasStaticShape())
          sourceShape = castSourceType.getShape();
      }
    }

    // If the source shape's dimension has a static shape, map the affine dim
    // expression to the known static size.
    for (unsigned i = 0; i < sourceShape.size(); i++) {
      if (sourceType.isDynamicDim(i))
        continue;
      if (auto affineDimExpr = dyn_cast<AffineDimExpr>(sourceMap.getResult(i)))
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
  auto sourceType = llvm::cast<RankedTensorType>(src.getType());
  Type resultType = sourceType;
  if (sourceType.hasStaticShape() && linalgOp.isDpsInit(opOperand)) {
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
    if (!affineExprToSize.contains(dimExpr) || !sourceType.isDynamicDim(i)) {
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
  if (linalgOp.isDpsInit(opOperand))
    resultTypes.push_back(resultType);
}

/// Static shapes for the operands can be inferred if any one of the operands
/// have a static shape. This can be done by referring to the affine dim
/// expressions for the operand.
struct InferStaticShapeOfOperands : public OpInterfaceRewritePattern<LinalgOp> {
  using OpInterfaceRewritePattern<LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!linalgOp.hasPureTensorSemantics())
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
    resultTypes.reserve(linalgOp.getNumDpsInits());

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
    Operation *newOp = clone(rewriter, linalgOp, resultTypes, newOperands);
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
// SoftmaxOp
//===----------------------------------------------------------------------===//

LogicalResult SoftmaxOp::verify() {
  ShapedType inputType = getInputOperandType();
  ShapedType outputType = getOutputOperandType();

  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> outputShape = outputType.getShape();
  if (failed(verifyCompatibleShape(inputShape, outputShape)))
    return emitOpError("incompatible output shape");

  int64_t inputRank = getInputOperandRank();
  int64_t dimension = getDimension();
  if ((dimension < 0) || (dimension >= inputRank))
    return emitOpError("incorrect dimension specified");

  return success();
}

SmallVector<Range> SoftmaxOp::getIterationDomain(OpBuilder &builder) {
  int64_t operandRank = getInputOperandRank();
  SmallVector<Range> loopBounds(operandRank);
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value source = getInput();
  for (auto dim : llvm::seq<int64_t>(0, operandRank)) {
    loopBounds[dim].offset = zero;
    loopBounds[dim].size = getDimValue(builder, loc, source, dim);
    loopBounds[dim].stride = one;
  }
  return loopBounds;
}

SmallVector<utils::IteratorType> SoftmaxOp::getLoopIteratorTypes() {
  SmallVector<utils::IteratorType> iteratorTypes(getInputOperandRank(),
                                                 utils::IteratorType::parallel);
  iteratorTypes[getDimension()] = utils::IteratorType::reduction;
  return iteratorTypes;
}

FailureOr<TilingResult>
SoftmaxOp::getTiledImplementation(OpBuilder &builder,
                                  ArrayRef<OpFoldResult> offsets,
                                  ArrayRef<OpFoldResult> sizes) {
  int64_t rank = getInputOperandRank();
  auto oneAttr = builder.getI64IntegerAttr(1);
  SmallVector<OpFoldResult> strides(rank, oneAttr);
  SmallVector<Value> tiledOperands;
  Operation *inputSlice =
      getSlice(builder, getLoc(), getInput(), offsets, sizes, strides);
  if (!inputSlice) {
    return emitOpError("failed to compute input slice");
  }
  tiledOperands.emplace_back(inputSlice->getResult(0));
  Operation *outputSlice =
      getSlice(builder, getLoc(), getOutput(), offsets, sizes, strides);
  if (!outputSlice) {
    return emitOpError("failed to compute output slice");
  }
  tiledOperands.emplace_back(outputSlice->getResult(0));

  SmallVector<Type, 4> resultTypes;
  if (hasPureTensorSemantics())
    resultTypes.push_back(tiledOperands[1].getType());
  Operation *tiledOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);

  return TilingResult{
      {tiledOp},
      SmallVector<Value>(tiledOp->getResults()),
      llvm::to_vector(ArrayRef<Operation *>{inputSlice, outputSlice})};
}

LogicalResult SoftmaxOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber == 0) {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }
  return failure();
}

// cast(dynamic) -> static.
LogicalResult SoftmaxOp::fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

LogicalResult
SoftmaxOp::reifyResultShapes(OpBuilder &b,
                             ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  SmallVector<OpFoldResult> shapes;
  Location loc = getOperation()->getLoc();
  IRRewriter rewriter(b);
  auto inputShapedType = llvm::cast<ShapedType>(getInputOperandType());
  auto outputShapedType = llvm::cast<ShapedType>(getOutputOperandType());
  for (int64_t dim : llvm::seq<int64_t>(0, getOutputOperandRank())) {
    if (!outputShapedType.isDynamicDim(dim)) {
      // Static dim: Return IntegerAttr.
      shapes.push_back(b.getIndexAttr(inputShapedType.getDimSize(dim)));
    } else {
      // Dynamic dim: Return Value.
      OpFoldResult ofr = createOrFoldDimOp(b, loc, getInput(), dim);
      shapes.push_back(getValueOrCreateConstantIndexOp(b, loc, ofr));
    }
  }
  reifiedReturnShapes.emplace_back(std::move(shapes));
  return success();
}

void SoftmaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  for (auto [index, operand] : llvm::enumerate(getDpsInputs())) {
    if (!llvm::isa<MemRefType>(operand.getType()))
      continue;
    effects.emplace_back(MemoryEffects::Read::get(),
                         &getOperation()->getOpOperand(index), /*stage=*/0,
                         /*effectOnFullRegion=*/true,
                         SideEffects::DefaultResource::get());
  }

  for (OpOperand &operand : getDpsInitsMutable()) {
    if (!llvm::isa<MemRefType>(operand.get().getType()))
      continue;
    effects.emplace_back(MemoryEffects::Read::get(), &operand, /*stage=*/0,
                         /*effectOnFullRegion=*/true,
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), &operand, /*stage=*/0,
                         /*effectOnFullRegion=*/true,
                         SideEffects::DefaultResource::get());
  }
}

// Helper functions for softmax decomposition.
// @{

// Helper function to produce the iterator types (reduction or parallel) and
// affine maps for the iterators used in the decomposition of softmax.
// This method creates:
// If allParallel == true:
// - iterator type: {parallel, ..., parallel}
// - affine maps:
// -- identity with inputRank dimensions.
// -- (d0, ..., dN) -> (d0, ..., d_dim-1, d_dim+1, ..., dN),
//    where N == inputRank.
//
// If allParallel == false:
// - iterator type at dim(i) == parallel for i != \p dim and
//   dim(dim) == reduction.
// - affine map:
// -- identity with inputRank dimensions.
// -- (d0, ..., dN) -> (d0, ..., d_dim-1, d_dim+1, ..., dN),
//    where N == inputRank.
static std::tuple<SmallVector<utils::IteratorType>, SmallVector<AffineMap>>
computeIteratorTypesAndIndexingMaps(OpBuilder &builder, int64_t inputRank,
                                    int64_t dim, bool allParallel = false) {
  SmallVector<utils::IteratorType> iteratorTypes(inputRank,
                                                 utils::IteratorType::parallel);
  if (!allParallel)
    iteratorTypes[dim] = utils::IteratorType::reduction;
  MLIRContext *ctxt = builder.getContext();
  auto identityMap = AffineMap::getMultiDimIdentityMap(inputRank, ctxt);
  SmallVector<AffineExpr, 2> affineExprs;
  for (int i = 0; i < inputRank; i++) {
    if (i != dim)
      affineExprs.push_back(mlir::getAffineDimExpr(i, ctxt));
  }
  auto reductionMap =
      AffineMap::get(inputRank, /*symbols=*/0, affineExprs, ctxt);
  SmallVector<AffineMap> indexingMaps{identityMap, reductionMap};
  return std::make_tuple(iteratorTypes, indexingMaps);
}

// Helper function to produce a linalg.generic that computes a reduction on
// dimension \p dim with the operation type \p T.
template <typename T>
static Value reduce(OpBuilder &builder, Location loc, Value input, Value output,
                    int64_t dim) {
  auto inputType = cast<ShapedType>(input.getType());
  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t inputRank = inputShape.size();
  auto [iteratorTypes, indexingMaps] =
      computeIteratorTypesAndIndexingMaps(builder, inputRank, dim);
  assert(indexingMaps.size() == 2 &&
         "We should have two maps: 1 for the input, 1 for the output");
  assert(indexingMaps[0].isIdentity() && "input map should be identity");

  auto genericOp = builder.create<linalg::GenericOp>(
      loc, output.getType(), input, output, indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value result = b.create<T>(loc, args[0], args[1]);
        b.create<linalg::YieldOp>(loc, result);
      });
  return genericOp.getResult(0);
}

/// Produce a linalg generic that computes the second step of the softmax
/// decomposition: res = exp(input - max), where \p max is the max of \p input
/// on dimension \p dim.
static Value buildSubAndExpOp(OpBuilder &builder, Location loc, Value input,
                              Value max, Value output, int64_t dim) {
  auto inputType = cast<ShapedType>(input.getType());
  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t inputRank = inputShape.size();
  auto [iteratorTypes, indexingMaps] = computeIteratorTypesAndIndexingMaps(
      builder, inputRank, dim, /*allParallel=*/true);
  assert(indexingMaps.size() == 2 && "We should have one map for each input");
  assert(indexingMaps[0].isIdentity() && "input map should be identity");
  // Add the affine map for the output argument.
  indexingMaps.push_back(indexingMaps[0]);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, input.getType(), ValueRange{input, max}, output, indexingMaps,
      iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
        Value diff = b.create<arith::SubFOp>(loc, args[0], args[1]);
        Value result = b.create<math::ExpOp>(loc, diff);
        b.create<linalg::YieldOp>(loc, result);
      });
  return genericOp.getResult(0);
}

/// Produce a linalg generic that computes the final step of the softmax
/// decomposition.
/// \returns  linalg.generic ins(\p numerator, \p denominator) outs(\p output) {
///   yield  n / d
/// }
static Value buildDivOp(OpBuilder &builder, Location loc, Value numerator,
                        Value denominator, Value output, int64_t dim) {
  auto inputType = cast<ShapedType>(numerator.getType());
  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t inputRank = inputShape.size();
  auto [iteratorTypes, indexingMaps] = computeIteratorTypesAndIndexingMaps(
      builder, inputRank, dim, /*allParallel=*/true);
  assert(indexingMaps.size() == 2 &&
         "We should have one map for each input (2)");
  assert(indexingMaps[0].isIdentity() && "Numerator map should be identity");
  // Add the affine map for the output tensor.
  indexingMaps.push_back(indexingMaps[0]);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, numerator.getType(), ValueRange{numerator, denominator}, output,
      indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value result = b.create<arith::DivFOp>(loc, args[0], args[1]);
        b.create<linalg::YieldOp>(loc, result);
      });
  return genericOp.getResult(0);
}
// @} End helper functions for softmax decomposition.

/// Given an N-dimensional tensor x, this method converts
/// softmax(x) to the following sequence of operations:
///
/// 1. Compute the max of x along dimension d. This results
///    in a N-1 dimensional tensor m.
///    m = max(x, dim = d)
///
/// 2. Subtract a broadcasted m from x and exponentiate. This results in
///    a N dimensional tensor z.
///    z = exp(x - m)
///
/// 3. Compute the sum of z along dimension d. This results in
///    a N-1 dimensional tensor l.
///    l = sum(z, dim = d)
///
/// 4. Divide z and l. This gives the N-dimensional softmax.
///    softmax = z / l
///
FailureOr<SmallVector<Value>> SoftmaxOp::decomposeOperation(OpBuilder &b) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(*this);
  Location loc = getLoc();
  Value input = getInput();
  ShapedType inputType = getInputOperandType();
  Type elementType = inputType.getElementType();
  int64_t reductionDim = getDimension();
  SmallVector<OpFoldResult> dims = tensor::getMixedSizes(b, loc, input);
  Value output = getOutput();
  dims.erase(dims.begin() + reductionDim);
  // Step 1: Compute max along dim.
  Value outputReduce = b.create<tensor::EmptyOp>(loc, dims, elementType);
  Value neutralForMaxF = arith::getIdentityValue(arith::AtomicRMWKind::maxnumf,
                                                 elementType, b, loc,
                                                 /*useOnlyFiniteValue=*/true);
  Value neutralForMaxFInit =
      b.create<linalg::FillOp>(loc, Value{neutralForMaxF}, outputReduce)
          .result();
  Value max =
      reduce<arith::MaxNumFOp>(b, loc, input, neutralForMaxFInit, reductionDim);

  // Step 2: Subtract max from input and exponentiate.
  Value numerator = buildSubAndExpOp(b, loc, input, max, output, reductionDim);

  // Step 3: Compute sum along dim.
  Value zero = arith::getIdentityValue(arith::AtomicRMWKind::addf, elementType,
                                       b, loc, /*useOnlyFiniteValue=*/true);
  Value zeroInit =
      b.create<linalg::FillOp>(loc, Value{zero}, outputReduce).result();
  Value denominator =
      reduce<arith::AddFOp>(b, loc, numerator, zeroInit, reductionDim);

  // Step 4: Compute softmax.
  Value result =
      buildDivOp(b, loc, numerator, denominator, output, reductionDim);
  return SmallVector<Value>{result};
}

//===----------------------------------------------------------------------===//
// WinogradFilterTransformOp
//===----------------------------------------------------------------------===//

LogicalResult WinogradFilterTransformOp::verify() {
  auto filterType = cast<ShapedType>(getFilter().getType());
  ArrayRef<int64_t> filterShape = filterType.getShape();
  int64_t filterH = filterShape[getFilterHDim()];
  int64_t filterW = filterShape[getFilterWDim()];
  int64_t r = getR();
  int64_t m = getM();

  if (filterH != r && filterH != 1)
    return emitOpError("expect filter height either equals to r or 1");
  if (filterW != r && filterW != 1)
    return emitOpError("expect filter width either equals to r or 1");
  if (filterH == 1 && filterW == 1)
    return emitOpError("expect either filter height or width equals to r");

  SmallVector<int64_t> expectedOutputShape;
  expectedOutputShape.push_back(filterH == r ? m + r - 1 : 1);
  expectedOutputShape.push_back(filterW == r ? m + r - 1 : 1);
  expectedOutputShape.push_back(filterShape[getFilterCDim()]);
  expectedOutputShape.push_back(filterShape[getFilterFDim()]);

  auto outputType = cast<ShapedType>(getOutput().getType());
  ArrayRef<int64_t> outputShape = outputType.getShape();
  if (failed(verifyCompatibleShape(expectedOutputShape, outputShape))) {
    return emitOpError("the output shape is not expected");
  }
  return success();
}

SmallVector<Range>
WinogradFilterTransformOp::getIterationDomain(OpBuilder &builder) {
  Location loc = getLoc();
  IntegerAttr zeroAttr = builder.getIndexAttr(0);
  IntegerAttr oneAttr = builder.getIndexAttr(1);
  Value filter = getFilter();
  int64_t filterRank = getFilterOperandRank();
  SmallVector<Range> loopBounds(filterRank);
  for (unsigned dim = 0; dim < filterRank; ++dim) {
    loopBounds[dim].offset = zeroAttr;
    loopBounds[dim].size = getDimValue(builder, loc, filter, dim);
    loopBounds[dim].stride = oneAttr;
  }
  return loopBounds;
}

SmallVector<utils::IteratorType>
WinogradFilterTransformOp::getLoopIteratorTypes() {
  int64_t filterRank = getFilterOperandRank();
  SmallVector<utils::IteratorType> iteratorTypes(filterRank,
                                                 utils::IteratorType::parallel);
  return iteratorTypes;
}

LogicalResult WinogradFilterTransformOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  IntegerAttr zeroAttr = builder.getI64IntegerAttr(0);
  ShapedType filterType = getFilterOperandType();
  ArrayRef<int64_t> filterShape = filterType.getShape();
  int64_t filterH = filterShape[getFilterHDim()];
  int64_t filterW = filterShape[getFilterWDim()];
  int64_t m = getM();
  int64_t r = getR();
  int64_t alpha = m + r - 1;
  int64_t alphaH = filterH != 1 ? alpha : 1;
  int64_t alphaW = filterW != 1 ? alpha : 1;
  IntegerAttr alphaHAttr = builder.getI64IntegerAttr(alphaH);
  IntegerAttr alphaWAttr = builder.getI64IntegerAttr(alphaW);

  resultOffsets.append(
      {zeroAttr, zeroAttr, offsets[getFilterCDim()], offsets[getFilterFDim()]});
  resultSizes.append(
      {alphaHAttr, alphaWAttr, sizes[getFilterCDim()], sizes[getFilterFDim()]});

  return success();
}

/// Implement tiling for winograd_filter_transform
/// The input of winograd_filter_transform is (F, KH, KW, C).
/// The output of winograd_filter_transform is (alphaH, alphaW, C, F)
/// Users can specify the tile sizes of F and C.
/// `offsets` are the values for the offsets of F, KH, KW, C for one tile.
/// `sizes` are the values for the sizes of F, KH, KW, C for one tile.
FailureOr<TilingResult> WinogradFilterTransformOp::getTiledImplementation(
    OpBuilder &builder, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  IntegerAttr oneAttr = builder.getI64IntegerAttr(1);
  IntegerAttr zeroAttr = builder.getI64IntegerAttr(0);
  ShapedType filterType = getFilterOperandType();
  ArrayRef<int64_t> filterShape = filterType.getShape();
  int64_t filterH = filterShape[getFilterHDim()];
  int64_t filterW = filterShape[getFilterWDim()];
  IntegerAttr filterHAttr = builder.getI64IntegerAttr(filterH);
  IntegerAttr filterWAttr = builder.getI64IntegerAttr(filterW);
  SmallVector<Value> tiledOperands;
  SmallVector<OpFoldResult> sliceOffsets, sliceSizes;

  sliceOffsets.append(
      {offsets[getFilterFDim()], zeroAttr, zeroAttr, offsets[getFilterCDim()]});
  sliceSizes.append({sizes[getFilterFDim()], filterHAttr, filterWAttr,
                     sizes[getFilterCDim()]});
  int64_t filterRank = getFilterOperandRank();
  SmallVector<OpFoldResult> filterStrides(filterRank, oneAttr);
  Location loc = getLoc();
  auto filterSlice = builder.create<tensor::ExtractSliceOp>(
      loc, getFilter(), sliceOffsets, sliceSizes, filterStrides);
  tiledOperands.emplace_back(filterSlice);

  SmallVector<OpFoldResult> resultOffsets, resultSizes;
  if (failed(getResultTilePosition(builder, 1, offsets, sizes, resultOffsets,
                                   resultSizes)))
    return failure();

  int64_t outputRank = getOutputOperandRank();
  SmallVector<OpFoldResult> outputStrides(outputRank, oneAttr);
  auto outputSlice = builder.create<tensor::ExtractSliceOp>(
      loc, getOutput(), resultOffsets, resultSizes, outputStrides);
  tiledOperands.emplace_back(outputSlice);

  SmallVector<Type> resultTypes;
  resultTypes.push_back(tiledOperands[1].getType());
  Operation *tiledOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);

  return TilingResult{
      {tiledOp},
      SmallVector<Value>(tiledOp->getResults()),
      llvm::to_vector(ArrayRef<Operation *>{filterSlice, outputSlice})};
}

//===----------------------------------------------------------------------===//
// WinogradInputTransformOp
//===----------------------------------------------------------------------===//

LogicalResult WinogradInputTransformOp::verify() {
  auto inputType = cast<ShapedType>(getInput().getType());
  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t inputH = inputShape[getInputHDim()];
  int64_t inputW = inputShape[getInputWDim()];
  int m = getM();
  int r = getR();
  int64_t tileSize = m + r - 1;

  auto outputType = cast<ShapedType>(getOutput().getType());
  ArrayRef<int64_t> outputShape = outputType.getShape();
  bool leftTransform = outputShape[getOutputAlphaHDim()] != 1;
  bool rightTransform = outputShape[getOutputAlphaWDim()] != 1;

  SmallVector<int64_t> expectedOutputShape(6, inputH);
  if (ShapedType::isDynamic(inputH)) {
    expectedOutputShape[getOutputAlphaHDim()] = tileSize;
    expectedOutputShape[getOutputTileHDim()] = ShapedType::kDynamic;
  } else {
    expectedOutputShape[getOutputAlphaHDim()] = leftTransform ? tileSize : 1;
    expectedOutputShape[getOutputTileHDim()] =
        leftTransform ? (inputH - (r - 1)) / m : inputH;
  }
  if (ShapedType::isDynamic(inputW)) {
    expectedOutputShape[getOutputAlphaWDim()] = tileSize;
    expectedOutputShape[getOutputTileWDim()] = ShapedType::kDynamic;
  } else {
    expectedOutputShape[getOutputAlphaWDim()] = rightTransform ? tileSize : 1;
    expectedOutputShape[getOutputTileWDim()] =
        rightTransform ? (inputW - (r - 1)) / m : inputW;
  }
  expectedOutputShape[getOutputNDim()] = inputShape[getInputNDim()];
  expectedOutputShape[getOutputCDim()] = inputShape[getInputCDim()];

  if (failed(verifyCompatibleShape(expectedOutputShape, outputShape))) {
    return emitOpError("the output shape is not expected");
  }
  return success();
}

SmallVector<Range>
WinogradInputTransformOp::getIterationDomain(OpBuilder &builder) {
  Location loc = getLoc();
  IntegerAttr zeroAttr = builder.getIndexAttr(0);
  IntegerAttr oneAttr = builder.getIndexAttr(1);
  Value output = getOutput();
  int64_t outputRank = getOutputOperandRank();
  SmallVector<Range> loopBounds(outputRank);
  for (unsigned dim = 0; dim < outputRank; ++dim) {
    loopBounds[dim].offset = zeroAttr;
    // alphaH, alphaW, tileH, tileW, N, C
    loopBounds[dim].size = getDimValue(builder, loc, output, dim);
    loopBounds[dim].stride = oneAttr;
  }
  return loopBounds;
}

SmallVector<utils::IteratorType>
WinogradInputTransformOp::getLoopIteratorTypes() {
  int64_t outputRank = getOutputOperandRank();
  SmallVector<utils::IteratorType> iteratorTypes(outputRank,
                                                 utils::IteratorType::parallel);
  return iteratorTypes;
}

LogicalResult WinogradInputTransformOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  IntegerAttr zeroAttr = builder.getI64IntegerAttr(0);
  ShapedType outputType = getOutputOperandType();
  ArrayRef<int64_t> outputShape = outputType.getShape();
  int64_t outputAlphaH = outputShape[getOutputAlphaHDim()];
  int64_t outputAlphaW = outputShape[getOutputAlphaWDim()];

  int64_t m = getM();
  int64_t r = getR();
  int64_t alpha = m + r - 1;
  int64_t alphaH = outputAlphaH != 1 ? alpha : 1;
  int64_t alphaW = outputAlphaW != 1 ? alpha : 1;

  IntegerAttr alphaHAttr = builder.getI64IntegerAttr(alphaH);
  IntegerAttr alphaWAttr = builder.getI64IntegerAttr(alphaW);

  resultOffsets.append({zeroAttr, zeroAttr, offsets[getOutputTileHDim()],
                        offsets[getOutputTileWDim()], offsets[getOutputNDim()],
                        offsets[getOutputCDim()]});
  resultSizes.append({alphaHAttr, alphaWAttr, sizes[getOutputTileHDim()],
                      sizes[getOutputTileWDim()], sizes[getOutputNDim()],
                      sizes[getOutputCDim()]});

  return success();
}

/// Implement tiling for winograd_input_transform
/// The input of winograd_input_transform is (N, H, W, C).
/// The output of winograd_input_transform is (alphaH, alphaW, tileH, tileW, N,
/// C) Users can specify the tile sizes of tileH, tileW, N, and C. `offsets` are
/// the values for the offsets of tileH, tileW, N, C for one tile. `sizes` are
/// the values for the sizes of tileH, tileW, N, C for one tile.
FailureOr<TilingResult>
WinogradInputTransformOp::getTiledImplementation(OpBuilder &builder,
                                                 ArrayRef<OpFoldResult> offsets,
                                                 ArrayRef<OpFoldResult> sizes) {
  IntegerAttr oneAttr = builder.getI64IntegerAttr(1);
  int64_t m = getM();
  int64_t r = getR();

  ShapedType outputType = getOutputOperandType();
  ArrayRef<int64_t> outputShape = outputType.getShape();
  int64_t alphaH = outputShape[getOutputAlphaHDim()];
  int64_t alphaW = outputShape[getOutputAlphaWDim()];

  Location loc = getLoc();
  MLIRContext *context = builder.getContext();
  auto identityAffineMap =
      AffineMap::get(1, 0, {builder.getAffineDimExpr(0)}, context);
  auto offsetAffineMap =
      AffineMap::get(1, 0, {builder.getAffineDimExpr(0) * m}, context);
  Value mappedOffsetH = affine::makeComposedAffineApply(
      builder, loc, (alphaH != 1 ? offsetAffineMap : identityAffineMap),
      offsets[getOutputTileHDim()]);
  Value mappedOffsetW = affine::makeComposedAffineApply(
      builder, loc, (alphaW != 1 ? offsetAffineMap : identityAffineMap),
      offsets[getOutputTileWDim()]);
  auto sizeAffineMap = AffineMap::get(
      1, 0, {builder.getAffineDimExpr(0) * m + (r - 1)}, context);
  Value mappedSizeH = affine::makeComposedAffineApply(
      builder, loc, sizeAffineMap, sizes[getOutputTileHDim()]);
  Value mappedSizeW = affine::makeComposedAffineApply(
      builder, loc, sizeAffineMap, sizes[getOutputTileWDim()]);

  SmallVector<Value> tiledOperands;
  SmallVector<OpFoldResult> sliceOffsets, sliceSizes;

  OpFoldResult offsetH = OpFoldResult(mappedOffsetH);
  OpFoldResult offsetW = OpFoldResult(mappedOffsetW);
  sliceOffsets.append(
      {offsets[getOutputNDim()], offsetH, offsetW, offsets[getOutputCDim()]});
  OpFoldResult sizeH =
      alphaH != 1 ? OpFoldResult(mappedSizeH) : OpFoldResult(oneAttr);
  OpFoldResult sizeW =
      alphaW != 1 ? OpFoldResult(mappedSizeW) : OpFoldResult(oneAttr);
  sliceSizes.append(
      {sizes[getOutputNDim()], sizeH, sizeW, sizes[getOutputCDim()]});
  int64_t inputRank = getInputOperandRank();
  SmallVector<OpFoldResult> inputStrides(inputRank, oneAttr);
  auto inputSlice = builder.create<tensor::ExtractSliceOp>(
      loc, getInput(), sliceOffsets, sliceSizes, inputStrides);
  tiledOperands.emplace_back(inputSlice);

  SmallVector<OpFoldResult> resultOffsets, resultSizes;
  if (failed(getResultTilePosition(builder, 1, offsets, sizes, resultOffsets,
                                   resultSizes)))
    return failure();

  int64_t outputRank = getOutputOperandRank();
  SmallVector<OpFoldResult> outputStrides(outputRank, oneAttr);
  auto outputSlice = builder.create<tensor::ExtractSliceOp>(
      loc, getOutput(), resultOffsets, resultSizes, outputStrides);
  tiledOperands.emplace_back(outputSlice);

  SmallVector<Type> resultTypes;
  resultTypes.push_back(tiledOperands[1].getType());
  Operation *tiledOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);

  return TilingResult{
      {tiledOp},
      SmallVector<Value>(tiledOp->getResults()),
      llvm::to_vector(ArrayRef<Operation *>{inputSlice, outputSlice})};
}

//===----------------------------------------------------------------------===//
// WinogradOutputTransformOp
//===----------------------------------------------------------------------===//

LogicalResult WinogradOutputTransformOp::verify() {
  auto valueType = cast<ShapedType>(getValue().getType());
  ArrayRef<int64_t> valueShape = valueType.getShape();
  int64_t valueH = valueShape[getValueAlphaHDim()];
  int64_t valueW = valueShape[getValueAlphaWDim()];
  int64_t valueTileH = valueShape[getValueTileHDim()];
  int64_t valueTileW = valueShape[getValueTileWDim()];
  int m = getM();
  int r = getR();
  bool leftTransform = valueH != 1;
  bool rightTransform = valueW != 1;

  int64_t outputRank = getOutputOperandRank();
  SmallVector<int64_t> expectedOutputShape(outputRank, valueH);
  if (ShapedType::isDynamic(valueH) || ShapedType::isDynamic(valueTileH)) {
    expectedOutputShape[getOutputHDim()] = ShapedType::kDynamic;
  } else {
    if (valueH != (leftTransform ? m + r - 1 : 1))
      return emitOpError("expect input height equals to input tile size");
    expectedOutputShape[getOutputHDim()] = (leftTransform ? m : 1) * valueTileH;
  }
  if (ShapedType::isDynamic(valueW) || ShapedType::isDynamic(valueTileW)) {
    expectedOutputShape[getOutputWDim()] = ShapedType::kDynamic;
  } else {
    if (valueW != (rightTransform ? m + r - 1 : 1))
      return emitOpError("expect input width equals to input tile size");
    expectedOutputShape[getOutputWDim()] =
        (rightTransform ? m : 1) * valueTileW;
  }
  expectedOutputShape[getOutputNDim()] = valueShape[getValueNDim()];
  expectedOutputShape[getOutputFDim()] = valueShape[getValueFDim()];

  auto outputType = cast<ShapedType>(getOutput().getType());
  ArrayRef<int64_t> outputShape = outputType.getShape();
  if (failed(verifyCompatibleShape(expectedOutputShape, outputShape))) {
    return emitOpError("the output shape is not expected");
  }
  return success();
}

SmallVector<Range>
WinogradOutputTransformOp::getIterationDomain(OpBuilder &builder) {
  Location loc = getLoc();
  IntegerAttr zeroAttr = builder.getIndexAttr(0);
  IntegerAttr oneAttr = builder.getIndexAttr(1);
  Value value = getValue();
  int64_t valueRank = getValueOperandRank();
  SmallVector<Range> loopBounds(valueRank);
  for (unsigned dim = 0; dim < valueRank; ++dim) {
    loopBounds[dim].offset = zeroAttr;
    // alphaH, alphaW, tileH, tileW, N, F
    loopBounds[dim].size = getDimValue(builder, loc, value, dim);
    loopBounds[dim].stride = oneAttr;
  }
  return loopBounds;
}

SmallVector<utils::IteratorType>
WinogradOutputTransformOp::getLoopIteratorTypes() {
  int64_t valueRank = getValueOperandRank();
  SmallVector<utils::IteratorType> iteratorTypes(valueRank,
                                                 utils::IteratorType::parallel);
  return iteratorTypes;
}

LogicalResult WinogradOutputTransformOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  int64_t m = getM();

  Location loc = getLoc();
  MLIRContext *context = builder.getContext();
  auto identityAffineMap =
      AffineMap::get(1, 0, {builder.getAffineDimExpr(0)}, context);
  auto affineMap =
      AffineMap::get(1, 0, {builder.getAffineDimExpr(0) * m}, context);

  ShapedType valueType = getValueOperandType();
  ArrayRef<int64_t> valueShape = valueType.getShape();
  int64_t valueH = valueShape[0];
  int64_t valueW = valueShape[1];
  Value mappedOffsetH = affine::makeComposedAffineApply(
      builder, loc, (valueH != 1 ? affineMap : identityAffineMap),
      offsets[getValueTileHDim()]);
  Value mappedOffsetW = affine::makeComposedAffineApply(
      builder, loc, (valueW != 1 ? affineMap : identityAffineMap),
      offsets[getValueTileWDim()]);
  Value mappedSizeH = affine::makeComposedAffineApply(
      builder, loc, affineMap, sizes[getValueTileHDim()]);
  Value mappedSizeW = affine::makeComposedAffineApply(
      builder, loc, affineMap, sizes[getValueTileWDim()]);

  IntegerAttr oneAttr = builder.getI64IntegerAttr(1);
  OpFoldResult offsetH = OpFoldResult(mappedOffsetH);
  OpFoldResult offsetW = OpFoldResult(mappedOffsetW);
  OpFoldResult sizeH =
      valueH != 1 ? OpFoldResult(mappedSizeH) : OpFoldResult(oneAttr);
  OpFoldResult sizeW =
      valueW != 1 ? OpFoldResult(mappedSizeW) : OpFoldResult(oneAttr);

  resultOffsets.append(
      {offsets[getValueNDim()], offsetH, offsetW, offsets[getValueFDim()]});
  resultSizes.append(
      {sizes[getValueNDim()], sizeH, sizeW, sizes[getValueFDim()]});
  return success();
}

/// Implement tiling for winograd_output_transform
/// The input of winograd_output_transform is (alphaH, alphaW, tileH, tileW, N,
/// F). The output of winograd_output_transform is (N, H, W, F) Users can
/// specify the tile sizes of tileH, tileW, N, and F. `offsets` are the values
/// for the offsets of tileH, tileW, N, F for one tile. `sizes` are the values
/// for the sizes of tileH, tileW, N, F for one tile.
FailureOr<TilingResult> WinogradOutputTransformOp::getTiledImplementation(
    OpBuilder &builder, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  IntegerAttr oneAttr = builder.getI64IntegerAttr(1);
  IntegerAttr zeroAttr = builder.getI64IntegerAttr(0);
  Location loc = getLoc();
  SmallVector<Value> tiledOperands;
  SmallVector<OpFoldResult> sliceOffsets, sliceSizes;

  ShapedType valueType = getValueOperandType();
  ArrayRef<int64_t> valueShape = valueType.getShape();
  int64_t alphaH = valueShape[getValueAlphaHDim()];
  int64_t alphaW = valueShape[getValueAlphaWDim()];
  IntegerAttr alphaHAttr = builder.getI64IntegerAttr(alphaH);
  IntegerAttr alphaWAttr = builder.getI64IntegerAttr(alphaW);

  sliceOffsets.append({zeroAttr, zeroAttr, offsets[getValueTileHDim()],
                       offsets[getValueTileWDim()], offsets[getValueNDim()],
                       offsets[getValueFDim()]});
  sliceSizes.append({alphaHAttr, alphaWAttr, sizes[getValueTileHDim()],
                     sizes[getValueTileWDim()], sizes[getValueNDim()],
                     sizes[getValueFDim()]});
  int64_t valueRank = getValueOperandRank();
  SmallVector<OpFoldResult> sliceStrides(valueRank, oneAttr);
  auto valueSlice = builder.create<tensor::ExtractSliceOp>(
      loc, getValue(), sliceOffsets, sliceSizes, sliceStrides);
  tiledOperands.emplace_back(valueSlice);

  SmallVector<OpFoldResult> resultOffsets, resultSizes;
  if (failed(getResultTilePosition(builder, 1, offsets, sizes, resultOffsets,
                                   resultSizes)))
    return failure();

  int64_t outputRank = getOutputOperandRank();
  SmallVector<OpFoldResult> strides(outputRank, oneAttr);
  auto outputSlice = builder.create<tensor::ExtractSliceOp>(
      loc, getOutput(), resultOffsets, resultSizes, strides);
  tiledOperands.emplace_back(outputSlice);

  SmallVector<Type> resultTypes;
  resultTypes.push_back(tiledOperands[1].getType());
  Operation *tiledOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);

  return TilingResult{
      {tiledOp},
      SmallVector<Value>(tiledOp->getResults()),
      llvm::to_vector(ArrayRef<Operation *>{valueSlice, outputSlice})};
}

//===----------------------------------------------------------------------===//
// LinalgDialect
//===----------------------------------------------------------------------===//

void LinalgDialect::getCanonicalizationPatterns(
    RewritePatternSet &results) const {
  results.add<EraseDeadLinalgOp, FoldTensorCastConsumerOp,
              InferStaticShapeOfOperands>(getContext());
}

Operation *LinalgDialect::materializeConstant(OpBuilder &builder,
                                              Attribute value, Type type,
                                              Location loc) {
  return arith::ConstantOp::materialize(builder, value, type, loc);
}

/// Returns true if the result AffineExpr of the \p explicitMap is same as \p
/// defaultMap.
static bool isValidResultDimExprs(AffineMap explictMap, AffineMap defaultMap) {
  auto explicitRange = explictMap.getResults();
  auto defaultRange = defaultMap.getResults();
  DenseSet<AffineExpr> explicitSet(explicitRange.begin(), explicitRange.end());
  DenseSet<AffineExpr> defaultSet(defaultRange.begin(), defaultRange.end());
  llvm::set_union(explicitSet, defaultSet);
  return explicitSet == defaultSet;
}

/// Returns true if the \p explictMap is broadcasted with respect to the
/// \p defaultMap.
static bool isBroadcasted(AffineMap explictMap, AffineMap defaultMap) {
  return explictMap.getNumResults() < defaultMap.getNumResults();
}

/// Verifies the broadcast and transpose semantic sepecified by the explicit
/// indexing map for the MatmulOp \p op for each operand specified by \p
/// opIndex.
static LogicalResult verifyExtendedMatmulSemantic(MatmulOp matmulOp,
                                                  unsigned opIndex) {
  SmallVector<AffineMap, 3> opIndexingMaps = matmulOp.getIndexingMapsArray();
  SmallVector<AffineMap, 3> defaultIndexingMaps =
      matmulOp.getDefaultIndexingMaps(matmulOp->getContext());

  auto opIndexingMap = opIndexingMaps[opIndex];
  auto defaultIndexingMap = defaultIndexingMaps[opIndex];
  // Check general validity of indexing map results.
  if (!isValidResultDimExprs(opIndexingMap, defaultIndexingMap))
    return matmulOp->emitOpError()
           << "Unexpected dim expression in map result.";

  // Check if the requested broadcast is valid.
  if (isBroadcasted(opIndexingMap, defaultIndexingMap)) {
    if (!matmulOp.isValidLhsRhsBroadcastMap(opIndexingMap)) {
      return matmulOp->emitOpError()
             << "Invalid broadcast requested, should be (d2).";
    }
    return success();
  }
  return success();
}

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// MatMulOp
//===----------------------------------------------------------------------===//

/// Returns a list of AffineMap with the typical matmul indexing charactristic.
SmallVector<AffineMap> MatmulOp::getDefaultIndexingMaps(MLIRContext *context) {
  AffineExpr d0, d1, d2;
  SmallVector<AffineMap> indexingMaps;
  bindDims(context, d0, d1, d2);
  indexingMaps.push_back(AffineMap::get(3, 0, {d0, d2}, context));
  indexingMaps.push_back(AffineMap::get(3, 0, {d2, d1}, context));
  indexingMaps.push_back(AffineMap::get(3, 0, {d0, d1}, context));
  return indexingMaps;
}

SmallVector<utils::IteratorType> MatmulOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{utils::IteratorType::parallel,
                                          utils::IteratorType::parallel,
                                          utils::IteratorType::reduction};
}

unsigned MatmulOp::getNumRegionArgs() { return 3; }

std::string MatmulOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool MatmulOp::hasDynamicIndexingMaps() { return true; }

/// Check if the op has broadcast and/or transpose semantic. Returns true if
/// the user defined indexing maps are not equal to default map.
bool MatmulOp::hasUserDefinedMaps() {
  SmallVector<AffineMap, 3> defaultMaps =
      getDefaultIndexingMaps(this->getContext());
  SmallVector<AffineMap, 3> explicitMaps = getIndexingMapsArray();
  return defaultMaps != explicitMaps;
}

/// Implements the block region builder for the MatmulOp. This is called by
/// 'fillStructuredOpRegion'.
void MatmulOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block,
                             ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "MatmulOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(b, block);
  SmallVector<Value> yields;

  TypeFn castVal = TypeFn::cast_signed;
  auto castIter = llvm::find_if(attrs, [&](const NamedAttribute &attr) {
    return attr.getName() == "cast";
  });
  if (castIter != attrs.end()) {
    if (auto attr = llvm::dyn_cast<TypeFnAttr>(castIter->getValue()))
      castVal = attr.getValue();
  }

  Value value1 = helper.buildTypeFn(castVal, block.getArgument(2).getType(),
                                    block.getArgument(0));
  Value value2 = helper.buildTypeFn(castVal, block.getArgument(2).getType(),
                                    block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 =
      helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

/// Returns true if the given broadcast map \p bcastMap is valid for this op.
bool MatmulOp::isValidLhsRhsBroadcastMap(AffineMap bcastMap) {
  assert(bcastMap.getNumResults() == 1 && "Expected single result dim expr.");
  AffineExpr exp = bcastMap.getResult(0);
  // Invalid map if the common dimension of matmul not found.
  return exp.isFunctionOfDim(bcastMap.getNumDims() - 1);
}

FailureOr<ArrayAttr> parseIndexingMapsAttr(OpAsmParser &parser) {
  if (parser.parseOptionalKeyword("indexing_maps"))
    return ArrayAttr{
        nullptr}; // Success in case indexing_maps was not provided.

  ArrayAttr arrayAttr;
  if (parser.parseEqual() || parser.parseAttribute(arrayAttr))
    return failure();

  if (llvm::any_of(arrayAttr,
                   [](auto elt) { return !dyn_cast<AffineMapAttr>(elt); }))
    return parser.emitError(parser.getCurrentLocation())
           << "element of indexing_maps array is not an affine_map";

  return arrayAttr;
}

ParseResult MatmulOp::parse(OpAsmParser &parser, OperationState &result) {
  FailureOr<ArrayAttr> indexingMapsAttr = parseIndexingMapsAttr(parser);
  if (failed(indexingMapsAttr))
    return failure();

  if (*indexingMapsAttr == nullptr) {
    auto indexingMapAttrs = llvm::map_to_vector(
        MatmulOp::getDefaultIndexingMaps(parser.getContext()),
        [](AffineMap map) -> Attribute { return AffineMapAttr::get(map); });
    indexingMapsAttr = parser.getBuilder().getArrayAttr(indexingMapAttrs);
  }

  result.addAttribute("indexing_maps", *indexingMapsAttr);
  return parseNamedStructuredOp(parser, result, MatmulOp::getNumRegionArgs(),
                                MatmulOp::getRegionBuilder());
}

void MatmulOp::print(OpAsmPrinter &p) {
  SmallVector<StringRef, 3> elidedAttrs = {
      "operandSegmentSizes", "linalg.memoized_indexing_maps", "indexing_maps"};
  printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs(),
                         elidedAttrs);

  SmallVector<Attribute, 3> indexingMaps = llvm::map_to_vector(
      MatmulOp::getDefaultIndexingMaps(getContext()),
      [](AffineMap map) -> Attribute { return AffineMapAttr::get(map); });
  if (!llvm::equal(getIndexingMaps(), indexingMaps)) {
    p << " indexing_maps = [";
    llvm::interleaveComma(getIndexingMaps(), p,
                          [&](Attribute attr) { p.printAttribute(attr); });
    p << "]";
  }
}

/// Verify the user defined indexing maps.
LogicalResult MatmulOp::verify() {
  // Verification of pure matmul is handled by verifyStructuredOpInterface().
  if (!hasUserDefinedMaps())
    return success();

  for (unsigned opIndex = 0; opIndex < 2; opIndex++) {
    if (failed(verifyExtendedMatmulSemantic(*this, opIndex)))
      return failure();
  }
  return success();
}

LogicalResult MatmulOp::fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

void MatmulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (hasPureTensorSemantics())
    return;
  getGenericEffectsImpl(effects, cast<LinalgOp>(getOperation()));
}

Speculation::Speculatability MatmulOp::getSpeculatability() {
  return getGenericSpeculatabilityImpl(cast<LinalgOp>(getOperation()));
}

//===----------------------------------------------------------------------===//
// ContractOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> ContractOp::getIteratorTypesArray() {
  AffineMap outAffineMap = getIndexingMapsArray().pop_back_val();
  // On well-formed IR, indexing_maps is non-empty, contained affine_maps'
  // domains are all the same, and each implements a projected permutation.
  // Each iteration space dim must occur for at least one operand and either
  // takes part in a contraction/reduction or else has parallel iteration type.
  // We have that a dim is a contraction/reduction dim if and only if the dim
  // occurs for the output operand. We use this fact for fast inference:
  // NB: In case we allow dims to occur solely for one input, the above still
  //     holds: per the einsum semantics, these are reduction dims as well.
  SmallVector<bool> dimsInOutput(outAffineMap.getNumDims(), false);
  for (auto result : outAffineMap.getResults()) {
    auto dimExpr = dyn_cast<AffineDimExpr>(result);
    assert(dimExpr && "affine_map is a projected permutation");
    dimsInOutput[dimExpr.getPosition()] = true;
  }

  SmallVector<utils::IteratorType> iteratorTypes;
  for (auto dimOccursInOutput : dimsInOutput)
    iteratorTypes.push_back(dimOccursInOutput ? utils::IteratorType::parallel
                                              : utils::IteratorType::reduction);

  return iteratorTypes;
}

unsigned ContractOp::getNumRegionArgs() { return 3; }

/// Implement block region builder, which is called by 'fillStructuredOpRegion'.
void ContractOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block,
                               ArrayRef<NamedAttribute> attrs) {
  assert(block.getNumArguments() == 3 &&
         "ContractOp regionBuilder expects 3 args");
  RegionBuilderHelper helper(b, block);

  TypeFn castSignedness = TypeFn::cast_signed;
  auto castIter = llvm::find_if(attrs, [&](const NamedAttribute &attr) {
    return attr.getName() == "cast";
  });
  if (castIter != attrs.end()) {
    if (auto attr = llvm::dyn_cast<TypeFnAttr>(castIter->getValue()))
      castSignedness = attr.getValue();
  }

  // TODO: Support fields with operators besides mult & add.
  Type outType = block.getArgument(2).getType();
  Value lhsAtOutType =
      helper.buildTypeFn(castSignedness, outType, block.getArgument(0));
  Value rhsAtOutType =
      helper.buildTypeFn(castSignedness, outType, block.getArgument(1));
  Value productAtOutType =
      helper.buildBinaryFn(BinaryFn::mul, lhsAtOutType, rhsAtOutType);
  Value result = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2),
                                      productAtOutType);
  helper.yieldOutputs({result});
}

ParseResult ContractOp::parse(OpAsmParser &parser, OperationState &result) {
  FailureOr<ArrayAttr> indexingMapsAttr = parseIndexingMapsAttr(parser);
  if (failed(indexingMapsAttr) || *indexingMapsAttr == nullptr)
    return parser.emitError(parser.getCurrentLocation(),
                            "expected 'indexing_maps' attribute");
  result.addAttribute("indexing_maps", *indexingMapsAttr);

  return parseNamedStructuredOp(parser, result, getNumRegionArgs(),
                                regionBuilder);
}

void ContractOp::print(OpAsmPrinter &p) {
  p << " indexing_maps = [";
  llvm::interleaveComma(getIndexingMaps(), p,
                        [&](Attribute attr) { p.printAttribute(attr); });
  p << "]";
  printNamedStructuredOp(
      p, getOperation(), getInputs(), getOutputs(),
      /*elidedAttrs=*/{"indexing_maps", "operandSegmentSizes"});
}

LogicalResult ContractOp::verify() {
  int iterationSpaceDims = -1;
  // Map iter space dims to #occurrences in inputs' and output's affine_maps:
  // e.g., inOccurrences[0] will hold #times that dim (with index) 0 is used to
  // access an input operand (so occurrence count can be at most 2) and
  // outOccurrences[1] will indicate whether dim 1 occurred in the output, etc.
  SmallVector<size_t> inOccurrences;
  SmallVector<size_t> outOccurrences;

  // A helper so that for each operand's affine_map and type we check that ...
  auto checkAffineMapAndType = [&](AffineMap affineMap, Type operandType,
                                   bool isInput) -> LogicalResult {
    // ... the affine_map is a projected permutation;
    if (!affineMap.isProjectedPermutation())
      return emitError("provided affine_map is not a projected permutation");

    // ... the rank of the affine_map's results and corresponding type match;
    if (auto shapedType = dyn_cast<ShapedType>(operandType)) {
      if (affineMap.getNumResults() != shapedType.getRank())
        return emitError("ranks of shaped operand and results of corresponding "
                         "affine_map differ");
    } else if (affineMap.getNumResults() != 0) {
      return emitError("affine_map specifies shaped access while operand has "
                       "non-shaped type");
    }

    // ... the rank of the affine_map's domain is the same as those seen prior;
    if (iterationSpaceDims == -1) {
      iterationSpaceDims = affineMap.getNumDims();
      inOccurrences = SmallVector<size_t>(iterationSpaceDims, 0);
      outOccurrences = SmallVector<size_t>(iterationSpaceDims, 0);
    } else if (iterationSpaceDims != (int)affineMap.getNumDims()) {
      return emitError("iteration spaces of provided affine_maps differ");
    }

    // ... update counts of dims used to access either an input or the output.
    for (AffineExpr affineExpr : affineMap.getResults()) {
      auto affineDimExpr = dyn_cast<AffineDimExpr>(affineExpr);
      if (!affineDimExpr)
        llvm_unreachable("affine_map is a projected permutation");

      if (isInput)
        inOccurrences[affineDimExpr.getPosition()] += 1;
      else
        outOccurrences[affineDimExpr.getPosition()] += 1;
    }

    return success();
  };

  for (auto &&[affineMap, operandType, isInput] :
       llvm::zip(getIndexingMapsArray(), getOperandTypes(),
                 SmallVector<bool>{true, true, false})) {
    if (failed(checkAffineMapAndType(affineMap, operandType, isInput)))
      return failure(); // NB: checkAffineMapAndType will emit relevant error.
  }

  bool hasContractingDim = false;
  for (size_t dimIndex = 0; dimIndex < (size_t)iterationSpaceDims; dimIndex++) {
    size_t inOccCount = inOccurrences[dimIndex];
    size_t outOccCount = outOccurrences[dimIndex];

    // We have a contracting dim if and only if ...
    hasContractingDim |= inOccCount == 2 && outOccCount == 0;

    if (inOccCount == 0 && outOccCount == 0)
      return emitError() << "iteration space dim at index " << dimIndex
                         << " not used to access any operand";

    // NB: We disallow a dim which occurs for only one input operand and not
    //     for the output. In terms of einsum semantics such dims have a
    //     sensible meaning - namely an additional reduction per each such dim.
    //     By contrast, the ContractionOpInterface does not know about this
    //     iter type - cf. inferContractionDims' supported dim kinds. Similarly,
    //     while vector.contract's verifier accepts dims of this kind many of
    //     its lowerings give up on encountering these dims.
    // TODO: Remove following once we have comprehensive support for input-only
    //       reduction dims, at both the linalg- and vector-dialect levels.
    if (inOccCount == 1 && outOccCount != 1)
      return emitError()
             << "iteration space dim at index " << dimIndex
             << " is neither a contracting dim nor of parallel iteration type";
  }

  if (!hasContractingDim)
    return emitError("'indexing_maps' do not specify a contracting dimension");

  return success();
}

LogicalResult ContractOp::fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

void ContractOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (hasPureTensorSemantics())
    return;
  getGenericEffectsImpl(effects, cast<LinalgOp>(getOperation()));
}

Speculation::Speculatability ContractOp::getSpeculatability() {
  return getGenericSpeculatabilityImpl(cast<LinalgOp>(getOperation()));
}

} // namespace linalg
} // namespace mlir
