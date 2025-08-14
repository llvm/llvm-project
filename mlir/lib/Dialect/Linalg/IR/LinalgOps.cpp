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
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InterleavedRange.h"
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
            return tensor::DimOp::create(builder, loc, v, dim);
          })
          .Case<MemRefType>([&](MemRefType t) -> Value {
            return memref::DimOp::create(builder, loc, v, dim);
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
        return tensor::ExtractSliceOp::create(b, loc, source, offsets, sizes,
                                              strides);
      })
      .Case<MemRefType>([&](MemRefType type) -> Operation * {
        return memref::SubViewOp::create(b, loc, source, offsets, sizes,
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

using RegionBuilderFn = llvm::function_ref<void(
    ImplicitLocOpBuilder &, Block &, ArrayRef<NamedAttribute>,
    function_ref<InFlightDiagnostic()>)>;

/// Fills the region of a structured operation using the provided
/// `regionBuilder`. The method is used by both named structured ops created by
/// ods-gen and by manually defined C++ ops. It is called by both builders and
/// parsers and creates a block with arguments corresponding to the elemental
/// types of `inputTypes` and `outputTypes`.
static void fillStructuredOpRegion(OpBuilder &opBuilder, Region &region,
                                   TypeRange inputTypes, TypeRange outputTypes,
                                   ArrayRef<NamedAttribute> attrs,
                                   function_ref<InFlightDiagnostic()> emitError,
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
  regionBuilder(b, *body, attrs, emitError);

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
                         state.attributes.getAttrs(), /*emitError=*/{},
                         regionBuilder);
}

static void buildMatmulOp(OpBuilder &b, OperationState &state,
                          std::optional<TypeRange> resultTensorTypes,
                          ValueRange inputs, ValueRange outputs,
                          ArrayRef<NamedAttribute> attributes,
                          RegionBuilderFn regionBuilder,
                          ArrayRef<AffineMap> indexingMaps) {
  // Initialize indexingMaps attribute, for MatmulOp.
  SmallVector<Attribute, 3> indexingMapsAttrVal;
  indexingMapsAttrVal =
      llvm::map_to_vector(indexingMaps, [](AffineMap map) -> Attribute {
        return AffineMapAttr::get(map);
      });
  state.addAttribute("indexing_maps", b.getArrayAttr(indexingMapsAttrVal));
  return buildStructuredOp(b, state, resultTensorTypes, inputs, outputs,
                           attributes, regionBuilder);
}

static void buildBatchMatmulOp(OpBuilder &b, OperationState &state,
                               std::optional<TypeRange> resultTensorTypes,
                               ValueRange inputs, ValueRange outputs,
                               ArrayRef<NamedAttribute> attributes,
                               RegionBuilderFn regionBuilder,
                               ArrayRef<AffineMap> indexingMaps) {
  // Initialize indexingMaps attribute, for BatchMatmulOp.
  SmallVector<Attribute, 4> indexingMapsAttrVal;
  indexingMapsAttrVal =
      llvm::map_to_vector(indexingMaps, [](AffineMap map) -> Attribute {
        return AffineMapAttr::get(map);
      });
  state.addAttribute("indexing_maps", b.getArrayAttr(indexingMapsAttrVal));
  return buildStructuredOp(b, state, resultTensorTypes, inputs, outputs,
                           attributes, regionBuilder);
}

static void buildBatchReduceMatmulOp(OpBuilder &b, OperationState &state,
                                     std::optional<TypeRange> resultTensorTypes,
                                     ValueRange inputs, ValueRange outputs,
                                     ArrayRef<NamedAttribute> attributes,
                                     RegionBuilderFn regionBuilder,
                                     ArrayRef<AffineMap> indexingMaps) {
  // Initialize indexingMaps attribute, for BatchReduceMatmulOp.
  SmallVector<Attribute, 4> indexingMapsAttrVal;
  indexingMapsAttrVal =
      llvm::map_to_vector(indexingMaps, [](AffineMap map) -> Attribute {
        return AffineMapAttr::get(map);
      });
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
    RegionBuilderFn regionBuilder, SMLoc loc) {
  if (numRegionArgs != inputTypes.size() + outputTypes.size()) {
    return parser.emitError(
        parser.getCurrentLocation(),
        llvm::formatv("[parseNamedStructuredOpRegion] ods-gen generated "
                      "region expects {0} args, got {1}",
                      numRegionArgs, inputTypes.size() + outputTypes.size()));
  }

  OpBuilder opBuilder(parser.getContext());
  ParseResult result = success();
  fillStructuredOpRegion(
      opBuilder, region, inputTypes, outputTypes, attrs,
      [&]() {
        result = failure();
        return parser.emitError(loc);
      },
      regionBuilder);
  return result;
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
  SMLoc loc = parser.getCurrentLocation();
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
                                   regionBuilder, loc))
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
  Value buildUnaryFn(UnaryFn unaryFn, Value arg,
                     function_ref<InFlightDiagnostic()> emitError = {}) {
    if (!isFloatingPoint(arg)) {
      if (emitError) {
        emitError() << "unsupported non numeric type";
        return nullptr;
      }
      llvm_unreachable("unsupported non numeric type");
    }
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToEnd(&block);
    switch (unaryFn) {
    case UnaryFn::exp:
      return math::ExpOp::create(builder, arg.getLoc(), arg);
    case UnaryFn::log:
      return math::LogOp::create(builder, arg.getLoc(), arg);
    case UnaryFn::abs:
      return math::AbsFOp::create(builder, arg.getLoc(), arg);
    case UnaryFn::ceil:
      return math::CeilOp::create(builder, arg.getLoc(), arg);
    case UnaryFn::floor:
      return math::FloorOp::create(builder, arg.getLoc(), arg);
    case UnaryFn::negf:
      return arith::NegFOp::create(builder, arg.getLoc(), arg);
    case UnaryFn::reciprocal: {
      Attribute oneAttr = builder.getOneAttr(arg.getType());
      auto one = arith::ConstantOp::create(builder, arg.getLoc(),
                                           ::cast<TypedAttr>(oneAttr));
      return arith::DivFOp::create(builder, arg.getLoc(), one, arg);
    }
    case UnaryFn::round:
      return math::RoundOp::create(builder, arg.getLoc(), arg);
    case UnaryFn::sqrt:
      return math::SqrtOp::create(builder, arg.getLoc(), arg);
    case UnaryFn::rsqrt:
      return math::RsqrtOp::create(builder, arg.getLoc(), arg);
    case UnaryFn::square:
      return arith::MulFOp::create(builder, arg.getLoc(), arg, arg);
    case UnaryFn::tanh:
      return math::TanhOp::create(builder, arg.getLoc(), arg);
    case UnaryFn::erf:
      return math::ErfOp::create(builder, arg.getLoc(), arg);
    }
    if (emitError) {
      emitError() << "unsupported unary function";
      return nullptr;
    }
    llvm_unreachable("unsupported unary function");
  }

  // Build the binary functions defined by OpDSL.
  // If emitError is provided, an error will be emitted if the operation is not
  // supported and a nullptr will be returned, otherwise an assertion will be
  // raised.
  Value buildBinaryFn(BinaryFn binaryFn, Value arg0, Value arg1,
                      function_ref<InFlightDiagnostic()> emitError = {}) {
    bool allComplex = isComplex(arg0) && isComplex(arg1);
    bool allFloatingPoint = isFloatingPoint(arg0) && isFloatingPoint(arg1);
    bool allInteger = isInteger(arg0) && isInteger(arg1);
    bool allBool = allInteger && arg0.getType().getIntOrFloatBitWidth() == 1 &&
                   arg1.getType().getIntOrFloatBitWidth() == 1;
    if (!allComplex && !allFloatingPoint && !allInteger) {
      if (emitError) {
        emitError()
            << "Cannot build binary Linalg operation: expects allComplex, "
               "allFloatingPoint, or allInteger, got "
            << arg0.getType() << " and " << arg1.getType();
        return nullptr;
      }
      llvm_unreachable("unsupported non numeric type");
    }
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToEnd(&block);
    switch (binaryFn) {
    case BinaryFn::add:
      if (allComplex)
        return complex::AddOp::create(builder, arg0.getLoc(), arg0, arg1);
      if (allFloatingPoint)
        return arith::AddFOp::create(builder, arg0.getLoc(), arg0, arg1);
      if (allBool)
        return arith::OrIOp::create(builder, arg0.getLoc(), arg0, arg1);
      return arith::AddIOp::create(builder, arg0.getLoc(), arg0, arg1);
    case BinaryFn::sub:
      if (allComplex)
        return complex::SubOp::create(builder, arg0.getLoc(), arg0, arg1);
      if (allFloatingPoint)
        return arith::SubFOp::create(builder, arg0.getLoc(), arg0, arg1);
      if (allBool) {
        if (emitError) {
          emitError() << "unsupported operation: sub with bools";
          return nullptr;
        }
        llvm_unreachable("unsupported operation: sub with bools");
      }
      return arith::SubIOp::create(builder, arg0.getLoc(), arg0, arg1);
    case BinaryFn::mul:
      if (allComplex)
        return complex::MulOp::create(builder, arg0.getLoc(), arg0, arg1);
      if (allFloatingPoint)
        return arith::MulFOp::create(builder, arg0.getLoc(), arg0, arg1);
      if (allBool)
        return arith::AndIOp::create(builder, arg0.getLoc(), arg0, arg1);
      return arith::MulIOp::create(builder, arg0.getLoc(), arg0, arg1);
    case BinaryFn::div:
      if (allComplex)
        return complex::DivOp::create(builder, arg0.getLoc(), arg0, arg1);
      if (allFloatingPoint)
        return arith::DivFOp::create(builder, arg0.getLoc(), arg0, arg1);
      if (allBool) {
        if (emitError) {
          emitError() << "unsupported operation: div with bools";
          return nullptr;
        }
        llvm_unreachable("unsupported operation: div with bools");
      }
      return arith::DivSIOp::create(builder, arg0.getLoc(), arg0, arg1);
    case BinaryFn::div_unsigned:
      if (!allInteger || allBool) {
        if (emitError) {
          emitError() << "unsupported operation: unsigned div not on uint";
          return nullptr;
        }
        llvm_unreachable("unsupported operation: unsigned div not on uint");
      }
      return arith::DivUIOp::create(builder, arg0.getLoc(), arg0, arg1);
    case BinaryFn::max_signed:
      assert(!allComplex);
      if (allFloatingPoint)
        return arith::MaximumFOp::create(builder, arg0.getLoc(), arg0, arg1);
      return arith::MaxSIOp::create(builder, arg0.getLoc(), arg0, arg1);
    case BinaryFn::min_signed:
      assert(!allComplex);
      if (allFloatingPoint)
        return arith::MinimumFOp::create(builder, arg0.getLoc(), arg0, arg1);
      return arith::MinSIOp::create(builder, arg0.getLoc(), arg0, arg1);
    case BinaryFn::max_unsigned:
      assert(!allComplex);
      if (allFloatingPoint)
        return arith::MaximumFOp::create(builder, arg0.getLoc(), arg0, arg1);
      return arith::MaxUIOp::create(builder, arg0.getLoc(), arg0, arg1);
    case BinaryFn::min_unsigned:
      assert(!allComplex);
      if (allFloatingPoint)
        return arith::MinimumFOp::create(builder, arg0.getLoc(), arg0, arg1);
      return arith::MinUIOp::create(builder, arg0.getLoc(), arg0, arg1);
    case BinaryFn::powf:
      assert(allFloatingPoint);
      return math::PowFOp::create(builder, arg0.getLoc(), arg0, arg1);
    }
    if (emitError) {
      emitError() << "unsupported binary function";
      return nullptr;
    }
    llvm_unreachable("unsupported binary function");
  }

  // Build the ternary functions defined by OpDSL.
  Value buildTernaryFn(TernaryFn ternaryFn, Value arg0, Value arg1, Value arg2,
                       function_ref<InFlightDiagnostic()> emitError = {}) {
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
      return arith::SelectOp::create(builder, arg0.getLoc(), arg0, arg1, arg2);
    }
    if (emitError) {
      emitError() << "unsupported ternary function";
      return nullptr;
    }
    llvm_unreachable("unsupported ternary function");
  }

  // Build the type functions defined by OpDSL.
  Value buildTypeFn(TypeFn typeFn, Type toType, Value operand,
                    function_ref<InFlightDiagnostic()> emitError = {}) {
    switch (typeFn) {
    case TypeFn::cast_signed:
      return cast(toType, operand, false);
    case TypeFn::cast_unsigned:
      return cast(toType, operand, true);
    }
    if (emitError) {
      emitError() << "unsupported type conversion function";
      return nullptr;
    }
    llvm_unreachable("unsupported type conversion function");
  }

  void yieldOutputs(ValueRange values) {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToEnd(&block);
    Location loc = builder.getUnknownLoc();
    YieldOp::create(builder, loc, values);
  }

  Value constant(const std::string &value) {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToEnd(&block);
    Location loc = builder.getUnknownLoc();
    Attribute valueAttr = parseAttribute(value, builder.getContext());
    return arith::ConstantOp::create(builder, loc,
                                     ::cast<TypedAttr>(valueAttr));
  }

  Value index(int64_t dim) {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToEnd(&block);
    return IndexOp::create(builder, builder.getUnknownLoc(), dim);
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
    if (isa<UnknownLoc>(loc)) {
      if (operand.getDefiningOp())
        loc = operand.getDefiningOp()->getLoc();
      else if (operand.getParentBlock() &&
               operand.getParentBlock()->getParentOp())
        loc = operand.getParentBlock()->getParentOp()->getLoc();
    }
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

      newInit = TensorReshapeOp::create(
          rewriter, loc, reshapeOp.getResultType(), oldFill.output(),
          reshapeOp.getReassociation(), reshapeOp.getOutputShape(),
          reshapeOp.getStaticOutputShape());
    } else {
      newInit = TensorReshapeOp::create(
          rewriter, loc, reshapeOp.getResultType(), oldFill.output(),
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

    auto emptyTensor =
        tensor::EmptyOp::create(rewriter, padOp.getLoc(), reifiedShape.front(),
                                padOp.getResultType().getElementType());
    Value replacement =
        FillOp::create(rewriter, fillOp.getLoc(), ValueRange{padValue},
                       ValueRange{emptyTensor})
            .getResult(0);
    if (replacement.getType() != padOp.getResultType()) {
      replacement = tensor::CastOp::create(rewriter, fillOp.getLoc(),
                                           padOp.getResultType(), replacement);
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
            tensor::DimOp::create(rewriter, loc, srcPadOp.getSource(), i)
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
                                                linalg::PackOp packOp) {
  auto fillOp = packOp.getSource().getDefiningOp<FillOp>();
  if (!fillOp)
    return failure();

  if (auto paddingValue = packOp.getPaddingValue())
    if (!isEqualConstantIntOrValue(paddingValue, fillOp.value()))
      return failure();

  Value packOpDest = packOp.getDest();
  if (!packOpDest.hasOneUse())
    return failure();

  return linalg::FillOp::create(rewriter, packOp.getLoc(), fillOp.getInputs(),
                                packOp.getDest());
}

/// Wrapper pattern that applies foldFillPackIntoFillOp method.
struct FoldFillWithPack : public OpRewritePattern<linalg::PackOp> {
public:
  FoldFillWithPack(MLIRContext *context)
      : OpRewritePattern<linalg::PackOp>(context) {}

  LogicalResult matchAndRewrite(linalg::PackOp packOp,
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

    Value outsConcat = tensor::ConcatOp::create(rewriter, concatOp.getLoc(),
                                                concatOp.getDim(), allOuts);
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
  genericAttrNamesSet.insert_range(genericAttrNames);
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

/// Remove linalg operations that are just copying the values from inputs to
/// results. In the memref case, the operation must be copying to and from the
/// same value. Requirements are:
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
      if (linalgOp.getNumDpsInputs() != 1 || linalgOp.getNumDpsInits() != 1 ||
          linalgOp.getDpsInputOperand(0)->get() !=
              linalgOp.getDpsInitOperand(0)->get()) {
        return rewriter.notifyMatchFailure(
            linalgOp, "expected single input and output to be the same value");
      }

      auto yieldArg = dyn_cast<BlockArgument>(yieldOp.getOperand(0));
      if (!yieldArg || yieldArg.getOwner() != &body) {
        return rewriter.notifyMatchFailure(linalgOp,
                                           "cannot fold fill-like op");
      }

      rewriter.eraseOp(linalgOp);
      return success();
    }

    if (!linalgOp.hasPureTensorSemantics()) {
      return rewriter.notifyMatchFailure(
          linalgOp, "mixed semantics is not supported yet");
    }

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
          returnedArg = sparse_tensor::ConvertOp::create(
              rewriter, linalgOp.getLoc(), resultType, returnedArg);
        else {
          if (!tensor::CastOp::areCastCompatible(returnedArg.getType(),
                                                 resultType))
            return failure();
          returnedArg = tensor::CastOp::create(rewriter, linalgOp.getLoc(),
                                               resultType, returnedArg);
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
  YieldOp::create(b, result.location, payloadOp->getResults());
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
                         linalg::YieldOp::create(b, loc, args[0]);
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
        dims.push_back(tensor::DimOp::create(rewriter, loc, broadcastInput, i)
                           ->getResult(0));
      } else {
        dims.push_back(IntegerAttr::get(IndexType::get(ctx),
                                        broadcastInputTy.getDimSize(i)));
      }
    }
    SmallVector<OpFoldResult> transposeResultShapes =
        applyPermutation(dims, resultPerms);
    Value transposeInit = tensor::EmptyOp::create(
        rewriter, transposeOp.getLoc(), transposeResultShapes,
        broadcastInputTy.getElementType());

    // Create broadcast(transpose(input)).
    Value transposeResult =
        TransposeOp::create(rewriter, loc, broadcastOp.getInput(),
                            transposeInit, resultPerms)
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

/// Fold back-to-back broadcasts together.
struct FoldBroadcasts : OpRewritePattern<linalg::BroadcastOp> {
  using OpRewritePattern<linalg::BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BroadcastOp broadcastOp,
                                PatternRewriter &rewriter) const override {
    auto defBroadcastOp = broadcastOp.getInput().getDefiningOp<BroadcastOp>();
    if (!defBroadcastOp)
      return failure();
    ArrayRef<int64_t> defDimensions = defBroadcastOp.getDimensions();
    ArrayRef<int64_t> dimensions = broadcastOp.getDimensions();
    SmallVector<int64_t> foldedDims(dimensions);
    Value init = broadcastOp.getInit();
    int64_t initRank = cast<ShapedType>(init.getType()).getRank();
    // Mapping from input dims to init dims.
    SmallVector<int64_t> dimMap;
    for (auto dim : llvm::seq<int64_t>(0, initRank)) {
      if (!llvm::is_contained(dimensions, dim))
        dimMap.push_back(dim);
    }
    for (auto dim : defDimensions)
      foldedDims.push_back(dimMap[dim]);

    llvm::sort(foldedDims);
    rewriter.replaceOpWithNewOp<BroadcastOp>(
        broadcastOp, defBroadcastOp.getInput(), init, foldedDims);
    return success();
  }
};

void BroadcastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<EraseIdentityLinalgOp<BroadcastOp>, FoldBroadcasts>(context);
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

OpFoldResult IndexOp::fold(FoldAdaptor adaptor) {
  auto linalgOp = dyn_cast_or_null<LinalgOp>((*this)->getParentOp());
  // Bail out if `linalg.index` does not have a proper parent yet at this
  // point, e.g., when calling `createOrFold` during IR construction in
  // `genericOp::build`.
  if (!linalgOp)
    return OpFoldResult{};

  // Index of unit dims is always 0.
  SmallVector<int64_t, 4> loopBounds = linalgOp.getStaticLoopRanges();
  uint64_t dim = getDim();
  assert(dim < loopBounds.size() && "Dim is out of bounds");
  if (loopBounds[dim] == 1)
    return IntegerAttr::get(IndexType::get(getContext()), 0);

  return OpFoldResult{};
}

/////// Operations corresponding to library calls defined with Tablegen ////////

#include "mlir/Dialect/Linalg/IR/LinalgNamedStructuredOps.yamlgen.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgOps.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgRelayoutOps.cpp.inc"

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
  llvm::replace(name, '.', '_');
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
        tensor::CastOp::create(rewriter, loc, resultType, outOperand->get());
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
    Value castBack = tensor::CastOp::create(
        rewriter, loc, resultValue.getType(), newOp->getResult(resultNumber));

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
  resultType = RankedTensorType::get(newShape, sourceType.getElementType(),
                                     sourceType.getEncoding());
  if (newOperandNeeded) {
    changeNeeded = true;
    // Get the new operand value given its size and element type by
    // casting it.
    Value newOperand = tensor::CastOp::create(rewriter, loc, resultType, src);
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
              ? tensor::CastOp::create(rewriter, loc, oldType, newResult)
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
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
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

  auto genericOp = linalg::GenericOp::create(
      builder, loc, output.getType(), input, output, indexingMaps,
      iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
        Value result = T::create(b, loc, args[0], args[1]);
        linalg::YieldOp::create(b, loc, result);
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
  auto genericOp = linalg::GenericOp::create(
      builder, loc, input.getType(), ValueRange{input, max}, output,
      indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value diff = arith::SubFOp::create(b, loc, args[0], args[1]);
        Value result = math::ExpOp::create(b, loc, diff);
        linalg::YieldOp::create(b, loc, result);
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
  auto genericOp = linalg::GenericOp::create(
      builder, loc, numerator.getType(), ValueRange{numerator, denominator},
      output, indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value result = arith::DivFOp::create(b, loc, args[0], args[1]);
        linalg::YieldOp::create(b, loc, result);
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
  Value outputReduce = tensor::EmptyOp::create(b, loc, dims, elementType);
  Value neutralForMaxF = arith::getIdentityValue(arith::AtomicRMWKind::maxnumf,
                                                 elementType, b, loc,
                                                 /*useOnlyFiniteValue=*/true);
  Value neutralForMaxFInit =
      linalg::FillOp::create(b, loc, Value{neutralForMaxF}, outputReduce)
          .result();
  Value max =
      reduce<arith::MaxNumFOp>(b, loc, input, neutralForMaxFInit, reductionDim);

  // Step 2: Subtract max from input and exponentiate.
  Value numerator = buildSubAndExpOp(b, loc, input, max, output, reductionDim);

  // Step 3: Compute sum along dim.
  Value zero = arith::getIdentityValue(arith::AtomicRMWKind::addf, elementType,
                                       b, loc, /*useOnlyFiniteValue=*/true);
  Value zeroInit =
      linalg::FillOp::create(b, loc, Value{zero}, outputReduce).result();
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
  WinogradConv2DFmr fmr = getFmr();
  int64_t m, r;
  std::tie(m, r) = getFmrFromWinogradConv2DFmr(fmr);

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
  WinogradConv2DFmr fmr = getFmr();
  int64_t m, r;
  std::tie(m, r) = getFmrFromWinogradConv2DFmr(fmr);
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
  auto filterSlice = tensor::ExtractSliceOp::create(
      builder, loc, getFilter(), sliceOffsets, sliceSizes, filterStrides);
  tiledOperands.emplace_back(filterSlice);

  SmallVector<OpFoldResult> resultOffsets, resultSizes;
  if (failed(getResultTilePosition(builder, 1, offsets, sizes, resultOffsets,
                                   resultSizes)))
    return failure();

  int64_t outputRank = getOutputOperandRank();
  SmallVector<OpFoldResult> outputStrides(outputRank, oneAttr);
  auto outputSlice = tensor::ExtractSliceOp::create(
      builder, loc, getOutput(), resultOffsets, resultSizes, outputStrides);
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
  WinogradConv2DFmr fmr = getFmr();
  int64_t m, r;
  std::tie(m, r) = getFmrFromWinogradConv2DFmr(fmr);
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

  WinogradConv2DFmr fmr = getFmr();
  int64_t m, r;
  std::tie(m, r) = getFmrFromWinogradConv2DFmr(fmr);
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
  WinogradConv2DFmr fmr = getFmr();
  int64_t m, r;
  std::tie(m, r) = getFmrFromWinogradConv2DFmr(fmr);

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
  auto inputSlice = tensor::ExtractSliceOp::create(
      builder, loc, getInput(), sliceOffsets, sliceSizes, inputStrides);
  tiledOperands.emplace_back(inputSlice);

  SmallVector<OpFoldResult> resultOffsets, resultSizes;
  if (failed(getResultTilePosition(builder, 1, offsets, sizes, resultOffsets,
                                   resultSizes)))
    return failure();

  int64_t outputRank = getOutputOperandRank();
  SmallVector<OpFoldResult> outputStrides(outputRank, oneAttr);
  auto outputSlice = tensor::ExtractSliceOp::create(
      builder, loc, getOutput(), resultOffsets, resultSizes, outputStrides);
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
  WinogradConv2DFmr fmr = getFmr();
  int64_t m, r;
  std::tie(m, r) = getFmrFromWinogradConv2DFmr(fmr);
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
  WinogradConv2DFmr fmr = getFmr();
  int64_t m, r;
  std::tie(m, r) = getFmrFromWinogradConv2DFmr(fmr);

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
  auto valueSlice = tensor::ExtractSliceOp::create(
      builder, loc, getValue(), sliceOffsets, sliceSizes, sliceStrides);
  tiledOperands.emplace_back(valueSlice);

  SmallVector<OpFoldResult> resultOffsets, resultSizes;
  if (failed(getResultTilePosition(builder, 1, offsets, sizes, resultOffsets,
                                   resultSizes)))
    return failure();

  int64_t outputRank = getOutputOperandRank();
  SmallVector<OpFoldResult> strides(outputRank, oneAttr);
  auto outputSlice = tensor::ExtractSliceOp::create(
      builder, loc, getOutput(), resultOffsets, resultSizes, strides);
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
// TODO: Merge with the LinalgDialect block at the bottom
//===----------------------------------------------------------------------===//

// Returns true if the result expression of `subMap` are a subset of `fullMap`.
static bool areResultExprsSubsetOf(AffineMap subMap, AffineMap fullMap) {
  auto explicitRange = subMap.getResults();
  auto defaultRange = fullMap.getResults();
  DenseSet<AffineExpr> explicitSet(explicitRange.begin(), explicitRange.end());
  DenseSet<AffineExpr> defaultSet(defaultRange.begin(), defaultRange.end());
  llvm::set_union(explicitSet, defaultSet);
  return explicitSet == defaultSet;
}

/// Check if the user defined map is valid broadcast map. Here broadcast
/// indexing maps are defined in context of corresponding default indexing maps
/// for the given Op. This way the check becomes very simple i.e just check the
/// number of result dims.
/// Returns true if the explictMap is broadcasted with respect to the
/// defaultMap.
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
  if (!areResultExprsSubsetOf(opIndexingMap, defaultIndexingMap))
    return matmulOp->emitOpError()
           << "Unexpected dim expression in map result.";

  if (isBroadcasted(opIndexingMap, defaultIndexingMap)) {
    if (!matmulOp.isValidLhsRhsBroadcastMap(opIndexingMap)) {
      return matmulOp->emitOpError()
             << "Invalid broadcast requested, should be (d2).";
    }
    return success();
  }
  return success();
}

// Check general validity of input indexing map of
// BatchMatmulOp/BatchReduceMatmulOp.
template <typename OpTy>
static LogicalResult verifyInputMaps(OpTy batchVariantMatmulOp,
                                     AffineMap opIndexingMap,
                                     AffineMap defaultIndexingMap, bool isLHS) {
  assert((isa<BatchMatmulOp>(batchVariantMatmulOp) ||
          isa<BatchReduceMatmulOp>(batchVariantMatmulOp)) &&
         "Expected BatchMatmulOp or BatchReduceMatmulOp");
  // Check the result dims are valid.
  if (!areResultExprsSubsetOf(opIndexingMap, defaultIndexingMap))
    return batchVariantMatmulOp->emitOpError()
           << "Unexpected result dim expression (outside the set of default "
              "result dims).";

  // Check for valid number of result dims of input maps.
  if (opIndexingMap.getNumResults() > 3)
    return batchVariantMatmulOp->emitOpError()
           << "no. of result dim expressions exceeds 3.";

  auto hasValidBatchDim = [](AffineMap map) {
    AffineExpr batchDim = map.getResult(0);
    return batchDim.isFunctionOfDim(0);
  };

  // Check if the requested broadcast is valid.
  if (isBroadcasted(opIndexingMap, defaultIndexingMap)) {
    if (!batchVariantMatmulOp.isValidLhsRhsBroadcastMap(opIndexingMap, isLHS))
      return batchVariantMatmulOp->emitOpError()
             << "Invalid broadcast requested.";
  } else if (!hasValidBatchDim(opIndexingMap)) {
    return batchVariantMatmulOp->emitOpError()
           << "Invalid batch dimension expression.";
  }
  return success();
}

/// This function checks if the given AffineMap for the output of a
/// BatchMatmulOp/BatchReduceMatmulOp has exactly the desired number of result
/// dimensions and if the output map result dimensions are valid.
template <typename OpTy>
static LogicalResult verifyOutputMap(OpTy batchVariantMatmulOp,
                                     AffineMap opIndexingMap) {
  assert((isa<BatchMatmulOp>(batchVariantMatmulOp) ||
          isa<BatchReduceMatmulOp>(batchVariantMatmulOp)) &&
         "Expected BatchMatmulOp or BatchReduceMatmulOp");
  if (isa<BatchMatmulOp>(batchVariantMatmulOp) &&
      opIndexingMap.getNumResults() != 3) {

    return batchVariantMatmulOp->emitOpError()
           << "expects 3 dims, but got (" << opIndexingMap.getNumResults()
           << ").";
  }
  if (isa<BatchReduceMatmulOp>(batchVariantMatmulOp) &&
      opIndexingMap.getNumResults() != 2) {
    return batchVariantMatmulOp->emitOpError()
           << "expects 2 dims, but got (" << opIndexingMap.getNumResults()
           << ").";
  }

  auto areValidOutputResultDim = [&](AffineMap outputMap) {
    return isa<BatchMatmulOp>(batchVariantMatmulOp)
               ? outputMap.getResult(0).isFunctionOfDim(0) &&
                     outputMap.getResult(1).isFunctionOfDim(1) &&
                     outputMap.getResult(2).isFunctionOfDim(2)
               : outputMap.getResult(0).isFunctionOfDim(1) &&
                     outputMap.getResult(1).isFunctionOfDim(2);
  };

  if (!areValidOutputResultDim(opIndexingMap)) {
    return batchVariantMatmulOp->emitOpError()
           << "Invalid output map result dimension.";
  }

  return success();
}

/// Verifies the broadcast and transpose semantic specified by the explicit
/// indexing map for the BatchMatmulOp/BatchReduceMatmulOp op for each operand
/// specified by opIndex.
template <typename OpTy>
static LogicalResult
verifyExtendedBatchVariantMatmulSemantic(OpTy batchVariantMatmulOp,
                                         unsigned opIndex) {
  SmallVector<AffineMap, 3> opIndexingMaps =
      batchVariantMatmulOp.getIndexingMapsArray();
  SmallVector<AffineMap, 3> defaultIndexingMaps =
      batchVariantMatmulOp.getDefaultIndexingMaps(
          batchVariantMatmulOp->getContext());

  if (opIndexingMaps.size() != 3)
    return batchVariantMatmulOp->emitOpError()
           << "Indexing_map attribute must have 3 affine maps.";

  auto opIndexingMap = opIndexingMaps[opIndex];
  auto defaultIndexingMap = defaultIndexingMaps[opIndex];

  if (opIndex == 2 &&
      failed(verifyOutputMap(batchVariantMatmulOp, opIndexingMap)))
    return failure();

  if (opIndex != 2 &&
      failed(verifyInputMaps(batchVariantMatmulOp, opIndexingMap,
                             defaultIndexingMap, opIndex == 0)))
    return failure();

  return success();
}

namespace mlir {
namespace linalg {

std::optional<WinogradConv2DFmr> getWinogradConv2DFmr(int64_t m, int64_t r) {
  if (m == 2 && r == 3)
    return WinogradConv2DFmr::F_2_3;
  if (m == 4 && r == 3)
    return WinogradConv2DFmr::F_4_3;
  if (m == 2 && r == 5)
    return WinogradConv2DFmr::F_2_5;
  return std::nullopt;
}

std::pair<int64_t, int64_t> getFmrFromWinogradConv2DFmr(WinogradConv2DFmr fmr) {
  switch (fmr) {
  case WinogradConv2DFmr::F_2_3:
    return {2, 3};
  case WinogradConv2DFmr::F_4_3:
    return {4, 3};
  case WinogradConv2DFmr::F_2_5:
    return {2, 5};
  }
}

//===----------------------------------------------------------------------===//
// MatMulOp
//===----------------------------------------------------------------------===//

static FailureOr<SmallVector<SmallVector<int64_t>>>
getAffineResultPositions(ArrayAttr maps) {
  SmallVector<SmallVector<int64_t>> positions;
  for (auto map : maps) {
    AffineMapAttr attr = dyn_cast<AffineMapAttr>(map);
    if (!attr)
      return failure();
    SmallVector<int64_t> pos;
    for (auto result : attr.getAffineMap().getResults()) {
      auto dim = dyn_cast<AffineDimExpr>(result);
      if (!dim)
        return failure();
      pos.push_back(dim.getPosition());
    }
    positions.push_back(pos);
  }
  return positions;
}

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

bool MatmulOp::isDefaultIndexingMaps(Attribute attr) {
  ArrayAttr maps = dyn_cast<ArrayAttr>(attr);
  if (!maps)
    return false;
  if (maps.size() != 3)
    return false;
  auto positions = getAffineResultPositions(maps);
  if (failed(positions))
    return false;
  return (*positions)[0] == SmallVector<int64_t>{0, 2} &&
         (*positions)[1] == SmallVector<int64_t>{2, 1} &&
         (*positions)[2] == SmallVector<int64_t>{0, 1};
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
                             ArrayRef<NamedAttribute> attrs,
                             function_ref<InFlightDiagnostic()> emitError) {
  if (emitError && block.getNumArguments() != 3) {
    emitError() << "MatmulOp regionBuilder expects 3 args, got "
                << block.getNumArguments();
    return;
  }
  assert(block.getNumArguments() == 3 &&
         "MatmulOp regionBuilder expects 3 args");
  RegionBuilderHelper helper(b, block);
  SmallVector<Value> yields;

  TypeFn castVal = TypeFn::cast_signed;
  const auto *castIter = llvm::find_if(attrs, [&](const NamedAttribute &attr) {
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
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2, emitError);
  if (!value3)
    return;
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2),
                                      value3, emitError);
  if (!value4)
    return;
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

/// Returns true if the given bcastMap map is a valid broadcast map. A valid
/// broadcast map must include K dimension.
/// TODO: Strict inclusion of K dimension in the broadcast map is not
/// necessary for both input matrices simultaneously. We can relax this
/// condition to have K dimension for one input matrix map and infer the K
/// dimension for other input matrix map from the one already having K
/// dimension.
bool MatmulOp::isValidLhsRhsBroadcastMap(AffineMap bcastMap) {
  assert(bcastMap.getNumResults() == 1 && "Expected single result dim expr.");
  AffineExpr expr = bcastMap.getResult(0);
  // Invalid map if the common dimension of matmul not found.
  return expr.isFunctionOfDim(bcastMap.getNumDims() - 1);
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
  SmallVector<Attribute, 3> indexingMaps = llvm::map_to_vector<3>(
      MatmulOp::getDefaultIndexingMaps(getContext()),
      [](AffineMap map) -> Attribute { return AffineMapAttr::get(map); });
  if (!llvm::equal(getIndexingMaps(), indexingMaps))
    p << " indexing_maps = " << llvm::interleaved_array(getIndexingMaps());

  std::array<StringRef, 3> elidedAttrs = {
      "operandSegmentSizes", "linalg.memoized_indexing_maps", "indexing_maps"};
  printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs(),
                         elidedAttrs);
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

SmallVector<AffineMap>
MatmulTransposeAOp::getDefaultIndexingMaps(OpBuilder &builder) {
  AffineExpr d0, d1, d2;
  MLIRContext *context = builder.getContext();
  bindDims(context, d0, d1, d2);
  AffineMap mapLHS = AffineMap::get(3, 0, {d2, d0}, context);
  AffineMap mapRHS = AffineMap::get(3, 0, {d2, d1}, context);
  AffineMap mapOut = AffineMap::get(3, 0, {d0, d1}, context);
  return {mapLHS, mapRHS, mapOut};
}

bool MatmulTransposeAOp::isDefaultIndexingMaps(Attribute attr) {
  ArrayAttr maps = dyn_cast<ArrayAttr>(attr);
  if (!maps)
    return false;
  if (maps.size() != 3)
    return false;
  auto positions = getAffineResultPositions(maps);
  if (failed(positions))
    return false;
  return (*positions)[0] == SmallVector<int64_t>{2, 0} &&
         (*positions)[1] == SmallVector<int64_t>{2, 1} &&
         (*positions)[2] == SmallVector<int64_t>{0, 1};
}

void linalg::MatmulTransposeAOp::build(OpBuilder &builder,
                                       OperationState &result,
                                       ValueRange inputs, ValueRange outputs,
                                       ArrayRef<NamedAttribute> attributes) {
  buildMatmulOp(builder, result, std::nullopt, inputs, outputs, attributes,
                MatmulOp::getRegionBuilder(), getDefaultIndexingMaps(builder));
}

MatmulTransposeAOp
MatmulTransposeAOp::create(OpBuilder &builder, Location location,
                           ValueRange inputs, ValueRange outputs,
                           ArrayRef<NamedAttribute> attributes) {
  OperationState state(location, getOperationName());
  build(builder, state, inputs, outputs, attributes);
  auto res = dyn_cast<MatmulTransposeAOp>(builder.create(state));
  assert(res && "builder didn't return the right type");
  return res;
}

void linalg::MatmulTransposeAOp::build(OpBuilder &builder,
                                       OperationState &result,
                                       TypeRange resultTensorTypes,
                                       ValueRange inputs, ValueRange outputs,
                                       ArrayRef<NamedAttribute> attributes) {
  buildMatmulOp(builder, result, resultTensorTypes, inputs, outputs, attributes,
                MatmulOp::getRegionBuilder(), getDefaultIndexingMaps(builder));
}

MatmulTransposeAOp
MatmulTransposeAOp::create(OpBuilder &builder, Location location,
                           TypeRange resultTensorTypes, ValueRange inputs,
                           ValueRange outputs,
                           ArrayRef<NamedAttribute> attributes) {
  OperationState state(location, getOperationName());
  build(builder, state, resultTensorTypes, inputs, outputs, attributes);
  auto res = dyn_cast<MatmulTransposeAOp>(builder.create(state));
  assert(res && "builder didn't return the right type");
  return res;
}

void linalg::MatmulTransposeAOp::build(OpBuilder &builder,
                                       OperationState &result,
                                       TypeRange resultTensorTypes,
                                       ValueRange inputs, ValueRange outputs,
                                       Attribute cast,
                                       ArrayRef<NamedAttribute> attributes) {
  result.addAttribute("cast", cast);
  buildMatmulOp(builder, result, resultTensorTypes, inputs, outputs, attributes,
                MatmulOp::getRegionBuilder(), getDefaultIndexingMaps(builder));
}

MatmulTransposeAOp
MatmulTransposeAOp::create(OpBuilder &builder, Location location,
                           TypeRange resultTensorTypes, ValueRange inputs,
                           ValueRange outputs, Attribute cast,
                           ArrayRef<NamedAttribute> attributes) {
  OperationState state(location, getOperationName());
  build(builder, state, resultTensorTypes, inputs, outputs, cast, attributes);
  auto res = dyn_cast<MatmulTransposeAOp>(builder.create(state));
  assert(res && "builder didn't return the right type");
  return res;
}

bool MatmulTransposeAOp::classof(Operation *op) {
  return dyn_cast_or_null<linalg::MatmulOp>(op) &&
         MatmulTransposeAOp::isDefaultIndexingMaps(
             op->getAttr("indexing_maps"));
}

SmallVector<AffineMap>
MatmulTransposeBOp::getDefaultIndexingMaps(OpBuilder &builder) {
  AffineExpr d0, d1, d2;
  MLIRContext *context = builder.getContext();
  bindDims(context, d0, d1, d2);
  AffineMap mapLHS = AffineMap::get(3, 0, {d0, d2}, context);
  AffineMap mapRHS = AffineMap::get(3, 0, {d1, d2}, context);
  AffineMap mapOut = AffineMap::get(3, 0, {d0, d1}, context);
  return {mapLHS, mapRHS, mapOut};
}

bool MatmulTransposeBOp::isDefaultIndexingMaps(Attribute attr) {
  ArrayAttr maps = dyn_cast<ArrayAttr>(attr);
  if (!maps)
    return false;
  if (maps.size() != 3)
    return false;
  auto positions = getAffineResultPositions(maps);
  if (failed(positions))
    return false;
  return (*positions)[0] == SmallVector<int64_t>{0, 2} &&
         (*positions)[1] == SmallVector<int64_t>{1, 2} &&
         (*positions)[2] == SmallVector<int64_t>{0, 1};
}

void linalg::MatmulTransposeBOp::build(OpBuilder &builder,
                                       OperationState &result,
                                       ValueRange inputs, ValueRange outputs,
                                       ArrayRef<NamedAttribute> attributes) {
  buildMatmulOp(builder, result, std::nullopt, inputs, outputs, attributes,
                MatmulOp::getRegionBuilder(), getDefaultIndexingMaps(builder));
}

MatmulTransposeBOp
MatmulTransposeBOp::create(OpBuilder &builder, Location location,
                           ValueRange inputs, ValueRange outputs,
                           ArrayRef<NamedAttribute> attributes) {
  OperationState state(location, getOperationName());
  build(builder, state, inputs, outputs, attributes);
  auto res = dyn_cast<MatmulTransposeBOp>(builder.create(state));
  assert(res && "builder didn't return the right type");
  return res;
}

void linalg::MatmulTransposeBOp::build(OpBuilder &builder,
                                       OperationState &result,
                                       TypeRange resultTensorTypes,
                                       ValueRange inputs, ValueRange outputs,
                                       ArrayRef<NamedAttribute> attributes) {
  buildMatmulOp(builder, result, resultTensorTypes, inputs, outputs, attributes,
                MatmulOp::getRegionBuilder(), getDefaultIndexingMaps(builder));
}

MatmulTransposeBOp
MatmulTransposeBOp::create(OpBuilder &builder, Location location,
                           TypeRange resultTensorTypes, ValueRange inputs,
                           ValueRange outputs,
                           ArrayRef<NamedAttribute> attributes) {
  OperationState state(location, getOperationName());
  build(builder, state, resultTensorTypes, inputs, outputs, attributes);
  auto res = dyn_cast<MatmulTransposeBOp>(builder.create(state));
  assert(res && "builder didn't return the right type");
  return res;
}

void linalg::MatmulTransposeBOp::build(OpBuilder &builder,
                                       OperationState &result,
                                       TypeRange resultTensorTypes,
                                       ValueRange inputs, ValueRange outputs,
                                       Attribute cast,
                                       ArrayRef<NamedAttribute> attributes) {
  result.addAttribute("cast", cast);
  buildMatmulOp(builder, result, resultTensorTypes, inputs, outputs, attributes,
                MatmulOp::getRegionBuilder(), getDefaultIndexingMaps(builder));
}

MatmulTransposeBOp
MatmulTransposeBOp::create(OpBuilder &builder, Location location,
                           TypeRange resultTensorTypes, ValueRange inputs,
                           ValueRange outputs, Attribute cast,
                           ArrayRef<NamedAttribute> attributes) {
  OperationState state(location, getOperationName());
  build(builder, state, resultTensorTypes, inputs, outputs, cast, attributes);
  auto res = dyn_cast<MatmulTransposeBOp>(builder.create(state));
  assert(res && "builder didn't return the right type");
  return res;
}

bool MatmulTransposeBOp::classof(Operation *op) {
  return dyn_cast_or_null<linalg::MatmulOp>(op) &&
         MatmulTransposeBOp::isDefaultIndexingMaps(
             op->getAttr("indexing_maps"));
}

SmallVector<AffineMap>
BatchMatmulTransposeAOp::getDefaultIndexingMaps(OpBuilder &builder) {
  AffineExpr d0, d1, d2, d3;
  MLIRContext *context = builder.getContext();
  bindDims(context, d0, d1, d2, d3);
  AffineMap mapLHS = AffineMap::get(4, 0, {d0, d3, d1}, context);
  AffineMap mapRHS = AffineMap::get(4, 0, {d0, d3, d2}, context);
  AffineMap mapOut = AffineMap::get(4, 0, {d0, d1, d2}, context);
  return {mapLHS, mapRHS, mapOut};
}

bool BatchMatmulTransposeAOp::isDefaultIndexingMaps(Attribute attr) {
  ArrayAttr maps = dyn_cast<ArrayAttr>(attr);
  if (!maps)
    return false;
  if (maps.size() != 3)
    return false;
  auto positions = getAffineResultPositions(maps);
  if (failed(positions))
    return false;
  return (*positions)[0] == SmallVector<int64_t>{0, 3, 1} &&
         (*positions)[1] == SmallVector<int64_t>{0, 3, 2} &&
         (*positions)[2] == SmallVector<int64_t>{0, 1, 2};
}

void linalg::BatchMatmulTransposeAOp::build(
    OpBuilder &builder, OperationState &result, ValueRange inputs,
    ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
  buildMatmulOp(builder, result, std::nullopt, inputs, outputs, attributes,
                BatchMatmulOp::getRegionBuilder(),
                getDefaultIndexingMaps(builder));
}

BatchMatmulTransposeAOp
BatchMatmulTransposeAOp::create(OpBuilder &builder, Location location,
                                ValueRange inputs, ValueRange outputs,
                                ArrayRef<NamedAttribute> attributes) {
  OperationState state(location, getOperationName());
  build(builder, state, inputs, outputs, attributes);
  auto res = dyn_cast<BatchMatmulTransposeAOp>(builder.create(state));
  assert(res && "builder didn't return the right type");
  return res;
}

void linalg::BatchMatmulTransposeAOp::build(
    OpBuilder &builder, OperationState &result, TypeRange resultTensorTypes,
    ValueRange inputs, ValueRange outputs,
    ArrayRef<NamedAttribute> attributes) {
  buildMatmulOp(builder, result, resultTensorTypes, inputs, outputs, attributes,
                BatchMatmulOp::getRegionBuilder(),
                getDefaultIndexingMaps(builder));
}

BatchMatmulTransposeAOp
BatchMatmulTransposeAOp::create(OpBuilder &builder, Location location,
                                TypeRange resultTensorTypes, ValueRange inputs,
                                ValueRange outputs,
                                ArrayRef<NamedAttribute> attributes) {
  OperationState state(location, getOperationName());
  build(builder, state, resultTensorTypes, inputs, outputs, attributes);
  auto res = dyn_cast<BatchMatmulTransposeAOp>(builder.create(state));
  assert(res && "builder didn't return the right type");
  return res;
}

void linalg::BatchMatmulTransposeAOp::build(
    OpBuilder &builder, OperationState &result, TypeRange resultTensorTypes,
    ValueRange inputs, ValueRange outputs, Attribute cast,
    ArrayRef<NamedAttribute> attributes) {
  result.addAttribute("cast", cast);
  buildMatmulOp(builder, result, resultTensorTypes, inputs, outputs, attributes,
                BatchMatmulOp::getRegionBuilder(),
                getDefaultIndexingMaps(builder));
}

BatchMatmulTransposeAOp
BatchMatmulTransposeAOp::create(OpBuilder &builder, Location location,
                                TypeRange resultTensorTypes, ValueRange inputs,
                                ValueRange outputs, Attribute cast,
                                ArrayRef<NamedAttribute> attributes) {
  OperationState state(location, getOperationName());
  build(builder, state, resultTensorTypes, inputs, outputs, cast, attributes);
  auto res = dyn_cast<BatchMatmulTransposeAOp>(builder.create(state));
  assert(res && "builder didn't return the right type");
  return res;
}

bool BatchMatmulTransposeAOp::classof(Operation *op) {
  return dyn_cast_or_null<linalg::BatchMatmulOp>(op) &&
         BatchMatmulTransposeAOp::isDefaultIndexingMaps(
             op->getAttr("indexing_maps"));
}

SmallVector<AffineMap>
BatchMatmulTransposeBOp::getDefaultIndexingMaps(OpBuilder &builder) {
  AffineExpr d0, d1, d2, d3;
  MLIRContext *context = builder.getContext();
  bindDims(context, d0, d1, d2, d3);
  AffineMap mapLHS = AffineMap::get(4, 0, {d0, d1, d3}, context);
  AffineMap mapRHS = AffineMap::get(4, 0, {d0, d2, d3}, context);
  AffineMap mapOut = AffineMap::get(4, 0, {d0, d1, d2}, context);
  return {mapLHS, mapRHS, mapOut};
}

bool BatchMatmulTransposeBOp::isDefaultIndexingMaps(Attribute attr) {
  ArrayAttr maps = dyn_cast<ArrayAttr>(attr);
  if (!maps)
    return false;
  if (maps.size() != 3)
    return false;
  auto positions = getAffineResultPositions(maps);
  if (failed(positions))
    return false;
  return (*positions)[0] == SmallVector<int64_t>{0, 1, 3} &&
         (*positions)[1] == SmallVector<int64_t>{0, 2, 3} &&
         (*positions)[2] == SmallVector<int64_t>{0, 1, 2};
}

void linalg::BatchMatmulTransposeBOp::build(
    OpBuilder &builder, OperationState &result, ValueRange inputs,
    ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
  buildMatmulOp(builder, result, std::nullopt, inputs, outputs, attributes,
                BatchMatmulOp::getRegionBuilder(),
                getDefaultIndexingMaps(builder));
}

BatchMatmulTransposeBOp
BatchMatmulTransposeBOp::create(OpBuilder &builder, Location location,
                                ValueRange inputs, ValueRange outputs,
                                ArrayRef<NamedAttribute> attributes) {
  OperationState state(location, getOperationName());
  build(builder, state, inputs, outputs, attributes);
  auto res = dyn_cast<BatchMatmulTransposeBOp>(builder.create(state));
  assert(res && "builder didn't return the right type");
  return res;
}

void linalg::BatchMatmulTransposeBOp::build(
    OpBuilder &builder, OperationState &result, TypeRange resultTensorTypes,
    ValueRange inputs, ValueRange outputs,
    ArrayRef<NamedAttribute> attributes) {
  buildMatmulOp(builder, result, resultTensorTypes, inputs, outputs, attributes,
                BatchMatmulOp::getRegionBuilder(),
                getDefaultIndexingMaps(builder));
}

BatchMatmulTransposeBOp
BatchMatmulTransposeBOp::create(OpBuilder &builder, Location location,
                                TypeRange resultTensorTypes, ValueRange inputs,
                                ValueRange outputs,
                                ArrayRef<NamedAttribute> attributes) {
  OperationState state(location, getOperationName());
  build(builder, state, resultTensorTypes, inputs, outputs, attributes);
  auto res = dyn_cast<BatchMatmulTransposeBOp>(builder.create(state));
  assert(res && "builder didn't return the right type");
  return res;
}

void linalg::BatchMatmulTransposeBOp::build(
    OpBuilder &builder, OperationState &result, TypeRange resultTensorTypes,
    ValueRange inputs, ValueRange outputs, Attribute cast,
    ArrayRef<NamedAttribute> attributes) {
  result.addAttribute("cast", cast);
  buildMatmulOp(builder, result, resultTensorTypes, inputs, outputs, attributes,
                BatchMatmulOp::getRegionBuilder(),
                getDefaultIndexingMaps(builder));
}

BatchMatmulTransposeBOp
BatchMatmulTransposeBOp::create(OpBuilder &builder, Location location,
                                TypeRange resultTensorTypes, ValueRange inputs,
                                ValueRange outputs, Attribute cast,
                                ArrayRef<NamedAttribute> attributes) {
  OperationState state(location, getOperationName());
  build(builder, state, resultTensorTypes, inputs, outputs, cast, attributes);
  auto res = dyn_cast<BatchMatmulTransposeBOp>(builder.create(state));
  assert(res && "builder didn't return the right type");
  return res;
}

bool BatchMatmulTransposeBOp::classof(Operation *op) {
  return dyn_cast_or_null<linalg::BatchMatmulOp>(op) &&
         BatchMatmulTransposeBOp::isDefaultIndexingMaps(
             op->getAttr("indexing_maps"));
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
                               ArrayRef<NamedAttribute> attrs,
                               function_ref<InFlightDiagnostic()> emitError) {
  if (emitError && block.getNumArguments() != 3) {
    emitError() << "ContractOp regionBuilder expects 3 args, got "
                << block.getNumArguments();
    return;
  }
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
  Value productAtOutType = helper.buildBinaryFn(BinaryFn::mul, lhsAtOutType,
                                                rhsAtOutType, emitError);
  if (!productAtOutType)
    return;
  Value result = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2),
                                      productAtOutType, emitError);
  if (!result)
    return;
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
  p << " indexing_maps = " << llvm::interleaved_array(getIndexingMaps());
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

//===----------------------------------------------------------------------===//
// Implementation of BatchMatmulOp
//===----------------------------------------------------------------------===//
SmallVector<AffineMap>
BatchMatmulOp::getDefaultIndexingMaps(MLIRContext *context) {
  AffineExpr d0, d1, d2, d3;
  SmallVector<AffineMap> indexingMaps;
  bindDims(context, d0, d1, d2, d3);
  indexingMaps.push_back(AffineMap::get(4, 0, {d0, d1, d3}, context));
  indexingMaps.push_back(AffineMap::get(4, 0, {d0, d3, d2}, context));
  indexingMaps.push_back(AffineMap::get(4, 0, {d0, d1, d2}, context));
  return indexingMaps;
}

bool BatchMatmulOp::isDefaultIndexingMaps(Attribute attr) {
  ArrayAttr maps = dyn_cast<ArrayAttr>(attr);
  if (!maps)
    return false;
  if (maps.size() != 3)
    return false;
  auto positions = getAffineResultPositions(maps);
  if (failed(positions))
    return false;
  return (*positions)[0] == SmallVector<int64_t>{0, 1, 3} &&
         (*positions)[1] == SmallVector<int64_t>{0, 3, 2} &&
         (*positions)[2] == SmallVector<int64_t>{0, 1, 2};
}

SmallVector<utils::IteratorType> BatchMatmulOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{
      utils::IteratorType::parallel, utils::IteratorType::parallel,
      utils::IteratorType::parallel, utils::IteratorType::reduction};
}

unsigned BatchMatmulOp::getNumRegionArgs() { return 3; }

std::string BatchMatmulOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

/// Check if the op has broadcast and/or transpose semantic. Returns true if
/// the user defined indexing maps are not equal to default map.
bool BatchMatmulOp::hasUserDefinedMaps() {
  SmallVector<AffineMap, 3> defaultMaps =
      getDefaultIndexingMaps(this->getContext());
  SmallVector<AffineMap, 3> explicitMaps = getIndexingMapsArray();
  return defaultMaps != explicitMaps;
}

/// Returns true if the given bcastMap map is a valid broadcast map. A valid
/// broadcast map must include K dimension.
/// TODO: Strict inclusion of K dimension in the broadcast map is not
/// necessary for both input matrices simultaneously. We can relax this
/// condition to have K dimension for one input matrix map and infer the K
/// dimension for other input matrix map from the one already having K
/// dimension.
bool BatchMatmulOp::isValidLhsRhsBroadcastMap(AffineMap bcastMap, bool isLHS) {
  assert(bcastMap.getNumResults() < 3 &&
         "Expected less than 3 result dim expr.");
  bool isValid = false;
  enum Indices { batchPos, mPos, nPos, kPos };
  if (bcastMap.getNumResults() == 1) {
    AffineExpr expr = bcastMap.getResult(0);
    isValid = expr.isFunctionOfDim(kPos);
  } else if (bcastMap.getNumResults() == 2) {
    AffineExpr expr0 = bcastMap.getResult(0);
    AffineExpr expr1 = bcastMap.getResult(1);
    isValid =
        isLHS ? ((expr0.isFunctionOfDim(batchPos) ||
                  expr0.isFunctionOfDim(mPos)) &&
                 expr1.isFunctionOfDim(kPos))
              : ((expr0.isFunctionOfDim(batchPos) &&
                  expr1.isFunctionOfDim(kPos)) ||
                 (expr0.isFunctionOfDim(kPos) && expr1.isFunctionOfDim(nPos)));
  }
  return isValid;
}

void BatchMatmulOp::regionBuilder(
    ImplicitLocOpBuilder &b, Block &block, ArrayRef<NamedAttribute> attrs,
    function_ref<InFlightDiagnostic()> emitError) {
  if (emitError && block.getNumArguments() != 3) {
    emitError() << "BatchMatmulOp regionBuilder expects 3 args, got "
                << block.getNumArguments();
    return;
  }
  assert(block.getNumArguments() == 3 &&
         "BatchMatmulOp regionBuilder expects 3 args");
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

  auto toType = block.getArgument(2).getType();
  Value castValA = helper.buildTypeFn(castVal, toType, block.getArgument(0));
  Value castValB = helper.buildTypeFn(castVal, toType, block.getArgument(1));
  Value mulVal = helper.buildBinaryFn(BinaryFn::mul, castValA, castValB);
  Value addVal =
      helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), mulVal);
  yields.push_back(addVal);
  helper.yieldOutputs(yields);
}

ParseResult BatchMatmulOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<Attribute, 3> indexingMapsAttr;
  Attribute mapAttr;
  if (succeeded(parser.parseOptionalKeyword("indexing_maps"))) {
    if (parser.parseEqual())
      return failure();

    if (parser.parseLSquare())
      return failure();

    do {
      if (parser.parseAttribute(mapAttr))
        return failure();
      if (!isa<AffineMapAttr>(mapAttr)) {
        return parser.emitError(parser.getCurrentLocation(),
                                "expected affine map attribute");
      }
      indexingMapsAttr.push_back(mapAttr);

      if (parser.parseOptionalComma())
        break;
    } while (true);

    if (parser.parseRSquare())
      return failure();
  }
  // Initialize indexingMaps, if not supplied explicitly.
  if (indexingMapsAttr.empty()) {
    indexingMapsAttr = llvm::map_to_vector(
        BatchMatmulOp::getDefaultIndexingMaps(parser.getContext()),
        [](AffineMap map) -> Attribute { return AffineMapAttr::get(map); });
  }
  result.addAttribute("indexing_maps",
                      parser.getBuilder().getArrayAttr(indexingMapsAttr));

  return ::parseNamedStructuredOp(parser, result,
                                  BatchMatmulOp::getNumRegionArgs(),
                                  BatchMatmulOp::getRegionBuilder());
}

void BatchMatmulOp::print(OpAsmPrinter &p) {
  SmallVector<Attribute, 3> indexingMaps = llvm::map_to_vector<3>(
      BatchMatmulOp::getDefaultIndexingMaps(getContext()),
      [](AffineMap map) -> Attribute { return AffineMapAttr::get(map); });
  if (!llvm::equal(getIndexingMaps(), indexingMaps))
    p << " indexing_maps = " << llvm::interleaved_array(getIndexingMaps());

  std::array<StringRef, 3> elidedAttrs = {
      "operandSegmentSizes", "linalg.memoized_indexing_maps", "indexing_maps"};
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs(),
                           elidedAttrs);
}

/// Verify the user defined indexing maps.
LogicalResult BatchMatmulOp::verify() {
  // Verification of pure batch_matmul is handled by
  // verifyStructuredOpInterface().
  if (!hasUserDefinedMaps())
    return success();

  for (unsigned opIndex = 0; opIndex < 3; opIndex++) {
    if (failed(verifyExtendedBatchVariantMatmulSemantic(*this, opIndex)))
      return failure();
  }
  return success();
}

LogicalResult BatchMatmulOp::fold(FoldAdaptor,
                                  SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

void BatchMatmulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (hasPureTensorSemantics())
    return;
  getGenericEffectsImpl(effects, cast<LinalgOp>(getOperation()));
}

Speculation::Speculatability BatchMatmulOp::getSpeculatability() {
  return getGenericSpeculatabilityImpl(cast<LinalgOp>(getOperation()));
}

//===----------------------------------------------------------------------===//
// ElementwiseOp
//===----------------------------------------------------------------------===//
//
namespace {
struct ArityGroupAndKind {
  // The enum class {Unary, Binary, Ternary, ..}
  ElementwiseArityGroup arityGroup;

  // The kind (e.g. `exp` or `add`) belonging to the arity group.
  union Kind {
    UnaryFn unaryFn;
    BinaryFn binaryFn;
    TernaryFn ternaryFn;
  } kind;
};

unsigned getArityGroupAsUInt(ElementwiseArityGroup arityGroup) {
  return static_cast<unsigned>(arityGroup);
}
} // namespace

static ArityGroupAndKind getArityGroupAndKind(ElementwiseKind kind) {
  constexpr int lastUnary = static_cast<int>(ElementwiseCaseLimits::LastUnary);
  constexpr int lastBinary =
      static_cast<int>(ElementwiseCaseLimits::LastBinary);
  constexpr int lastTernary =
      static_cast<int>(ElementwiseCaseLimits::LastTernary);

  int val = static_cast<int>(kind);
  ArityGroupAndKind result;

  if (val < lastUnary) {
    result.arityGroup = ElementwiseArityGroup::Unary;
    result.kind.unaryFn = static_cast<UnaryFn>(val);
    return result;
  }
  if (val < lastBinary) {
    result.arityGroup = ElementwiseArityGroup::Binary;
    result.kind.binaryFn = static_cast<BinaryFn>(val - lastUnary);
    return result;
  }
  if (val >= lastTernary) {
    llvm_unreachable("unhandled ElementwiseFn");
  }
  result.arityGroup = ElementwiseArityGroup::Ternary;
  result.kind.ternaryFn = static_cast<TernaryFn>(val - lastBinary);
  return result;
}

SmallVector<utils::IteratorType> ElementwiseOp::getIteratorTypesArray() {
  auto rank = getResultRank();
  return SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel);
}

SmallVector<AffineMap>
ElementwiseOp::getDefaultIndexingMaps(unsigned numMaps, unsigned numDims,
                                      MLIRContext *context) {
  auto map = AffineMap::getMultiDimIdentityMap(numDims, context);
  return SmallVector<AffineMap>(numMaps, map);
}

ParseResult ElementwiseOp::parse(OpAsmParser &parser, OperationState &result) {
  // Expect e.g. `kind = #linalg.elemwise_kind<add>`
  Attribute attr;
  mlir::linalg::ElementwiseKind elemwiseKindVal;
  if (parser.parseKeyword("kind") || parser.parseEqual())
    return failure();

  if (succeeded(parser.parseAttribute(attr))) {
    auto elemwiseKindAttr = dyn_cast<ElementwiseKindAttr>(attr);
    if (!elemwiseKindAttr)
      return parser.emitError(parser.getCurrentLocation(),
                              "expected ElementwiseKind attribute");
    elemwiseKindVal = elemwiseKindAttr.getValue();
  } else {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected operation 'kind' attribute");
  }
  result.addAttribute(
      "kind", ElementwiseKindAttr::get(parser.getContext(), elemwiseKindVal));

  // Parse optional `indexing_maps`
  SmallVector<Attribute, 3> indexingMapsAttr;
  Attribute mapAttr;
  if (succeeded(parser.parseOptionalKeyword("indexing_maps"))) {
    if (parser.parseEqual())
      return failure();
    if (parser.parseLSquare())
      return failure();
    do {
      if (parser.parseAttribute(mapAttr))
        return failure();
      if (!isa<AffineMapAttr>(mapAttr))
        return parser.emitError(parser.getCurrentLocation(),
                                "expected affine map attribute");
      indexingMapsAttr.push_back(mapAttr);
      if (parser.parseOptionalComma())
        break;
    } while (true);
    if (parser.parseRSquare())
      return failure();
  }
  // At this stage of parsing the only way to infer number of region
  // args is through op kind, as input output tensors are not parsed yet.
  auto arityGroupAndKind = getArityGroupAndKind(elemwiseKindVal);
  int numRegionArgs =
      getArityGroupAsUInt(arityGroupAndKind.arityGroup) + 1 /*output*/;
  if (parseNamedStructuredOp(parser, result, numRegionArgs,
                             ElementwiseOp::getRegionBuilder())) {
    return parser.emitError(parser.getCurrentLocation(),
                            "unable to parse elemwise op");
  }

  // Initialize indexingMaps, if not supplied explicitly.
  if (indexingMapsAttr.empty()) {
    // We need to infer the numDims of the indexing maps from the output
    // type which is already parsed by now.
    auto resultType = result.operands[result.operands.size() - 1].getType();
    auto shapedType = llvm::dyn_cast<ShapedType>(resultType);
    if (!shapedType)
      return parser.emitError(parser.getCurrentLocation(),
                              "return type needs to be shaped type");
    auto numDims = shapedType.getRank();
    indexingMapsAttr = llvm::map_to_vector(
        ElementwiseOp::getDefaultIndexingMaps(numRegionArgs, numDims,
                                              parser.getContext()),
        [](AffineMap map) -> Attribute { return AffineMapAttr::get(map); });
  }

  result.addAttribute("indexing_maps",
                      parser.getBuilder().getArrayAttr(indexingMapsAttr));
  return success();
}

void ElementwiseOp::print(OpAsmPrinter &p) {
  p << " kind=";
  p.printAttribute(getKindAttr());
  SmallVector<StringRef, 3> elidedAttrs = {"operandSegmentSizes", "kind",
                                           "indexing_maps"};
  unsigned arity =
      getArityGroupAsUInt(getArityGroupAndKind(getKind()).arityGroup);
  unsigned numDims = getResultRank();

  SmallVector<Attribute, 3> indexingMaps = llvm::map_to_vector<3>(
      ElementwiseOp::getDefaultIndexingMaps(arity + 1 /*output*/, numDims,
                                            getContext()),
      [](AffineMap map) -> Attribute { return AffineMapAttr::get(map); });

  if (!llvm::equal(getIndexingMaps(), indexingMaps))
    p << " indexing_maps = " << llvm::interleaved_array(getIndexingMaps());

  printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs(),
                         elidedAttrs);
}

LogicalResult ElementwiseOp::verify() {
  // All necessary checks are done either by
  // - EnumAttr (e.g. unknown operation kind)
  // - verifyStructuredOpInterface (incorrect map, sizes).
  return success();
}

/// Implements the block region builder for the ElementwiseOp. This is called by
/// 'fillStructuredOpRegion'.
void ElementwiseOp::regionBuilder(
    ImplicitLocOpBuilder &b, Block &block, ArrayRef<NamedAttribute> attrs,
    function_ref<InFlightDiagnostic()> emitError) {
  ElementwiseKind elemwiseKind;
  for (auto attr : attrs) {
    if (attr.getName() == b.getStringAttr("kind")) {
      auto kindAttr = dyn_cast<ElementwiseKindAttr>(attr.getValue());
      assert(kindAttr && "op kind attribute incorrectly set");
      elemwiseKind = kindAttr.getValue();
      break;
    }
  }

  ArityGroupAndKind groupAndKind = getArityGroupAndKind(elemwiseKind);
  auto arityGroup = groupAndKind.arityGroup;
  auto kind = groupAndKind.kind;
  if (emitError && block.getNumArguments() !=
                       getArityGroupAsUInt(arityGroup) + 1 /*output*/) {
    emitError() << "Elementwise regionBuilder expects "
                << (getArityGroupAsUInt(arityGroup) + 1) << " args, got "
                << block.getNumArguments();
    return;
  }
  assert(block.getNumArguments() ==
             getArityGroupAsUInt(arityGroup) + 1 /*output*/
         && "Elementwise regionBuilder number of block args mismatch");

  RegionBuilderHelper helper(b, block);
  SmallVector<Value> yields;
  Value result;

  if (arityGroup == ElementwiseArityGroup::Unary) {
    result = helper.buildUnaryFn(kind.unaryFn, block.getArgument(0));

  } else if (arityGroup == ElementwiseArityGroup::Binary) {
    result = helper.buildBinaryFn(kind.binaryFn, block.getArgument(0),
                                  block.getArgument(1));

  } else if (arityGroup == ElementwiseArityGroup::Ternary) {
    result = helper.buildTernaryFn(kind.ternaryFn, block.getArgument(0),
                                   block.getArgument(1), block.getArgument(2));

  } else {
    assert(false && "found unhandled category in elemwise");
  }

  yields.push_back(result);
  helper.yieldOutputs(yields);
}

LogicalResult ElementwiseOp::fold(FoldAdaptor,
                                  SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

void ElementwiseOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (hasPureTensorSemantics())
    return;
  getGenericEffectsImpl(effects, cast<LinalgOp>(getOperation()));
}

Speculation::Speculatability ElementwiseOp::getSpeculatability() {
  return getGenericSpeculatabilityImpl(cast<LinalgOp>(getOperation()));
}

//===----------------------------------------------------------------------===//
// PackOp/UnPackOp Common
//===----------------------------------------------------------------------===//

template <typename OpTy, typename>
SmallVector<int64_t>
getPackedOuterShapeWithoutTransposition(OpTy packOrUnPack) {
  RankedTensorType packedType = (std::is_same<OpTy, PackOp>::value)
                                    ? packOrUnPack.getDestType()
                                    : packOrUnPack.getSourceType();
  RankedTensorType unpackedType = (std::is_same<OpTy, PackOp>::value)
                                      ? packOrUnPack.getSourceType()
                                      : packOrUnPack.getDestType();
  SmallVector<int64_t> result(
      packedType.getShape().take_front(unpackedType.getRank()));
  if (!packOrUnPack.getOuterDimsPerm().empty()) {
    applyPermutationToVector(
        result, invertPermutationVector(packOrUnPack.getOuterDimsPerm()));
  }
  return result;
}
template SmallVector<int64_t>
    getPackedOuterShapeWithoutTransposition<PackOp>(PackOp);
template SmallVector<int64_t>
    getPackedOuterShapeWithoutTransposition<UnPackOp>(UnPackOp);

// Given the (potentially) updated packed type, `newPackedTy`, generates an
// updated mixed-tile-sizes attribute. A tile size is updated only
// when:
//  * a dim from newPackedTy is static, and
//  * the corresponding size from mixedTiles is still dynamic.
// Otherwise, the original tile size is preserved.
// Note - packed-type-dim and mixed-tile-size should always match!
static SmallVector<OpFoldResult>
getNewMixedTileSizes(PatternRewriter &rewriter, Type newPackedTy,
                     SmallVector<OpFoldResult> mixedTiles) {
  SmallVector<OpFoldResult> newMixedTileSizes;
  for (auto it : llvm::zip(cast<ShapedType>(newPackedTy)
                               .getShape()
                               .take_back(mixedTiles.size()),
                           mixedTiles)) {
    int64_t shape = std::get<0>(it);
    if (shape == ShapedType::kDynamic) {
      newMixedTileSizes.push_back(std::get<1>(it));
      continue;
    }

    // If the current result dim is static, update the dynamic mixed-size
    // (provided the original value is dynamic).
    OpFoldResult tile = std::get<1>(it);
    if (Attribute attr = llvm::dyn_cast_if_present<Attribute>(tile)) {
      // Already a constant
      newMixedTileSizes.push_back(tile);
    } else {
      assert(getConstantIntValue(tile).value() == shape &&
             "tile size and dim size don't match!");
      newMixedTileSizes.push_back(
          (rewriter.getIntegerAttr(rewriter.getIndexType(), shape)));
    }
  }

  return newMixedTileSizes;
}

template <typename OpTy>
static LogicalResult
reifyResultShapesImpl(OpTy op, OpBuilder &builder,
                      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  int64_t destRank = op.getDestRank();
  reifiedReturnShapes.resize(1, SmallVector<OpFoldResult>(destRank));
  reifiedReturnShapes[0] =
      tensor::getMixedSizes(builder, op.getLoc(), op.getDest());
  return success();
}

template <typename OpTy>
static DenseMap<int64_t, OpFoldResult> getDimAndTileMappingImpl(OpTy op) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  DenseMap<int64_t, OpFoldResult> dimAndTileMapping;
  ArrayRef<int64_t> dimsToTile = op.getInnerDimsPos();
  SmallVector<OpFoldResult> tiles = op.getMixedTiles();
  assert(tiles.size() == dimsToTile.size() &&
         "tiles must match indices of dimension to block");
  // bind the dimension `i` with the tile factor.
  for (auto i : llvm::seq<int64_t>(0, dimsToTile.size()))
    dimAndTileMapping[dimsToTile[i]] = tiles[i];
  return dimAndTileMapping;
}

template <typename OpTy>
static SmallVector<OpFoldResult> getMixedTilesImpl(OpTy op) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  Builder builder(op);
  SmallVector<OpFoldResult> mixedInnerTiles;
  unsigned dynamicValIndex = 0;
  for (int64_t staticTile : op.getStaticInnerTiles()) {
    if (ShapedType::isStatic(staticTile))
      mixedInnerTiles.push_back(builder.getI64IntegerAttr(staticTile));
    else
      mixedInnerTiles.push_back(op.getInnerTiles()[dynamicValIndex++]);
  }
  return mixedInnerTiles;
}

template <typename OpTy>
static SmallVector<int64_t> getStaticTilesImpl(OpTy op) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  SmallVector<Value> dynamicTiles;
  SmallVector<int64_t> staticTiles;
  dispatchIndexOpFoldResults(op.getMixedTiles(), dynamicTiles, staticTiles);
  return staticTiles;
}

/// Returns true if `dimsPos` is invalid. It is invalid when:
/// a) It contains duplicate.
/// b) At least one dimension is out of bound (`dimPos` is >= 0 and < rank).
/// c) The number of elements in `dimsPos` is > than `rank`.
static bool isInvalidPackingPosSpecification(ArrayRef<int64_t> dimsPos,
                                             size_t rank) {
  size_t dimsPosSize = dimsPos.size();
  if (dimsPosSize > rank)
    return true;
  DenseSet<int64_t> uniqued(llvm::from_range, dimsPos);
  if (dimsPosSize != uniqued.size())
    return true;
  return llvm::any_of(dimsPos, [rank](int64_t dimPos) {
    return dimPos < 0 || dimPos >= static_cast<int64_t>(rank);
  });
}

template <typename OpTy>
static LogicalResult commonVerifierPackAndUnPackOp(OpTy packOrUnPack) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  Operation *op = packOrUnPack.getOperation();

  // Return true if we have a zero-value tile.
  auto hasZeros = [&](ArrayRef<OpFoldResult> tiles) {
    return llvm::any_of(tiles, isZeroInteger);
  };

  // Verify tiles. Do not allow zero tiles.
  SmallVector<OpFoldResult> mixedTiles = packOrUnPack.getMixedTiles();
  if (hasZeros(mixedTiles))
    return op->emitError("invalid zero tile factor");

  // Verify inner_dims_pos and outer_dims_perm.
  RankedTensorType unpackedType = (std::is_same<OpTy, PackOp>::value)
                                      ? packOrUnPack.getSourceType()
                                      : packOrUnPack.getDestType();
  size_t unpackedRank = unpackedType.getRank();
  ArrayRef<int64_t> innerDimsPos = packOrUnPack.getInnerDimsPos();
  ArrayRef<int64_t> outerDimPerm = packOrUnPack.getOuterDimsPerm();
  if (isInvalidPackingPosSpecification(innerDimsPos, unpackedRank))
    return op->emitError("invalid inner_dims_pos vector");
  if (isInvalidPackingPosSpecification(outerDimPerm, unpackedRank))
    return op->emitError("invalid outer_dims_perm vector");
  if (!outerDimPerm.empty() && outerDimPerm.size() != unpackedRank)
    return op->emitError("outer_dims_perm must be a permutation or empty");

  // Tiling factors must be less than or equal to the input rank for pack (or
  // output rank for unpack), and must match the number of `inner_dims_pos`.
  if (mixedTiles.size() > unpackedRank) {
    return op->emitError("tiling factors must be less than or equal to the "
                         "input rank for pack or output rank for unpack");
  }
  if (mixedTiles.size() != innerDimsPos.size()) {
    return op->emitError(
        "tiling factors must equal the number of dimensions to tile");
  }

  ShapedType packedType = (std::is_same<OpTy, PackOp>::value)
                              ? packOrUnPack.getDestType()
                              : packOrUnPack.getSourceType();
  size_t packedRank = packedType.getRank();
  // Require output rank to match input rank + number of blocking factors.
  size_t expectedPackedRank = unpackedRank + mixedTiles.size();
  if (expectedPackedRank != packedRank) {
    return op->emitError(
               "packed rank != (unpacked rank + num tiling factors), got ")
           << packedRank << " != " << expectedPackedRank;
  }

  // Verify result shape is greater than the minimum expected
  // by the pack operation, and that the output shape
  // represents full tiles.
  RankedTensorType expectedPackedType = PackOp::inferPackedType(
      unpackedType, packOrUnPack.getStaticTiles(), innerDimsPos, outerDimPerm);
  if (!llvm::all_of(
          llvm::zip(packedType.getShape().take_back(mixedTiles.size()),
                    mixedTiles),
          [](std::tuple<int64_t, OpFoldResult> it) {
            int64_t shape = std::get<0>(it);
            if (Attribute attr =
                    llvm::dyn_cast_if_present<Attribute>(std::get<1>(it))) {
              IntegerAttr intAttr = dyn_cast_or_null<IntegerAttr>(attr);
              int64_t staticTileSize = intAttr.getValue().getSExtValue();
              return shape == staticTileSize;
            }
            return ShapedType::isDynamic(shape);
          })) {
    return op->emitError("mismatch in inner tile sizes specified and shaped of "
                         "tiled dimension in the packed type");
  }
  if (failed(verifyCompatibleShape(expectedPackedType.getShape(),
                                   packedType.getShape()))) {
    return op->emitError("expected ")
           << expectedPackedType << " for the packed domain value, got "
           << packedType;
  }
  return success();
}

namespace {
/// Subset of PackOp/UnPackOp fields used to compute the result of applying
/// various permutations to the op.
// TODO: Add linalg.transpose + pack/unpack folding patterns that just reuse
// these. These may or may not become true foldings / canonicalizations
// depending on how aggressive we want to be in automatically folding
// transposes.
struct PackOrUnPackTransposeResult {
  SmallVector<int64_t> innerDimsPos;
  SmallVector<OpFoldResult> innerTiles;
  SmallVector<int64_t> outerDimsPerm;
};
} // namespace

template <typename OpTy>
static PackOrUnPackTransposeResult
commonPermutationOfPackAndUnPackOp(OpTy packOrUnPackOp,
                                   ArrayRef<int64_t> innerPermutation,
                                   ArrayRef<int64_t> outerPermutation) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  assert((!innerPermutation.empty() || !outerPermutation.empty()) &&
         "some permutation must be non-empty");
  PackOrUnPackTransposeResult metadata;
  metadata.innerDimsPos =
      SmallVector<int64_t>(packOrUnPackOp.getInnerDimsPos());
  metadata.innerTiles =
      SmallVector<OpFoldResult>(packOrUnPackOp.getMixedTiles());
  int64_t numOuterDims = std::is_same<OpTy, PackOp>::value
                             ? packOrUnPackOp.getSourceRank()
                             : packOrUnPackOp.getDestRank();
  metadata.outerDimsPerm =
      packOrUnPackOp.getOuterDimsPerm().empty()
          ? llvm::to_vector(llvm::seq<int64_t>(0, numOuterDims))
          : SmallVector<int64_t>(packOrUnPackOp.getOuterDimsPerm());
  if (!innerPermutation.empty()) {
    assert(innerPermutation.size() == metadata.innerDimsPos.size() &&
           isPermutationVector(innerPermutation) &&
           "invalid inner permutation");
    applyPermutationToVector(metadata.innerDimsPos, innerPermutation);
    applyPermutationToVector(metadata.innerTiles, innerPermutation);
  }
  if (!outerPermutation.empty()) {
    assert(outerPermutation.size() == metadata.outerDimsPerm.size() &&
           isPermutationVector(outerPermutation) &&
           "invalid outer permutation");
    applyPermutationToVector(metadata.outerDimsPerm, outerPermutation);
  }
  return metadata;
}

//===----------------------------------------------------------------------===//
// PackOp
//===----------------------------------------------------------------------===//

void PackOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "pack");
}

void PackOp::build(OpBuilder &builder, OperationState &state, Value source,
                   Value dest, ArrayRef<int64_t> innerDimsPos,
                   ArrayRef<OpFoldResult> innerTiles,
                   std::optional<Value> paddingValue,
                   ArrayRef<int64_t> outerDimsPerm) {
  assert(innerDimsPos.size() == innerTiles.size() &&
         "number of tile sizes specified must match the specified number of "
         "original dimensions to be tiled");
  SmallVector<int64_t> staticTileSizes;
  SmallVector<Value> dynamicTileSizes;
  dispatchIndexOpFoldResults(innerTiles, dynamicTileSizes, staticTileSizes);
  build(builder, state, dest.getType(), source, dest,
        paddingValue ? *paddingValue : nullptr,
        outerDimsPerm.empty() ? nullptr
                              : builder.getDenseI64ArrayAttr(outerDimsPerm),
        builder.getDenseI64ArrayAttr(innerDimsPos), dynamicTileSizes,
        builder.getDenseI64ArrayAttr(staticTileSizes));
}

LogicalResult
PackOp::reifyResultShapes(OpBuilder &builder,
                          ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return reifyResultShapesImpl(*this, builder, reifiedReturnShapes);
}

DenseMap<int64_t, OpFoldResult> PackOp::getDimAndTileMapping() {
  return getDimAndTileMappingImpl(*this);
}

SmallVector<OpFoldResult> PackOp::getMixedTiles() {
  return getMixedTilesImpl(*this);
}

SmallVector<int64_t> PackOp::getStaticTiles() {
  return getStaticTilesImpl(*this);
}

ArrayRef<int64_t> PackOp::getAllOuterDims() {
  ShapedType inputType = getSourceType();
  int64_t inputRank = inputType.getRank();
  return getDestType().getShape().take_front(inputRank);
}

SmallVector<int64_t> PackOp::getTiledOuterDims() {
  auto innerDimsPos = getInnerDimsPos();
  auto packedShape = getDestType().getShape();
  SmallVector<int64_t> res;

  for (auto index : innerDimsPos)
    res.push_back(packedShape[index]);

  return res;
}

bool PackOp::requirePaddingValue(ArrayRef<int64_t> inputShape,
                                 ArrayRef<int64_t> innerDimsPos,
                                 ArrayRef<int64_t> outputShape,
                                 ArrayRef<int64_t> outerDimsPerm,
                                 ArrayRef<OpFoldResult> innerTiles) {
  SmallVector<int64_t> outputTileSizes(
      outputShape.take_front(inputShape.size()));
  if (!outerDimsPerm.empty()) {
    assert(outerDimsPerm.size() == outputTileSizes.size() &&
           "expected output and outer_dims_perm to have same size");
    applyPermutationToVector(outputTileSizes,
                             invertPermutationVector(outerDimsPerm));
  }
  for (auto [pos, tileSize] : llvm::zip_equal(innerDimsPos, innerTiles)) {
    if (ShapedType::isDynamic(inputShape[pos]))
      continue;
    std::optional<int64_t> constantTile = getConstantIntValue(tileSize);

    if (!constantTile) {
      if (ShapedType::isStatic(outputTileSizes[pos]) &&
          (inputShape[pos] % outputTileSizes[pos] != 0))
        return true;
    } else if (inputShape[pos] % (*constantTile) != 0) {
      return true;
    }
  }
  return false;
}

LogicalResult PackOp::verify() {
  if (failed(commonVerifierPackAndUnPackOp(*this)))
    return failure();

  // Verify padding value, and bail out if the tile does not divide the
  // dimension fully. In the case of dynamic tile factors or dimensions, having
  // a partial tile is undefined behavior.
  auto paddingValue = getPaddingValue();
  if (paddingValue &&
      paddingValue.getType() != getSourceType().getElementType()) {
    return emitOpError("expected padding_value has ")
           << getSourceType().getElementType()
           << " but got: " << paddingValue.getType();
  }

  if (!paddingValue &&
      requirePaddingValue(getSourceType().getShape(), getInnerDimsPos(),
                          getDestType().getShape(), getOuterDimsPerm(),
                          getMixedTiles())) {
    return emitOpError(
        "invalid tile factor or output size provided. Only full tiles are "
        "supported when padding_value is not set");
  }
  return success();
}

/// Converts OpFoldResults to int64_t shape entries, unconditionally mapping all
/// Value's to kDynamic, even if they are arith.constant values.
static SmallVector<int64_t>
asShapeWithAnyValueAsDynamic(ArrayRef<OpFoldResult> ofrs) {
  SmallVector<int64_t> result;
  for (auto o : ofrs) {
    // Have to do this first, as getConstantIntValue special-cases constants.
    if (llvm::dyn_cast_if_present<Value>(o))
      result.push_back(ShapedType::kDynamic);
    else
      result.push_back(getConstantIntValue(o).value_or(ShapedType::kDynamic));
  }
  return result;
}

/// Helper for PackOp::{getResultShape,inferPackedType}. Returns the shape of
/// the packed type. Having a shared helper helps implement these two methods in
/// a way that ensures that they agree on which dimensions are dynamic.
static SmallVector<int64_t> getPackOpResultTypeShape(
    ArrayRef<int64_t> sourceShape, ArrayRef<int64_t> innerTileSizes,
    ArrayRef<int64_t> innerDimsPos, ArrayRef<int64_t> outerDimsPerm) {
  SmallVector<int64_t> resultShape = llvm::to_vector(sourceShape);
  for (auto tiledDim : llvm::enumerate(llvm::to_vector(innerDimsPos))) {
    if (ShapedType::isDynamic(resultShape[tiledDim.value()]))
      continue;
    if (ShapedType::isDynamic(innerTileSizes[tiledDim.index()])) {
      resultShape[tiledDim.value()] = ShapedType::kDynamic;
      continue;
    }
    resultShape[tiledDim.value()] = llvm::divideCeilSigned(
        resultShape[tiledDim.value()], innerTileSizes[tiledDim.index()]);
  }

  // Swap tile loops if outer_dims_perm is available.
  if (!outerDimsPerm.empty())
    applyPermutationToVector(resultShape, outerDimsPerm);

  // Append the inner tile dimensions.
  resultShape.append(innerTileSizes.begin(), innerTileSizes.end());
  return resultShape;
}

SmallVector<OpFoldResult> PackOp::getResultShape(
    OpBuilder &builder, Location loc, ArrayRef<OpFoldResult> sourceDims,
    ArrayRef<OpFoldResult> innerTileSizes, ArrayRef<int64_t> innerDimsPos,
    ArrayRef<int64_t> outerDimsPerm) {
  SmallVector<OpFoldResult> resultDims = llvm::to_vector(sourceDims);

  AffineExpr s0, s1;
  bindSymbols(builder.getContext(), s0, s1);
  AffineExpr ceilDivExpr = s0.ceilDiv(s1);
  for (auto tiledDim : llvm::enumerate(llvm::to_vector(innerDimsPos))) {
    resultDims[tiledDim.value()] = affine::makeComposedFoldedAffineApply(
        builder, loc, ceilDivExpr,
        {resultDims[tiledDim.value()], innerTileSizes[tiledDim.index()]});
  }
  if (!outerDimsPerm.empty())
    applyPermutationToVector(resultDims, outerDimsPerm);
  resultDims.append(innerTileSizes.begin(), innerTileSizes.end());

  SmallVector<int64_t> resultTypeShape =
      getPackOpResultTypeShape(asShapeWithAnyValueAsDynamic(sourceDims),
                               asShapeWithAnyValueAsDynamic(innerTileSizes),
                               innerDimsPos, outerDimsPerm);

  // Fix-up `resultDims` to ensure that they are Value's if and only if the
  // result type shape says it's a dynamic dim. This is needed as callers may
  // use dispatchIndexOpFoldResults on the result, and rely on exact number of
  // dynamic dims returned by that.
  for (unsigned i = 0; i < resultDims.size(); ++i) {
    if (ShapedType::isStatic(resultTypeShape[i]))
      continue;
    resultDims[i] =
        getValueOrCreateConstantIndexOp(builder, loc, resultDims[i]);
  }

  return resultDims;
}

/// Get the expected packed type based on source type, tile factors, position of
/// the inner tiles and permutation of the outer tiled loop.
RankedTensorType PackOp::inferPackedType(RankedTensorType sourceType,
                                         ArrayRef<int64_t> innerTileSizes,
                                         ArrayRef<int64_t> innerDimsPos,
                                         ArrayRef<int64_t> outerDimsPerm) {
  SmallVector<int64_t> resultShape = getPackOpResultTypeShape(
      sourceType.getShape(), innerTileSizes, innerDimsPos, outerDimsPerm);
  return RankedTensorType::get(resultShape, sourceType.getElementType());
}

Value PackOp::createDestinationTensor(OpBuilder &b, Location loc, Value source,
                                      ArrayRef<OpFoldResult> innerTileSizes,
                                      ArrayRef<int64_t> innerDimsPos,
                                      ArrayRef<int64_t> outerDimsPerm) {
  AffineExpr dim0, dim1;
  bindDims(b.getContext(), dim0, dim1);
  auto ceilDiv = [&](OpFoldResult v1, OpFoldResult v2) -> OpFoldResult {
    return affine::makeComposedFoldedAffineApply(b, loc, dim0.ceilDiv(dim1),
                                                 {v1, v2});
  };

  SmallVector<OpFoldResult> mixedSizes;
  for (auto [index, value] : llvm::enumerate(
           llvm::cast<RankedTensorType>(source.getType()).getShape())) {
    if (ShapedType::isDynamic(value))
      mixedSizes.push_back(
          tensor::DimOp::create(b, loc, source, index).getResult());
    else
      mixedSizes.push_back(b.getIndexAttr(value));
  }
  for (auto it : llvm::zip(innerDimsPos, innerTileSizes)) {
    int64_t dimPos = std::get<0>(it);
    OpFoldResult tileSize = std::get<1>(it);
    mixedSizes[dimPos] = ceilDiv(mixedSizes[dimPos], tileSize);
  }
  if (!outerDimsPerm.empty())
    applyPermutationToVector<OpFoldResult>(mixedSizes, outerDimsPerm);

  mixedSizes.append(innerTileSizes.begin(), innerTileSizes.end());
  auto elemType = llvm::cast<ShapedType>(source.getType()).getElementType();
  return tensor::EmptyOp::create(b, loc, mixedSizes, elemType);
}

PackOp PackOp::createTransposedClone(OpBuilder &b, Location loc,
                                     ArrayRef<int64_t> innerPermutation,
                                     ArrayRef<int64_t> outerPermutation) {
  PackOrUnPackTransposeResult metadata = commonPermutationOfPackAndUnPackOp(
      *this, innerPermutation, outerPermutation);
  Value transposedDest =
      createDestinationTensor(b, loc, getSource(), metadata.innerTiles,
                              metadata.innerDimsPos, metadata.outerDimsPerm);
  return PackOp::create(b, loc, getSource(), transposedDest,
                        metadata.innerDimsPos, metadata.innerTiles,
                        getPaddingValue(), metadata.outerDimsPerm);
}

/// Returns true if the tiles and the tiled dims are constant.
template <typename OpTy>
bool areTilesAndTiledDimsAllConstant(OpTy op) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  ShapedType packedType = (std::is_same<OpTy, PackOp>::value)
                              ? op.getDestType()
                              : op.getSourceType();
  SmallVector<OpFoldResult> mixedTiles = op.getMixedTiles();
  for (auto [dimDest, tile] : llvm::zip(
           packedType.getShape().take_back(mixedTiles.size()), mixedTiles)) {
    std::optional<int64_t> constTileSize = getConstantIntValue(tile);
    if (!constTileSize || ShapedType::isDynamic(dimDest))
      return false;
  }
  return true;
}

Speculation::Speculatability PackOp::getSpeculatability() {
  if (getPaddingValue())
    return Speculation::Speculatable;

  // The verifier rejects already operations if we can statically prove that the
  // sizes of the tiles do not divide perfectly the dimension; thus, check only
  // to have constant tiles and tiled inner dimensions.
  if (!areTilesAndTiledDimsAllConstant(*this))
    return Speculation::NotSpeculatable;

  return Speculation::Speculatable;
}

// Return true if `inner_dims_pos` and `outer_dims_perm` target the same
// dimensions for pack and unpack.
static bool hasSameInnerOuterAttribute(PackOp packOp, UnPackOp unPackOp) {
  if (packOp.getInnerDimsPos() != unPackOp.getInnerDimsPos())
    return false;
  if (packOp.getOuterDimsPerm() == unPackOp.getOuterDimsPerm())
    return true;
  // Outer dims permutation is optional.
  // To compare unbalanced pack-unpack pair, treat no permutation as equal to
  // identity permutation.
  return isIdentityPermutation(packOp.getOuterDimsPerm()) &&
         isIdentityPermutation(unPackOp.getOuterDimsPerm());
}

// Return true if pack and unpack have the same tiles.
// Same SSA values or same integer constants.
static bool haveSameTiles(PackOp packOp, UnPackOp unPackOp) {
  auto packTiles = packOp.getMixedTiles();
  auto unPackTiles = unPackOp.getMixedTiles();
  if (packTiles.size() != unPackTiles.size())
    return false;
  for (size_t i = 0, e = packTiles.size(); i < e; i++) {
    if (!isEqualConstantIntOrValue(packTiles[i], unPackTiles[i]))
      return false;
  }
  return true;
}

/// Returns true if the pack op does not need a padding value.
static bool paddingIsNotNeeded(PackOp op) {
  auto srcType = op.getSourceType();
  if (llvm::any_of(op.getInnerDimsPos(),
                   [&](int64_t pos) { return srcType.isDynamicDim(pos); }))
    return false;
  if (ShapedType::isDynamicShape(op.getStaticInnerTiles()))
    return false;
  return !PackOp::requirePaddingValue(
      srcType.getShape(), op.getInnerDimsPos(), op.getDestType().getShape(),
      op.getOuterDimsPerm(), op.getMixedTiles());
}

/// Returns true if the `srcShape` or `destShape` is different from the one in
/// `packOp` and populates each with the inferred static shape.
static bool inferStaticShape(PackOp packOp, SmallVectorImpl<int64_t> &srcShape,
                             SmallVectorImpl<int64_t> &destShape) {
  bool changeNeeded = false;
  srcShape.assign(packOp.getSourceType().getShape().begin(),
                  packOp.getSourceType().getShape().end());
  destShape.assign(packOp.getDestType().getShape().begin(),
                   packOp.getDestType().getShape().end());
  llvm::SmallSetVector<int64_t, 4> innerDims;
  innerDims.insert_range(packOp.getInnerDimsPos());
  SmallVector<int64_t> inverseOuterDimsPerm;
  if (!packOp.getOuterDimsPerm().empty())
    inverseOuterDimsPerm = invertPermutationVector(packOp.getOuterDimsPerm());
  int srcRank = packOp.getSourceRank();
  for (auto i : llvm::seq<int64_t>(0, srcRank)) {
    if (innerDims.contains(i))
      continue;
    int64_t srcPos = i;
    int64_t destPos = i;
    if (!inverseOuterDimsPerm.empty())
      destPos = inverseOuterDimsPerm[srcPos];
    if (ShapedType::isDynamic(srcShape[srcPos]) ==
        ShapedType::isDynamic(destShape[destPos])) {
      continue;
    }
    int64_t size = srcShape[srcPos];
    if (ShapedType::isDynamic(size))
      size = destShape[destPos];
    srcShape[srcPos] = size;
    destShape[destPos] = size;
    changeNeeded = true;
  }
  return changeNeeded;
}

LogicalResult PackOp::canonicalize(PackOp packOp, PatternRewriter &rewriter) {
  // Fold an pack(unpack(x)) to x.
  if (auto unPackOp = packOp.getSource().getDefiningOp<UnPackOp>()) {
    if (unPackOp.getSourceType() != packOp.getDestType())
      return failure();
    if (packOp.getPaddingValue() ||
        !hasSameInnerOuterAttribute(packOp, unPackOp) ||
        !haveSameTiles(packOp, unPackOp))
      return failure();
    rewriter.replaceOp(packOp, unPackOp.getSource());
    return success();
  }

  // Fold optional PaddingValue operand away if padding is not needed.
  if (packOp.getPaddingValue() && paddingIsNotNeeded(packOp)) {
    rewriter.startOpModification(packOp);
    packOp.getPaddingValueMutable().clear();
    rewriter.finalizeOpModification(packOp);
    return success();
  }

  // Insert tensor.cast ops if static shape inference is available..
  SmallVector<int64_t> srcShape, destShape;
  if (inferStaticShape(packOp, srcShape, destShape)) {
    Location loc = packOp.getLoc();
    Value source = packOp.getSource();
    if (srcShape != packOp.getSourceType().getShape()) {
      auto newSrcType = packOp.getSourceType().clone(srcShape);
      source =
          tensor::CastOp::create(rewriter, loc, newSrcType, packOp.getSource());
    }
    Value dest = packOp.getDest();
    RankedTensorType originalResultType = packOp.getDestType();
    bool needUpdateDestType = (destShape != originalResultType.getShape());
    if (needUpdateDestType) {
      auto newDestType = packOp.getDestType().clone(destShape);
      dest =
          tensor::CastOp::create(rewriter, loc, newDestType, packOp.getDest());
    }
    rewriter.modifyOpInPlace(packOp, [&] {
      packOp.getSourceMutable().assign(source);
      packOp.getDestMutable().assign(dest);
      packOp.getResult().setType(cast<RankedTensorType>(dest.getType()));
    });
    // Insert a cast if needed
    if (needUpdateDestType) {
      rewriter.setInsertionPointAfter(packOp);
      auto castOp =
          tensor::CastOp::create(rewriter, loc, originalResultType, packOp);
      rewriter.replaceAllUsesExcept(packOp, castOp, castOp);
    }
    return success();
  }

  return failure();
}

template <typename PackOrUnpackOp>
static bool isLikePadUnPad(PackOrUnpackOp packOp,
                           RankedTensorType packedTensorType) {
  static_assert(std::is_same<PackOrUnpackOp, PackOp>::value ||
                    std::is_same<PackOrUnpackOp, UnPackOp>::value,
                "Function meant for pack/unpack");
  // This is a pad if packing only adds ones and we don't transpose dimensions.

  // Check that we are not transposing any dimensions.
  ArrayRef<int64_t> innerDimsPos = packOp.getInnerDimsPos();
  int64_t numPackedDims = innerDimsPos.size();
  auto orderedDims = llvm::to_vector<4>(llvm::seq<int64_t>(0, numPackedDims));
  if (orderedDims != innerDimsPos) {
    // Dimensions don't happen in order.
    return false;
  }

  ArrayRef<int64_t> packedShape = packedTensorType.getShape();
  int64_t packedRank = packedTensorType.getRank();
  // At this point we know that we are taking numPackedDims outer
  // dimensions and pushing them all the way as the inner most dimensions.
  // What's left on the outer most dimensions is, in this order:
  // - the factor of the packed dimensions, then
  // - the untouched dimensions
  // This shifting inward of dimensions is a no-op (as opposed to a transpose)
  // if all the dimensions that bubble outerward are ones.
  // Therefore check that all the dimensions but the numPackedDims inner most
  // ones are ones.
  return llvm::all_of(
      llvm::seq<int64_t>(0, packedRank - numPackedDims),
      [&packedShape](int64_t i) { return packedShape[i] == 1; });
}

bool PackOp::isLikePad() {
  auto packedTensorType =
      llvm::cast<RankedTensorType>((*this)->getResultTypes().front());
  return isLikePadUnPad(*this, packedTensorType);
}

OpFoldResult PackOp::fold(FoldAdaptor adaptor) {
  std::optional<Attribute> paddingValue;
  if (auto pad = adaptor.getPaddingValue())
    paddingValue = pad;
  if (OpFoldResult reshapedSource = reshapeConstantSource(
          llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getSource()),
          getDestType(), paddingValue))
    return reshapedSource;
  return {};
}

/// Folds a tensor.cast op into a consuming PackOp op if the
/// `tensor.cast` has source that is more static than the consuming op.
///
/// Example:
/// ```mlir
///   %1 = tensor.cast %0 : tensor<8x16xf32> to tensor<?x?xf32>
///   %2 = tensor.pack %1 ... : tensor<?x?xf32> ...
/// ```
///
/// folds into:
///
/// ```mlir
///   %2 = tensor.pack %0 ... : tensor<8x16xf32> ...
/// ```
struct FoldTensorCastPackOp : public OpRewritePattern<PackOp> {
  using OpRewritePattern<PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PackOp op,
                                PatternRewriter &rewriter) const override {
    if (!tensor::hasFoldableTensorCastOperand(op))
      return failure();

    SmallVector<Type> newResultTypes(op->getResultTypes());
    SmallVector<Value> newOperands =
        tensor::getUpdatedOperandsAfterCastOpFolding(op, newResultTypes);

    // Get the updated mixed-tile-sizes attribute.
    SmallVector<OpFoldResult> newMixedTileSizes =
        getNewMixedTileSizes(rewriter, newResultTypes[0], op.getMixedTiles());

    // Clone op.
    // TODO: Strictly speaking, discardable attributes should be _discarded_ at
    // this point. However, in practice, we use them for things that we'd like
    // to preserve. Implement a better abstraction.
    PackOp newOp =
        PackOp::create(rewriter, op.getLoc(), newOperands[0], newOperands[1],
                       op.getInnerDimsPos(), newMixedTileSizes,
                       op.getPaddingValue(), op.getOuterDimsPerm());
    newOp->setDiscardableAttrs(op->getDiscardableAttrDictionary());

    // Replace op.
    Value oldResult = op.getResult();
    Value newResult = newOp.getResult();
    Value replacement =
        (newResult.getType() != oldResult.getType())
            ? tensor::CastOp::create(rewriter, op->getLoc(),
                                     oldResult.getType(), newResult)
            : newResult;

    rewriter.replaceOp(op, {replacement});

    return success();
  }
};

//===----------------------------------------------------------------------===//
// UnPackOp
//===----------------------------------------------------------------------===//

void UnPackOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "unpack");
}

LogicalResult
UnPackOp::reifyResultShapes(OpBuilder &builder,
                            ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return reifyResultShapesImpl(*this, builder, reifiedReturnShapes);
}

DenseMap<int64_t, OpFoldResult> UnPackOp::getDimAndTileMapping() {
  return getDimAndTileMappingImpl(*this);
}

SmallVector<OpFoldResult> UnPackOp::getMixedTiles() {
  return getMixedTilesImpl(*this);
}

SmallVector<int64_t> UnPackOp::getStaticTiles() {
  return getStaticTilesImpl(*this);
}

ArrayRef<int64_t> UnPackOp::getAllOuterDims() {
  ShapedType destType = getDestType();
  int64_t destRank = destType.getRank();
  return getSourceType().getShape().take_front(destRank);
}

SmallVector<int64_t> UnPackOp::getTiledOuterDims() {
  auto innerDimsPos = getInnerDimsPos();
  SmallVector<int64_t> outerDims(getAllOuterDims());
  SmallVector<int64_t> res;

  // Recover the original order of the outer dims.
  SmallVector<int64_t> outerDimPermInv(getOuterDimsPerm());
  invertPermutationVector(outerDimPermInv);
  if (!outerDimPermInv.empty())
    applyPermutationToVector(outerDims, outerDimPermInv);

  // Collect the outer dims corresponding to the tilled inner dims.
  for (auto index : innerDimsPos)
    res.push_back(outerDims[index]);

  return res;
}

LogicalResult UnPackOp::verify() {
  return commonVerifierPackAndUnPackOp(*this);
}

Speculation::Speculatability UnPackOp::getSpeculatability() {
  // See PackOp::getSpeculatability.
  if (!areTilesAndTiledDimsAllConstant(*this))
    return Speculation::NotSpeculatable;

  return Speculation::Speculatable;
}

void UnPackOp::build(OpBuilder &builder, OperationState &state, Value source,
                     Value dest, ArrayRef<int64_t> innerDimsPos,
                     ArrayRef<OpFoldResult> innerTiles,
                     ArrayRef<int64_t> outerDimsPerm) {
  assert(innerDimsPos.size() == innerTiles.size() &&
         "number of tile sizes specified must match the specified number of "
         "original dimensions to be tiled");
  SmallVector<int64_t> staticTileSizes;
  SmallVector<Value> dynamicTileSizes;
  dispatchIndexOpFoldResults(innerTiles, dynamicTileSizes, staticTileSizes);
  build(builder, state, dest.getType(), source, dest,
        outerDimsPerm.empty() ? nullptr
                              : builder.getDenseI64ArrayAttr(outerDimsPerm),
        builder.getDenseI64ArrayAttr(innerDimsPos), dynamicTileSizes,
        builder.getDenseI64ArrayAttr(staticTileSizes));
}

Value UnPackOp::createDestinationTensor(OpBuilder &b, Location loc,
                                        Value source,
                                        ArrayRef<OpFoldResult> innerTileSizes,
                                        ArrayRef<int64_t> innerDimsPos,
                                        ArrayRef<int64_t> outerDimsPerm) {
  AffineExpr sym0, sym1;
  bindSymbols(b.getContext(), sym0, sym1);
  auto dimMul = [&](OpFoldResult v1, OpFoldResult v2) -> OpFoldResult {
    return affine::makeComposedFoldedAffineApply(b, loc, sym0 * sym1, {v1, v2});
  };

  SmallVector<OpFoldResult> mixedSizes;
  auto srcType = llvm::cast<RankedTensorType>(source.getType());
  for (auto i :
       llvm::seq<unsigned>(0, srcType.getRank() - innerTileSizes.size())) {
    if (srcType.isDynamicDim(i))
      mixedSizes.push_back(
          tensor::DimOp::create(b, loc, source, i).getResult());
    else
      mixedSizes.push_back(b.getIndexAttr(srcType.getDimSize(i)));
  }
  if (!outerDimsPerm.empty()) {
    applyPermutationToVector<OpFoldResult>(
        mixedSizes, invertPermutationVector(outerDimsPerm));
  }

  for (auto [dimPos, tileSize] : llvm::zip_equal(innerDimsPos, innerTileSizes))
    mixedSizes[dimPos] = dimMul(mixedSizes[dimPos], tileSize);

  auto elemType = srcType.getElementType();
  return tensor::EmptyOp::create(b, loc, mixedSizes, elemType);
}

UnPackOp UnPackOp::createTransposedClone(OpBuilder &b, Location loc,
                                         Value transposedSource,
                                         ArrayRef<int64_t> innerPermutation,
                                         ArrayRef<int64_t> outerPermutation) {
  PackOrUnPackTransposeResult metadata = commonPermutationOfPackAndUnPackOp(
      *this, innerPermutation, outerPermutation);
  return UnPackOp::create(b, loc, transposedSource, getDest(),
                          metadata.innerDimsPos, metadata.innerTiles,
                          metadata.outerDimsPerm);
}

/// Returns true if the `srcShape` or `destShape` is different from the one in
/// `op` and populates each with the inferred static shape.
static bool inferStaticShape(UnPackOp op, SmallVectorImpl<int64_t> &srcShape,
                             SmallVectorImpl<int64_t> &destShape) {
  bool changeNeeded = false;
  srcShape.assign(op.getSourceType().getShape().begin(),
                  op.getSourceType().getShape().end());
  destShape.assign(op.getDestType().getShape().begin(),
                   op.getDestType().getShape().end());
  llvm::SmallSetVector<int64_t, 4> innerDims;
  innerDims.insert_range(op.getInnerDimsPos());
  SmallVector<int64_t> inverseOuterDimsPerm;
  if (!op.getOuterDimsPerm().empty())
    inverseOuterDimsPerm = invertPermutationVector(op.getOuterDimsPerm());
  int destRank = op.getDestRank();
  for (auto i : llvm::seq<int64_t>(0, destRank)) {
    if (innerDims.contains(i))
      continue;
    int64_t srcPos = i;
    int64_t destPos = i;
    if (!inverseOuterDimsPerm.empty())
      srcPos = inverseOuterDimsPerm[destPos];
    if (ShapedType::isDynamic(srcShape[srcPos]) ==
        ShapedType::isDynamic(destShape[destPos])) {
      continue;
    }
    int64_t size = srcShape[srcPos];
    if (ShapedType::isDynamic(size))
      size = destShape[destPos];
    srcShape[srcPos] = size;
    destShape[destPos] = size;
    changeNeeded = true;
  }
  return changeNeeded;
}

LogicalResult UnPackOp::canonicalize(UnPackOp unPackOp,
                                     PatternRewriter &rewriter) {
  /// unpack(pack(x)) -> x
  if (PackOp packOp = unPackOp.getSource().getDefiningOp<PackOp>()) {
    if (packOp.getSourceType() != unPackOp.getDestType())
      return failure();
    if (packOp.getPaddingValue() ||
        !hasSameInnerOuterAttribute(packOp, unPackOp) ||
        !haveSameTiles(packOp, unPackOp))
      return failure();
    rewriter.replaceOp(unPackOp, packOp.getSource());
    return success();
  }
  /// unpack(destinationStyleOp(x)) -> unpack(x)
  if (auto dstStyleOp =
          unPackOp.getDest().getDefiningOp<DestinationStyleOpInterface>()) {
    auto destValue = cast<OpResult>(unPackOp.getDest());
    Value newDest = dstStyleOp.getDpsInits()[destValue.getResultNumber()];
    rewriter.modifyOpInPlace(unPackOp,
                             [&]() { unPackOp.setDpsInitOperand(0, newDest); });
    return success();
  }
  /// extract_slice(unpack(x into y)) -> unpack(x into extract_slice(y))
  if (unPackOp->hasOneUse()) {
    auto extractSliceUser =
        dyn_cast<tensor::ExtractSliceOp>(*unPackOp->getUsers().begin());
    if (extractSliceUser && unPackOp.canFoldSliceOp(extractSliceUser)) {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(unPackOp);
      auto newDest = tensor::ExtractSliceOp::create(
          rewriter, unPackOp->getLoc(), unPackOp.getDest(),
          extractSliceUser.getMixedOffsets(), extractSliceUser.getMixedSizes(),
          extractSliceUser.getMixedStrides());
      rewriter.modifyOpInPlace(unPackOp, [&]() {
        unPackOp.setDpsInitOperand(0, newDest);
        unPackOp.getResult().setType(newDest.getType());
      });
      rewriter.replaceOp(extractSliceUser, unPackOp);
      return success();
    }
  }

  // Insert tensor.cast ops if static shape inference is available..
  SmallVector<int64_t> srcShape, destShape;
  if (inferStaticShape(unPackOp, srcShape, destShape)) {
    Location loc = unPackOp.getLoc();
    Value source = unPackOp.getSource();
    if (srcShape != unPackOp.getSourceType().getShape()) {
      auto newSrcType = unPackOp.getSourceType().clone(srcShape);
      source = tensor::CastOp::create(rewriter, loc, newSrcType,
                                      unPackOp.getSource());
    }
    Value dest = unPackOp.getDest();
    if (destShape != unPackOp.getDestType().getShape()) {
      auto newDestType = unPackOp.getDestType().clone(destShape);
      dest = tensor::CastOp::create(rewriter, loc, newDestType,
                                    unPackOp.getDest());
    }
    Value newOp = UnPackOp::create(
        rewriter, loc, source, dest, unPackOp.getInnerDimsPos(),
        unPackOp.getMixedTiles(), unPackOp.getOuterDimsPerm());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(
        unPackOp, unPackOp.getResult().getType(), newOp);
    return success();
  }

  return failure();
}

bool UnPackOp::canFoldSliceOp(tensor::ExtractSliceOp sliceOp) {
  // Rank-reduced folding is not supported.
  if (sliceOp.getResultType().getRank() != this->getDestType().getRank())
    return false;
  if (!areAllConstantIntValue(sliceOp.getMixedOffsets(), 0) ||
      !areAllConstantIntValue(sliceOp.getMixedStrides(), 1))
    return false;
  RankedTensorType unpackedTypeAfterFold = sliceOp.getResultType();
  SmallVector<int64_t> outerShapeWithoutTranspose =
      getPackedOuterShapeWithoutTransposition(*this);
  for (auto [pos, tileSize] :
       llvm::zip_equal(this->getInnerDimsPos(), this->getStaticInnerTiles())) {
    if (unpackedTypeAfterFold.isDynamicDim(pos))
      return false;
    if (ShapedType::isDynamic(outerShapeWithoutTranspose[pos]))
      return false;
    if (ShapedType::isDynamic(tileSize))
      return false;
    int64_t paddingSize = outerShapeWithoutTranspose[pos] * tileSize -
                          unpackedTypeAfterFold.getDimSize(pos);
    if (paddingSize >= tileSize)
      return false;
  }
  return true;
}

bool UnPackOp::isLikeUnPad() {
  RankedTensorType packedTensorType = getSourceType();
  return isLikePadUnPad(*this, packedTensorType);
}

OpFoldResult UnPackOp::fold(FoldAdaptor adaptor) {
  if (OpFoldResult reshapedSource = reshapeConstantSource(
          llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getSource()),
          getResult().getType()))
    return reshapedSource;
  return {};
}

/// Folds a tensor.cast op into a consuming UnPackOp op if the
/// `tensor.cast` has source that is more static than the consuming op.
///
/// Example:
/// ```mlir
///   %1 = tensor.cast %0 : tensor<1x1x8x1xi32> to tensor<1x1x?x1xi32>
///   %2 = tensor.unpack %1 ... : tensor<1x1x?x1xi32> -> tensor<7x?xi32>
/// ```
///
/// folds into:
///
/// ```mlir
///   %2 = tensor.unpack %0  ... tensor<1x1x8x1xi32> -> tensor<7x?xi32>
/// ```
struct FoldTensorCastUnPackOp : public OpRewritePattern<UnPackOp> {
  using OpRewritePattern<UnPackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnPackOp op,
                                PatternRewriter &rewriter) const override {
    if (!tensor::hasFoldableTensorCastOperand(op))
      return failure();

    SmallVector<Type> newResultTypes(op->getResultTypes());
    SmallVector<Value> newOperands =
        tensor::getUpdatedOperandsAfterCastOpFolding(op, newResultTypes);
    Value sourceTensor = newOperands[0];

    // Get the updated mixed-tile-sizes attribute.
    SmallVector<OpFoldResult> newMixedTileSizes = getNewMixedTileSizes(
        rewriter, sourceTensor.getType(), op.getMixedTiles());

    // Clone op.
    // TODO: Strictly speaking, discardable attributes should be _discarded_ at
    // this point. However, in practice, we use them for things that we'd like
    // to preserve. Implement a better abstraction.
    UnPackOp newOp = UnPackOp::create(rewriter, op.getLoc(), sourceTensor,
                                      newOperands[1], op.getInnerDimsPos(),
                                      newMixedTileSizes, op.getOuterDimsPerm());
    newOp->setDiscardableAttrs(op->getDiscardableAttrDictionary());

    // Replace op.
    Value oldResult = op.getResult();
    Value newResult = newOp.getResult();
    Value replacement =
        (newResult.getType() != oldResult.getType())
            ? tensor::CastOp::create(rewriter, op->getLoc(),
                                     oldResult.getType(), newResult)
            : newResult;

    rewriter.replaceOp(op, {replacement});

    return success();
  }
};

//===----------------------------------------------------------------------===//
// BatchReduceMatmulOp
//===----------------------------------------------------------------------===//
SmallVector<utils::IteratorType> BatchReduceMatmulOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{
      utils::IteratorType::reduction, utils::IteratorType::parallel,
      utils::IteratorType::parallel, utils::IteratorType::reduction};
}

SmallVector<AffineMap>
BatchReduceMatmulOp::getDefaultIndexingMaps(MLIRContext *context) {
  AffineExpr d0, d1, d2, d3;
  SmallVector<AffineMap> indexingMaps;
  bindDims(context, d0, d1, d2, d3);
  indexingMaps.push_back(AffineMap::get(4, 0, {d0, d1, d3}, context));
  indexingMaps.push_back(AffineMap::get(4, 0, {d0, d3, d2}, context));
  indexingMaps.push_back(AffineMap::get(4, 0, {d1, d2}, context));
  return indexingMaps;
}

bool BatchReduceMatmulOp::isDefaultIndexingMaps(Attribute attr) {
  ArrayAttr maps = dyn_cast<ArrayAttr>(attr);
  if (!maps)
    return false;
  if (maps.size() != 3)
    return false;
  auto positions = getAffineResultPositions(maps);
  if (failed(positions))
    return false;
  return (*positions)[0] == SmallVector<int64_t>{0, 1, 3} &&
         (*positions)[1] == SmallVector<int64_t>{0, 3, 2} &&
         (*positions)[2] == SmallVector<int64_t>{1, 2};
}
unsigned BatchReduceMatmulOp::getNumRegionArgs() { return 3; }

std::string BatchReduceMatmulOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

/// Check if the op has broadcast and/or transpose semantic. Returns true if
/// the user defined indexing maps are not equal to default map.
bool BatchReduceMatmulOp::hasUserDefinedMaps() {
  SmallVector<AffineMap, 3> defaultMaps =
      getDefaultIndexingMaps(this->getContext());
  SmallVector<AffineMap, 3> explicitMaps = getIndexingMapsArray();
  return defaultMaps != explicitMaps;
}

/// Returns true if the given bcastMap map is a valid broadcast map. A valid
/// broadcast map must include K dimension.
/// TODO: Strict inclusion of K dimension in the broadcast map is not
/// necessary for both input matrices simultaneously. We can relax this
/// condition to have K dimension for one input matrix map and infer the K
/// dimension for other input matrix map from the one already having K
/// dimension.
bool BatchReduceMatmulOp::isValidLhsRhsBroadcastMap(AffineMap bcastMap,
                                                    bool isLHS) {
  assert(bcastMap.getNumResults() < 3 &&
         "Expected less than 3 result dim expr.");
  bool isValid = false;
  enum Indices { batchPos, mPos, nPos, kPos };
  if (bcastMap.getNumResults() == 1) {
    AffineExpr expr = bcastMap.getResult(0);
    isValid = expr.isFunctionOfDim(kPos);
  } else if (bcastMap.getNumResults() == 2) {
    AffineExpr expr0 = bcastMap.getResult(0);
    AffineExpr expr1 = bcastMap.getResult(1);
    isValid =
        isLHS ? ((expr0.isFunctionOfDim(batchPos) ||
                  expr0.isFunctionOfDim(mPos)) &&
                 expr1.isFunctionOfDim(kPos))
              : ((expr0.isFunctionOfDim(batchPos) &&
                  expr1.isFunctionOfDim(kPos)) ||
                 (expr0.isFunctionOfDim(kPos) && expr1.isFunctionOfDim(nPos)));
  }
  return isValid;
}

void BatchReduceMatmulOp::regionBuilder(
    ImplicitLocOpBuilder &b, Block &block, ArrayRef<NamedAttribute> attrs,
    function_ref<InFlightDiagnostic()> emitError) {
  if (emitError && block.getNumArguments() != 3) {
    emitError() << "BatchReduceMatmulOp regionBuilder expects 3 args, got "
                << block.getNumArguments();
    return;
  }
  assert(block.getNumArguments() == 3 &&
         "BatchReduceMatmulOp regionBuilder expects 3 args");
  RegionBuilderHelper helper(b, block);
  SmallVector<Value> yields;

  auto toType = block.getArgument(2).getType();
  Value castValA =
      helper.buildTypeFn(TypeFn::cast_signed, toType, block.getArgument(0));
  Value castValB =
      helper.buildTypeFn(TypeFn::cast_signed, toType, block.getArgument(1));
  Value mulVal = helper.buildBinaryFn(BinaryFn::mul, castValA, castValB);
  Value addVal =
      helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), mulVal);
  yields.push_back(addVal);
  helper.yieldOutputs(yields);
}

ParseResult BatchReduceMatmulOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  SmallVector<Attribute, 3> indexingMapsAttr;
  Attribute mapAttr;
  if (succeeded(parser.parseOptionalKeyword("indexing_maps"))) {
    if (parser.parseEqual())
      return failure();
    if (parser.parseLSquare())
      return failure();

    do {
      if (parser.parseAttribute(mapAttr))
        return failure();
      if (!isa<AffineMapAttr>(mapAttr)) {
        return parser.emitError(parser.getCurrentLocation(),
                                "expected affine map attribute");
      }
      indexingMapsAttr.push_back(mapAttr);

      if (parser.parseOptionalComma())
        break;
    } while (true);

    if (parser.parseRSquare())
      return failure();
  }
  // Initialize indexingMaps, if not supplied explicitly.
  if (indexingMapsAttr.empty()) {
    indexingMapsAttr = llvm::map_to_vector(
        BatchReduceMatmulOp::getDefaultIndexingMaps(parser.getContext()),
        [](AffineMap map) -> Attribute { return AffineMapAttr::get(map); });
  }
  result.addAttribute("indexing_maps",
                      parser.getBuilder().getArrayAttr(indexingMapsAttr));
  return ::parseNamedStructuredOp(parser, result,
                                  BatchReduceMatmulOp::getNumRegionArgs(),
                                  BatchReduceMatmulOp::getRegionBuilder());
}

void BatchReduceMatmulOp::print(OpAsmPrinter &p) {
  SmallVector<Attribute, 3> indexingMaps = llvm::map_to_vector(
      BatchReduceMatmulOp::getDefaultIndexingMaps(getContext()),
      [](AffineMap map) -> Attribute { return AffineMapAttr::get(map); });

  if (!llvm::equal(getIndexingMaps(), indexingMaps)) {
    p << " indexing_maps = [";
    llvm::interleaveComma(getIndexingMaps(), p,
                          [&](Attribute attr) { p.printAttribute(attr); });
    p << "]";
  }

  SmallVector<StringRef, 3> elidedAttrs = {
      "operandSegmentSizes", "linalg.memoized_indexing_maps", "indexing_maps"};
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs(),
                           elidedAttrs);
}

/// Verify the user defined indexing maps.
LogicalResult BatchReduceMatmulOp::verify() {
  // Verification of pure batch_reduce_matmul is handled by
  // verifyStructuredOpInterface().
  if (!hasUserDefinedMaps())
    return success();

  for (unsigned opIndex = 0; opIndex < 3; opIndex++) {
    if (failed(verifyExtendedBatchVariantMatmulSemantic(*this, opIndex)))
      return failure();
  }
  return success();
}
LogicalResult BatchReduceMatmulOp::fold(FoldAdaptor,
                                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void BatchReduceMatmulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (hasPureTensorSemantics())
    return;
  getGenericEffectsImpl(effects, cast<LinalgOp>(getOperation()));
}

Speculation::Speculatability BatchReduceMatmulOp::getSpeculatability() {
  return getGenericSpeculatabilityImpl(cast<LinalgOp>(getOperation()));
}

} // namespace linalg
} // namespace mlir

//===----------------------------------------------------------------------===//
// LinalgDialect
//===----------------------------------------------------------------------===//

void LinalgDialect::getCanonicalizationPatterns(
    RewritePatternSet &results) const {
  results.add<EraseDeadLinalgOp, FoldTensorCastConsumerOp, FoldTensorCastPackOp,
              FoldTensorCastUnPackOp, InferStaticShapeOfOperands>(getContext());
}

Operation *LinalgDialect::materializeConstant(OpBuilder &builder,
                                              Attribute value, Type type,
                                              Location loc) {
  return arith::ConstantOp::materialize(builder, value, type, loc);
}
