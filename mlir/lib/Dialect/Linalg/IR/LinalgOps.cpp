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

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::linalg;

/// Forward declarations.

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
    TypeRange outputTypes, ValueRange captures = {},
    std::function<void(unsigned, unsigned)> errorHandler = nullptr);

/// Generic entry point to create both the region and the block of a LinalgOp.
template <typename NamedStructuredOpType>
static void
createAndFillStructuredOpRegion(OpBuilder &opBuilder, OperationState &result,
                                TypeRange inputTypes, TypeRange outputTypes,
                                ValueRange captures = {});

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
                             ArrayRef<OpAsmParser::OperandType> captures = {});

static ParseResult
parseNamedStructuredOpResults(OpAsmParser &parser,
                              SmallVectorImpl<Type> &resultTypes);

template <typename NamedStructuredOpType>
static ParseResult
parseNamedStructuredOp(OpAsmParser &parser, OperationState &result,
                       ArrayRef<OpAsmParser::OperandType> captures = {});

static void printNamedStructuredOpResults(OpAsmPrinter &p,
                                          TypeRange resultTypes);

template <typename NamedStructuredOpType>
static void printNamedStructuredOp(OpAsmPrinter &p, NamedStructuredOpType op);

/// Helper function to convert a Value into an OpFoldResult, if the Value is
/// known to be a constant index value.
static SmallVector<OpFoldResult> getAsOpFoldResult(ArrayRef<Value> values) {
  return llvm::to_vector<4>(
      llvm::map_range(values, [](Value v) -> OpFoldResult {
        APInt intValue;
        if (v.getType().isa<IndexType>() &&
            matchPattern(v, m_ConstantInt(&intValue))) {
          return IntegerAttr::get(v.getType(), intValue.getSExtValue());
        }
        return v;
      }));
}

/// Helper function to convert a vector of `OpFoldResult`s into a vector of
/// `Value`s.
static SmallVector<Value> getAsValues(OpBuilder &b, Location loc,
                                      ArrayRef<OpFoldResult> valueOrAttrVec) {
  return llvm::to_vector<4>(
      llvm::map_range(valueOrAttrVec, [&](OpFoldResult value) -> Value {
        if (auto attr = value.dyn_cast<Attribute>())
          return b.create<ConstantIndexOp>(loc,
                                           attr.cast<IntegerAttr>().getInt());
        return value.get<Value>();
      }));
}

/// Helper function to dispatch an OpFoldResult into either the `dynamicVec` if
/// it is a Value or into `staticVec` if it is an IntegerAttr.
/// In the case of a Value, a copy of the `sentinel` value is also pushed to
/// `staticVec`. This is useful to extract mixed static and dynamic entries that
/// come from an AttrSizedOperandSegments trait.
static void dispatchIndexOpFoldResult(OpFoldResult ofr,
                                      SmallVectorImpl<Value> &dynamicVec,
                                      SmallVectorImpl<int64_t> &staticVec,
                                      int64_t sentinel) {
  if (auto v = ofr.dyn_cast<Value>()) {
    dynamicVec.push_back(v);
    staticVec.push_back(sentinel);
    return;
  }
  APInt apInt = ofr.dyn_cast<Attribute>().cast<IntegerAttr>().getValue();
  staticVec.push_back(apInt.getSExtValue());
}

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
      BlockArgument newBbArg =
          body->insertArgument(bbArgIndex, castOp.getOperand().getType());
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
// and bind by name to math functions in the DSL as:
//   `applyfn__{fnName}`
// Examples:
//   `applyfn__add`
//   `applyfn__mul`
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
  RegionBuilderHelper(Block &block) : block(block) {}

  // Generates operations to cast the given operand to a specified type.
  // If the cast cannot be performed, a warning will be issued and the
  // operand returned as-is (which will presumably yield a verification
  // issue downstream).
  Value cast(Type toType, Value operand) {
    OpBuilder builder = getBuilder(operand);
    auto loc = operand.getLoc();

    if (operand.getType() == toType)
      return operand;
    if (auto toIntType = toType.dyn_cast<IntegerType>()) {
      // If operand is floating point, cast directly to the int type.
      if (operand.getType().isa<FloatType>())
        return builder.create<FPToSIOp>(loc, toType, operand);
      if (auto fromIntType = operand.getType().dyn_cast<IntegerType>()) {
        // Either sign extend or truncate.
        if (toIntType.getWidth() > fromIntType.getWidth())
          return builder.create<SignExtendIOp>(loc, toType, operand);
        else if (toIntType.getWidth() < fromIntType.getWidth())
          return builder.create<TruncateIOp>(loc, toType, operand);
      }
    } else if (auto toFloatType = toType.dyn_cast<FloatType>()) {
      // If operand is integer, cast directly to the float type.
      // Note that it is unclear how to cast from BF16<->FP16.
      if (operand.getType().isa<IntegerType>())
        return builder.create<SIToFPOp>(loc, toFloatType, operand);
      if (auto fromFloatType = operand.getType().dyn_cast<FloatType>()) {
        if (toFloatType.getWidth() > fromFloatType.getWidth())
          return builder.create<FPExtOp>(loc, toFloatType, operand);
        else if (toFloatType.getWidth() < fromFloatType.getWidth())
          return builder.create<FPTruncOp>(loc, toFloatType, operand);
      }
    }

    emitWarning(operand.getLoc()) << "could not cast operand of type "
                                  << operand.getType() << " to " << toType;
    return operand;
  }

  Value applyfn__add(Value lhs, Value rhs) {
    OpBuilder builder = getBuilder(lhs);
    if (isFloatingPoint(lhs))
      return builder.create<AddFOp>(lhs.getLoc(), lhs, rhs);
    else if (isInteger(lhs))
      return builder.create<AddIOp>(lhs.getLoc(), lhs, rhs);
    llvm_unreachable("unsupported non numeric type");
  }

  Value applyfn__mul(Value lhs, Value rhs) {
    OpBuilder builder = getBuilder(lhs);
    if (isFloatingPoint(lhs))
      return builder.create<MulFOp>(lhs.getLoc(), lhs, rhs);
    else if (isInteger(lhs))
      return builder.create<MulIOp>(lhs.getLoc(), lhs, rhs);
    llvm_unreachable("unsupported non numeric type");
  }

  void yieldOutputs(ValueRange values) {
    assert(!values.empty() && "linalg ops must yield outputs");
    if (values.empty())
      return;
    Value first = values.front();
    OpBuilder builder = getBuilder(first);
    builder.create<YieldOp>(first.getLoc(), values);
  }

private:
  Block &block;

  bool isFloatingPoint(Value value) { return value.getType().isa<FloatType>(); }
  bool isInteger(Value value) { return value.getType().isa<IntegerType>(); }

  OpBuilder getBuilder(Value value) {
    OpBuilder builder(value.getContext());
    builder.setInsertionPointToEnd(&block);
    return builder;
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//
void CopyOp::regionBuilder(Block &block, ValueRange captures) {
  using namespace edsc::intrinsics;
  assert(block.getNumArguments() == 2 && "CopyOp regionBuilder expects 2 args");
  (linalg_yield(block.getArgument(0)));
}

void CopyOp::build(OpBuilder &builder, OperationState &result, Value input,
                   Value output, AffineMap inputPermutation,
                   AffineMap outputPermutation,
                   ArrayRef<NamedAttribute> namedAttrs) {
  result.addOperands({input, output});
  result.addAttributes(namedAttrs);
  if (inputPermutation)
    result.addAttribute("inputPermutation",
                        AffineMapAttr::get(inputPermutation));
  if (outputPermutation)
    result.addAttribute("outputPermutation",
                        AffineMapAttr::get(outputPermutation));
  result.addRegion();
  fillStructuredOpRegion<CopyOp>(builder, *result.regions.front(),
                                 TypeRange{input.getType()},
                                 TypeRange{output.getType()});
}

ParseResult parseCopyOpRegion(OpAsmParser &parser, Region &r, Type inputType,
                              Type outputType) {
  OpBuilder opBuilder(parser.getBuilder().getContext());
  fillStructuredOpRegion<CopyOp>(opBuilder, r, TypeRange{inputType},
                                 TypeRange{outputType});
  return success();
}

/// CopyOp region is elided when printing.
void printCopyOpRegion(OpAsmPrinter &, Operation *, Region &, Type, Type) {}

static LogicalResult verify(CopyOp op) {
  auto outputViewType = op.getOutputShapedType(0);
  auto inputViewType = op.getInputShapedType(0);
  if (inputViewType.getElementType() != outputViewType.getElementType())
    return op.emitOpError("expects views of the same type");
  if (inputViewType.getRank() != outputViewType.getRank())
    return op.emitOpError("expects views of the same rank");
  auto rank = op.getNumParallelLoops();
  auto inputPermutationMap = op.inputPermutation();
  if (inputPermutationMap) {
    if (inputPermutationMap->getNumInputs() != rank)
      return op.emitOpError("expects optional input_permutation map of rank ")
             << rank;
    if (!inputPermutationMap->isPermutation())
      return op.emitOpError(
          "expects optional input_permutation map to be a permutation");
  }
  auto outputPermutationMap = op.outputPermutation();
  if (outputPermutationMap) {
    if (outputPermutationMap->getNumInputs() != rank)
      return op.emitOpError("expects optional output_permutation map of rank ")
             << rank;
    if (!outputPermutationMap->isPermutation())
      return op.emitOpError(
          "expects optional output_permutation map to be a permutation");
  }
  if (rank == 0 && inputPermutationMap)
    return op.emitOpError("expected no input permutation when rank == 0");
  if (rank == 0 && outputPermutationMap)
    return op.emitOpError("expected no output permutation when rank == 0");
  return success();
}

void CopyOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), input(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), output(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// FillOp
//===----------------------------------------------------------------------===//
void FillOp::regionBuilder(Block &block, ValueRange captures) {
  using namespace edsc::intrinsics;
  assert(captures.size() == 1 && "FillOp regionBuilder expects 1 capture");
  (linalg_yield(captures));
}

void FillOp::build(OpBuilder &builder, OperationState &result, Value output,
                   Value value) {
  build(builder, result, output.getType().dyn_cast<RankedTensorType>(), output,
        value);
  fillStructuredOpRegion<FillOp>(builder, *result.regions.front(), TypeRange{},
                                 TypeRange{output.getType()}, value);
}

ParseResult parseFillOpRegion(OpAsmParser &parser, Region &r, Type outputType,
                              OpAsmParser::OperandType valueRef) {
  OpBuilder opBuilder(parser.getBuilder().getContext());
  // Resolve `valueRef` into `value` at parse time so we can build the region
  // with captures.
  SmallVector<Value> value;
  parser.resolveOperand(valueRef, getElementTypeOrSelf(outputType), value);
  fillStructuredOpRegion<FillOp>(opBuilder, r, TypeRange{},
                                 TypeRange{outputType}, value);
  return success();
}

/// FillOp region is elided when printing.
void printFillOpRegion(OpAsmPrinter &, Operation *, Region &, Type, Value) {}

static LogicalResult verify(FillOp op) {
  auto viewType = op.getOutputShapedType(0);
  auto fillType = op.value().getType();
  if (viewType.getElementType() != fillType)
    return op.emitOpError("expects fill type to match view elemental type");
  if (!op.getNumResults() && !viewType.isa<MemRefType>()) {
    return op.emitOpError(
        "expected fill op with no result value to use memref type");
  }
  return success();
}

void FillOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (output().getType().isa<MemRefType>())
    effects.emplace_back(MemoryEffects::Write::get(), output(),
                         SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// GenericOps
//===----------------------------------------------------------------------===//
void GenericOp::build(
    OpBuilder &builder, OperationState &result, TypeRange resultTensorTypes,
    ValueRange inputs, ValueRange outputs, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<StringRef> iteratorTypes, StringRef doc, StringRef libraryCall,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild) {
  build(builder, result, resultTensorTypes, inputs, outputs,
        builder.getAffineMapArrayAttr(indexingMaps),
        builder.getStrArrayAttr(iteratorTypes),
        doc.empty() ? StringAttr() : builder.getStringAttr(doc),
        libraryCall.empty() ? StringAttr()
                            : builder.getStringAttr(libraryCall));
  if (!bodyBuild)
    return;

  SmallVector<Type, 4> blockArgTypes;
  for (ValueRange container : {inputs, outputs})
    for (Value v : container)
      blockArgTypes.push_back(v.getType().cast<ShapedType>().getElementType());

  OpBuilder::InsertionGuard guard(builder);
  auto &region = *result.regions.front();
  Block *bodyBlock = builder.createBlock(&region, region.end(), blockArgTypes);
  bodyBuild(builder, result.location, bodyBlock->getArguments());
}

void GenericOp::build(
    OpBuilder &builder, OperationState &result, ValueRange inputs,
    ValueRange outputs, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<StringRef> iteratorTypes, StringRef doc, StringRef libraryCall,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild) {
  build(builder, result, TypeRange{}, inputs, outputs, indexingMaps,
        iteratorTypes, doc, libraryCall, bodyBuild);
}

void GenericOp::build(
    OpBuilder &builder, OperationState &result, ValueRange inputs,
    ValueRange outputs, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<StringRef> iteratorTypes,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild) {
  build(builder, result, inputs, outputs, indexingMaps, iteratorTypes,
        /*doc=*/"",
        /*libraryCall=*/"", bodyBuild);
}

void GenericOp::build(
    OpBuilder &builder, OperationState &result, TypeRange resultTensorTypes,
    ValueRange inputs, ValueRange outputs, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<StringRef> iteratorTypes,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild) {
  build(builder, result, resultTensorTypes, inputs, outputs, indexingMaps,
        iteratorTypes,
        /*doc=*/"",
        /*libraryCall=*/"", bodyBuild);
}
void IndexedGenericOp::build(
    OpBuilder &builder, OperationState &result, TypeRange resultTensorTypes,
    ValueRange inputs, ValueRange outputs, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<StringRef> iteratorTypes, StringRef doc, StringRef libraryCall,
    function_ref<void(OpBuilder &, Location, ValueRange, ValueRange)>
        bodyBuild) {
  build(builder, result, resultTensorTypes, inputs, outputs,
        builder.getAffineMapArrayAttr(indexingMaps),
        builder.getStrArrayAttr(iteratorTypes),
        doc.empty() ? StringAttr() : builder.getStringAttr(doc),
        libraryCall.empty() ? StringAttr()
                            : builder.getStringAttr(libraryCall));
  if (!bodyBuild)
    return;

  unsigned nLoops = iteratorTypes.size();
  SmallVector<Type, 4> blockArgTypes(nLoops, builder.getIndexType());
  for (ValueRange container : {inputs, outputs})
    for (Value v : container)
      blockArgTypes.push_back(v.getType().cast<ShapedType>().getElementType());

  OpBuilder::InsertionGuard guard(builder);
  auto &region = *result.regions.front();
  Block *bodyBlock = builder.createBlock(&region, region.end(), blockArgTypes);
  bodyBuild(builder, result.location,
            bodyBlock->getArguments().take_front(nLoops),
            bodyBlock->getArguments().drop_front(nLoops));
}

void IndexedGenericOp::build(
    OpBuilder &builder, OperationState &result, ValueRange inputs,
    ValueRange outputs, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<StringRef> iteratorTypes, StringRef doc, StringRef libraryCall,
    function_ref<void(OpBuilder &, Location, ValueRange, ValueRange)>
        bodyBuild) {
  build(builder, result, TypeRange{}, inputs, outputs, indexingMaps,
        iteratorTypes, doc, libraryCall, bodyBuild);
}

void IndexedGenericOp::build(
    OpBuilder &builder, OperationState &result, ValueRange inputs,
    ValueRange outputs, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<StringRef> iteratorTypes,
    function_ref<void(OpBuilder &, Location, ValueRange, ValueRange)>
        bodyBuild) {
  build(builder, result, inputs, outputs, indexingMaps, iteratorTypes,
        /*doc=*/"", /*libraryCall=*/"", bodyBuild);
}

void IndexedGenericOp::build(
    OpBuilder &builder, OperationState &result, TypeRange resultTensorTypes,
    ValueRange inputs, ValueRange outputs, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<StringRef> iteratorTypes,
    function_ref<void(OpBuilder &, Location, ValueRange, ValueRange)>
        bodyBuild) {
  build(builder, result, resultTensorTypes, inputs, outputs, indexingMaps,
        iteratorTypes,
        /*doc=*/"",
        /*libraryCall=*/"", bodyBuild);
}

template <typename GenericOpType>
static void printGenericOp(OpAsmPrinter &p, GenericOpType op) {
  p << op.getOperationName() << " ";

  // Print extra attributes.
  auto genericAttrNames = op.linalgTraitAttrNames();

  llvm::StringSet<> genericAttrNamesSet;
  genericAttrNamesSet.insert(genericAttrNames.begin(), genericAttrNames.end());
  SmallVector<NamedAttribute, 8> genericAttrs;
  for (auto attr : op->getAttrs())
    if (genericAttrNamesSet.count(attr.first.strref()) > 0)
      genericAttrs.push_back(attr);
  if (!genericAttrs.empty()) {
    auto genericDictAttr = DictionaryAttr::get(op.getContext(), genericAttrs);
    p << genericDictAttr;
  }

  // Printing is shared with named ops, except for the region and attributes
  printCommonStructuredOpParts(p, op);

  genericAttrNames.push_back("operand_segment_sizes");
  genericAttrNamesSet.insert(genericAttrNames.back());

  bool hasExtraAttrs = false;
  for (NamedAttribute n : op->getAttrs()) {
    if ((hasExtraAttrs = !genericAttrNamesSet.contains(n.first.strref())))
      break;
  }
  if (hasExtraAttrs) {
    p << " attrs = ";
    p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/genericAttrNames);
  }

  // Print region.
  if (!op.region().empty())
    p.printRegion(op.region());

  // Print results.
  printNamedStructuredOpResults(p, op.result_tensors().getTypes());
}

static void print(OpAsmPrinter &p, GenericOp op) { printGenericOp(p, op); }

static void print(OpAsmPrinter &p, IndexedGenericOp op) {
  printGenericOp(p, op);
}

static ParseResult parseGenericOp(OpAsmParser &parser, OperationState &result) {
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
  getGenericEffectsImpl(effects, getOperation()->getResults(),
                        getInputBuffers(), getOutputBuffers());
}

void IndexedGenericOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getGenericEffectsImpl(effects, getOperation()->getResults(),
                        getInputBuffers(), getOutputBuffers());
}

template <typename GenericOpType>
static LogicalResult verifyGenericOp(GenericOpType op) {
  return success();
}

static LogicalResult verify(GenericOp op) { return verifyGenericOp(op); }

static LogicalResult verify(IndexedGenericOp op) { return verifyGenericOp(op); }

namespace {

/// Replace indexed_generic ops by generic ops that access the iteration indices
/// using index operation calls.
struct ConvertIndexedToGenericOp : OpRewritePattern<IndexedGenericOp> {
  using OpRewritePattern<IndexedGenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IndexedGenericOp indexedOp,
                                PatternRewriter &rewriter) const override {
    // Replace all uses of the index block arguments.
    BlockAndValueMapping bvm;
    if (Block *body = indexedOp.getBody()) {
      rewriter.setInsertionPointToStart(body);
      for (const auto &en : llvm::enumerate(
               body->getArguments().take_front(indexedOp.getNumLoops()))) {
        Value index = rewriter.create<IndexOp>(indexedOp.getLoc(), en.index());
        bvm.map(en.value(), index);
      }
    }

    // Create a generic replacement operation and clone the body.
    rewriter.setInsertionPointAfter(indexedOp);
    SmallVector<StringRef> iterators = llvm::to_vector<4>(
        indexedOp.iterator_types().getAsValueRange<StringAttr>());
    GenericOp genericOp = rewriter.create<GenericOp>(
        indexedOp.getLoc(), indexedOp->getResultTypes(), indexedOp.getInputs(),
        indexedOp.getOutputs(), indexedOp.getIndexingMaps(), iterators);
    Region &genericRegion = genericOp.region();
    Region &indexedRegion = indexedOp.region();
    rewriter.cloneRegionBefore(indexedRegion, genericRegion,
                               genericRegion.begin(), bvm);

    rewriter.replaceOp(indexedOp, genericOp->getResults());
    return success();
  }
};
} // namespace

void IndexedGenericOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  results.add<ConvertIndexedToGenericOp>(context);
}

//===----------------------------------------------------------------------===//
// InitTensorOp
//===----------------------------------------------------------------------===//
void InitTensorOp::build(OpBuilder &b, OperationState &result,
                         ArrayRef<OpFoldResult> sizes, Type elementType,
                         ArrayRef<NamedAttribute> attrs) {
  unsigned rank = sizes.size();
  SmallVector<Value, 4> dynamicSizes;
  SmallVector<int64_t, 4> staticSizes;
  for (unsigned i = 0; i < rank; ++i) {
    dispatchIndexOpFoldResult(sizes[i], dynamicSizes, staticSizes,
                              ShapedType::kDynamicSize);
  }
  auto resultType = RankedTensorType ::get(staticSizes, elementType);
  build(b, result, resultType, dynamicSizes, b.getI64ArrayAttr(staticSizes));
  result.addAttributes(attrs);
}

static LogicalResult verify(InitTensorOp op) {
  RankedTensorType resultType = op.getType();
  SmallVector<int64_t, 4> staticSizes = llvm::to_vector<4>(llvm::map_range(
      op.static_sizes().cast<ArrayAttr>(),
      [](Attribute a) -> int64_t { return a.cast<IntegerAttr>().getInt(); }));

  if (failed(verifyListOfOperandsOrIntegers(op, "sizes", resultType.getRank(),
                                            op.static_sizes(), op.sizes(),
                                            ShapedType::isDynamic)))
    return failure();

  if (op.static_sizes().size() != static_cast<unsigned>(resultType.getRank()))
    return op->emitError("expected ")
           << resultType.getRank() << " sizes values";

  Type expectedType =
      InitTensorOp::inferResultType(staticSizes, resultType.getElementType());
  if (resultType != expectedType) {
    return op.emitError("specified type ")
           << resultType << " does not match the inferred type "
           << expectedType;
  }
  return success();
}

Type InitTensorOp::inferResultType(ArrayRef<int64_t> staticSizes,
                                   Type elementType) {
  return RankedTensorType::get(staticSizes, elementType);
}

namespace {
/// Change the type of the result of a `linalg.init_tensor` by making the result
/// type statically sized along dimension that in the original operation where
/// defined as dynamic, but the size was defined using a `constant` op. For
/// example
///
///  %c5 = constant 5: index
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
      if (auto constantIndexOp = sizeOperand.getDefiningOp<ConstantIndexOp>()) {
        staticSizes.push_back(constantIndexOp.getValue());
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
/// subtensor of this is also needed only for its shape. The result can be
/// replaced by a new init_tensor operation of the same size as the subtensor
/// op.
struct FoldInitTensorWithSubTensorOp : public OpRewritePattern<SubTensorOp> {
  using OpRewritePattern<SubTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubTensorOp subtensorOp,
                                PatternRewriter &rewriter) const override {
    if (!subtensorOp.source().getDefiningOp<linalg::InitTensorOp>())
      return failure();
    rewriter.replaceOpWithNewOp<linalg::InitTensorOp>(
        subtensorOp, subtensorOp.sizes(),
        llvm::to_vector<4>(llvm::map_range(
            subtensorOp.static_sizes(),
            [](Attribute attr) { return attr.cast<IntegerAttr>().getInt(); })),
        subtensorOp.getSourceType().getElementType());
    return success();
  }
};

struct FoldInitTensorWithTensorReshapeOp
    : public OpRewritePattern<TensorReshapeOp> {
  using OpRewritePattern<TensorReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    if (!reshapeOp.src().getDefiningOp<InitTensorOp>())
      return failure();
    Location loc = reshapeOp.getLoc();
    SmallVector<SmallVector<Value>, 4> resultShapes;
    if (failed(reshapeOp.reifyReturnTypeShapesPerResultDim(rewriter,
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
} // namespace

void InitTensorOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.add<FoldInitTensorWithSubTensorOp, FoldInitTensorWithTensorReshapeOp,
              ReplaceStaticShapeDims>(context);
}

LogicalResult InitTensorOp::reifyReturnTypeShapesPerResultDim(
    OpBuilder &builder,
    SmallVectorImpl<SmallVector<Value>> &reifiedReturnShapes) {
  auto shapes = llvm::to_vector<4>(llvm::map_range(
      llvm::seq<int64_t>(0, getType().getRank()), [&](int64_t dim) -> Value {
        if (isDynamicSize(dim))
          return getDynamicSize(dim);
        return builder.create<ConstantIndexOp>(getLoc(), getStaticSize(dim));
      }));
  reifiedReturnShapes.emplace_back(std::move(shapes));
  return success();
}

//===----------------------------------------------------------------------===//
// PadTensorOp
//===----------------------------------------------------------------------===//

/// Extract int64_t values from the assumed ArrayAttr of IntegerAttr.
static SmallVector<int64_t, 4> extractFromI64ArrayAttr(Attribute attr) {
  return llvm::to_vector<4>(
      llvm::map_range(attr.cast<ArrayAttr>(), [](Attribute a) -> int64_t {
        return a.cast<IntegerAttr>().getInt();
      }));
}

static LogicalResult verify(PadTensorOp op) {
  auto sourceType = op.source().getType().cast<RankedTensorType>();
  auto resultType = op.result().getType().cast<RankedTensorType>();
  auto expectedType = PadTensorOp::inferResultType(
      sourceType, extractFromI64ArrayAttr(op.static_low()),
      extractFromI64ArrayAttr(op.static_high()));
  for (int i = 0, e = sourceType.getRank(); i < e; ++i) {
    if (resultType.getDimSize(i) == expectedType.getDimSize(i))
      continue;
    if (expectedType.isDynamicDim(i))
      continue;
    return op.emitError("specified type ")
           << resultType << " does not match the inferred type "
           << expectedType;
  }

  auto &region = op.region();
  unsigned rank = resultType.getRank();
  Block &block = region.front();
  if (block.getNumArguments() != rank)
    return op.emitError("expected the block to have ") << rank << " arguments";

  // Note: the number and type of yield values are checked in the YieldOp.
  for (auto en : llvm::enumerate(block.getArgumentTypes())) {
    if (!en.value().isIndex())
      return op.emitOpError("expected block argument ")
             << (en.index() + 1) << " to be an index";
  }

  return success();
}

RankedTensorType PadTensorOp::inferResultType(RankedTensorType sourceType,
                                              ArrayRef<int64_t> staticLow,
                                              ArrayRef<int64_t> staticHigh) {
  unsigned rank = sourceType.getRank();
  assert(staticLow.size() == rank && "unexpected staticLow size mismatch");
  assert(staticHigh.size() == rank && "unexpected staticHigh size mismatch");

  SmallVector<int64_t, 4> resultShape;
  for (auto i : llvm::seq<unsigned>(0, rank)) {
    if (sourceType.isDynamicDim(i) ||
        staticLow[i] == ShapedType::kDynamicSize ||
        staticHigh[i] == ShapedType::kDynamicSize) {
      resultShape.push_back(ShapedType::kDynamicSize);
    } else {
      int64_t size = sourceType.getDimSize(i) + staticLow[i] + staticHigh[i];
      resultShape.push_back(size);
    }
  }

  return RankedTensorType::get(resultShape, sourceType.getElementType());
}

void PadTensorOp::build(OpBuilder &b, OperationState &result, Value source,
                        ArrayRef<int64_t> staticLow,
                        ArrayRef<int64_t> staticHigh, ValueRange low,
                        ValueRange high, ArrayRef<NamedAttribute> attrs) {
  auto sourceType = source.getType().cast<RankedTensorType>();
  auto resultType = inferResultType(sourceType, staticLow, staticHigh);
  build(b, result, resultType, source, low, high, b.getI64ArrayAttr(staticLow),
        b.getI64ArrayAttr(staticHigh));
  result.addAttributes(attrs);
}

void PadTensorOp::build(OpBuilder &b, OperationState &result, Value source,
                        ValueRange low, ValueRange high,
                        ArrayRef<NamedAttribute> attrs) {
  auto sourceType = source.getType().cast<RankedTensorType>();
  unsigned rank = sourceType.getRank();
  SmallVector<int64_t, 4> staticVector(ShapedType::kDynamicSize, rank);
  build(b, result, source, staticVector, staticVector, low, high, attrs);
}

void PadTensorOp::build(OpBuilder &b, OperationState &result, Type resultType,
                        Value source, ArrayRef<OpFoldResult> low,
                        ArrayRef<OpFoldResult> high,
                        ArrayRef<NamedAttribute> attrs) {
  assert(resultType.isa<RankedTensorType>());
  auto sourceType = source.getType().cast<RankedTensorType>();
  unsigned rank = sourceType.getRank();
  SmallVector<Value, 4> dynamicLow, dynamicHigh;
  SmallVector<int64_t, 4> staticLow, staticHigh;
  for (unsigned i = 0; i < rank; ++i) {
    // staticLow and staticHigh have full information of the padding config.
    // This will grow staticLow and staticHigh with 1 value. If the config is
    // dynamic (ie not a constant), dynamicLow and dynamicHigh will grow with 1
    // value as well.
    dispatchIndexOpFoldResult(low[i], dynamicLow, staticLow,
                              ShapedType::kDynamicSize);
    dispatchIndexOpFoldResult(high[i], dynamicHigh, staticHigh,
                              ShapedType::kDynamicSize);
  }
  if (!resultType) {
    resultType =
        PadTensorOp::inferResultType(sourceType, staticLow, staticHigh);
  }
  build(b, result, resultType, source, dynamicLow, dynamicHigh,
        b.getI64ArrayAttr(staticLow), b.getI64ArrayAttr(staticHigh));
}

PadTensorOp PadTensorOp::createPadScalarOp(Type type, Value source, Value pad,
                                           ArrayRef<OpFoldResult> low,
                                           ArrayRef<OpFoldResult> high,
                                           Location loc, OpBuilder &builder) {
  auto padTensorOp =
      builder.create<linalg::PadTensorOp>(loc, type, source, low, high);
  int rank = padTensorOp.getResultType().getRank();
  SmallVector<Type, 4> blockArgTypes;
  blockArgTypes.assign(rank, builder.getIndexType());
  auto &region = padTensorOp.region();
  // `builder.createBlock` changes the insertion point within the block. Create
  // a guard to reset the insertion point of the builder after it is destroyed.
  OpBuilder::InsertionGuard guard(builder);
  builder.createBlock(&region, region.end(), blockArgTypes);
  builder.create<linalg::YieldOp>(loc, pad);
  return padTensorOp;
}

PadTensorOp PadTensorOp::createPadHighOp(Type type, Value source, Value pad,
                                         Location loc, OpBuilder &builder) {
  SmallVector<OpFoldResult, 4> low, high;
  auto rankedTensorType = type.cast<RankedTensorType>();
  assert(rankedTensorType.hasStaticShape());
  int rank = rankedTensorType.getRank();
  for (int i = 0; i < rank; ++i) {
    auto dimOp = builder.createOrFold<memref::DimOp>(loc, source, i);
    auto resultDimSize = builder.createOrFold<ConstantIndexOp>(
        loc, rankedTensorType.getDimSize(i));
    auto highValue = builder.createOrFold<SubIOp>(loc, resultDimSize, dimOp);
    high.push_back(highValue);
    low.push_back(builder.createOrFold<ConstantIndexOp>(loc, 0));
  }
  return PadTensorOp::createPadScalarOp(type, source, pad, low, high, loc,
                                        builder);
}

LogicalResult PadTensorOp::reifyReturnTypeShapesPerResultDim(
    OpBuilder &b, SmallVectorImpl<SmallVector<Value>> &reifiedReturnShapes) {
  Location loc = getLoc();
  auto lowPad = getMixedLowPad();
  auto highPad = getMixedHighPad();
  SmallVector<Value> shapes;
  for (auto dim : llvm::seq<int64_t>(0, getSourceType().getRank())) {
    // Shape along each dimension is source dim + low pad + high pad.
    SmallVector<Value> mapOperands;
    mapOperands.push_back(b.createOrFold<memref::DimOp>(loc, source(), dim));
    AffineExpr expr = b.getAffineDimExpr(0);
    unsigned numSymbols = 0;
    auto addOpFoldResult = [&](OpFoldResult valueOrAttr) {
      if (Value v = valueOrAttr.dyn_cast<Value>()) {
        expr = expr + b.getAffineSymbolExpr(numSymbols++);
        mapOperands.push_back(v);
        return;
      }
      int64_t staticValue =
          valueOrAttr.get<Attribute>().cast<IntegerAttr>().getInt();
      expr = expr + staticValue;
    };
    addOpFoldResult(lowPad[dim]);
    addOpFoldResult(highPad[dim]);
    shapes.push_back(applyMapToValues(
        b, loc, AffineMap::get(1, numSymbols, expr), mapOperands)[0]);
  }
  reifiedReturnShapes.emplace_back(std::move(shapes));
  return success();
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

Optional<SmallVector<ReassociationIndices>>
mlir::linalg::getReassociationIndicesForReshape(ShapedType sourceType,
                                                ShapedType targetType) {
  // Make the sourceType greater rank than the targetType. If they are same
  // rank, then its an unsupported reshape op.
  if (sourceType.getRank() == targetType.getRank())
    return llvm::None;
  if (sourceType.getRank() < targetType.getRank())
    std::swap(sourceType, targetType);

  ArrayRef<int64_t> sourceShape = sourceType.getShape();
  ArrayRef<int64_t> targetShape = targetType.getShape();
  unsigned sourceDim = 0;
  SmallVector<ReassociationIndices> reassociationMap;
  reassociationMap.reserve(targetType.getRank());

  ReassociationIndices currIndices;
  int64_t prodOfCollapsedDims = 1;
  while (sourceDim < sourceShape.size()) {
    unsigned targetDim = reassociationMap.size();

    // If all the dimensions of the targetShape are exhausted, then the
    // remaining dims in the source shape must be all 1s. So for such cases, set
    // 1 as the target shape. The actual reassociation indices will be handled
    // later.
    int64_t currTargetShape =
        (targetDim < targetType.getRank() ? targetShape[targetDim] : 1);
    while (sourceShape[sourceDim] != ShapedType::kDynamicSize &&
           prodOfCollapsedDims * sourceShape[sourceDim] < currTargetShape &&
           sourceDim < sourceShape.size()) {
      prodOfCollapsedDims *= sourceShape[sourceDim];
      currIndices.push_back(sourceDim++);
    }

    // If the current expanded dimension is dynamic, then the collapsed
    // dimensions should also be dynamic and product of all previous unprocessed
    // dimensions of the expanded shape should be 1.
    if (sourceShape[sourceDim] == ShapedType::kDynamicSize &&
        (currTargetShape != ShapedType::kDynamicSize ||
         prodOfCollapsedDims != 1))
      return llvm::None;

    // If the collapsed dim is dynamic, the current expanded dim should also
    // be dynamic.
    if (currTargetShape == ShapedType::kDynamicSize &&
        sourceShape[sourceDim] != ShapedType::kDynamicSize)
      return llvm::None;

    // For static shapes, if the product of dimensions of the expanded shape
    // should match the collapsed dimension shape.
    if (prodOfCollapsedDims * sourceShape[sourceDim] != currTargetShape)
      return llvm::None;

    currIndices.push_back(sourceDim++);
    // If the reassociation is empty but the currIndices is not, this by
    // definition is folding unit-dimensions with the result being scalar type.
    // So only append the `currIndices` if reassociation map is not empty.
    if (targetDim == targetShape.size()) {
      if (!reassociationMap.empty() && !currIndices.empty())
        reassociationMap.back().append(currIndices.begin(), currIndices.end());
      // Break out of the loops. We should be done here.
      break;
    }
    reassociationMap.emplace_back(ReassociationIndices{});
    std::swap(reassociationMap.back(), currIndices);
    prodOfCollapsedDims = 1;
  }
  // All the dimensions in the two shapes must have been processed.
  if (reassociationMap.size() != targetShape.size() ||
      sourceDim != sourceShape.size())
    return llvm::None;
  return reassociationMap;
}

template <typename ReshapeLikeOp>
static void print(OpAsmPrinter &p, ReshapeLikeOp op) {
  p << op.getOperationName() << ' ' << op.src() << " [";

  llvm::interleaveComma(op.reassociation(), p, [&](const Attribute &attr) {
    p << '[';
    auto arrayAttr = attr.template cast<ArrayAttr>();
    llvm::interleaveComma(arrayAttr, p, [&](const Attribute &attr) {
      p << attr.cast<IntegerAttr>().getInt();
    });
    p << ']';
  });

  p << "] ";
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{op.getReassociationAttrName()});
  p << ": " << op.src().getType() << " into " << op.getType();
}

static void print(OpAsmPrinter &p, linalg::ReshapeOp op) {
  print<linalg::ReshapeOp>(p, op);
}

static void print(OpAsmPrinter &p, linalg::TensorReshapeOp op) {
  print<linalg::TensorReshapeOp>(p, op);
}

static ParseResult parseReshapeLikeOp(OpAsmParser &parser,
                                      OperationState &result) {
  // Parse the operand.
  OpAsmParser::OperandType src;
  if (parser.parseOperand(src))
    return failure();

  // Parse reassociation indices.
  Builder &b = parser.getBuilder();
  SmallVector<Attribute, 4> reassociation;
  if (parser.parseLSquare())
    return failure();

  while (true) {
    if (succeeded(parser.parseOptionalRSquare()))
      break;
    if (parser.parseLSquare())
      return failure();
    SmallVector<int64_t> indices;
    while (true) {
      int64_t index;
      if (parser.parseInteger(index))
        return failure();
      indices.push_back(index);

      if (succeeded(parser.parseOptionalComma()))
        continue;
      if (failed(parser.parseRSquare()))
        return failure();
      break;
    }
    reassociation.push_back(b.getI64ArrayAttr(indices));
    if (succeeded(parser.parseOptionalComma()))
      continue;
    if (failed(parser.parseRSquare()))
      return failure();
    break;
  }

  result.addAttribute(ReshapeOp::getReassociationAttrName(),
                      b.getArrayAttr(reassociation));

  // Parse optional attributes.
  parser.parseOptionalAttrDict(result.attributes);

  // Parse types.
  Type srcType;
  Type resultType;
  if (parser.parseColon() || parser.parseType(srcType) ||
      parser.resolveOperand(src, srcType, result.operands) ||
      parser.parseKeyword("into") || parser.parseType(resultType))
    return failure();
  result.addTypes(resultType);
  return success();
}

/// Collapse reassociation maps that are used in pair of reshape ops where one
/// is a producer and other is the consumer. Only valid to use this method when
/// both the producer and consumer are collapsing dimensions or both are
/// expanding dimensions.
///
/// For example,
///   mapsProducer = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>,
///                   affine_map<(d0, d1, d2, d3, d4) -> (d2)>,
///                   affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>]
///   mapsConsumer = [affine_map<(d0, d1, d2) -> (d0, d1)>,
///                   affine_map<(d0, d1, d2) -> (d2)>]
///
/// is folded into
///
///   result = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
///             affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>]
static Optional<SmallVector<ReassociationIndices>>
collapseReassociationIndices(ArrayRef<AffineMap> mapsProducer,
                             ArrayRef<AffineMap> mapsConsumer,
                             MLIRContext *context) {
  // Make the producer the larger sized vector. If they are of same size, the
  // resulting reshape is not a supported reshape op.
  if (mapsProducer.size() == mapsConsumer.size())
    return llvm::None;
  if (mapsProducer.size() < mapsConsumer.size())
    std::swap(mapsProducer, mapsConsumer);

  // Handle the corner case of the result being a rank 0 shaped type. Return an
  // empty reassociation.
  if (mapsConsumer.empty())
    return SmallVector<ReassociationIndices>{};
  if (mapsProducer.size() != mapsConsumer[0].getNumDims())
    return llvm::None;

  unsigned currDim = 0;
  SmallVector<ReassociationIndices> reassociationMaps;
  for (AffineMap rhs : mapsConsumer) {
    ReassociationIndices reassociations;
    for (AffineExpr rhsExpr : rhs.getResults()) {
      AffineDimExpr dimExpr = rhsExpr.cast<AffineDimExpr>();
      for (int i = 0, e = mapsProducer[dimExpr.getPosition()].getNumResults();
           i < e; ++i)
        reassociations.push_back(currDim++);
    }
    reassociationMaps.push_back(std::move(reassociations));
  }
  return reassociationMaps;
}

namespace {
/// Pattern to collapse producer/consumer reshape ops that are both collapsing
/// dimensions or are both expanding dimensions.
template <typename ReshapeOpTy>
struct CollapseReshapeOps : public OpRewritePattern<ReshapeOpTy> {
  using OpRewritePattern<ReshapeOpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(ReshapeOpTy reshapeOp,
                                PatternRewriter &rewriter) const override {
    auto srcReshapeOp = reshapeOp.src().template getDefiningOp<ReshapeOpTy>();
    if (!srcReshapeOp)
      return failure();

    ShapedType srcReshapeSrcType = srcReshapeOp.getSrcType();
    ShapedType intermediateType = reshapeOp.getSrcType();
    ShapedType resultType = reshapeOp.getResultType();

    auto areReshapeOpsFoldable = [](ShapedType largerType,
                                    ShapedType intermediateType,
                                    ShapedType smallerType) -> bool {
      return largerType.getRank() > intermediateType.getRank() &&
             intermediateType.getRank() > smallerType.getRank();
    };
    Optional<SmallVector<ReassociationIndices>> reassociationIndices =
        llvm::None;
    // Check if producer and consumer are both expanding dims or both collapsing
    // dims. In this case, try to compose the affine maps. This works for
    // dynamic shapes too.
    if (areReshapeOpsFoldable(resultType, intermediateType,
                              srcReshapeSrcType) ||
        areReshapeOpsFoldable(srcReshapeSrcType, intermediateType,
                              resultType)) {
      reassociationIndices = collapseReassociationIndices(
          srcReshapeOp.getReassociationMaps(), reshapeOp.getReassociationMaps(),
          rewriter.getContext());
    }
    if (!reassociationIndices) {
      // If the source reshape can be collapsed/expanded into the target reshape
      // they can still be folded. This can only be reasoned about statically
      // for cases where
      // - either all shapes are static, or
      // - The number of dynamic dimensions matches in the source of source and
      //   result with all other dimensions being 1.
      reassociationIndices =
          getReassociationIndicesForReshape(srcReshapeSrcType, resultType);
    }
    if (!reassociationIndices)
      return failure();
    rewriter.replaceOpWithNewOp<ReshapeOpTy>(
        reshapeOp, resultType, srcReshapeOp.src(), *reassociationIndices);
    return success();
  }
};
} // namespace

template <typename ReshapeOpTy>
static OpFoldResult foldReshapeOp(ReshapeOpTy reshapeOp,
                                  ArrayRef<Attribute> operands) {
  // Fold producer-consumer reshape ops that where the operand type of the
  // producer is same as the return type of the consumer.
  ReshapeOpTy reshapeSrcOp =
      reshapeOp.src().template getDefiningOp<ReshapeOpTy>();
  if (reshapeSrcOp && reshapeSrcOp.getSrcType() == reshapeOp.getResultType())
    return reshapeSrcOp.src();
  // Reshape of a constant can be replaced with a new constant.
  if (auto elements = operands.front().dyn_cast_or_null<DenseElementsAttr>()) {
    return elements.reshape(
        reshapeOp.getResult().getType().template cast<ShapedType>());
  }
  return nullptr;
}

/// Return true if the reassociation specification is valid, false otherwise.
/// When false, the `invalidIndex` integer pointer is optionally filled with the
/// index of the offending reassociation map.
static bool isReassociationValid(ArrayRef<AffineMap> reassociation,
                                 int *invalidIndex = nullptr) {
  if (reassociation.empty())
    return true;
  unsigned nDims = reassociation[0].getNumDims();
  unsigned nextExpectedDim = 0;
  for (auto it : llvm::enumerate(reassociation)) {
    auto m = it.value();
    if (m.getNumDims() != nDims || m.getNumSymbols() != 0) {
      if (invalidIndex)
        *invalidIndex = it.index();
      return false;
    }
    for (auto e : m.getResults()) {
      auto d = e.dyn_cast<AffineDimExpr>();
      if (!d || d.getPosition() != nextExpectedDim++) {
        if (invalidIndex)
          *invalidIndex = it.index();
        return false;
      }
    }
  }
  if (nextExpectedDim != nDims) {
    if (invalidIndex)
      *invalidIndex = reassociation.size() - 1;
    return false;
  }
  return true;
}

/// Detect whether memref dims [dim, dim + extent) can be reshaped without
/// copies.
static bool isReshapableDimBand(unsigned dim, unsigned extent,
                                ArrayRef<int64_t> sizes,
                                ArrayRef<AffineExpr> strides) {
  assert(sizes.size() == strides.size() && "mismatched ranks");
  // off by 1 indexing to avoid out of bounds
  //                       V
  for (auto idx = dim, e = dim + extent; idx + 1 < e; ++idx) {
    // Only bands of static shapes are reshapable. This is due to the fact that
    // there is no relation between dynamic sizes and dynamic strides: we do not
    // have enough information to know whether a "-1" size corresponds to the
    // proper symbol in the AffineExpr of a stride.
    if (ShapedType::isDynamic(sizes[dim + 1]))
      return false;
    // TODO: Refine this by passing the proper nDims and nSymbols so we can
    // simplify on the fly and catch more reshapable cases.
    if (strides[idx] != strides[idx + 1] * sizes[idx + 1])
      return false;
  }
  return true;
}

/// Compute the MemRefType obtained by applying the `reassociation` (which is
/// expected to be valid) to `type`.
/// If `type` is Contiguous MemRefType, this always produce a contiguous
/// MemRefType.
static MemRefType
computeReshapeCollapsedType(MemRefType type,
                            ArrayRef<AffineMap> reassociation) {
  auto sizes = type.getShape();
  AffineExpr offset;
  SmallVector<AffineExpr, 4> strides;
  auto status = getStridesAndOffset(type, strides, offset);
  (void)status;
  assert(succeeded(status) && "expected strided memref");

  SmallVector<int64_t, 4> newSizes;
  newSizes.reserve(reassociation.size());
  SmallVector<AffineExpr, 4> newStrides;
  newStrides.reserve(reassociation.size());

  // Use the fact that reassociation is valid to simplify the logic: only use
  // each map's rank.
  assert(isReassociationValid(reassociation) && "invalid reassociation");
  unsigned currentDim = 0;
  for (AffineMap m : reassociation) {
    unsigned dim = m.getNumResults();
    int64_t size = 1;
    AffineExpr stride = strides[currentDim + dim - 1];
    if (!isReshapableDimBand(currentDim, dim, sizes, strides)) {
      size = ShapedType::kDynamicSize;
      stride = AffineExpr();
    } else {
      for (unsigned d = 0; d < dim; ++d)
        size *= sizes[currentDim + d];
    }
    newSizes.push_back(size);
    newStrides.push_back(stride);
    currentDim += dim;
  }

  // Early-exit: if `type` is contiguous, the result must be contiguous.
  if (canonicalizeStridedLayout(type).getAffineMaps().empty())
    return MemRefType::Builder(type).setShape(newSizes).setAffineMaps({});

  // Convert back to int64_t because we don't have enough information to create
  // new strided layouts from AffineExpr only. This corresponds to a case where
  // copies may be necessary.
  int64_t intOffset = ShapedType::kDynamicStrideOrOffset;
  if (auto o = offset.dyn_cast<AffineConstantExpr>())
    intOffset = o.getValue();
  SmallVector<int64_t, 4> intStrides;
  intStrides.reserve(strides.size());
  for (auto stride : newStrides) {
    if (auto cst = stride.dyn_cast_or_null<AffineConstantExpr>())
      intStrides.push_back(cst.getValue());
    else
      intStrides.push_back(ShapedType::kDynamicStrideOrOffset);
  }
  auto layout =
      makeStridedLinearLayoutMap(intStrides, intOffset, type.getContext());
  return canonicalizeStridedLayout(
      MemRefType::Builder(type).setShape(newSizes).setAffineMaps({layout}));
}


template <typename AffineExprTy>
unsigned getMaxPosOfType(ArrayRef<ReassociationExprs> exprArrays) {
  unsigned pos = 0;
  for (const auto &exprs : exprArrays) {
    for (auto expr : exprs) {
      expr.walk([&pos](AffineExpr e) {
        if (auto d = e.dyn_cast<AffineExprTy>())
          pos = std::max(pos, d.getPosition());
      });
    }
  }
  return pos;
}

static SmallVector<AffineMap, 4>
getSymbolLessAffineMaps(ArrayRef<ReassociationExprs> reassociation) {
  unsigned maxDim = getMaxPosOfType<AffineDimExpr>(reassociation);
  assert(getMaxPosOfType<AffineSymbolExpr>(reassociation) == 0 &&
         "Expected symbol-less expressions");
  SmallVector<AffineMap, 4> maps;
  maps.reserve(reassociation.size());
  for (const auto &exprs : reassociation) {
    assert(!exprs.empty());
    maps.push_back(AffineMap::get(maxDim + 1, 0, exprs, exprs[0].getContext()));
  }
  return maps;
}

static SmallVector<ReassociationIndices, 2> convertReassociationMapsToIndices(
    OpBuilder &b, ArrayRef<ReassociationExprs> reassociationExprs) {
  SmallVector<ReassociationIndices, 2> reassociationIndices;
  for (const auto &exprs : reassociationExprs) {
    ReassociationIndices indices;
    indices.reserve(exprs.size());
    for (const auto &expr : exprs)
      indices.push_back(expr.cast<AffineDimExpr>().getPosition());
    reassociationIndices.push_back(indices);
  }
  return reassociationIndices;
}

static SmallVector<SmallVector<AffineExpr, 2>, 2>
convertReassociationIndicesToExprs(
    OpBuilder &b, ArrayRef<ReassociationIndices> reassociationIndices) {
  SmallVector<SmallVector<AffineExpr, 2>, 2> reassociationMaps;
  for (const auto &indices : reassociationIndices) {
    SmallVector<AffineExpr, 2> reassociationMap;
    reassociationMap.reserve(indices.size());
    for (int64_t index : indices)
      reassociationMap.push_back(b.getAffineDimExpr(index));
    reassociationMaps.push_back(std::move(reassociationMap));
  }
  return reassociationMaps;
}

SmallVector<AffineMap, 4> ReshapeOp::getReassociationMaps() {
  return getSymbolLessAffineMaps(getReassociationExprs());
}
SmallVector<ReassociationExprs, 4> ReshapeOp::getReassociationExprs() {
  OpBuilder b(this->getContext());
  return convertReassociationIndicesToExprs(b, getReassociationIndices());
}
SmallVector<AffineMap, 4> TensorReshapeOp::getReassociationMaps() {
  return getSymbolLessAffineMaps(getReassociationExprs());
}
SmallVector<ReassociationExprs, 4> TensorReshapeOp::getReassociationExprs() {
  OpBuilder b(this->getContext());
  return convertReassociationIndicesToExprs(b, getReassociationIndices());
}
/// For reshape op compute the shape at dimension `dimIndex` of the output in
/// terms of shape of the `src`, when the reshape op is a collapsing
/// operation. It is the product of the shape of the collapsed dimensions of the
/// `src`.
static OpFoldResult
getCollapsedOutputDimFromInputShape(OpBuilder &builder, Location loc,
                                    int64_t dimIndex, Value src,
                                    ArrayRef<AffineMap> reassociationMap) {
  AffineMap map = reassociationMap[dimIndex];
  unsigned startPos =
      map.getResults().front().cast<AffineDimExpr>().getPosition();
  unsigned endPos = map.getResults().back().cast<AffineDimExpr>().getPosition();
  AffineExpr expr;
  SmallVector<Value, 2> dynamicDims;
  for (auto dim : llvm::seq(startPos, endPos + 1)) {
    dynamicDims.push_back(builder.createOrFold<memref::DimOp>(loc, src, dim));
    AffineExpr currExpr = builder.getAffineSymbolExpr(dim - startPos);
    expr = (expr ? expr * currExpr : currExpr);
  }
  return applyMapToValues(builder, loc,
                          AffineMap::get(0, endPos - startPos + 1, expr),
                          dynamicDims)[0];
}

/// Given the `src` of a collapsing reshape op and its reassociation maps,
/// compute the shape of the result of the reshape.
static SmallVector<OpFoldResult, 4> getCollapsedOutputShapeFromInputShape(
    OpBuilder &builder, Location loc, Value src,
    ArrayRef<int64_t> dstStaticShape, ArrayRef<AffineMap> reassociation) {
  return llvm::to_vector<4>(llvm::map_range(
      llvm::seq<int64_t>(0, dstStaticShape.size()), [&](int64_t dim) {
        return getCollapsedOutputDimFromInputShape(builder, loc, dim, src,
                                                   reassociation);
      }));
}

/// Compute a map that for a given dimension of the expanded type gives the
/// dimension in the collapsed type it maps to. Essentially its the inverse of
/// the `reassocation` maps.
static llvm::DenseMap<int64_t, int64_t>
getExpandedDimToCollapsedDimMap(ArrayRef<AffineMap> reassociation) {
  llvm::DenseMap<int64_t, int64_t> expandedDimToCollapsedDim;
  for (auto map : enumerate(reassociation)) {
    unsigned startPos =
        map.value().getResults().front().cast<AffineDimExpr>().getPosition();
    unsigned endPos =
        map.value().getResults().back().cast<AffineDimExpr>().getPosition();
    for (auto dim : llvm::seq(startPos, endPos + 1)) {
      expandedDimToCollapsedDim[dim] = map.index();
    }
  }
  return expandedDimToCollapsedDim;
}

/// For an expanding reshape op, compute the value for a dimension of the output
/// from the shape of the input.
static OpFoldResult getExpandedOutputDimFromInputShape(
    OpBuilder &builder, Location loc, int64_t dimIndex, Value src,
    ArrayRef<int64_t> dstStaticShape, ArrayRef<AffineMap> reassociation,
    llvm::DenseMap<int64_t, int64_t> &expandedDimToCollapsedDim) {
  if (!ShapedType::isDynamic(dstStaticShape[dimIndex])) {
    return builder.getI64IntegerAttr(dstStaticShape[dimIndex]);
  }
  unsigned sourceDimPos = expandedDimToCollapsedDim[dimIndex];
  unsigned startPos = reassociation[sourceDimPos]
                          .getResults()
                          .front()
                          .cast<AffineDimExpr>()
                          .getPosition();
  unsigned endPos = reassociation[sourceDimPos]
                        .getResults()
                        .back()
                        .cast<AffineDimExpr>()
                        .getPosition();
  int64_t linearizedStaticDim = 1;
  for (auto d :
       llvm::enumerate(dstStaticShape.slice(startPos, endPos - startPos + 1))) {
    if (d.index() + startPos == static_cast<unsigned>(dimIndex))
      continue;
    assert(!ShapedType::isDynamic(d.value()) &&
           "single dimension cannot be expanded into multiple dynamic "
           "dimensions");
    linearizedStaticDim *= d.value();
  }
  Value sourceDim = builder.create<memref::DimOp>(loc, src, sourceDimPos);
  return applyMapToValues(
      builder, loc,
      AffineMap::get(
          0, 1, builder.getAffineSymbolExpr(0).floorDiv(linearizedStaticDim)),
      sourceDim)[0];
}

/// Given the `src` of an expanding reshape op, the reassociation maps and the
/// result type, compute the shape of the result of the reshape.
static SmallVector<OpFoldResult, 4> getExpandedOutputShapeFromInputShape(
    OpBuilder &builder, Location loc, Value src,
    ArrayRef<int64_t> dstStaticShape, ArrayRef<AffineMap> reassociation) {
  llvm::DenseMap<int64_t, int64_t> expandedDimToCollapsedDim =
      getExpandedDimToCollapsedDimMap(reassociation);
  return llvm::to_vector<4>(llvm::map_range(
      llvm::seq<int64_t>(0, dstStaticShape.size()), [&](int64_t dim) {
        return getExpandedOutputDimFromInputShape(builder, loc, dim, src,
                                                  dstStaticShape, reassociation,
                                                  expandedDimToCollapsedDim);
      }));
}

static SmallVector<OpFoldResult, 4>
getReshapeOutputShapeFromInputShape(OpBuilder &builder, Location loc, Value src,
                                    ArrayRef<int64_t> dstStaticShape,
                                    ArrayRef<AffineMap> reassocation) {
  return dstStaticShape.size() >
                 static_cast<size_t>(src.getType().cast<ShapedType>().getRank())
             ? getExpandedOutputShapeFromInputShape(
                   builder, loc, src, dstStaticShape, reassocation)
             : getCollapsedOutputShapeFromInputShape(
                   builder, loc, src, dstStaticShape, reassocation);
}

static ArrayAttr
getReassociationIndicesAttribute(OpBuilder &b,
                                 ArrayRef<ReassociationIndices> reassociation) {
  SmallVector<Attribute, 4> reassociationAttr =
      llvm::to_vector<4>(llvm::map_range(
          reassociation, [&](ReassociationIndices indices) -> Attribute {
            return b.getI64ArrayAttr(indices).cast<Attribute>();
          }));
  return b.getArrayAttr(reassociationAttr);
}

void mlir::linalg::ReshapeOp::build(
    OpBuilder &b, OperationState &result, Value src,
    ArrayRef<ReassociationIndices> reassociation,
    ArrayRef<NamedAttribute> attrs) {
  auto memRefType = src.getType().cast<MemRefType>();
  auto resultType = computeReshapeCollapsedType(
      memRefType, getSymbolLessAffineMaps(
                      convertReassociationIndicesToExprs(b, reassociation)));
  build(b, result, resultType, src, attrs);
  result.addAttribute(ReshapeOp::getReassociationAttrName(),
                      getReassociationIndicesAttribute(b, reassociation));
}

void mlir::linalg::ReshapeOp::build(
    OpBuilder &b, OperationState &result, Type resultType, Value src,
    ArrayRef<ReassociationIndices> reassociation,
    ArrayRef<NamedAttribute> attrs) {
  build(b, result, resultType, src, attrs);
  result.addAttribute(ReshapeOp::getReassociationAttrName(),
                      getReassociationIndicesAttribute(b, reassociation));
}

Value mlir::linalg::ReshapeOp::getViewSource() { return src(); }

/// Verify that shapes of the reshaped types using following rules
/// 1) if a dimension in the collapsed type is static, then the corresponding
///    dimensions in the expanded shape should be
///    a) static
///    b) the product should be same as the collaped shape.
/// 2) if a dimension in the collaped type is dynamic, one and only one of the
///    corresponding dimensions in the expanded type should be dynamic. This
///    rule is only needed with reshape operations that are expanding.
template <typename OpTy>
static LogicalResult verifyReshapeLikeShapes(OpTy op, ShapedType collapsedType,
                                             ShapedType expandedType,
                                             bool isExpandingReshape) {
  ArrayRef<int64_t> collapsedShape = collapsedType.getShape();
  ArrayRef<int64_t> expandedShape = expandedType.getShape();
  unsigned expandedDimStart = 0;
  for (auto map : llvm::enumerate(op.getReassociationMaps())) {
    Optional<int64_t> dynamicShape;
    int64_t linearizedStaticShape = 1;
    for (auto dim : llvm::enumerate(expandedShape.slice(
             expandedDimStart, map.value().getNumResults()))) {
      if (ShapedType::isDynamic(dim.value())) {
        if (isExpandingReshape && dynamicShape) {
          return op->emitOpError("invalid to have a single dimension (")
                 << map.index() << ") expanded into multiple dynamic dims ("
                 << expandedDimStart + dynamicShape.getValue() << ","
                 << expandedDimStart + dim.index() << ")";
        }
        dynamicShape = dim.index();
      } else {
        linearizedStaticShape *= dim.value();
      }
    }
    if (dynamicShape) {
      if (!ShapedType::isDynamic(collapsedShape[map.index()])) {
        return op->emitOpError("expected dimension ")
               << map.index()
               << " of collapsed type to be dynamic since one or more of the "
                  "corresponding dimensions in the expanded type is dynamic";
      }
    } else {
      if (collapsedShape[map.index()] != linearizedStaticShape) {
        return op->emitOpError("expected dimension ")
               << map.index() << " of collapsed type to be static value of "
               << linearizedStaticShape << " ";
      }
    }
    expandedDimStart += map.value().getNumResults();
  }
  return success();
}

// Common verifier for reshape-like types. Fills `expandedType` and
// `collapsedType` with the proper `src` or `result` type.
template <typename Op, typename T>
static LogicalResult verifyReshapeLikeTypes(Op op, T &expandedType,
                                            T &collapsedType) {
  expandedType = op.getSrcType();
  collapsedType = op.getResultType();
  unsigned expandedRank = expandedType.getRank();
  unsigned collapsedRank = collapsedType.getRank();
  bool isCollapse = expandedRank > collapsedRank;
  if (!isCollapse) {
    std::swap(expandedRank, collapsedRank);
    std::swap(expandedType, collapsedType);
  }
  if (expandedRank == 0)
    return op.emitOpError("expected non-zero memref ranks");
  if (expandedRank == collapsedRank)
    return op.emitOpError("expected to collapse or expand dims");

  if (collapsedRank == 0) {
    // If collapsed rank is 0, then expanded type must be static shaped and of
    // sizes 1.
    if (llvm::any_of(expandedType.getShape(),
                     [](int64_t dim) -> bool { return dim != 1; }))
      return op.emitOpError("invalid to reshape tensor/memref with non-unit "
                            "extent dimensions to zero-rank tensor/memref");
    return success();
  }
  if (collapsedRank != op.reassociation().size())
    return op.emitOpError("expected rank of the collapsed type(")
           << collapsedRank << ") to be the number of reassociation maps("
           << op.reassociation().size() << ")";
  auto maps = op.getReassociationMaps();
  for (auto it : llvm::enumerate(maps))
    if (it.value().getNumDims() != expandedRank)
      return op.emitOpError("expected reassociation map #")
             << it.index() << " of same rank as expanded memref("
             << expandedRank << "), but got " << it.value().getNumDims();
  int invalidIdx = 0;
  if (!isReassociationValid(maps, &invalidIdx))
    return op.emitOpError("expected reassociation map #")
           << invalidIdx << " to be valid and contiguous";
  return verifyReshapeLikeShapes(op, collapsedType, expandedType, !isCollapse);
}

static LogicalResult verify(ReshapeOp op) {
  MemRefType expandedType, collapsedType;
  if (failed(verifyReshapeLikeTypes(op, expandedType, collapsedType)))
    return failure();
  auto maps = op.getReassociationMaps();
  MemRefType expectedType = computeReshapeCollapsedType(expandedType, maps);
  if (collapsedType != expectedType)
    return op.emitOpError("expected collapsed type to be ")
           << expectedType << ", but got " << collapsedType;
  return success();
}

void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<CollapseReshapeOps<ReshapeOp>>(context);
}

//===----------------------------------------------------------------------===//
// TensorReshapeOp
//===----------------------------------------------------------------------===//

/// Compute the RankedTensorType obtained by applying `reassociation` to `type`.
static RankedTensorType
computeTensorReshapeCollapsedType(RankedTensorType type,
                                  ArrayRef<AffineMap> reassociation) {
  auto shape = type.getShape();
  SmallVector<int64_t, 4> newShape;
  newShape.reserve(reassociation.size());

  // Use the fact that reassociation is valid to simplify the logic: only use
  // each map's rank.
  assert(isReassociationValid(reassociation) && "invalid reassociation");
  unsigned currentDim = 0;
  for (AffineMap m : reassociation) {
    unsigned dim = m.getNumResults();
    auto band = shape.slice(currentDim, dim);
    int64_t size = 1;
    if (llvm::is_contained(band, ShapedType::kDynamicSize))
      size = ShapedType::kDynamicSize;
    else
      for (unsigned d = 0; d < dim; ++d)
        size *= shape[currentDim + d];
    newShape.push_back(size);
    currentDim += dim;
  }

  return RankedTensorType::get(newShape, type.getElementType());
}

void mlir::linalg::TensorReshapeOp::build(
    OpBuilder &b, OperationState &result, Value src,
    ArrayRef<ReassociationIndices> reassociation,
    ArrayRef<NamedAttribute> attrs) {
  auto resultType = computeTensorReshapeCollapsedType(
      src.getType().cast<RankedTensorType>(),
      getSymbolLessAffineMaps(
          convertReassociationIndicesToExprs(b, reassociation)));
  build(b, result, resultType, src, attrs);
  result.addAttribute(ReshapeOp::getReassociationAttrName(),
                      getReassociationIndicesAttribute(b, reassociation));
}

void mlir::linalg::TensorReshapeOp::build(
    OpBuilder &b, OperationState &result, Type resultType, Value src,
    ArrayRef<ReassociationIndices> reassociation,
    ArrayRef<NamedAttribute> attrs) {
  build(b, result, resultType, src, attrs);
  result.addAttribute(ReshapeOp::getReassociationAttrName(),
                      getReassociationIndicesAttribute(b, reassociation));
}

static LogicalResult verify(TensorReshapeOp op) {
  RankedTensorType expandedType, collapsedType;
  if (failed(verifyReshapeLikeTypes(op, expandedType, collapsedType)))
    return failure();

  auto maps = op.getReassociationMaps();
  RankedTensorType expectedType =
      computeTensorReshapeCollapsedType(expandedType, maps);
  if (collapsedType != expectedType)
    return op.emitOpError("expected collapsed type to be ")
           << expectedType << ", but got " << collapsedType;
  return success();
}

namespace {
/// Reshape of a splat constant can be replaced with a constant of the result
/// type.
struct FoldReshapeWithConstant : OpRewritePattern<TensorReshapeOp> {
  using OpRewritePattern<TensorReshapeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TensorReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    DenseElementsAttr attr;
    if (!matchPattern(reshapeOp.src(), m_Constant(&attr)))
      return failure();
    if (!attr || !attr.isSplat())
      return failure();
    DenseElementsAttr newAttr = DenseElementsAttr::getFromRawBuffer(
        reshapeOp.getResultType(), attr.getRawData(), true);
    rewriter.replaceOpWithNewOp<ConstantOp>(reshapeOp, newAttr);
    return success();
  }
};

/// Fold linalg.fill -> linalg.tensor_reshape chain.
///
/// For such op chains, we can create new linalg.fill ops with the result
/// type of the linalg.tensor_reshape op.
struct FoldFillWithTensorReshape : OpRewritePattern<TensorReshapeOp> {
  using OpRewritePattern<TensorReshapeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TensorReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    auto oldFill = reshapeOp.src().getDefiningOp<FillOp>();
    if (!oldFill)
      return failure();

    Location loc = oldFill.getLoc();
    auto newInit = rewriter.create<TensorReshapeOp>(
        loc, reshapeOp.getResultType(), oldFill.output(),
        reshapeOp.reassociation());
    rewriter.replaceOpWithNewOp<FillOp>(reshapeOp, newInit, oldFill.value());

    return success();
  }
};
} // namespace

void TensorReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.add<CollapseReshapeOps<TensorReshapeOp>, FoldFillWithTensorReshape,
              FoldInitTensorWithTensorReshapeOp, FoldReshapeWithConstant>(
      context);
}

LogicalResult TensorReshapeOp::reifyReturnTypeShapesPerResultDim(
    OpBuilder &b, SmallVectorImpl<SmallVector<Value>> &reifiedReturnShapes) {
  auto resultShape =
      getAsValues(b, getLoc(),
                  getReshapeOutputShapeFromInputShape(
                      b, getLoc(), src(), getResultType().getShape(),
                      getReassociationMaps()));
  reifiedReturnShapes.emplace_back(std::move(resultShape));
  return success();
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, linalg::YieldOp op) {
  p << op.getOperationName();
  if (op.getNumOperands() > 0)
    p << ' ' << op.getOperands();
  p.printOptionalAttrDict(op->getAttrs());
  if (op.getNumOperands() > 0)
    p << " : " << op.getOperandTypes();
}

static ParseResult parseYieldOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 2> opInfo;
  SmallVector<Type, 2> types;
  llvm::SMLoc loc = parser.getCurrentLocation();
  return failure(parser.parseOperandList(opInfo) ||
                 parser.parseOptionalAttrDict(result.attributes) ||
                 (!opInfo.empty() && parser.parseColonTypeList(types)) ||
                 parser.resolveOperands(opInfo, types, loc, result.operands));
}

// Check the operand number and types must match the element types of the
// LinalgOp interface's shaped operands.
static LogicalResult verifyYield(linalg::YieldOp op,
                                 LinalgOp linalgOpInterface) {
  auto nOutputs = linalgOpInterface.getNumOutputs();
  if (op.getNumOperands() != nOutputs)
    return op.emitOpError("expected number of yield values (")
           << nOutputs << ") to match the number of operands of the enclosing "
           << "LinalgOp (" << op.getNumOperands() << ")";

  for (unsigned i = 0; i != nOutputs; ++i) {
    auto elementType =
        linalgOpInterface.getOutputShapedType(i).getElementType();
    if (op.getOperand(i).getType() != elementType)
      return op.emitOpError("type of yield operand ")
             << (i + 1) << " (" << op.getOperand(i).getType()
             << ") doesn't match "
             << "the element type of the enclosing linalg.generic op ("
             << elementType << ")";
  }
  return success();
}

static LogicalResult verify(linalg::YieldOp op) {
  auto *parentOp = op->getParentOp();
  if (parentOp->getNumRegions() != 1 || parentOp->getRegion(0).empty())
    return op.emitOpError("expected single non-empty parent region");

  if (auto linalgOp = dyn_cast<LinalgOp>(parentOp))
    return verifyYield(op, cast<LinalgOp>(parentOp));

  if (auto padTensorOp = dyn_cast<linalg::PadTensorOp>(parentOp)) {
    if (op.getNumOperands() != 1)
      return op.emitOpError("expected single yield operand (got ")
             << op->getNumOperands() << ")";
    if (op.getOperand(0).getType() !=
        padTensorOp.getType().cast<ShapedType>().getElementType())
      return op.emitOpError("expected yield type to match shape element type");
    return success();
  }

  if (auto tiledLoopOp = dyn_cast<linalg::TiledLoopOp>(parentOp)) {
    // Check if output args with tensor types match results types.
    SmallVector<Value, 2> tensorOuts;
    llvm::copy_if(
        tiledLoopOp.outputs(), std::back_inserter(tensorOuts),
        [&](Value out) { return out.getType().isa<RankedTensorType>(); });
    if (tensorOuts.size() != op.values().size())
      return op.emitOpError("expected number of tensor output args = ")
             << tensorOuts.size() << " to match the number of yield operands = "
             << op.values().size();

    TypeRange tensorTypes(llvm::makeArrayRef(tensorOuts));
    for (auto &item :
         llvm::enumerate(llvm::zip(tensorTypes, op.getOperandTypes()))) {
      Type outType, resultType;
      unsigned index = item.index();
      std::tie(outType, resultType) = item.value();
      if (outType != resultType)
        return op.emitOpError("expected yield operand ")
               << index << " with type = " << resultType
               << " to match output arg type = " << outType;
    }
    return success();
  }
  return op.emitOpError("expected parent op with LinalgOp interface");
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

  // Add output types for `RankedTensorType` output arguments.
  for (Value output : outputs) {
    Type outputType = output.getType();
    if (outputType.isa<RankedTensorType>())
      result.addTypes(outputType);
  }

  OpBuilder::InsertionGuard guard(builder);
  unsigned numIVs = steps.size();
  SmallVector<Type, 8> argTypes(numIVs, builder.getIndexType());
  for (Type type : TypeRange(inputs))
    argTypes.push_back(type);
  for (Type type : TypeRange(outputs))
    argTypes.push_back(type);
  Region *bodyRegion = result.addRegion();
  Block *bodyBlock = builder.createBlock(bodyRegion, {}, argTypes);

  if (bodyBuilderFn) {
    builder.setInsertionPointToStart(bodyBlock);
    bodyBuilderFn(builder, result.location,
                  bodyBlock->getArguments().take_front(numIVs),
                  bodyBlock->getArguments().slice(numIVs, inputs.size()),
                  bodyBlock->getArguments().take_back(outputs.size()));
    TiledLoopOp::ensureTerminator(*bodyRegion, builder, result.location);
  }
}

static void print(OpAsmPrinter &p, TiledLoopOp op) {
  p << op.getOperationName() << " (" << op.getInductionVars() << ") = ("
    << op.lowerBound() << ") to (" << op.upperBound() << ") step (" << op.step()
    << ")";

  if (!op.inputs().empty()) {
    p << " ins (";
    llvm::interleaveComma(llvm::zip(op.getRegionInputArgs(), op.inputs()), p,
                          [&](auto it) {
                            p << std::get<0>(it) << " = " << std::get<1>(it)
                              << ": " << std::get<1>(it).getType();
                          });
    p << ")";
  }
  if (!op.outputs().empty()) {
    p << " outs (";
    llvm::interleaveComma(llvm::zip(op.getRegionOutputArgs(), op.outputs()), p,
                          [&](auto it) {
                            p << std::get<0>(it) << " = " << std::get<1>(it)
                              << ": " << std::get<1>(it).getType();
                          });
    p << ")";
  }

  if (llvm::any_of(op.iterator_types(), [](Attribute attr) {
        return attr.cast<StringAttr>().getValue() !=
               getParallelIteratorTypeName();
      })) {
    p << " iterators" << op.iterator_types() << "";
  }

  p.printRegion(op.region(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(
      op->getAttrs(), /*elidedAttrs=*/{TiledLoopOp::getOperandSegmentSizeAttr(),
                                       getIteratorTypesAttrName()});
}

static ParseResult parseTiledLoopOp(OpAsmParser &parser,
                                    OperationState &result) {
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
  SmallVector<OpAsmParser::OperandType, 4> inputs, input_region_args;
  SmallVector<Type, 4> inputTypes;
  if (succeeded(parser.parseOptionalKeyword("ins"))) {
    llvm::SMLoc inputsOperandsLoc = parser.getCurrentLocation();

    if (parser.parseAssignmentListWithTypes(input_region_args, inputs,
                                            inputTypes))
      return failure();

    if (parser.resolveOperands(inputs, inputTypes, inputsOperandsLoc,
                               result.operands))
      return failure();
  }

  // Parse output tensors.
  SmallVector<OpAsmParser::OperandType, 4> outputs, output_region_args;
  SmallVector<Type, 4> outputTypes;
  if (succeeded(parser.parseOptionalKeyword("outs"))) {
    llvm::SMLoc outputsOperandsLoc = parser.getCurrentLocation();

    if (parser.parseAssignmentListWithTypes(output_region_args, outputs,
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
  SmallVector<Attribute, 4> iterTypes;
  if (succeeded(parser.parseOptionalKeyword("iterators"))) {
    StringAttr iterType;

    if (parser.parseLSquare() || parser.parseAttribute(iterType))
      return failure();
    iterTypes.push_back(iterType);
    for (int i = 1, e = ivs.size(); i < e; ++i) {
      if (parser.parseComma() || parser.parseAttribute(iterType))
        return failure();
      iterTypes.push_back(iterType);
    }
    if (parser.parseRSquare())
      return failure();
  } else {
    auto parallelIter = builder.getStringAttr(getParallelIteratorTypeName());
    iterTypes = SmallVector<Attribute, 4>(ivs.size(), parallelIter);
  }
  result.addAttribute(getIteratorTypesAttrName(),
                      builder.getArrayAttr(iterTypes));
  result.addAttribute(
      TiledLoopOp::getOperandSegmentSizeAttr(),
      builder.getI32VectorAttr({static_cast<int32_t>(lower.size()),
                                static_cast<int32_t>(upper.size()),
                                static_cast<int32_t>(steps.size()),
                                static_cast<int32_t>(inputs.size()),
                                static_cast<int32_t>(outputs.size())}));

  // Parse the body.
  Region *body = result.addRegion();

  SmallVector<Type, 4> region_types(ivs.size(), builder.getIndexType());
  region_types.append(inputTypes);
  region_types.append(outputTypes);

  SmallVector<OpAsmParser::OperandType, 4> region_args(ivs);
  region_args.append(input_region_args);
  region_args.append(output_region_args);

  if (parser.parseRegion(*body, region_args, region_types))
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

static LogicalResult verify(TiledLoopOp op) {
  // Check if iterator types are provided for every loop dimension.
  if (op.iterator_types().size() != op.getNumLoops())
    return op.emitOpError("expected iterator types array attribute size = ")
           << op.iterator_types().size()
           << " to match the number of loops = " << op.getNumLoops();

  // Check if types of input arguments match region args types.
  for (auto &item :
       llvm::enumerate(llvm::zip(op.inputs(), op.getRegionInputArgs()))) {
    Value input, inputRegionArg;
    unsigned index = item.index();
    std::tie(input, inputRegionArg) = item.value();
    if (input.getType() != inputRegionArg.getType())
      return op.emitOpError("expected input arg ")
             << index << " with type = " << input.getType()
             << " to match region arg " << index + op.getNumLoops()
             << " type = " << inputRegionArg.getType();
  }

  // Check if types of input arguments match region args types.
  for (auto &item :
       llvm::enumerate(llvm::zip(op.outputs(), op.getRegionOutputArgs()))) {
    Value output, outputRegionArg;
    unsigned index = item.index();
    std::tie(output, outputRegionArg) = item.value();
    if (output.getType() != outputRegionArg.getType())
      return op.emitOpError("expected output arg ")
             << index << " with type = " << output.getType()
             << " to match region arg "
             << index + op.getNumLoops() + op.inputs().size()
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
    for (auto en : llvm::enumerate(
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
        newInputs, tiledLoop.outputs(), tiledLoop.iterator_types());

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
    for (auto en : llvm::enumerate(
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
        tiledLoop.inputs(), newOutputOperands, tiledLoop.iterator_types());

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

void TiledLoopOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<TiledLoopInputsFolder, TiledLoopResultsFolder>(context);
}

LogicalResult TiledLoopOp::fold(ArrayRef<Attribute>,
                                SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCastInTiledLoopOp(*this);
}

//===----------------------------------------------------------------------===//
// IndexOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(IndexOp op) {
  auto linalgOp = dyn_cast<LinalgOp>(op->getParentOp());
  if (!linalgOp)
    return op.emitOpError("expected parent op with LinalgOp interface");
  if (linalgOp.getNumLoops() <= op.dim())
    return op.emitOpError("expected dim (")
           << op.dim() << ") to be lower than the number of loops ("
           << linalgOp.getNumLoops() << ") of the enclosing LinalgOp";
  return success();
}

/////// Operations corresponding to library calls defined with Tablegen ////////

template <typename LinalgPoolingOp>
static LogicalResult verifyStrideOrDilation(LinalgPoolingOp op,
                                            ArrayRef<Attribute> attrs,
                                            bool isStride) {
  auto strideOrDilation = isStride ? "stride" : "dilation";
  if (attrs.size() != op.getNumWindowLoops())
    return op.emitOpError("expects num ")
           << strideOrDilation
           << "s equal to number of window dimensions: " << attrs.size()
           << " vs " << op.getNumWindowLoops();
  return success();
}

void ConvOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), input(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), filter(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), output(),
                       SideEffects::DefaultResource::get());
}

static LogicalResult verify(ConvOp op) {
  auto oType = op.output().getType().cast<MemRefType>();
  auto fType = op.filter().getType().cast<MemRefType>();
  auto iType = op.input().getType().cast<MemRefType>();
  if (oType.getElementType() != iType.getElementType() ||
      oType.getElementType() != fType.getElementType())
    return op.emitOpError("expects memref elemental types to match");
  if (oType.getRank() != iType.getRank() || oType.getRank() != fType.getRank())
    return op.emitOpError("expects memref ranks to match");
  if (auto strides = op.strides()) {
    if (failed(verifyStrideOrDilation(op, strides->getValue(),
                                      /*isStride=*/true)))
      return failure();
  }
  if (auto dilations = op.dilations()) {
    if (failed(verifyStrideOrDilation(op, dilations->getValue(),
                                      /*isStride=*/false)))
      return failure();
  }
  return success();
}

template <typename PoolingOp>
static LogicalResult verifySingleInputPoolingOp(PoolingOp op) {
  auto inputType = op.input().getType().template cast<MemRefType>();
  auto outputType = op.output().getType().template cast<MemRefType>();
  if (outputType.getElementType() != inputType.getElementType())
    return op.emitOpError("expects memref elemental types to match");

  auto windowDimsType = op.windowDims().getType().template cast<MemRefType>();
  if (outputType.getRank() != inputType.getRank() ||
      outputType.getRank() != windowDimsType.getRank())
    return op.emitOpError("expects memref ranks to match");

  if (auto strides = op.strides()) {
    if (failed(verifyStrideOrDilation(op, strides->getValue(),
                                      /*isStride=*/true)))
      return failure();
  }
  if (auto dilations = op.dilations()) {
    if (failed(verifyStrideOrDilation(op, dilations->getValue(),
                                      /*isStride=*/false)))
      return failure();
  }
  return success();
}

#define DEFINE_POOLING_OP_GET_EFFECTS(OP_NAME)                                 \
  void OP_NAME::getEffects(                                                    \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>      \
          &effects) {                                                          \
    effects.emplace_back(MemoryEffects::Read::get(), input(),                  \
                         SideEffects::DefaultResource::get());                 \
    effects.emplace_back(MemoryEffects::Write::get(), output(),                \
                         SideEffects::DefaultResource::get());                 \
  }

static LogicalResult verify(PoolingMaxOp op) {
  return verifySingleInputPoolingOp(op);
}
static LogicalResult verify(PoolingMinOp op) {
  return verifySingleInputPoolingOp(op);
}
static LogicalResult verify(PoolingSumOp op) {
  return verifySingleInputPoolingOp(op);
}

DEFINE_POOLING_OP_GET_EFFECTS(PoolingMaxOp)
DEFINE_POOLING_OP_GET_EFFECTS(PoolingMinOp)
DEFINE_POOLING_OP_GET_EFFECTS(PoolingSumOp)

namespace {
struct EraseDeadLinalgOp;
struct FoldTensorCastOp;
} // namespace

#include "mlir/Dialect/Linalg/IR/LinalgNamedStructuredOps.tcgen.cpp.inc"
#include "mlir/Dialect/Linalg/IR/LinalgNamedStructuredOps.yamlgen.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgOps.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"

/// Return the dims that are `iteratorTypeName` loops in the LinalgOp `op`.
/// Assumes `op` is a LinalgOp.
void mlir::linalg::getDimsOfType(Operation *op, StringRef iteratorTypeName,
                                 SmallVectorImpl<AffineExpr> &res) {
  if (!cast<LinalgOp>(op).iterator_types())
    return;

  unsigned dim = 0;
  MLIRContext *ctx = op->getContext();
  for (auto tn :
       cast<LinalgOp>(op).iterator_types().getAsValueRange<StringAttr>()) {
    if (tn == iteratorTypeName)
      res.push_back(getAffineDimExpr(dim, ctx));
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

template <typename PoolingOp>
SmallVector<AffineExpr, 4>
mlir::linalg::weightedPoolingInputIndex(PoolingOp op,
                                        ArrayRef<AffineExpr> outputDims,
                                        ArrayRef<AffineExpr> windowDims) {
  assert(outputDims.size() == windowDims.size());
  SmallVector<AffineExpr, 4> res;
  res.reserve(outputDims.size());
  for (unsigned i = 0, e = outputDims.size(); i < e; ++i) {
    // TODO: add a level of indirection to linalg.generic.
    auto expr = op.getStride(i) * outputDims[i] +
                op.getDilation(i) * windowDims[i] - op.getLowPad(i);
    res.push_back(expr);
  }
  return res;
}

#define INSTANTIATE_WEIGHTED_POOLING_INPUT_INDEX(OP_TYPE)                      \
  template SmallVector<AffineExpr, 4>                                          \
  mlir::linalg::weightedPoolingInputIndex<OP_TYPE>(                            \
      OP_TYPE op, ArrayRef<AffineExpr> outputDims,                             \
      ArrayRef<AffineExpr> windowDims);

INSTANTIATE_WEIGHTED_POOLING_INPUT_INDEX(ConvOp)
INSTANTIATE_WEIGHTED_POOLING_INPUT_INDEX(PoolingMaxOp)
INSTANTIATE_WEIGHTED_POOLING_INPUT_INDEX(PoolingMinOp)
INSTANTIATE_WEIGHTED_POOLING_INPUT_INDEX(PoolingSumOp)

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

// TODO: Consider making all this boilerplate easy to autogenerate
// with Tablegen. This seems a desirable property in the context of
// OpInterfaces where a Linalg "named" op **isa** LinalgOp.
OpFoldResult ReshapeOp::fold(ArrayRef<Attribute> operands) {
  if (succeeded(foldMemRefCast(*this)))
    return getResult();
  return foldReshapeOp(*this, operands);
}
OpFoldResult TensorReshapeOp::fold(ArrayRef<Attribute> operands) {
  return foldReshapeOp(*this, operands);
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
static void
fillStructuredOpRegion(OpBuilder &opBuilder, Region &region,
                       TypeRange inputTypes, TypeRange outputTypes,
                       ValueRange captures,
                       std::function<void(unsigned, unsigned)> errorHandler) {
  assert(llvm::all_of(inputTypes, [](Type t) { return t.isa<ShapedType>(); }));
  assert(llvm::all_of(outputTypes, [](Type t) { return t.isa<ShapedType>(); }));

  // TODO: atm all operands go through getElementTypeOrSelf,
  // reconsider when we have evidence we need to.
  SmallVector<Type, 8> argTypes;
  for (auto containers : {inputTypes, outputTypes})
    for (auto t : containers)
      argTypes.push_back(getElementTypeOrSelf(t));

  // RAII.
  OpBuilder::InsertionGuard guard(opBuilder);
  Block *body = opBuilder.createBlock(&region, /*insertPt=*/{}, argTypes);
  unsigned actual = body->getNumArguments();
  unsigned expected = NamedStructuredOpType::getNumRegionArgs();
  if (expected != actual) {
    if (errorHandler)
      errorHandler(expected, actual);
    return;
  }

  opBuilder.setInsertionPointToStart(body);
  mlir::edsc::ScopedContext scope(opBuilder, opBuilder.getUnknownLoc());
  NamedStructuredOpType::regionBuilder(*body, captures);

  // indexing_maps is an auto-generated method.

  // iterator_types is an auto-generated method.
}

/// Generic entry point to create both the region and the block of a LinalgOp.
template <typename NamedStructuredOpType>
void createAndFillStructuredOpRegion(OpBuilder &opBuilder,
                                     OperationState &result,
                                     TypeRange inputTypes,
                                     TypeRange outputTypes,
                                     ValueRange captures) {
  Region &region = *result.addRegion();
  fillStructuredOpRegion<NamedStructuredOpType>(
      opBuilder, region, inputTypes, outputTypes, captures,
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
  llvm::SMLoc inputsOperandsLoc, outputsOperandsLoc;
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
                             ArrayRef<OpAsmParser::OperandType> captures) {
  ParseResult res = success();
  OpBuilder opBuilder(parser.getBuilder().getContext());
  // Resolve `captures` into `capturedValues` at parse time so we can build the
  // region with captures.
  SmallVector<Value> capturedValues;
  fillStructuredOpRegion<NamedStructuredOpType>(
      opBuilder, region, inputTypes, outputTypes, capturedValues,
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
  if (succeeded(parser.parseOptionalArrow()))
    if (parser.parseTypeList(resultTypes))
      return failure();
  return success();
}

template <typename NamedStructuredOpType>
static ParseResult
parseNamedStructuredOp(OpAsmParser &parser, OperationState &result,
                       ArrayRef<OpAsmParser::OperandType> captures) {
  // TODO: Enable when ods-gen supports captures.
  assert(captures.empty() && "unexpected captures for named structured ops");
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
          parser, *region, inputTypes, outputTypes, captures))
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
  p << op.getOperationName();
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
    for (Value v : op.getShapedOperands()) {
      // Linalg "inputs" may be either tensor or memref type.
      // tensor<0xelt_type> is a convention that may not always mean
      // "0 iterations". Only erase in cases we see memref<...x0x...>.
      auto mt = v.getType().dyn_cast<MemRefType>();
      if (!mt)
        continue;
      if (llvm::is_contained(mt.getShape(), 0)) {
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
        llvm::any_of(op.getShapedOperands(), [&](Value v) {
          if (v.isa<BlockArgument>())
            return false;
          auto castOp = v.getDefiningOp<tensor::CastOp>();
          return castOp && canFoldIntoConsumerOp(castOp);
        });
    if (!hasTensorCastOperand)
      return failure();

    SmallVector<Type, 4> newResultTypes;
    newResultTypes.reserve(op->getNumResults());
    SmallVector<Value, 4> newOperands;
    newOperands.reserve(op->getNumOperands());
    // Inputs may fold.
    for (Value v : op.getInputs()) {
      auto tensorCastOp = v.getDefiningOp<tensor::CastOp>();
      newOperands.push_back(
          canFoldIntoConsumerOp(tensorCastOp) ? tensorCastOp.source() : v);
    }
    // Init tensors may fold, in which case the resultType must also change.
    for (Value v : op.getOutputs()) {
      auto tensorCastOp = v.getDefiningOp<tensor::CastOp>();
      bool fold = canFoldIntoConsumerOp(tensorCastOp);
      newOperands.push_back(fold ? tensorCastOp.getOperand() : v);
      newResultTypes.push_back(newOperands.back().getType());
    }
    auto extraOperands = op.getAssumedNonShapedOperands();
    newOperands.append(extraOperands.begin(), extraOperands.end());
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

namespace {
// Deduplicate redundant args of a linalg op.
// An arg is redundant if it has the same Value and indexing map as another.
struct DeduplicateInputs : public OpInterfaceRewritePattern<LinalgOp> {
  using OpInterfaceRewritePattern<LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LinalgOp op,
                                PatternRewriter &rewriter) const override {
    // This pattern reduces the number of arguments of an op, which breaks
    // the invariants of semantically charged named ops.
    if (!isa<GenericOp, IndexedGenericOp>(op))
      return failure();

    // Associate each input to an equivalent "canonical" input that has the same
    // Value and indexing map.
    //
    // In the non-duplicate case, input `i` will have canonical input `i`. But
    // in the case of duplicated inputs, the canonical input could be some other
    // input `< i`. That is, a later input will have some earlier input as its
    // canonical input.
    llvm::SmallDenseMap<std::pair<Value, AffineMap>, int> canonicalInput;
    // For later remapping tasks like deduplicating payload block arguments,
    // having a simple "inputIndex -> canonicalInputIndex" integer mapping is
    // convenient.
    SmallVector<int, 6> canonicalInputIndices;
    for (int i = 0, e = op.getNumInputs(); i != e; i++) {
      Value input = op.getInput(i);
      AffineMap indexingMap = op.getInputIndexingMap(i);
      // STL-like maps have a convenient behavior for our use case here. In the
      // case of duplicate keys, the insertion is rejected, and the returned
      // iterator gives access to the value already in the map.
      auto pair = canonicalInput.insert({{input, indexingMap}, i});
      canonicalInputIndices.push_back(pair.first->second);
    }

    // If there are no duplicate args, then bail out.
    if (canonicalInput.size() == op.getNumInputs())
      return failure();

    // The operands for the newly canonicalized op.
    SmallVector<Value, 6> newOperands;
    for (auto v : llvm::enumerate(op.getInputs()))
      if (canonicalInputIndices[v.index()] == static_cast<int>(v.index()))
        newOperands.push_back(v.value());
    llvm::append_range(newOperands, op.getOutputs());
    llvm::append_range(newOperands, op.getAssumedNonShapedOperands());

    // Clone the old op with new operands.
    Operation *newOp =
        op.clone(rewriter, op->getLoc(), op->getResultTypes(), newOperands);
    auto newLinalgOp = cast<LinalgOp>(newOp);

    // Repair the indexing maps by filtering out the ones that have been
    // eliminated.
    SmallVector<AffineMap, 6> newIndexingMaps;
    for (int i = 0, e = newLinalgOp.getNumInputs(); i != e; i++)
      if (canonicalInputIndices[i] == i)
        newIndexingMaps.push_back(newLinalgOp.getIndexingMap(i));
    for (int i = 0, e = newLinalgOp.getNumOutputs(); i != e; i++)
      newIndexingMaps.push_back(newLinalgOp.getOutputIndexingMap(i));
    newOp->setAttr("indexing_maps",
                   rewriter.getAffineMapArrayAttr(newIndexingMaps));

    // Set the number of inputs to the new value. The `clone` call above kept
    // the value from the original op.
    newLinalgOp.setNumInputs(canonicalInput.size());

    // linalg.indexed_generic payloads have additional arguments prepended to
    // the block arg list.
    int bbArgBaseOffset = newLinalgOp.getNumPayloadInductionVariables();

    // Repair the payload entry block by RAUW'ing redundant arguments and
    // erasing them.
    Block &payload = newOp->getRegion(0).front();
    for (int i = 0, e = op.getNumInputs(); i < e; i++) {
      // Iterate in reverse, so that we erase later args first, preventing the
      // argument list from shifting unexpectedly and invalidating all our
      // indices.
      int reversed = e - i - 1;
      int canonicalIndex = canonicalInputIndices[reversed];
      if (canonicalInputIndices[reversed] == reversed)
        continue;
      payload.getArgument(bbArgBaseOffset + reversed)
          .replaceAllUsesWith(
              payload.getArgument(bbArgBaseOffset + canonicalIndex));
      payload.eraseArgument(bbArgBaseOffset + reversed);
    }

    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

/// Remove generic/indexed_generic operations (on tensors) that are just copying
/// the values from inputs to the results. Requirements are
/// 1) All iterator types are parallel
/// 2) The body contains just a yield operation with the yielded values being
///    the arguments corresponding to the operands.
struct RemoveIdentityLinalgOps : public OpInterfaceRewritePattern<LinalgOp> {
  using OpInterfaceRewritePattern<LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LinalgOp op,
                                PatternRewriter &rewriter) const override {
    if (auto copyOp = dyn_cast<CopyOp>(*op)) {
      assert(copyOp.hasBufferSemantics());
      if (copyOp.input() == copyOp.output() &&
          copyOp.inputPermutation() == copyOp.outputPermutation()) {
        rewriter.eraseOp(op);
        return success();
      }
    }

    if (!isa<GenericOp, IndexedGenericOp>(op))
      return failure();
    if (!op.hasTensorSemantics())
      return failure();
    // Check all indexing maps are identity.
    if (llvm::any_of(op.getIndexingMaps(),
                     [](AffineMap map) { return !map.isIdentity(); }))
      return failure();

    // Check that the body of the linalg operation is just a linalg.yield
    // operation.
    Block &body = op->getRegion(0).front();
    if (!llvm::hasSingleElement(body))
      return failure();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yieldOp)
      return failure();

    // Get the argument number of the returned values. That is the operand
    // number to use for replacing uses of this operation.
    unsigned numIndexArgs = op.getNumPayloadInductionVariables();
    SmallVector<Value, 4> returnedArgs;
    for (Value yieldVal : yieldOp.values()) {
      auto yieldArg = yieldVal.dyn_cast<BlockArgument>();
      if (!yieldArg || yieldArg.getOwner() != &body)
        return failure();
      unsigned argumentNumber = yieldArg.getArgNumber();
      if (argumentNumber < numIndexArgs)
        return failure();
      returnedArgs.push_back(op->getOperand(argumentNumber - numIndexArgs));
    }
    if (returnedArgs.size() != op.getOperation()->getNumResults())
      return failure();
    rewriter.replaceOp(op, returnedArgs);
    return success();
  }
};
} // namespace

#define CANONICALIZERS_AND_FOLDERS(XXX)                                        \
  void XXX::getCanonicalizationPatterns(RewritePatternSet &results,            \
                                        MLIRContext *context) {                \
    results.add<DeduplicateInputs, EraseDeadLinalgOp, FoldTensorCastOp,        \
                RemoveIdentityLinalgOps>(context);                             \
  }                                                                            \
                                                                               \
  LogicalResult XXX::fold(ArrayRef<Attribute>,                                 \
                          SmallVectorImpl<OpFoldResult> &) {                   \
    return foldMemRefCast(*this);                                              \
  }

CANONICALIZERS_AND_FOLDERS(ConvOp)
CANONICALIZERS_AND_FOLDERS(PoolingMaxOp)
CANONICALIZERS_AND_FOLDERS(PoolingMinOp)
CANONICALIZERS_AND_FOLDERS(PoolingSumOp)
CANONICALIZERS_AND_FOLDERS(CopyOp)
CANONICALIZERS_AND_FOLDERS(FillOp)
CANONICALIZERS_AND_FOLDERS(GenericOp)

// All named ops canonicalizers and folders are auto-generated in the
// .cpp.inc.
