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
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

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
///    someop(memrefcast) -> someop
/// ```
/// It folds the source of the memref_cast into the root operation directly.
static LogicalResult foldMemRefCast(Operation *op) {
  bool folded = false;
  for (OpOperand &operand : op->getOpOperands()) {
    auto castOp = operand.get().getDefiningOp<MemRefCastOp>();
    if (castOp && canFoldIntoConsumerOp(castOp)) {
      operand.set(castOp.getOperand());
      folded = true;
    }
  }
  return success(folded);
}

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
        libraryCall.empty() ? StringAttr() : builder.getStringAttr(libraryCall),
        ArrayAttr());
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
        libraryCall.empty() ? StringAttr() : builder.getStringAttr(libraryCall),
        ArrayAttr());
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
  for (auto attr : op.getAttrs())
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
  for (NamedAttribute n : op.getAttrs()) {
    if ((hasExtraAttrs = !genericAttrNamesSet.contains(n.first.strref())))
      break;
  }
  if (hasExtraAttrs) {
    p << " attrs = ";
    p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/genericAttrNames);
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

namespace {

template <typename GenericOpType>
struct AnnotationsVerifier {
  static LogicalResult verify(GenericOpType op) { return success(); }
};

template <>
LogicalResult AnnotationsVerifier<GenericOp>::verify(GenericOp op) {
  ArrayAttr sparseAttr = op.sparseAttr();
  if (!sparseAttr)
    return success();
  // Verify consistency of sparse annotations.
  if (!op.hasTensorSemantics())
    return op.emitOpError("expected sparse annotations on tensors only");
  if (op.getNumOutputs() != 1)
    return op.emitOpError("expected single output tensor");
  unsigned numTensors = op.getNumShapedOperands();
  if (sparseAttr.size() != numTensors)
    return op.emitOpError("expected one sparse annotation for each tensor");
  for (unsigned t = 0; t < numTensors; t++) {
    auto dimAttr = sparseAttr[t].dyn_cast_or_null<ArrayAttr>();
    if (!dimAttr)
      return op.emitOpError("expected sparse annotation array for tensor ")
             << t;
    unsigned rank = op.getShapedType(t).getRank();
    if (dimAttr.size() != rank)
      return op.emitOpError("expected sparse annotation with rank ")
             << rank << " for tensor " << t;
    // Per-dimension annotations for each tensor consist of only "D" or "S".
    for (unsigned d = 0; d < rank; d++) {
      if (isDenseDim(dimAttr[d])) {
        continue;
      } else if (isSparseDim(dimAttr[d])) {
        if (t == numTensors - 1)
          return op.emitOpError("sparse output tensors not supported (yet)");
        continue;
      }
      return op.emitOpError("expected sparse annotation at position ")
             << d << " for tensor " << t;
    }
  }
  return success();
}

} // namespace

template <typename GenericOpType>
static LogicalResult verifyGenericOp(GenericOpType op) {
  if (failed(AnnotationsVerifier<GenericOpType>::verify(op)))
    return failure();

  return success();
}

static LogicalResult verify(GenericOp op) { return verifyGenericOp(op); }

static LogicalResult verify(IndexedGenericOp op) { return verifyGenericOp(op); }

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
    // staticLow and staticHigh have full information of the padding config.
    // This will grow staticLow and staticHigh with 1 value. If the config is
    // dynamic (ie not a constant), dynamicLow and dynamicHigh will grow with 1
    // value as well.
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

static Value getCollapsedInitTensor(OpBuilder &builder,
                                    TensorReshapeOp reshapeOp) {
  Location loc = reshapeOp.getLoc();
  SmallVector<Value, 4> dynamicShapes;
  SmallVector<int64_t, 4> staticShapes;
  auto reassociation = reshapeOp.getReassociationMaps();
  Value src = reshapeOp.src();
  RankedTensorType srcType = reshapeOp.getSrcType();
  ArrayRef<int64_t> srcShape = srcType.getShape();
  for (auto map : reassociation) {
    Value linearizedDynamicDim = nullptr;
    int64_t linearizedStaticDim = 1;
    for (unsigned i : llvm::map_range(map.getResults(), [](AffineExpr e) {
           return e.cast<AffineDimExpr>().getPosition();
         })) {
      if (ShapedType::isDynamic(srcShape[i])) {
        Value shapeVal = builder.create<DimOp>(loc, src, i);
        if (linearizedDynamicDim) {
          linearizedDynamicDim =
              builder.create<MulIOp>(loc, linearizedDynamicDim, shapeVal);
        } else {
          linearizedDynamicDim = shapeVal;
        }
      } else {
        linearizedStaticDim *= srcShape[i];
      }
    }
    if (linearizedDynamicDim) {
      if (linearizedStaticDim != 1) {
        linearizedDynamicDim = builder.create<MulIOp>(
            loc, linearizedDynamicDim,
            builder.create<ConstantIndexOp>(loc, linearizedStaticDim));
      }
      dynamicShapes.push_back(linearizedDynamicDim);
      staticShapes.push_back(ShapedType::kDynamicSize);
    } else {
      staticShapes.push_back(linearizedStaticDim);
    }
  }
  return builder.create<InitTensorOp>(loc, dynamicShapes, staticShapes,
                                      srcType.getElementType());
}

static Value getExpandedInitTensor(OpBuilder &builder,
                                   TensorReshapeOp reshapeOp) {
  SmallVector<Value, 4> dynamicShapes;
  SmallVector<int64_t, 4> staticShapes;
  auto reassociation = reshapeOp.getReassociationMaps();
  Value src = reshapeOp.src();
  RankedTensorType srcType = reshapeOp.getSrcType();
  ArrayRef<int64_t> srcShape = srcType.getShape();
  ArrayRef<int64_t> dstShape = reshapeOp.getResultType().getShape();
  Location loc = reshapeOp.getLoc();
  for (auto map : enumerate(reassociation)) {
    int64_t linearizedStaticDim = 1;
    bool hasDynamic = false;
    for (unsigned i :
         llvm::map_range(map.value().getResults(), [](AffineExpr e) {
           return e.cast<AffineDimExpr>().getPosition();
         })) {
      if (ShapedType::isDynamic(dstShape[i])) {
        // Only one of the dimensions of the expanded shape should be dynamic.
        if (hasDynamic)
          return nullptr;
        hasDynamic = true;
        staticShapes.push_back(ShapedType::kDynamicSize);
        continue;
      }
      staticShapes.push_back(dstShape[i]);
      linearizedStaticDim *= dstShape[i];
    }
    if (hasDynamic) {
      // If the expanded dimensions has a dynamic shape, the src shape must be
      // dynamic as well.
      if (!ShapedType::isDynamic(srcShape[map.index()]))
        return nullptr;
      Value dynamicDim = builder.create<DimOp>(loc, src, map.index());
      if (linearizedStaticDim != 1) {
        dynamicDim = builder.create<UnsignedDivIOp>(
            loc, dynamicDim,
            builder.create<ConstantIndexOp>(loc, linearizedStaticDim));
      }
      dynamicShapes.push_back(dynamicDim);
    }
  }
  return builder.create<InitTensorOp>(loc, dynamicShapes, staticShapes,
                                      srcType.getElementType());
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

/// Canonicalize a `linalg.init_tensor` -> `dim` pattern by replacing the `dim`
/// with
/// - A constant value if the size is static along the dimension.
/// - The dynamic value that defines the size of the result of
///   `linalg.init_tensor` op.
struct ReplaceDimOfInitTensorOp : public OpRewritePattern<DimOp> {
  using OpRewritePattern<DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DimOp dimOp,
                                PatternRewriter &rewriter) const override {
    auto initTensorOp = dimOp.memrefOrTensor().getDefiningOp<InitTensorOp>();
    if (!initTensorOp)
      return failure();
    auto dimIndex = dimOp.index().getDefiningOp<ConstantIndexOp>();
    if (!dimIndex)
      return failure();
    int64_t index = dimIndex.getValue();
    if (!initTensorOp.isDynamicSize(index)) {
      rewriter.replaceOpWithNewOp<ConstantIndexOp>(
          dimOp, initTensorOp.getStaticSize(index));
    } else {
      rewriter.replaceOp(dimOp, initTensorOp.getDynamicSize(index));
    }
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
    SmallVector<Value, 4> resultShapeValues =
        reshapeOp.getOutputShape(rewriter, loc);
    Value initTensor = rewriter.create<InitTensorOp>(
        loc, resultShapeValues, reshapeOp.getResultType().getElementType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(
        reshapeOp, reshapeOp.getResultType(), initTensor);
    return success();
  }
};
} // namespace

void InitTensorOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results
      .insert<FoldInitTensorWithSubTensorOp, FoldInitTensorWithTensorReshapeOp,
              ReplaceDimOfInitTensorOp, ReplaceStaticShapeDims>(context);
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
    auto dimOp = builder.createOrFold<DimOp>(loc, source, i);
    auto resultDimSize = builder.createOrFold<ConstantIndexOp>(
        loc, rankedTensorType.getDimSize(i));
    auto highValue = builder.createOrFold<SubIOp>(loc, resultDimSize, dimOp);
    high.push_back(highValue);
    low.push_back(builder.createOrFold<ConstantIndexOp>(loc, 0));
  }
  return PadTensorOp::createPadScalarOp(type, source, pad, low, high, loc,
                                        builder);
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

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
static ArrayAttr collapseReassociationMaps(ArrayRef<AffineMap> mapsProducer,
                                           ArrayRef<AffineMap> mapsConsumer,
                                           MLIRContext *context) {
  // Handle the corner case of the result being a rank 0 shaped type. Return an
  // emtpy ArrayAttr.
  if (mapsConsumer.empty() && !mapsProducer.empty())
    return ArrayAttr::get(context, ArrayRef<Attribute>());
  if (mapsProducer.empty() || mapsConsumer.empty() ||
      mapsProducer[0].getNumDims() < mapsConsumer[0].getNumDims() ||
      mapsProducer.size() != mapsConsumer[0].getNumDims())
    return nullptr;
  unsigned numLhsDims = mapsProducer[0].getNumDims();
  unsigned currDim = 0;
  SmallVector<AffineExpr, 4> reassociations;
  SmallVector<Attribute, 4> reassociationMaps;
  for (AffineMap rhs : mapsConsumer) {
    for (AffineExpr rhsExpr : rhs.getResults()) {
      AffineDimExpr dimExpr = rhsExpr.cast<AffineDimExpr>();
      for (int i = 0, e = mapsProducer[dimExpr.getPosition()].getNumResults();
           i < e; ++i) {
        reassociations.push_back(getAffineDimExpr(currDim++, context));
      }
    }
    reassociationMaps.push_back(AffineMapAttr::get(AffineMap::get(
        numLhsDims, /*numSymbols =*/0, reassociations, context)));
    reassociations.clear();
  }
  return ArrayAttr::get(context, reassociationMaps);
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

    auto areReshapeOpsFoldable = [](ShapedType largerType,
                                    ShapedType intermediateType,
                                    ShapedType smallerType) -> bool {
      return largerType.getRank() > intermediateType.getRank() &&
             intermediateType.getRank() > smallerType.getRank();
    };
    // Check if producer and consumer are both expanding dims.
    if (areReshapeOpsFoldable(reshapeOp.getResultType(), reshapeOp.getSrcType(),
                              srcReshapeOp.getSrcType())) {
      rewriter.replaceOpWithNewOp<ReshapeOpTy>(
          reshapeOp, reshapeOp.getResultType(), srcReshapeOp.src(),
          collapseReassociationMaps(reshapeOp.getReassociationMaps(),
                                    srcReshapeOp.getReassociationMaps(),
                                    rewriter.getContext()));
      return success();
    }
    // Check if producer and consumer are both collapsing dims.
    if (areReshapeOpsFoldable(srcReshapeOp.getSrcType(), reshapeOp.getSrcType(),
                              reshapeOp.getResultType())) {
      rewriter.replaceOpWithNewOp<ReshapeOpTy>(
          reshapeOp, reshapeOp.getResultType(), srcReshapeOp.src(),
          collapseReassociationMaps(srcReshapeOp.getReassociationMaps(),
                                    reshapeOp.getReassociationMaps(),
                                    rewriter.getContext()));
      return success();
    }
    return failure();
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

/// Helper functions assert Attribute of the proper type in attr and returns the
/// corresponding vector.
/// TODO: this should be evolved into a generic
/// `getRangeOfType<AffineMap>(ArrayAttr attrs)` that does not copy.
static SmallVector<AffineMap, 4> getAffineMaps(ArrayAttr attrs) {
  return llvm::to_vector<8>(llvm::map_range(
      attrs, [](Attribute a) { return a.cast<AffineMapAttr>().getValue(); }));
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

static SmallVector<SmallVector<AffineExpr, 2>, 2>
convertReassociationIndicesToMaps(
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

/// For reshape op compute the shape at dimension `dimIndex` of the output in
/// terms of shape of the `src`, when the reshape op is a collapsing
/// operation. It is the product of the shape of the collapsed dimensions of the
/// `src`.
static Value
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
    dynamicDims.push_back(builder.create<DimOp>(loc, src, dim));
    AffineExpr currExpr = builder.getAffineSymbolExpr(dim - startPos);
    expr = (expr ? expr * currExpr : currExpr);
  }
  return applyMapToValues(builder, loc,
                          AffineMap::get(0, endPos - startPos + 1, expr),
                          dynamicDims)[0];
}

/// Given the `src` of a collapsing reshape op and its reassociation maps,
/// compute the shape of the result of the reshape.
static SmallVector<Value, 4> getCollapsedOutputShapeFromInputShape(
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
static Value getExpandedOutputDimFromInputShape(
    OpBuilder &builder, Location loc, int64_t dimIndex, Value src,
    ArrayRef<int64_t> dstStaticShape, ArrayRef<AffineMap> reassociation,
    llvm::DenseMap<int64_t, int64_t> &expandedDimToCollapsedDim) {
  if (!ShapedType::isDynamic(dstStaticShape[dimIndex])) {
    return builder.create<ConstantIndexOp>(loc, dstStaticShape[dimIndex]);
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
  Value sourceDim = builder.create<DimOp>(loc, src, sourceDimPos);
  return applyMapToValues(
      builder, loc,
      AffineMap::get(
          0, 1, builder.getAffineSymbolExpr(0).floorDiv(linearizedStaticDim)),
      sourceDim)[0];
}

/// Given the `src` of an expanding reshape op, the reassociation maps and the
/// result type, compute the shape of the result of the reshape.
static SmallVector<Value, 4> getExpandedOutputShapeFromInputShape(
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

SmallVector<Value, 4> mlir::linalg::getReshapeOutputShapeFromInputShape(
    OpBuilder &builder, Location loc, Value src,
    ArrayRef<int64_t> dstStaticShape, ArrayRef<AffineMap> reassocation) {
  return dstStaticShape.size() >
                 static_cast<size_t>(src.getType().cast<ShapedType>().getRank())
             ? getExpandedOutputShapeFromInputShape(
                   builder, loc, src, dstStaticShape, reassocation)
             : getCollapsedOutputShapeFromInputShape(
                   builder, loc, src, dstStaticShape, reassocation);
}

/// For a reshape op, compute the value of a given dimension of the output
/// (`dimIndex`) from the shape of the inputs and type of the result.
static Value getReshapeOutputDimFromInputShape(
    OpBuilder &builder, Location loc, int64_t dimIndex, Value src,
    ArrayRef<int64_t> dstStaticShape, ArrayRef<AffineMap> reassociation) {
  if (dstStaticShape.size() >
      static_cast<size_t>(src.getType().cast<ShapedType>().getRank())) {
    llvm::DenseMap<int64_t, int64_t> expandedDimToCollapsedDim =
        getExpandedDimToCollapsedDimMap(reassociation);
    return getExpandedOutputDimFromInputShape(builder, loc, dimIndex, src,
                                              dstStaticShape, reassociation,
                                              expandedDimToCollapsedDim);
  }
  return getCollapsedOutputDimFromInputShape(builder, loc, dimIndex, src,
                                             reassociation);
}

void mlir::linalg::ReshapeOp::build(OpBuilder &b, OperationState &result,
                                    Value src,
                                    ArrayRef<ReassociationExprs> reassociation,
                                    ArrayRef<NamedAttribute> attrs) {
  auto maps = getSymbolLessAffineMaps(reassociation);
  auto memRefType = src.getType().cast<MemRefType>();
  auto resultType = computeReshapeCollapsedType(memRefType, maps);
  build(b, result, resultType, src, attrs);
  result.addAttribute(ReshapeOp::getReassociationAttrName(),
                      b.getAffineMapArrayAttr(maps));
}

void mlir::linalg::ReshapeOp::build(OpBuilder &b, OperationState &result,
                                    Type resultType, Value src,
                                    ArrayRef<ReassociationExprs> reassociation,
                                    ArrayRef<NamedAttribute> attrs) {
  auto maps = getSymbolLessAffineMaps(reassociation);
  build(b, result, resultType, src, attrs);
  result.addAttribute(ReshapeOp::getReassociationAttrName(),
                      b.getAffineMapArrayAttr(maps));
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
      return op.emitOpError(
          "invalid to reshape tensor/memref with non-unit extent dimensions to "
          "zero-rank tensor/memref");
    return success();
  }
  if (collapsedRank != op.reassociation().size())
    return op.emitOpError("expected rank of the collapsed type(")
           << collapsedRank << ") to be the number of reassociation maps("
           << op.reassociation().size() << ")";
  auto maps = getAffineMaps(op.reassociation());
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
  auto maps = getAffineMaps(op.reassociation());
  MemRefType expectedType = computeReshapeCollapsedType(expandedType, maps);
  if (collapsedType != expectedType)
    return op.emitOpError("expected collapsed type to be ")
           << expectedType << ", but got " << collapsedType;
  return success();
}

void ReshapeOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  results.insert<CollapseReshapeOps<ReshapeOp>>(context);
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
    ArrayRef<ReassociationExprs> reassociation,
    ArrayRef<NamedAttribute> attrs) {
  auto maps = getSymbolLessAffineMaps(reassociation);
  auto resultType = computeTensorReshapeCollapsedType(
      src.getType().cast<RankedTensorType>(), maps);
  build(b, result, resultType, src, attrs);
  result.addAttribute(TensorReshapeOp::getReassociationAttrName(),
                      b.getAffineMapArrayAttr(maps));
}

void mlir::linalg::TensorReshapeOp::build(
    OpBuilder &b, OperationState &result, Type resultType, Value src,
    ArrayRef<ReassociationExprs> reassociation,
    ArrayRef<NamedAttribute> attrs) {
  auto maps = getSymbolLessAffineMaps(reassociation);
  build(b, result, resultType, src, attrs);
  result.addAttribute(TensorReshapeOp::getReassociationAttrName(),
                      b.getAffineMapArrayAttr(maps));
}

static LogicalResult verify(TensorReshapeOp op) {
  RankedTensorType expandedType, collapsedType;
  if (failed(verifyReshapeLikeTypes(op, expandedType, collapsedType)))
    return failure();
  auto maps = getAffineMaps(op.reassociation());
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

/// Canonicalize dim ops that use the output shape with dim of the input.
struct ReplaceDimOfReshapeOpResult : OpRewritePattern<DimOp> {
  using OpRewritePattern<DimOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(DimOp dimOp,
                                PatternRewriter &rewriter) const override {
    Value dimValue = dimOp.memrefOrTensor();
    Optional<int64_t> dimIndex = dimOp.getConstantIndex();
    if (!dimIndex)
      return failure();

    auto reshapeOp = dimValue.getDefiningOp<TensorReshapeOp>();
    if (!reshapeOp)
      return failure();

    rewriter.replaceOp(dimOp,
                       getReshapeOutputDimFromInputShape(
                           rewriter, dimOp.getLoc(), *dimIndex, reshapeOp.src(),
                           reshapeOp.getResultType().getShape(),
                           reshapeOp.getReassociationMaps()));
    return success();
  }
};
} // namespace

void TensorReshapeOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<CollapseReshapeOps<TensorReshapeOp>, FoldReshapeWithConstant,
                 ReplaceDimOfReshapeOpResult>(context);
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, linalg::YieldOp op) {
  p << op.getOperationName();
  if (op.getNumOperands() > 0)
    p << ' ' << op.getOperands();
  p.printOptionalAttrDict(op.getAttrs());
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
    return success();
  }
  return op.emitOpError("expected parent op with LinalgOp interface");
}

//===----------------------------------------------------------------------===//
// TiledLoopOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, TiledLoopOp op) {
  p << op.getOperationName() << " (" << op.getBody()->getArguments() << ") = ("
    << op.lowerBound() << ") to (" << op.upperBound() << ") step (" << op.step()
    << ")";

  if (!op.inputs().empty())
    p << " ins (" << op.inputs() << ")";
  if (!op.outputs().empty())
    p << " outs (" << op.outputs() << ")";

  if (llvm::any_of(op.iterator_types(), [](Attribute attr) {
        return attr.cast<StringAttr>().getValue() !=
               getParallelIteratorTypeName();
      })) {
    p << " iterators(" << op.iterator_types() << ")";
  }

  p.printRegion(op.region(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(
      op.getAttrs(), /*elidedAttrs=*/{TiledLoopOp::getOperandSegmentSizeAttr(),
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
  SmallVector<OpAsmParser::OperandType, 4> inputs;
  if (succeeded(parser.parseOptionalKeyword("ins"))) {
    SmallVector<Type, 4> inputTypes;
    llvm::SMLoc inputsOperandsLoc = parser.getCurrentLocation();

    if (parser.parseLParen() || parser.parseOperandList(inputs) ||
        parser.parseColonTypeList(inputTypes) || parser.parseRParen())
      return failure();

    if (parser.resolveOperands(inputs, inputTypes, inputsOperandsLoc,
                               result.operands))
      return failure();
  }

  // Parse output tensors.
  SmallVector<OpAsmParser::OperandType, 4> outputs;
  if (succeeded(parser.parseOptionalKeyword("outs"))) {
    SmallVector<Type, 4> outputTypes;
    llvm::SMLoc outputsOperandsLoc = parser.getCurrentLocation();

    if (parser.parseLParen() || parser.parseOperandList(outputs) ||
        parser.parseColonTypeList(outputTypes) || parser.parseRParen())
      return failure();

    if (parser.resolveOperands(outputs, outputTypes, outputsOperandsLoc,
                               result.operands))
      return failure();
    result.addTypes(outputTypes);
  }

  // Parse attributes.
  SmallVector<Attribute, 4> iterTypes;
  if (succeeded(parser.parseOptionalKeyword("iterators"))) {
    StringAttr iterType;

    if (parser.parseLParen() || parser.parseAttribute(iterType))
      return failure();
    iterTypes.push_back(iterType);
    for (int i = 1, e = ivs.size(); i < e; ++i) {
      if (parser.parseComma() || parser.parseAttribute(iterType))
        return failure();
      iterTypes.push_back(iterType);
    }
    if (parser.parseRParen())
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
  SmallVector<Type, 4> types(ivs.size(), builder.getIndexType());
  if (parser.parseRegion(*body, ivs, types))
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

static LogicalResult verify(TiledLoopOp op) { return success(); }

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
    if (failed(
            verifyStrideOrDilation(op, strides->getValue(), /*isStride=*/true)))
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
    if (failed(
            verifyStrideOrDilation(op, strides->getValue(), /*isStride=*/true)))
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

#include "mlir/Dialect/Linalg/IR/LinalgNamedStructuredOps.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgOps.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgSparseOps.cpp.inc"

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
    if (errorHandler) errorHandler(expected, actual);
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
  p.printOptionalAttrDict(op.getAttrs(),
                          /*elidedAttrs=*/{"operand_segment_sizes"});

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
struct EraseDeadLinalgOp : public RewritePattern {
  EraseDeadLinalgOp(PatternBenefit benefit = 1)
      : RewritePattern(benefit, MatchAnyOpTypeTag()) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = dyn_cast<LinalgOp>(op);
    if (!linalgOp)
      return failure();
    for (Value v : linalgOp.getShapedOperands()) {
      // Linalg "inputs" may be either tensor or memref type.
      // tensor<0xelt_type> is a convention that may not always mean
      // "0 iterations". Only erase in cases we see memref<...x0x...>.
      auto mt = v.getType().dyn_cast<MemRefType>();
      if (!mt)
        continue;
      if (llvm::is_contained(mt.getShape(), 0)) {
        rewriter.eraseOp(linalgOp);
        return success();
      }
    }
    return failure();
  }
};

struct FoldTensorCastOp : public RewritePattern {
  FoldTensorCastOp(PatternBenefit benefit = 1)
      : RewritePattern(benefit, MatchAnyOpTypeTag()) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = dyn_cast<LinalgOp>(op);
    if (!linalgOp)
      return failure();

    // If no operand comes from a tensor::CastOp and can be folded then fail.
    bool hasTensorCastOperand =
        llvm::any_of(linalgOp.getShapedOperands(), [&](Value v) {
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
    for (Value v : linalgOp.getInputs()) {
      auto tensorCastOp = v.getDefiningOp<tensor::CastOp>();
      newOperands.push_back(
          canFoldIntoConsumerOp(tensorCastOp) ? tensorCastOp.source() : v);
    }
    // Init tensors may fold, in which case the resultType must also change.
    for (Value v : linalgOp.getOutputs()) {
      auto tensorCastOp = v.getDefiningOp<tensor::CastOp>();
      bool fold = canFoldIntoConsumerOp(tensorCastOp);
      newOperands.push_back(fold ? tensorCastOp.getOperand() : v);
      newResultTypes.push_back(newOperands.back().getType());
    }
    auto extraOperands = linalgOp.getAssumedNonShapedOperands();
    newOperands.append(extraOperands.begin(), extraOperands.end());
    // Clone op.
    Operation *newOp =
        linalgOp.clone(rewriter, op->getLoc(), newResultTypes, newOperands);
    rewriter.replaceOp(op, newOp->getResults());

    return success();
  }
};

/// Replaces std.dim operations that use the result of a LinalgOp (on tensors)
/// with std.dim operations that use one of the arguments. For example,
///
/// %0 = linalg.matmul ins(%arg0, %arg1, ...)
/// %1 = dim %0, %c0
///
/// with
///
/// %1 = dim %arg0, %c0
///
/// where possible. With this the result of the `linalg.matmul` is not used in
/// dim operations. If the value produced is replaced with another value (say by
/// tiling `linalg.matmul`) will make the `linalg.matmul` truly dead instead of
/// used in a dim op that would prevent the DCE of this op.
struct ReplaceDimOfLinalgOpResult : public OpRewritePattern<DimOp> {
  using OpRewritePattern<DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DimOp dimOp,
                                PatternRewriter &rewriter) const override {
    Value dimValue = dimOp.memrefOrTensor();
    Optional<int64_t> dimIndex = dimOp.getConstantIndex();
    if (!dimIndex)
      return failure();
    auto linalgOp = dimValue.getDefiningOp<LinalgOp>();
    if (!linalgOp)
      return failure();

    unsigned resultIndex = dimValue.cast<OpResult>().getResultNumber();
    Optional<Value> operandDimValue = linalgOp.inferResultDimFromInputShapes(
        rewriter, dimOp.getLoc(), resultIndex,
        static_cast<unsigned>(*dimIndex));
    if (!operandDimValue) {
      // Its always possible to replace using the corresponding `outs`
      // parameter.
      operandDimValue = rewriter.create<DimOp>(
          dimOp.getLoc(), linalgOp.getOutput(resultIndex), *dimIndex);
    }
    rewriter.replaceOp(dimOp, *operandDimValue);
    return success();
  }
};

} // namespace

namespace {
// Deduplicate redundant args of a linalg op.
// An arg is redundant if it has the same Value and indexing map as another.
struct DeduplicateInputs : public RewritePattern {
  DeduplicateInputs(PatternBenefit benefit = 1)
      : RewritePattern(benefit, MatchAnyOpTypeTag()) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // This pattern reduces the number of arguments of an op, which breaks
    // the invariants of semantically charged named ops.
    if (!isa<GenericOp, IndexedGenericOp>(op))
      return failure();
    auto linalgOp = cast<LinalgOp>(op);

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
    for (int i = 0, e = linalgOp.getNumInputs(); i != e; i++) {
      Value input = linalgOp.getInput(i);
      AffineMap indexingMap = linalgOp.getInputIndexingMap(i);
      // STL-like maps have a convenient behavior for our use case here. In the
      // case of duplicate keys, the insertion is rejected, and the returned
      // iterator gives access to the value already in the map.
      auto pair = canonicalInput.insert({{input, indexingMap}, i});
      canonicalInputIndices.push_back(pair.first->second);
    }

    // If there are no duplicate args, then bail out.
    if (canonicalInput.size() == linalgOp.getNumInputs())
      return failure();

    // The operands for the newly canonicalized op.
    SmallVector<Value, 6> newOperands;
    for (auto v : llvm::enumerate(linalgOp.getInputs()))
      if (canonicalInputIndices[v.index()] == static_cast<int>(v.index()))
        newOperands.push_back(v.value());
    llvm::append_range(newOperands, linalgOp.getOutputs());
    llvm::append_range(newOperands, linalgOp.getAssumedNonShapedOperands());

    // Clone the old op with new operands.
    Operation *newOp = linalgOp.clone(rewriter, op->getLoc(),
                                      op->getResultTypes(), newOperands);
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
    for (int i = 0, e = linalgOp.getNumInputs(); i < e; i++) {
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
struct RemoveIdentityLinalgOps : public RewritePattern {
  RemoveIdentityLinalgOps(PatternBenefit benefit = 1)
      : RewritePattern(benefit, MatchAnyOpTypeTag()) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isa<GenericOp, IndexedGenericOp>(op))
      return failure();
    LinalgOp genericOp = cast<LinalgOp>(op);
    if (!genericOp.hasTensorSemantics())
      return failure();
    // Check all indexing maps are identity.
    if (llvm::any_of(genericOp.getIndexingMaps(),
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
    unsigned numIndexArgs = genericOp.getNumPayloadInductionVariables();
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
    if (returnedArgs.size() != genericOp.getOperation()->getNumResults())
      return failure();
    rewriter.replaceOp(genericOp, returnedArgs);
    return success();
  }
};
} // namespace

#define CANONICALIZERS_AND_FOLDERS(XXX)                                        \
  void XXX::getCanonicalizationPatterns(OwningRewritePatternList &results,     \
                                        MLIRContext *context) {                \
    results.insert<DeduplicateInputs, EraseDeadLinalgOp, FoldTensorCastOp,     \
                   RemoveIdentityLinalgOps>();                                 \
    results.insert<ReplaceDimOfLinalgOpResult>(context);                       \
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
CANONICALIZERS_AND_FOLDERS(IndexedGenericOp)

// All named ops canonicalizers and folders are auto-generated in the
// .cpp.inc.
