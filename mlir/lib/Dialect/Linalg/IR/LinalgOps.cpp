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
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::linalg;

/// Fully compose map with operands and canonicalize the result.
/// Return the `createOrFold`'ed AffineApply op.
static Value createFoldedComposedAffineApply(OpBuilder &b, Location loc,
                                             AffineMap map,
                                             ValueRange operandsRef) {
  SmallVector<Value, 4> operands(operandsRef.begin(), operandsRef.end());
  fullyComposeAffineMapAndOperands(&map, &operands);
  canonicalizeMapAndOperands(&map, &operands);
  return b.createOrFold<AffineApplyOp>(loc, map, operands);
}

SmallVector<Value, 4> mlir::linalg::applyMapToValues(OpBuilder &b, Location loc,
                                                     AffineMap map,
                                                     ValueRange values) {
  SmallVector<Value, 4> res;
  res.reserve(map.getNumResults());
  unsigned numDims = map.getNumDims(), numSym = map.getNumSymbols();
  // For each `expr` in `map`, applies the `expr` to the values extracted from
  // ranges. If the resulting application can be folded into a Value, the
  // folding occurs eagerly.
  for (auto expr : map.getResults()) {
    AffineMap map = AffineMap::get(numDims, numSym, expr);
    res.push_back(createFoldedComposedAffineApply(b, loc, map, values));
  }
  return res;
}

SmallVector<Value, 4> LinalgOp::createFlatListOfOperandDims(OpBuilder &b,
                                                            Location loc) {
  SmallVector<Value, 4> res;
  for (Value v : getShapedOperands()) {
    ShapedType t = v.getType().template cast<ShapedType>();
    for (unsigned i = 0, e = t.getRank(); i < e; ++i)
      res.push_back(b.create<DimOp>(loc, v, i));
  }
  return res;
}

SmallVector<Range, 4> LinalgOp::createLoopRanges(OpBuilder &b, Location loc) {
  AffineMap map = getLoopsToShapesMap();
  unsigned numDims = map.getNumDims(), numRes = map.getNumResults();
  auto viewSizes = createFlatListOfOperandDims(b, loc);
  SmallVector<Range, 4> res(numDims);
  Value zeroVal = b.create<ConstantIndexOp>(loc, 0);
  Value oneVal = b.create<ConstantIndexOp>(loc, 1);
  for (unsigned idx = 0; idx < numRes; ++idx) {
    auto result = map.getResult(idx);
    if (auto d = result.dyn_cast<AffineDimExpr>()) {
      if (res[d.getPosition()].offset)
        continue;
      res[d.getPosition()] = Range{zeroVal, viewSizes[idx], oneVal};
    }
  }
  return res;
}

/// Forward declarations.
template <typename NamedStructuredOpType>
static void buildNamedStructuredOpRegionAndAttributes(OpBuilder &opBuilder,
                                                      OperationState &result,
                                                      TypeRange inputTypes,
                                                      TypeRange outputTypes);

static ParseResult
parseCommonStructuredOpParts(OpAsmParser &parser, OperationState &result,
                             SmallVectorImpl<Type> &inputTypes,
                             SmallVectorImpl<Type> &outputTypes);

template <typename NamedStructuredOpType>
static ParseResult
parseNamedStructuredOpRegion(OpAsmParser &parser, Region &region,
                             TypeRange inputTypes, TypeRange outputTypes);
static ParseResult
parseNamedStructuredOpResults(OpAsmParser &parser,
                              SmallVectorImpl<Type> &resultTypes);

template <typename NamedStructuredOpType>
static ParseResult parseNamedStructuredOp(OpAsmParser &parser,
                                          OperationState &result);

template <typename NamedStructuredOpType>
static void printCommonStructuredOpParts(OpAsmPrinter &p,
                                         NamedStructuredOpType op);

static void printNamedStructuredOpResults(OpAsmPrinter &p,
                                          TypeRange resultTypes);

template <typename NamedStructuredOpType>
static void printNamedStructuredOp(OpAsmPrinter &p, NamedStructuredOpType op);

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

///////////////////// Operations defined with Tablegen /////////////////////////
// For such operations that do not correspond to library calls (i.e. defined in
// LinalgOps.td), we define an overloaded `print` function and a
// parse`className` function.

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
    auto genericDictAttr = DictionaryAttr::get(genericAttrs, op.getContext());
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

LogicalResult mlir::linalg::detail::verifyStructuredOpInterface(Operation *op) {
  LinalgOp linalgOp = cast<LinalgOp>(op);
  // Expect at least one shaped operand.
  // This means an op that constructs a tensor out of indices cannot be a
  // LinalgOp at the moment. For now this will have to be a special op until we
  // have output shape operands that are not tensors.
  auto nShapedOperands = linalgOp.getNumShapedOperands();
  if (nShapedOperands == 0)
    return linalgOp.emitOpError("expected at least 1 Shaped operand");
  if (failed(OpTrait::impl::verifyAtLeastNOperands(op, nShapedOperands)))
    return failure();
  // Should have at least one output tensor per result tensor.
  // Can also have outbut buffers that do not correspond to results.
  if (op->getNumResults() > linalgOp.getNumOutputTensors())
    return op->emitError("unexpected #results > #outputs");

  // All shaped operands must be indexed.
  if (linalgOp.indexing_maps().size() != linalgOp.getNumShapedOperands())
    return linalgOp.emitOpError("expected the number of indexing_map (")
           << linalgOp.indexing_maps().size()
           << ") to be equal to the number of shaped operands ("
           << linalgOp.getNumShapedOperands() << ")";

  SmallVector<AffineMap, 4> indexingMaps;
  indexingMaps.reserve(linalgOp.indexing_maps().size());
  for (auto en : llvm::enumerate(linalgOp.indexing_maps())) {
    auto idx = en.index();
    auto m = en.value().template cast<AffineMapAttr>().getValue();
    indexingMaps.push_back(m); // Save reference to map for further checks.
    auto shapedValue = linalgOp.getShapedType(idx);

    // Symbols disallowed.
    if (m.getNumSymbols() != 0)
      return linalgOp.emitOpError("unexpected symbols in indexing_map #")
             << idx;

    // Domain must be consistent.
    auto nLoops = linalgOp.getNumLoops();
    if (m.getNumDims() != nLoops)
      return linalgOp.emitOpError("expected indexing_map #")
             << idx << " to have " << nLoops
             << " dim(s) to match the number of loops";

    if (m.getNumResults() != shapedValue.getRank())
      return linalgOp.emitOpError("expected shaped value rank (")
             << shapedValue.getRank()
             << ") to match the result rank of indexing_map #" << idx << " ("
             << m.getNumResults() << ")";
  }

  SmallVector<AffineExpr, 4> redDims;
  linalgOp.getReductionDims(redDims);

  // Simplifying assumption: either full tensor or full buffer mode.
  // This allows simpler verification of output operands vs result types
  // without premature tracking of which operand is what in mixed-mode.
  // TODO: relax when mixed-mode needs to pass verification.
  if (linalgOp.getNumOutputBuffers() > 0 && linalgOp.getNumOutputTensors() > 0)
    return op->emitError("expected output operands to all have tensor type or "
                         "all have buffer type");

  for (auto it :
       llvm::zip(linalgOp.getOutputOpOperands(), op->getResultTypes())) {
    if (!std::get<0>(it).get().getType().isa<RankedTensorType>())
      continue;
    if (std::get<0>(it).get().getType() != std::get<1>(it))
      return op->emitError("expected type of operand #")
             << std::get<0>(it).getOperandNumber() << " ("
             << std::get<0>(it).get().getType() << ")"
             << " to match type of corresponding result (" << std::get<1>(it)
             << ")";
  }

  // Output tensor indexing map may not depend on reduction indices.
  for (OpOperand &opOperand : linalgOp.getOutputOpOperands()) {
    AffineMap outputMap = linalgOp.getIndexingMap(opOperand.getOperandNumber());
    for (auto expr : outputMap.getResults()) {
      for (auto dim : redDims) {
        unsigned pos = dim.cast<AffineDimExpr>().getPosition();
        if (expr.isFunctionOfDim(pos)) {
          std::string exprStr;
          {
            llvm::raw_string_ostream os(exprStr);
            os << expr;
          }
          return op->emitError(
                     "unexpected output tensor expression in indexing map #")
                 << (opOperand.getOperandNumber() - linalgOp.getNumInputs())
                 << " a.k.a '" << exprStr
                 << "' is function of reduction iterator 'd" << pos << "'";
        }
      }
    }
  }

  // Named ops that are defined manually have a region builder but no region at
  // this time. Assume the region is well-formed by specification.
  // TODO: use linalg-ods-gen for all ops when we have enough expressive power.
  if (linalgOp->getNumRegions() == 0) {
    assert(!linalgOp.getRegionBuilder() && "regionBuilder but no region");
    return success();
  }

  auto &region = linalgOp->getRegion(0);
  if (linalgOp->getNumRegions() > 1 || !llvm::hasSingleElement(region))
    return op->emitOpError("expected 1 region with 1 block");

  if (!linalgOp.getShapesToLoopsMap())
    return op->emitOpError("expected the shape-to-loops map to be non-null");

  // Simplifying assumption: bbargs match 1-1 with shape operands elemental
  // types.
  // TODO: once ranked shape types are plugged in, we may want to drop the
  // corresponding bbargs, that can never be read from. This will be subject to
  // consistency discussions (i.e. what to do with output tensors whose bbarg is
  // not used).
  Block &block = linalgOp->getRegion(0).front();
  unsigned numBBIvs = linalgOp.getNumPayloadInductionVariables();

  if (linalgOp.getNumShapedOperands() + numBBIvs != block.getNumArguments())
    return op->emitError("expected as many non-induction variable region "
                         "arguments as the number of shaped operands");

  // Note: the number and type of yield values are checked in the YieldOp.
  for (unsigned i = 0; i < numBBIvs; ++i)
    if (!block.getArgument(i).getType().isIndex())
      return op->emitOpError("expected index block argument #") << i;

  unsigned idx = 0;
  for (auto it : llvm::zip(linalgOp.getShapedOperandTypes(),
                           block.getArguments().drop_front(numBBIvs))) {
    if (std::get<0>(it).getElementType() != std::get<1>(it).getType())
      return op->emitError("expected type of bb argument #")
             << (idx + numBBIvs) << " (" << std::get<1>(it).getType() << ")"
             << " to match element type of corresponding shaped operand ("
             << std::get<0>(it).getElementType() << ")";
    ++idx;
  }

  return success();
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

static ParseResult parseInitTensorOp(OpAsmParser &parser,
                                     OperationState &result) {
  OpAsmParser::OperandType srcInfo;
  Type dstType;
  SmallVector<OpAsmParser::OperandType, 2> sizeInfo;
  IndexType indexType = parser.getBuilder().getIndexType();
  if (failed(parseListOfOperandsOrIntegers(
          parser, result, InitTensorOp::getStaticSizesAttrName(),
          ShapedType::kDynamicSize, sizeInfo)) ||
      failed(parser.parseOptionalAttrDict(result.attributes)) ||
      failed(parser.parseColonType(dstType)) ||
      failed(parser.resolveOperands(sizeInfo, indexType, result.operands)))
    return failure();
  return parser.addTypeToList(dstType, result.types);
}

static void print(OpAsmPrinter &p, InitTensorOp op) {
  p << op.getOperation()->getName() << ' ';
  printListOfOperandsOrIntegers(p, op.sizes(), op.static_sizes(),
                                ShapedType::isDynamic);
  p.printOptionalAttrDict(op.getAttrs(),
                          InitTensorOp::getStaticSizesAttrName());
  p << " : " << op.getType();
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
struct FoldWithTensorReshapeOp : public OpRewritePattern<TensorReshapeOp> {
  using OpRewritePattern<TensorReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    if (!reshapeOp.src().getDefiningOp<InitTensorOp>())
      return failure();
    RankedTensorType collapsedType = reshapeOp.getSrcType();
    RankedTensorType expandedType = reshapeOp.getResultType();
    bool isCollapsed = expandedType.getRank() < collapsedType.getRank();
    if (isCollapsed)
      std::swap(collapsedType, expandedType);
    Value initTensorOp = isCollapsed
                             ? getCollapsedInitTensor(rewriter, reshapeOp)
                             : getExpandedInitTensor(rewriter, reshapeOp);
    if (!initTensorOp)
      return failure();
    rewriter.replaceOp(reshapeOp, initTensorOp);
    return success();
  }
};
} // namespace

void InitTensorOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<FoldWithTensorReshapeOp, ReplaceDimOfInitTensorOp,
                 ReplaceStaticShapeDims>(context);
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
    return ArrayAttr::get(ArrayRef<Attribute>(), context);
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
  return ArrayAttr::get(reassociationMaps, context);
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
} // namespace

void TensorReshapeOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<CollapseReshapeOps<TensorReshapeOp>, FoldReshapeWithConstant>(
      context);
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//
void mlir::linalg::SliceOp::build(OpBuilder &b, OperationState &result,
                                  Value base, ValueRange indexings) {
  result.addOperands(base);
  result.addOperands(indexings);

  auto memRefType = base.getType().cast<MemRefType>();
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  auto res = getStridesAndOffset(memRefType, strides, offset);
  assert(succeeded(res) && strides.size() == indexings.size());
  (void)res;

  unsigned rank = memRefType.getRank();
  // TODO: propagate static size and stride information when available.
  SmallVector<int64_t, 4> sizes(rank, -1); // -1 encodes dynamic size.
  result.addTypes({MemRefType::Builder(memRefType)
                       .setShape(sizes)
                       .setAffineMaps(makeStridedLinearLayoutMap(
                           strides, offset, b.getContext()))});
}

static void print(OpAsmPrinter &p, SliceOp op) {
  auto indexings = op.indexings();
  p << SliceOp::getOperationName() << " " << op.view() << "[" << indexings
    << "] ";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getBaseViewType();
  if (!indexings.empty())
    p << ", " << op.indexings().getTypes();
  p << ", " << op.getType();
}

static ParseResult parseSliceOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType baseInfo;
  SmallVector<OpAsmParser::OperandType, 8> operands;
  SmallVector<Type, 8> types;
  if (parser.parseOperand(baseInfo) ||
      parser.parseOperandList(operands, OpAsmParser::Delimiter::Square) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(types))
    return failure();

  if (types.size() < 2)
    return parser.emitError(parser.getCurrentLocation(),
                            "expected at least input and result view types");

  ArrayRef<Type> indexingTypes = ArrayRef<Type>(types).drop_front().drop_back();
  return failure(
      parser.resolveOperand(baseInfo, types.front(), result.operands) ||
      (!operands.empty() &&
       parser.resolveOperands(operands, indexingTypes,
                              operands.front().location, result.operands)) ||
      parser.addTypeToList(types.back(), result.types));
}

static LogicalResult verify(SliceOp op) {
  unsigned rank = op.getBaseViewRank();
  if (rank != llvm::size(op.indexings()))
    return op.emitOpError("expected ")
           << rank << " indexings, got " << llvm::size(op.indexings());
  unsigned index = 0;
  for (auto indexing : op.indexings()) {
    if (indexing.getType().isa<IndexType>())
      --rank;
    ++index;
  }
  if (op.getRank() != rank)
    return op.emitOpError() << "expected rank of the view(" << op.getRank()
                            << ") to be the number of ranges(" << rank << ")";
  return success();
}

Value SliceOp::getViewSource() { return view(); }

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

  return op.emitOpError("expected parent op with LinalgOp interface");
}

/////// Operations corresponding to library calls defined with Tablegen ////////

void FillOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), output(),
                       SideEffects::DefaultResource::get());
}

static LogicalResult verify(FillOp op) {
  auto viewType = op.getOutputShapedType(0);
  auto fillType = op.value().getType();
  if (viewType.getElementType() != fillType)
    return op.emitOpError("expects fill type to match view elemental type");
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

#include "mlir/Dialect/Linalg/IR/LinalgStructuredOpsInterfaces.cpp.inc"

#include "mlir/Dialect/Linalg/IR/LinalgNamedStructuredOps.cpp.inc"

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
OpFoldResult SliceOp::fold(ArrayRef<Attribute>) {
  if (succeeded(foldMemRefCast(*this)))
    return getResult();
  return {};
}
OpFoldResult TensorReshapeOp::fold(ArrayRef<Attribute> operands) {
  return foldReshapeOp(*this, operands);
}

//===----------------------------------------------------------------------===//
// Auto-generated Linalg named ops.
//===----------------------------------------------------------------------===//

template <typename NamedStructuredOpType>
static void buildNamedStructuredOpRegionAndAttributesImpl(
    OpBuilder &opBuilder, Region &region, TypeRange inputTypes,
    TypeRange outputTypes,
    std::function<void(unsigned, unsigned)> errorHandler) {
  // TODO: atm all operands go through getElementTypeOrSelf,
  // reconsider when we have evidence we need to.
  SmallVector<Type, 8> argTypes;
  for (auto containers : {inputTypes, outputTypes})
    for (auto t : containers)
      argTypes.push_back(getElementTypeOrSelf(t));

  // RAII.
  OpBuilder::InsertionGuard guard(opBuilder);
  Block *body = opBuilder.createBlock(&region, {}, argTypes);
  unsigned actual = body->getNumArguments();
  unsigned expected = NamedStructuredOpType::getNumRegionArgs();
  if (expected != actual)
    return errorHandler(expected, actual);

  opBuilder.setInsertionPointToStart(body);
  mlir::edsc::ScopedContext scope(opBuilder, opBuilder.getUnknownLoc());
  NamedStructuredOpType::regionBuilder(*body);

  // indexing_maps is an auto-generated method.

  // iterator_types is an auto-generated method.
}

template <typename NamedStructuredOpType>
void buildNamedStructuredOpRegionAndAttributes(OpBuilder &opBuilder,
                                               OperationState &result,
                                               TypeRange inputTypes,
                                               TypeRange outputTypes) {
  Region &region = *result.addRegion();
  buildNamedStructuredOpRegionAndAttributesImpl<NamedStructuredOpType>(
      opBuilder, region, inputTypes, outputTypes,
      [&](unsigned expected, unsigned actual) {
        llvm::errs() << "region expects " << expected << " args, got "
                     << actual;
        assert(expected != actual && "incorrect number of arguments");
      });
}

template <typename NamedStructuredOpType>
static ParseResult
parseNamedStructuredOpRegion(OpAsmParser &parser, Region &region,
                             TypeRange inputTypes, TypeRange outputTypes) {
  ParseResult res = success();
  OpBuilder opBuilder(parser.getBuilder().getContext());
  buildNamedStructuredOpRegionAndAttributesImpl<NamedStructuredOpType>(
      opBuilder, region, inputTypes, outputTypes,
      [&](unsigned expected, unsigned actual) {
        res = parser.emitError(parser.getCurrentLocation(),
                               llvm::formatv("region expects {0} args, got {1}",
                                             expected, actual));
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
static ParseResult parseNamedStructuredOp(OpAsmParser &parser,
                                          OperationState &result) {
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
          parser, *region, inputTypes, outputTypes))
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
static void printCommonStructuredOpParts(OpAsmPrinter &p,
                                         NamedStructuredOpType op) {
  if (!op.inputs().empty())
    p << " ins(" << op.inputs() << " : " << op.inputs().getTypes() << ")";
  if (!op.outputs().empty())
    p << " outs(" << op.outputs() << " : " << op.outputs().getTypes() << ")";
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

/// Canonicalize a `linalgOp` -> `dim` pattern by replacing the `dim` arg
/// with the corresponding output tensor argument of the linalg op.
struct ReplaceDimOfLinalgResult : public OpRewritePattern<DimOp> {
  using OpRewritePattern<DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DimOp dimOp,
                                PatternRewriter &rewriter) const override {
    Value dimOpArg = dimOp.memrefOrTensor();
    auto linalgOp = dimOpArg.getDefiningOp<LinalgOp>();
    if (!linalgOp)
      return failure();

    auto results = linalgOp.getOperation()->getResults();
    int64_t id = std::distance(results.begin(), llvm::find(results, dimOpArg));
    auto outputTensors = linalgOp.getOutputTensors();
    rewriter.replaceOpWithNewOp<DimOp>(dimOp, outputTensors[id], dimOp.index());
    return success();
  }
};
} // namespace

#define CANONICALIZERS_AND_FOLDERS(XXX)                                        \
  void XXX::getCanonicalizationPatterns(OwningRewritePatternList &results,     \
                                        MLIRContext *context) {                \
    results.insert<DeduplicateInputs, EraseDeadLinalgOp, FoldTensorCastOp>();  \
    results.insert<ReplaceDimOfLinalgResult>(context);                         \
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
