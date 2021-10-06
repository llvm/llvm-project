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
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
    TypeRange outputTypes,
    std::function<void(unsigned, unsigned)> errorHandler = nullptr);

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
                             TypeRange inputTypes, TypeRange outputTypes);

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

/// Helper function to convert a vector of `OpFoldResult`s into a vector of
/// `Value`s.
static SmallVector<Value> getAsValues(OpBuilder &b, Location loc,
                                      ArrayRef<OpFoldResult> valueOrAttrVec) {
  return llvm::to_vector<4>(
      llvm::map_range(valueOrAttrVec, [&](OpFoldResult value) -> Value {
        return getValueOrCreateConstantIndexOp(b, loc, value);
      }));
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
  RegionBuilderHelper(MLIRContext *context, Block &block)
      : context(context), block(block) {}

  // Generates operations to cast the given operand to a specified type.
  // If the cast cannot be performed, a warning will be issued and the
  // operand returned as-is (which will presumably yield a verification
  // issue downstream).
  Value cast(Type toType, Value operand) {
    OpBuilder builder = getBuilder();
    auto loc = operand.getLoc();

    if (operand.getType() == toType)
      return operand;
    if (auto toIntType = toType.dyn_cast<IntegerType>()) {
      // If operand is floating point, cast directly to the int type.
      if (operand.getType().isa<FloatType>())
        return builder.create<FPToSIOp>(loc, toType, operand);
      // Cast index operands directly to the int type.
      if (operand.getType().isIndex())
        return builder.create<IndexCastOp>(loc, toType, operand);
      if (auto fromIntType = operand.getType().dyn_cast<IntegerType>()) {
        // Either sign extend or truncate.
        if (toIntType.getWidth() > fromIntType.getWidth())
          return builder.create<SignExtendIOp>(loc, toType, operand);
        if (toIntType.getWidth() < fromIntType.getWidth())
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
        if (toFloatType.getWidth() < fromFloatType.getWidth())
          return builder.create<FPTruncOp>(loc, toFloatType, operand);
      }
    }

    emitWarning(operand.getLoc()) << "could not cast operand of type "
                                  << operand.getType() << " to " << toType;
    return operand;
  }

  Value applyfn__add(Value lhs, Value rhs) {
    OpBuilder builder = getBuilder();
    if (isFloatingPoint(lhs))
      return builder.create<AddFOp>(lhs.getLoc(), lhs, rhs);
    if (isInteger(lhs))
      return builder.create<AddIOp>(lhs.getLoc(), lhs, rhs);
    llvm_unreachable("unsupported non numeric type");
  }

  Value applyfn__exp(Value x) {
    OpBuilder builder = getBuilder();
    if (isFloatingPoint(x))
      return builder.create<math::ExpOp>(x.getLoc(), x);
    llvm_unreachable("unsupported non numeric type");
  }

  Value applyfn__log(Value x) {
    OpBuilder builder = getBuilder();
    if (isFloatingPoint(x))
      return builder.create<math::LogOp>(x.getLoc(), x);
    llvm_unreachable("unsupported non numeric type");
  }

  Value applyfn__sub(Value lhs, Value rhs) {
    OpBuilder builder = getBuilder();
    if (isFloatingPoint(lhs))
      return builder.create<SubFOp>(lhs.getLoc(), lhs, rhs);
    if (isInteger(lhs))
      return builder.create<SubIOp>(lhs.getLoc(), lhs, rhs);
    llvm_unreachable("unsupported non numeric type");
  }

  Value applyfn__mul(Value lhs, Value rhs) {
    OpBuilder builder = getBuilder();
    if (isFloatingPoint(lhs))
      return builder.create<MulFOp>(lhs.getLoc(), lhs, rhs);
    if (isInteger(lhs))
      return builder.create<MulIOp>(lhs.getLoc(), lhs, rhs);
    llvm_unreachable("unsupported non numeric type");
  }

  Value applyfn__max(Value lhs, Value rhs) {
    if (isFloatingPoint(lhs))
      return emitCmpFAndSelect(lhs, rhs, CmpFPredicate::OGT);
    if (isInteger(lhs))
      return emitCmpIAndSelect(lhs, rhs, CmpIPredicate::sgt);
    llvm_unreachable("unsupported non numeric type");
  }

  Value applyfn__min(Value lhs, Value rhs) {
    if (isFloatingPoint(lhs))
      return emitCmpFAndSelect(lhs, rhs, CmpFPredicate::OLT);
    if (isInteger(lhs))
      return emitCmpIAndSelect(lhs, rhs, CmpIPredicate::slt);
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

  Value constant(std::string value) {
    OpBuilder builder = getBuilder();
    Location loc = builder.getUnknownLoc();
    Attribute valueAttr = parseAttribute(value, builder.getContext());
    return builder.create<ConstantOp>(loc, valueAttr.getType(), valueAttr);
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

  Value emitCmpFAndSelect(Value lhs, Value rhs, CmpFPredicate predicate) {
    OpBuilder builder = getBuilder();
    Value condition = builder.create<CmpFOp>(lhs.getLoc(), predicate, lhs, rhs);
    return builder.create<SelectOp>(lhs.getLoc(), condition, lhs, rhs);
  }
  Value emitCmpIAndSelect(Value lhs, Value rhs, CmpIPredicate predicate) {
    OpBuilder builder = getBuilder();
    Value condition = builder.create<CmpIOp>(lhs.getLoc(), predicate, lhs, rhs);
    return builder.create<SelectOp>(lhs.getLoc(), condition, lhs, rhs);
  }

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
// CopyOp
//===----------------------------------------------------------------------===//
void CopyOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(block.getNumArguments() == 2 && "CopyOp regionBuilder expects 2 args");
  b.create<linalg::YieldOp>(block.getArgument(0));
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
  OpBuilder opBuilder(parser.getContext());
  fillStructuredOpRegion<CopyOp>(opBuilder, r, TypeRange{inputType},
                                 TypeRange{outputType});
  return success();
}

/// CopyOp region is elided when printing.
void printCopyOpRegion(OpAsmPrinter &, Operation *, Region &, Type, Type) {}

static LogicalResult verify(CopyOp op) {
  OpOperand *output = op.getOutputOperand(0);
  OpOperand *input = op.getInputOperand(0);
  if (getElementTypeOrSelf(input->get()) != getElementTypeOrSelf(output->get()))
    return op.emitOpError("expects views of the same type");
  if (op.getRank(input) != op.getRank(output))
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

namespace {
/// Remove copy operations that copy data inplace. Requirements are:
/// 1) The input and output values are identical.
/// 2) The input and output permutation maps are identical.
struct EraseIdentityCopyOp : public OpRewritePattern<CopyOp> {
  using OpRewritePattern<CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    assert(copyOp.hasBufferSemantics());
    if (copyOp.input() == copyOp.output() &&
        copyOp.inputPermutation() == copyOp.outputPermutation()) {
      rewriter.eraseOp(copyOp);
      return success();
    }
    return failure();
  }
};
} // namespace

void CopyOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<EraseIdentityCopyOp>(context);
}

//===----------------------------------------------------------------------===//
// FillOp
//===----------------------------------------------------------------------===//
void FillOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(block.getNumArguments() == 2 && "FillOp regionBuilder expects 2 args");
  b.create<linalg::YieldOp>(block.getArgument(0));
}

void FillOp::build(OpBuilder &builder, OperationState &result, Value value,
                   Value output) {
  build(builder, result, output.getType().dyn_cast<RankedTensorType>(), value,
        output);
  fillStructuredOpRegion<FillOp>(builder, *result.regions.front(),
                                 TypeRange{value.getType()},
                                 TypeRange{output.getType()}, {});
}

ParseResult parseFillOpRegion(OpAsmParser &parser, Region &r, Type valueType,
                              Type outputType) {
  OpBuilder opBuilder(parser.getContext());
  fillStructuredOpRegion<FillOp>(opBuilder, r, TypeRange{valueType},
                                 TypeRange{outputType});
  return success();
}

/// FillOp region is elided when printing.
void printFillOpRegion(OpAsmPrinter &, Operation *, Region &, Type, Type) {}

static LogicalResult verify(FillOp op) {
  OpOperand *output = op.getOutputOperand(0);
  Type fillType = op.value().getType();
  if (getElementTypeOrSelf(output->get()) != fillType)
    return op.emitOpError("expects fill type to match view elemental type");
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
  for (ValueRange container : {inputs, outputs})
    for (Value v : container)
      blockArgTypes.push_back(getElementTypeOrSelf(v));

  OpBuilder::InsertionGuard guard(builder);
  auto &region = *result.regions.front();
  Block *bodyBlock = builder.createBlock(&region, region.end(), blockArgTypes);
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

static void print(OpAsmPrinter &p, GenericOp op) {
  p << " ";

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
  SmallVector<Value> inputBuffers = getInputBufferOperands();
  SmallVector<Value> outputBuffers = getOutputBufferOperands();
  getGenericEffectsImpl(effects, getOperation()->getResults(), inputBuffers,
                        outputBuffers);
}

template <typename GenericOpType>
static LogicalResult verifyGenericOp(GenericOpType op) {
  return success();
}

static LogicalResult verify(GenericOp op) { return verifyGenericOp(op); }

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
      if (!llvm::is_contained(odsAttrs, kv.first.c_str())) {
        newOp->setAttr(kv.first, kv.second);
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
    if (!genericOp.hasTensorSemantics())
      return failure();
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

    // Get the argument number of the returned values. That is the operand
    // number to use for replacing uses of this operation.
    SmallVector<Value> returnedArgs;
    for (Value yieldVal : yieldOp.values()) {
      auto yieldArg = yieldVal.dyn_cast<BlockArgument>();
      if (!yieldArg || yieldArg.getOwner() != &body)
        return failure();
      unsigned argumentNumber = yieldArg.getArgNumber();
      returnedArgs.push_back(genericOp->getOperand(argumentNumber));
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
    if (failed(reshapeOp.reifyResultShapes(rewriter, resultShapes)) ||
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
              FoldInitTensorWithTensorReshapeOp<TensorExpandShapeOp>,
              FoldInitTensorWithTensorReshapeOp<TensorCollapseShapeOp>,
              ReplaceStaticShapeDims>(context);
}

LogicalResult InitTensorOp::reifyResultShapes(
    OpBuilder &builder, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
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

// TODO: Replace custom<InferType> directive with AllTypesMatch as soon as it
// supports optional types.
void printInferType(OpAsmPrinter &printer, Operation *op, Value optOperand,
                    Type typeToInfer, Type typeToInferFrom) {}

ParseResult parseInferType(OpAsmParser &parser,
                           Optional<OpAsmParser::OperandType> optOperand,
                           Type &typeToInfer, Type typeToInferFrom) {
  if (optOperand)
    typeToInfer = typeToInferFrom;
  return success();
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
                                              ArrayRef<int64_t> staticHigh,
                                              ArrayRef<int64_t> resultShape) {
  unsigned rank = sourceType.getRank();
  assert(staticLow.size() == rank && "unexpected staticLow size mismatch");
  assert(staticHigh.size() == rank && "unexpected staticHigh size mismatch");
  assert((resultShape.empty() || resultShape.size() == rank) &&
         "unexpected resultShape size mismatch");

  SmallVector<int64_t, 4> inferredShape;
  for (auto i : llvm::seq<unsigned>(0, rank)) {
    if (sourceType.isDynamicDim(i) ||
        staticLow[i] == ShapedType::kDynamicSize ||
        staticHigh[i] == ShapedType::kDynamicSize) {
      inferredShape.push_back(resultShape.empty() ? ShapedType::kDynamicSize
                                                  : resultShape[i]);
    } else {
      int64_t size = sourceType.getDimSize(i) + staticLow[i] + staticHigh[i];
      assert((resultShape.empty() || size == resultShape[i] ||
              resultShape[i] == ShapedType::kDynamicSize) &&
             "mismatch between inferred shape and result shape");
      inferredShape.push_back(size);
    }
  }

  return RankedTensorType::get(inferredShape, sourceType.getElementType());
}

void PadTensorOp::build(OpBuilder &b, OperationState &result, Value source,
                        ArrayRef<int64_t> staticLow,
                        ArrayRef<int64_t> staticHigh, ValueRange low,
                        ValueRange high, bool nofold,
                        ArrayRef<NamedAttribute> attrs) {
  auto sourceType = source.getType().cast<RankedTensorType>();
  auto resultType = inferResultType(sourceType, staticLow, staticHigh);
  build(b, result, resultType, source, low, high, b.getI64ArrayAttr(staticLow),
        b.getI64ArrayAttr(staticHigh), nofold ? b.getUnitAttr() : UnitAttr());
  result.addAttributes(attrs);
}

void PadTensorOp::build(OpBuilder &b, OperationState &result, Value source,
                        ValueRange low, ValueRange high, bool nofold,
                        ArrayRef<NamedAttribute> attrs) {
  auto sourceType = source.getType().cast<RankedTensorType>();
  unsigned rank = sourceType.getRank();
  SmallVector<int64_t, 4> staticVector(rank, ShapedType::kDynamicSize);
  build(b, result, source, staticVector, staticVector, low, high, nofold,
        attrs);
}

void PadTensorOp::build(OpBuilder &b, OperationState &result, Type resultType,
                        Value source, ArrayRef<OpFoldResult> low,
                        ArrayRef<OpFoldResult> high, bool nofold,
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
        b.getI64ArrayAttr(staticLow), b.getI64ArrayAttr(staticHigh),
        nofold ? b.getUnitAttr() : UnitAttr());
  result.addAttributes(attrs);
}

PadTensorOp PadTensorOp::createPadScalarOp(Type type, Value source, Value pad,
                                           ArrayRef<OpFoldResult> low,
                                           ArrayRef<OpFoldResult> high,
                                           bool nofold, Location loc,
                                           OpBuilder &builder) {
  auto padTensorOp =
      builder.create<linalg::PadTensorOp>(loc, type, source, low, high, nofold);
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
                                         bool nofold, Location loc,
                                         OpBuilder &builder) {
  SmallVector<OpFoldResult, 4> low, high;
  auto rankedTensorType = type.cast<RankedTensorType>();
  assert(rankedTensorType.hasStaticShape());
  int rank = rankedTensorType.getRank();
  for (int i = 0; i < rank; ++i) {
    auto dimOp = builder.createOrFold<tensor::DimOp>(loc, source, i);
    auto resultDimSize = builder.createOrFold<ConstantIndexOp>(
        loc, rankedTensorType.getDimSize(i));
    auto highValue = builder.createOrFold<SubIOp>(loc, resultDimSize, dimOp);
    high.push_back(highValue);
    low.push_back(builder.createOrFold<ConstantIndexOp>(loc, 0));
  }
  return PadTensorOp::createPadScalarOp(type, source, pad, low, high, nofold,
                                        loc, builder);
}

LogicalResult PadTensorOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  Location loc = getLoc();
  auto lowPad = getMixedLowPad();
  auto highPad = getMixedHighPad();
  SmallVector<Value> shapes;
  for (auto dim : llvm::seq<int64_t>(0, getSourceType().getRank())) {
    // Shape along each dimension is source dim + low pad + high pad.
    SmallVector<Value> mapOperands;
    mapOperands.push_back(b.createOrFold<tensor::DimOp>(loc, source(), dim));
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
// Methods related to PadTensor tiling.
//===----------------------------------------------------------------------===//

SmallVector<Value> PadTensorOp::getDestinationOperands(OpBuilder &b) {
  ReifiedRankedShapedTypeDims reifiedShapes;
  (void)reifyResultShapes(b, reifiedShapes);
  SmallVector<OpFoldResult> mixedSizes = getAsOpFoldResult(reifiedShapes[0]);
  Value initTensor = b.create<InitTensorOp>(getLoc(), mixedSizes,
                                            getResultType().getElementType());
  return {initTensor};
}

SmallVector<StringRef> PadTensorOp::getLoopIteratorTypes() {
  SmallVector<StringRef> iteratorTypes(getResultType().getRank(),
                                       getParallelIteratorTypeName());
  return iteratorTypes;
}

SmallVector<Range> PadTensorOp::getLoopBounds(OpBuilder &b) {
  ReifiedRankedShapedTypeDims reifiedShapes;
  (void)reifyResultShapes(b, reifiedShapes);
  Value zero = b.create<ConstantIndexOp>(getLoc(), 0);
  Value one = b.create<ConstantIndexOp>(getLoc(), 1);
  // Initialize all the ranges to {zero, one, one}. All the `ub`s are
  // overwritten.
  SmallVector<Range> loopRanges(reifiedShapes[0].size(), {zero, one, one});
  for (auto ub : enumerate(reifiedShapes[0]))
    loopRanges[ub.index()].size = ub.value();
  return loopRanges;
}

Operation *PadTensorOp::getTiledImplementation(OpBuilder &b, ValueRange dest,
                                               ArrayRef<OpFoldResult> offsets,
                                               ArrayRef<OpFoldResult> sizes) {
  // Only constant padding value supported.
  Value padValue = getConstantPaddingValue();
  if (!padValue)
    return nullptr;

  // Helper variables and functions for various arithmetic operations. These are
  // used extensively for computing new offset/length and padding values.
  Location loc = getLoc();
  AffineExpr dim0, dim1;
  bindDims(b.getContext(), dim0, dim1);
  // Add two integers.
  auto addMap = AffineMap::get(2, 0, {dim0 + dim1});
  auto add = [&](Value v1, Value v2) {
    return b.createOrFold<AffineApplyOp>(loc, addMap, ValueRange{v1, v2});
  };
  // Subtract two integers.
  auto subMap = AffineMap::get(2, 0, {dim0 - dim1});
  auto sub = [&](Value v1, Value v2) {
    return b.createOrFold<AffineApplyOp>(loc, subMap, ValueRange{v1, v2});
  };
  // Take the minimum of two integers.
  auto idMap = AffineMap::getMultiDimIdentityMap(2, b.getContext());
  auto min = [&](Value v1, Value v2) {
    return b.createOrFold<AffineMinOp>(loc, idMap, ValueRange{v1, v2});
  };
  // Take the maximum of two integers.
  auto max = [&](Value v1, Value v2) {
    return b.createOrFold<AffineMaxOp>(loc, idMap, ValueRange{v1, v2});
  };
  // Zero index-typed integer.
  auto zero = b.create<ConstantIndexOp>(loc, 0);

  // Helper function for filling static/dynamic low/high padding indices vectors
  // of PadTensorOp.
  auto appendIndex = [&](Value val, SmallVector<Value> &dynIndices,
                         SmallVector<int64_t> &staticIndices) {
    if (auto constInt = getConstantIntValue(val)) {
      staticIndices.push_back(*constInt);
    } else {
      staticIndices.push_back(ShapedType::kDynamicSize);
      dynIndices.push_back(val);
    }
  };

  // Compute new offsets, lengths, low padding, high padding.
  SmallVector<OpFoldResult> newOffsets, newLengths, newStrides;
  SmallVector<Value> newLows, newHighs;
  SmallVector<int64_t> staticNewLows, staticNewHighs;
  // Set to true if the original data source is not read at all.
  bool hasZeroLen = false;
  // Same as hasZeroLen, but for dynamic dimension sizes. This condition
  // is true if the original data source turns out to be unused at runtime.
  Value dynHasZeroLenCond;

  int64_t rank = getSourceType().getRank();
  for (unsigned dim = 0; dim < rank; ++dim) {
    auto low = getValueOrCreateConstantIndexOp(b, loc, getMixedLowPad()[dim]);
    bool hasLowPad = getConstantIntValue(low) != static_cast<int64_t>(0);
    auto high = getValueOrCreateConstantIndexOp(b, loc, getMixedHighPad()[dim]);
    bool hasHighPad = getConstantIntValue(high) != static_cast<int64_t>(0);
    auto offset = getValueOrCreateConstantIndexOp(b, loc, offsets[dim]);
    auto length = getValueOrCreateConstantIndexOp(b, loc, sizes[dim]);
    auto srcSize = b.createOrFold<tensor::DimOp>(loc, source(), dim);

    // The new amount of low padding is `low - offset`. Except for the case
    // where none of the low padding is read. In that case, the new amount of
    // low padding is zero.
    //
    // Optimization: If low = 0, then newLow = 0.
    Value newLow = hasLowPad ? max(zero, sub(low, offset)) : zero;
    appendIndex(newLow, newLows, staticNewLows);

    // Start reading the data from position `offset - low`. Since the original
    // read may have started in the low padding zone, this value could be
    // negative. Therefore, start reading from:
    //
    // max(offset - low, 0)
    //
    // The original read could also have started in the high padding zone.
    // In that case, set the offset to the end of source tensor. The new
    // ExtractSliceOp length will be zero in that case. (Effectively reading no
    // data from the source.)
    //
    // Optimization: If low = 0, then the formula can be simplified.
    Value newOffset = hasLowPad ? min(max(sub(offset, low), zero), srcSize)
                                : min(offset, srcSize);
    newOffsets.push_back(getAsOpFoldResult(newOffset));

    // The original ExtractSliceOp was reading until position `offset + length`.
    // Therefore, the corresponding position within the source tensor is:
    //
    // offset + length - low
    //
    // In case the original ExtractSliceOp stopped reading within the low
    // padding zone, this value can be negative. In that case, the end position
    // of the read should be zero. (Similar to newOffset.)
    //
    // The original read could also have stopped in the high padding zone.
    // In that case, set the end positition of the read should be the end of the
    // source tensor. (Similar to newOffset.)
    //
    // endLoc = min(max(offset - low + length, 0), srcSize)
    //
    // The new ExtractSliceOp length is `endLoc - newOffset`.
    //
    // Optimization: If low = 0, then the formula can be simplified.
    Value endLoc = hasLowPad
                       ? min(max(add(sub(offset, low), length), zero), srcSize)
                       : min(add(offset, length), srcSize);
    Value newLength = sub(endLoc, newOffset);
    newLengths.push_back(getAsOpFoldResult(newLength));

    // Check if newLength is zero. In that case, no SubTensorOp should be
    // executed.
    if (auto newLengthInt = getConstantIntValue(newLength)) {
      hasZeroLen |= *newLengthInt == 0;
    } else {
      Value check = b.create<CmpIOp>(loc, CmpIPredicate::eq, newLength, zero);
      dynHasZeroLenCond = dynHasZeroLenCond
                              ? b.create<OrOp>(loc, check, dynHasZeroLenCond)
                              : check;
    }

    // The amount of high padding is simply the number of elements remaining,
    // so that the result has the same length as the original ExtractSliceOp.
    // As an optimization, if the original high padding is zero, then the new
    // high padding must also be zero.
    Value newHigh = hasHighPad ? sub(sub(length, newLength), newLow) : zero;
    appendIndex(newHigh, newHighs, staticNewHighs);

    // Only unit stride supported.
    newStrides.push_back(b.getIndexAttr(1));
  }

  // The shape of the result can be obtained from the sizes passed in.
  SmallVector<Value> dynDims;
  SmallVector<int64_t> shape;
  dispatchIndexOpFoldResults(sizes, dynDims, shape, ShapedType::kDynamicSize);
  RankedTensorType resultType =
      RankedTensorType::get(shape, getResultType().getElementType());

  // Insert cast to ensure that types match. (May be folded away.)
  auto castResult = [&](Value val) -> Operation * {
    auto castOp = b.create<tensor::CastOp>(loc, resultType, val);
    return castOp;
  };

  // In cases where the original data source is unused: Emit a GenerateOp and
  // do not generate a SliceOp. (The result shape of the SliceOp would
  // have a dimension of size 0, the semantics of which is unclear.)
  auto createGenerateOp = [&]() {
    // Create GenerateOp.
    auto generateOp = b.create<tensor::GenerateOp>(
        loc, resultType, dynDims,
        [&](OpBuilder &builder, Location gLoc, ValueRange indices) {
          builder.create<tensor::YieldOp>(gLoc, padValue);
        });
    return castResult(generateOp);
  };

  // Emit a SliceOp and a PadTensorOp. Should not be used in cases where
  // the result shape of the new SliceOp has a zero dimension.
  auto createPadTensorOfSubTensor = [&]() {
    // Create pad_tensor(subtensor(x)).
    auto newSliceOp = b.create<tensor::ExtractSliceOp>(
        loc, source(), newOffsets, newLengths, newStrides);
    auto newPadTensorOp = b.create<PadTensorOp>(
        loc, newSliceOp, staticNewLows, staticNewHighs, newLows, newHighs);

    // Copy region to new PadTensorOp.
    BlockAndValueMapping bvm;
    region().cloneInto(&newPadTensorOp.getRegion(), bvm);

    // Cast result and return.
    return castResult(newPadTensorOp);
  };

  // Rewrite subtensor(pad_tensor(x)) into a GenerateOp it is statically known
  // that the original data source x is not used.
  if (hasZeroLen) {
    return createGenerateOp();
  }

  // If there are dynamic dimensions: Generate an scf.if check to avoid creating
  // SliceOps with result dimensions of size 0 at runtime.
  if (dynHasZeroLenCond) {
    auto result = b.create<scf::IfOp>(
        loc, resultType, dynHasZeroLenCond,
        /*thenBuilder=*/
        [&](OpBuilder &b, Location loc) {
          b.create<scf::YieldOp>(loc, createGenerateOp()->getResult(0));
        },
        /*elseBuilder=*/
        [&](OpBuilder &b, Location loc) {
          b.create<scf::YieldOp>(loc,
                                 createPadTensorOfSubTensor()->getResult(0));
        });
    return result;
  }
  return createPadTensorOfSubTensor();
}

namespace {
// Folds linalg.pad_tensor when padding is static zeros and the attribute
// doesn't request otherwise.
struct FoldStaticZeroPadding : public OpRewritePattern<PadTensorOp> {
  using OpRewritePattern<PadTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PadTensorOp padTensorOp,
                                PatternRewriter &rewriter) const override {
    if (!padTensorOp.hasZeroLowPad() || !padTensorOp.hasZeroHighPad())
      return failure();
    if (padTensorOp.nofold())
      return failure();
    rewriter.replaceOpWithNewOp<tensor::CastOp>(
        padTensorOp, padTensorOp.result().getType(), padTensorOp.source());
    return success();
  }
};

// Fold CastOp into PadTensorOp when adding static information.
struct FoldSourceTensorCast : public OpRewritePattern<PadTensorOp> {
  using OpRewritePattern<PadTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PadTensorOp padTensorOp,
                                PatternRewriter &rewriter) const override {
    auto castOp = padTensorOp.source().getDefiningOp<tensor::CastOp>();
    if (!tensor::canFoldIntoConsumerOp(castOp))
      return failure();

    auto newResultType = PadTensorOp::inferResultType(
        castOp.source().getType().cast<RankedTensorType>(),
        extractFromI64ArrayAttr(padTensorOp.static_low()),
        extractFromI64ArrayAttr(padTensorOp.static_high()),
        padTensorOp.getResultType().getShape());

    if (newResultType == padTensorOp.getResultType()) {
      rewriter.updateRootInPlace(padTensorOp, [&]() {
        padTensorOp.sourceMutable().assign(castOp.source());
      });
    } else {
      auto newOp = rewriter.create<PadTensorOp>(
          padTensorOp->getLoc(), newResultType, padTensorOp.source(),
          padTensorOp.low(), padTensorOp.high(), padTensorOp.static_low(),
          padTensorOp.static_high(), padTensorOp.nofold());
      BlockAndValueMapping mapper;
      padTensorOp.getRegion().cloneInto(&newOp.getRegion(), mapper);

      rewriter.replaceOpWithNewOp<tensor::CastOp>(
          padTensorOp, padTensorOp.getResultType(), newOp);
    }
    return success();
  }
};

// Fold CastOp using the result of PadTensorOp back into the latter if it adds
// static information.
struct FoldTargetTensorCast : public OpRewritePattern<PadTensorOp> {
  using OpRewritePattern<PadTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PadTensorOp padTensorOp,
                                PatternRewriter &rewriter) const override {
    if (!padTensorOp.result().hasOneUse())
      return failure();
    auto tensorCastOp =
        dyn_cast<tensor::CastOp>(*padTensorOp->getUsers().begin());
    if (!tensorCastOp)
      return failure();
    if (!tensor::preservesStaticInformation(padTensorOp.result().getType(),
                                            tensorCastOp.dest().getType()))
      return failure();

    auto replacementOp = rewriter.create<PadTensorOp>(
        padTensorOp.getLoc(), tensorCastOp.dest().getType(),
        padTensorOp.source(), padTensorOp.low(), padTensorOp.high(),
        padTensorOp.static_low(), padTensorOp.static_high(),
        padTensorOp.nofold());
    replacementOp.region().takeBody(padTensorOp.region());

    rewriter.replaceOp(padTensorOp, replacementOp.result());
    rewriter.replaceOp(tensorCastOp, replacementOp.result());
    return success();
  }
};
} // namespace

void PadTensorOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<FoldStaticZeroPadding, FoldSourceTensorCast>(context);
  results.add<FoldTargetTensorCast>(context);
}

/// Return the padding value of the PadTensorOp if it constant. In this context,
/// "constant" means an actual constant or "defined outside of the block".
///
/// Values are considered constant in three cases:
///  - A ConstantLike value.
///  - A basic block argument from a different block.
///  - A value defined outside of the block.
///
/// If the padding value is not constant, an empty Value is returned.
Value PadTensorOp::getConstantPaddingValue() {
  auto yieldOp = dyn_cast<YieldOp>(getRegion().front().getTerminator());
  if (!yieldOp || yieldOp.values().size() != 1)
    return {};
  Value padValue = yieldOp.values().front();
  // Check if yield value is a constant.
  if (matchPattern(padValue, m_Constant()))
    return padValue;
  // Check if yield value is defined inside the PadTensorOp block.
  if (padValue.getParentBlock() == &getRegion().front())
    return {};
  // Else: Yield value defined outside of the PadTensorOp block.
  return padValue;
}

OpFoldResult PadTensorOp::fold(ArrayRef<Attribute>) {
  if (getResultType().hasStaticShape() && getResultType() == getSourceType() &&
      !nofold())
    return source();
  return {};
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, linalg::TensorExpandShapeOp op) {
  ::mlir::printReshapeOp<linalg::TensorExpandShapeOp>(p, op);
}

static void print(OpAsmPrinter &p, linalg::TensorCollapseShapeOp op) {
  ::mlir::printReshapeOp<linalg::TensorCollapseShapeOp>(p, op);
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

SmallVector<AffineMap, 4> TensorCollapseShapeOp::getReassociationMaps() {
  return getSymbolLessAffineMaps(getReassociationExprs());
}
SmallVector<ReassociationExprs, 4>
TensorCollapseShapeOp::getReassociationExprs() {
  return convertReassociationIndicesToExprs(getContext(),
                                            getReassociationIndices());
}
SmallVector<AffineMap, 4> TensorExpandShapeOp::getReassociationMaps() {
  return getSymbolLessAffineMaps(getReassociationExprs());
}
SmallVector<ReassociationExprs, 4>
TensorExpandShapeOp::getReassociationExprs() {
  return convertReassociationIndicesToExprs(getContext(),
                                            getReassociationIndices());
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
  for (auto dim : llvm::seq_inclusive(startPos, endPos)) {
    dynamicDims.push_back(builder.createOrFold<tensor::DimOp>(loc, src, dim));
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
    for (auto dim : llvm::seq_inclusive(startPos, endPos)) {
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
  Value sourceDim = builder.create<tensor::DimOp>(loc, src, sourceDimPos);
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

void mlir::linalg::TensorCollapseShapeOp::build(
    OpBuilder &b, OperationState &result, Value src,
    ArrayRef<ReassociationIndices> reassociation,
    ArrayRef<NamedAttribute> attrs) {
  auto resultType = computeTensorReshapeCollapsedType(
      src.getType().cast<RankedTensorType>(),
      getSymbolLessAffineMaps(
          convertReassociationIndicesToExprs(b.getContext(), reassociation)));
  build(b, result, resultType, src, attrs);
  result.addAttribute(getReassociationAttrName(),
                      getReassociationIndicesAttribute(b, reassociation));
}

void mlir::linalg::TensorExpandShapeOp::build(
    OpBuilder &b, OperationState &result, Value src,
    ArrayRef<ReassociationIndices> reassociation,
    ArrayRef<NamedAttribute> attrs) {
  auto resultType = computeTensorReshapeCollapsedType(
      src.getType().cast<RankedTensorType>(),
      getSymbolLessAffineMaps(
          convertReassociationIndicesToExprs(b.getContext(), reassociation)));
  build(b, result, resultType, src, attrs);
  result.addAttribute(getReassociationAttrName(),
                      getReassociationIndicesAttribute(b, reassociation));
}

template <typename TensorReshapeOp,
          bool isExpansion =
              std::is_same<TensorReshapeOp, TensorExpandShapeOp>::value>
static LogicalResult verifyTensorReshapeOp(TensorReshapeOp op,
                                           RankedTensorType expandedType,
                                           RankedTensorType collapsedType) {
  if (failed(
          verifyReshapeLikeTypes(op, expandedType, collapsedType, isExpansion)))
    return failure();

  auto maps = op.getReassociationMaps();
  RankedTensorType expectedType =
      computeTensorReshapeCollapsedType(expandedType, maps);
  if (collapsedType != expectedType)
    return op.emitOpError("expected collapsed type to be ")
           << expectedType << ", but got " << collapsedType;
  return success();
}

static LogicalResult verify(TensorExpandShapeOp op) {
  return verifyTensorReshapeOp(op, op.getResultType(), op.getSrcType());
}

static LogicalResult verify(TensorCollapseShapeOp op) {
  return verifyTensorReshapeOp(op, op.getSrcType(), op.getResultType());
}

namespace {
/// Reshape of a splat constant can be replaced with a constant of the result
/// type.
template <typename TensorReshapeOp>
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
} // namespace

void TensorExpandShapeOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results
      .add<CollapseReshapeOps<TensorExpandShapeOp>,
           CollapseMixedReshapeOps<TensorExpandShapeOp, TensorCollapseShapeOp>,
           FoldFillWithTensorReshape<TensorExpandShapeOp>,
           FoldInitTensorWithTensorReshapeOp<TensorExpandShapeOp>,
           FoldReshapeWithConstant<TensorExpandShapeOp>>(context);
}

void TensorCollapseShapeOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results
      .add<CollapseReshapeOps<TensorCollapseShapeOp>,
           CollapseMixedReshapeOps<TensorCollapseShapeOp, TensorExpandShapeOp>,
           FoldFillWithTensorReshape<TensorCollapseShapeOp>,
           FoldInitTensorWithTensorReshapeOp<TensorCollapseShapeOp>,
           FoldReshapeWithConstant<TensorCollapseShapeOp>>(context);
}

LogicalResult TensorExpandShapeOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  auto resultShape =
      getAsValues(b, getLoc(),
                  getReshapeOutputShapeFromInputShape(
                      b, getLoc(), src(), getResultType().getShape(),
                      getReassociationMaps()));
  reifiedReturnShapes.emplace_back(std::move(resultShape));
  return success();
}

LogicalResult TensorCollapseShapeOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
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
  p << " (" << op.getInductionVars() << ") = (" << op.lowerBound() << ") to ("
    << op.upperBound() << ") step (" << op.step() << ")";

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
      }))
    p << " iterators" << op.iterator_types() << "";

  if (op.distribution_types().hasValue())
    p << " distribution" << op.distribution_types().getValue() << "";

  p.printRegion(op.region(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(
      op->getAttrs(), /*elidedAttrs=*/{TiledLoopOp::getOperandSegmentSizeAttr(),
                                       getIteratorTypesAttrName(),
                                       getDistributionTypesAttrName()});
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

void TiledLoopOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<TiledLoopInputsFolder, TiledLoopResultsFolder,
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
OpFoldResult TensorExpandShapeOp::fold(ArrayRef<Attribute> operands) {
  return foldReshapeOp<TensorExpandShapeOp, TensorCollapseShapeOp>(*this,
                                                                   operands);
}
OpFoldResult TensorCollapseShapeOp::fold(ArrayRef<Attribute> operands) {
  return foldReshapeOp<TensorCollapseShapeOp, TensorExpandShapeOp>(*this,
                                                                   operands);
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
                       std::function<void(unsigned, unsigned)> errorHandler) {
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
  ImplicitLocOpBuilder b(opBuilder.getUnknownLoc(), opBuilder);
  NamedStructuredOpType::regionBuilder(b, *body);

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
      opBuilder, region, inputTypes, outputTypes,
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
                             TypeRange inputTypes, TypeRange outputTypes) {
  ParseResult res = success();
  OpBuilder opBuilder(parser.getContext());
  // Resolve `captures` into `capturedValues` at parse time so we can build the
  // region with captures.
  SmallVector<Value> capturedValues;
  fillStructuredOpRegion<NamedStructuredOpType>(
      opBuilder, region, inputTypes, outputTypes,
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

static llvm::SmallVector<int64_t> getIndicesVector(int start, int end) {
  return llvm::to_vector<2>(llvm::seq<int64_t>(start, end));
}

LogicalResult matchAndReplaceDepthwiseConv(Operation *operation, Value input,
                                           Value kernel, Value iZp, Value kZp,
                                           Value init, Attribute stride,
                                           Attribute dilation,
                                           PatternRewriter &rewriter) {
  Location loc = operation->getLoc();
  auto linalgOp = dyn_cast<LinalgOp>(operation);
  // Exit out on the memref version of this operation.
  if (!linalgOp || !linalgOp.hasTensorSemantics())
    return failure();

  auto result = operation->getResult(0);

  auto kernelTy = kernel.getType().dyn_cast<RankedTensorType>();
  auto initTy = init.getType().dyn_cast<RankedTensorType>();
  auto resultTy = result.getType().template dyn_cast<RankedTensorType>();
  if (!kernelTy || !initTy || !resultTy)
    return failure();

  if (kernelTy.getDimSize(3) != 1)
    return failure();

  // Collapse kernel dims.
  SmallVector<ReassociationIndices, 4> collapsedKernelDims = {
      getIndicesVector(0, 1), getIndicesVector(1, 2), getIndicesVector(2, 4)};
  auto newKernelTy = RankedTensorType::get(
      {kernelTy.getDimSize(0), kernelTy.getDimSize(1), kernelTy.getDimSize(2)},
      kernelTy.getElementType());
  auto collapsedKernel = rewriter.create<linalg::TensorCollapseShapeOp>(
      loc, newKernelTy, kernel, collapsedKernelDims);

  // Collapse init dims.
  SmallVector<ReassociationIndices, 4> collapsedInitDims = {
      getIndicesVector(0, 1), getIndicesVector(1, 2), getIndicesVector(2, 3),
      getIndicesVector(3, 5)};
  auto newInitTy =
      RankedTensorType::get({initTy.getDimSize(0), initTy.getDimSize(1),
                             initTy.getDimSize(2), initTy.getDimSize(3)},
                            initTy.getElementType());
  auto collapsedInit = rewriter.create<linalg::TensorCollapseShapeOp>(
      loc, newInitTy, init, collapsedInitDims);

  Value newConv;
  if (isa<DepthwiseConv2DNhwcOp>(operation)) {
    newConv = rewriter
                  .create<DepthwiseConv2DNhwOp>(
                      loc, newInitTy, ValueRange{input, collapsedKernel},
                      ValueRange{collapsedInit}, stride, dilation)
                  .getResult(0);
  } else if (isa<DepthwiseConv2DNhwcQOp>(operation)) {
    newConv =
        rewriter
            .create<DepthwiseConv2DNhwQOp>(
                loc, newInitTy, ValueRange{input, collapsedKernel, iZp, kZp},
                ValueRange{collapsedInit}, stride, dilation)
            .getResult(0);
  }

  if (!newConv)
    return failure();

  // Expand dimensions back out to
  rewriter.replaceOpWithNewOp<linalg::TensorExpandShapeOp>(
      operation, resultTy, newConv, collapsedInitDims);
  return success();
}

struct SimplifyDepthwiseConvOp
    : public OpRewritePattern<DepthwiseConv2DNhwcOp> {
  using OpRewritePattern<DepthwiseConv2DNhwcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DepthwiseConv2DNhwcOp op,
                                PatternRewriter &rewriter) const override {
    Operation *operation = op.getOperation();
    Value input = op.getInputOperand(0)->get();
    Value kernel = op.getInputOperand(1)->get();
    Value init = op.getOutputOperand(0)->get();

    auto stride = op.strides();
    auto dilation = op.dilations();

    return matchAndReplaceDepthwiseConv(operation, input, kernel, nullptr,
                                        nullptr, init, stride, dilation,
                                        rewriter);
  }
};

struct SimplifyDepthwiseConvQOp
    : public OpRewritePattern<DepthwiseConv2DNhwcQOp> {
  using OpRewritePattern<DepthwiseConv2DNhwcQOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DepthwiseConv2DNhwcQOp op,
                                PatternRewriter &rewriter) const override {
    Operation *operation = op.getOperation();
    Value input = op.getInputOperand(0)->get();
    Value kernel = op.getInputOperand(1)->get();
    Value iZp = op.getInputOperand(2)->get();
    Value kZp = op.getInputOperand(3)->get();
    Value init = op.getOutputOperand(0)->get();

    auto stride = op.strides();
    auto dilation = op.dilations();

    return matchAndReplaceDepthwiseConv(operation, input, kernel, iZp, kZp,
                                        init, stride, dilation, rewriter);
  }
};

} // namespace

#define LINALGOP_FOLDERS(XXX)                                                  \
  LogicalResult XXX::fold(ArrayRef<Attribute>,                                 \
                          SmallVectorImpl<OpFoldResult> &) {                   \
    return foldMemRefCast(*this);                                              \
  }

LINALGOP_FOLDERS(ConvOp)
LINALGOP_FOLDERS(CopyOp)
LINALGOP_FOLDERS(FillOp)
LINALGOP_FOLDERS(GenericOp)

// All named ops canonicalizers and folders are auto-generated in the
// .cpp.inc.

//===----------------------------------------------------------------------===//
// LinalgDialect
//===----------------------------------------------------------------------===//

void LinalgDialect::getCanonicalizationPatterns(
    RewritePatternSet &results) const {
  results.add<EraseDeadLinalgOp, FoldTensorCastOp, SimplifyDepthwiseConvOp,
              SimplifyDepthwiseConvQOp>(getContext());
}
