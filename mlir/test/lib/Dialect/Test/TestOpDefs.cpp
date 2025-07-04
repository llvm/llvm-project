//===- TestOpDefs.cpp - MLIR Test Dialect Operation Hooks -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "TestOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"

using namespace mlir;
using namespace test;

//===----------------------------------------------------------------------===//
// TestBranchOp
//===----------------------------------------------------------------------===//

SuccessorOperands TestBranchOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return SuccessorOperands(getTargetOperandsMutable());
}

//===----------------------------------------------------------------------===//
// TestProducingBranchOp
//===----------------------------------------------------------------------===//

SuccessorOperands TestProducingBranchOp::getSuccessorOperands(unsigned index) {
  assert(index <= 1 && "invalid successor index");
  if (index == 1)
    return SuccessorOperands(getFirstOperandsMutable());
  return SuccessorOperands(getSecondOperandsMutable());
}

//===----------------------------------------------------------------------===//
// TestInternalBranchOp
//===----------------------------------------------------------------------===//

SuccessorOperands TestInternalBranchOp::getSuccessorOperands(unsigned index) {
  assert(index <= 1 && "invalid successor index");
  if (index == 0)
    return SuccessorOperands(0, getSuccessOperandsMutable());
  return SuccessorOperands(1, getErrorOperandsMutable());
}

//===----------------------------------------------------------------------===//
// TestCallOp
//===----------------------------------------------------------------------===//

LogicalResult TestCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  if (!symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(*this, fnAttr))
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";
  return success();
}

//===----------------------------------------------------------------------===//
// FoldToCallOp
//===----------------------------------------------------------------------===//

namespace {
struct FoldToCallOpPattern : public OpRewritePattern<FoldToCallOp> {
  using OpRewritePattern<FoldToCallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FoldToCallOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::CallOp>(op, TypeRange(),
                                              op.getCalleeAttr(), ValueRange());
    return success();
  }
};
} // namespace

void FoldToCallOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.add<FoldToCallOpPattern>(context);
}

//===----------------------------------------------------------------------===//
// IsolatedRegionOp - test parsing passthrough operands
//===----------------------------------------------------------------------===//

ParseResult IsolatedRegionOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  // Parse the input operand.
  OpAsmParser::Argument argInfo;
  argInfo.type = parser.getBuilder().getIndexType();
  if (parser.parseOperand(argInfo.ssaName) ||
      parser.resolveOperand(argInfo.ssaName, argInfo.type, result.operands))
    return failure();

  // Parse the body region, and reuse the operand info as the argument info.
  Region *body = result.addRegion();
  return parser.parseRegion(*body, argInfo, /*enableNameShadowing=*/true);
}

void IsolatedRegionOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printOperand(getOperand());
  p.shadowRegionArgs(getRegion(), getOperand());
  p << ' ';
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// SSACFGRegionOp
//===----------------------------------------------------------------------===//

RegionKind SSACFGRegionOp::getRegionKind(unsigned index) {
  return RegionKind::SSACFG;
}

//===----------------------------------------------------------------------===//
// GraphRegionOp
//===----------------------------------------------------------------------===//

RegionKind GraphRegionOp::getRegionKind(unsigned index) {
  return RegionKind::Graph;
}

//===----------------------------------------------------------------------===//
// IsolatedGraphRegionOp
//===----------------------------------------------------------------------===//

RegionKind IsolatedGraphRegionOp::getRegionKind(unsigned index) {
  return RegionKind::Graph;
}

//===----------------------------------------------------------------------===//
// AffineScopeOp
//===----------------------------------------------------------------------===//

ParseResult AffineScopeOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the body region, and reuse the operand info as the argument info.
  Region *body = result.addRegion();
  return parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{});
}

void AffineScopeOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// TestRemoveOpWithInnerOps
//===----------------------------------------------------------------------===//

namespace {
struct TestRemoveOpWithInnerOps
    : public OpRewritePattern<TestOpWithRegionPattern> {
  using OpRewritePattern<TestOpWithRegionPattern>::OpRewritePattern;

  void initialize() { setDebugName("TestRemoveOpWithInnerOps"); }

  LogicalResult matchAndRewrite(TestOpWithRegionPattern op,
                                PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// TestOpWithRegionPattern
//===----------------------------------------------------------------------===//

void TestOpWithRegionPattern::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<TestRemoveOpWithInnerOps>(context);
}

//===----------------------------------------------------------------------===//
// TestOpWithRegionFold
//===----------------------------------------------------------------------===//

OpFoldResult TestOpWithRegionFold::fold(FoldAdaptor adaptor) {
  return getOperand();
}

//===----------------------------------------------------------------------===//
// TestOpConstant
//===----------------------------------------------------------------------===//

OpFoldResult TestOpConstant::fold(FoldAdaptor adaptor) { return getValue(); }

//===----------------------------------------------------------------------===//
// TestOpWithVariadicResultsAndFolder
//===----------------------------------------------------------------------===//

LogicalResult TestOpWithVariadicResultsAndFolder::fold(
    FoldAdaptor adaptor, SmallVectorImpl<OpFoldResult> &results) {
  for (Value input : this->getOperands()) {
    results.push_back(input);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TestOpInPlaceFold
//===----------------------------------------------------------------------===//

OpFoldResult TestOpInPlaceFold::fold(FoldAdaptor adaptor) {
  // Exercise the fact that an operation created with createOrFold should be
  // allowed to access its parent block.
  assert(getOperation()->getBlock() &&
         "expected that operation is not unlinked");

  if (adaptor.getOp() && !getProperties().attr) {
    // The folder adds "attr" if not present.
    getProperties().attr = dyn_cast_or_null<IntegerAttr>(adaptor.getOp());
    return getResult();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// OpWithInferTypeInterfaceOp
//===----------------------------------------------------------------------===//

LogicalResult OpWithInferTypeInterfaceOp::inferReturnTypes(
    MLIRContext *, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  if (operands[0].getType() != operands[1].getType()) {
    return emitOptionalError(location, "operand type mismatch ",
                             operands[0].getType(), " vs ",
                             operands[1].getType());
  }
  inferredReturnTypes.assign({operands[0].getType()});
  return success();
}

//===----------------------------------------------------------------------===//
// OpWithShapedTypeInferTypeInterfaceOp
//===----------------------------------------------------------------------===//

LogicalResult OpWithShapedTypeInferTypeInterfaceOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  // Create return type consisting of the last element of the first operand.
  auto operandType = operands.front().getType();
  auto sval = dyn_cast<ShapedType>(operandType);
  if (!sval)
    return emitOptionalError(location, "only shaped type operands allowed");
  int64_t dim = sval.hasRank() ? sval.getShape().front() : ShapedType::kDynamic;
  auto type = IntegerType::get(context, 17);

  Attribute encoding;
  if (auto rankedTy = dyn_cast<RankedTensorType>(sval))
    encoding = rankedTy.getEncoding();
  inferredReturnShapes.push_back(ShapedTypeComponents({dim}, type, encoding));
  return success();
}

LogicalResult OpWithShapedTypeInferTypeInterfaceOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    llvm::SmallVectorImpl<Value> &shapes) {
  shapes = SmallVector<Value, 1>{
      builder.createOrFold<tensor::DimOp>(getLoc(), operands.front(), 0)};
  return success();
}

//===----------------------------------------------------------------------===//
// OpWithResultShapeInterfaceOp
//===----------------------------------------------------------------------===//

LogicalResult OpWithResultShapeInterfaceOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    llvm::SmallVectorImpl<Value> &shapes) {
  Location loc = getLoc();
  shapes.reserve(operands.size());
  for (Value operand : llvm::reverse(operands)) {
    auto rank = cast<RankedTensorType>(operand.getType()).getRank();
    auto currShape = llvm::to_vector<4>(
        llvm::map_range(llvm::seq<int64_t>(0, rank), [&](int64_t dim) -> Value {
          return builder.createOrFold<tensor::DimOp>(loc, operand, dim);
        }));
    shapes.push_back(builder.create<tensor::FromElementsOp>(
        getLoc(), RankedTensorType::get({rank}, builder.getIndexType()),
        currShape));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// OpWithResultShapePerDimInterfaceOp
//===----------------------------------------------------------------------===//

LogicalResult OpWithResultShapePerDimInterfaceOp::reifyResultShapes(
    OpBuilder &builder, ReifiedRankedShapedTypeDims &shapes) {
  Location loc = getLoc();
  shapes.reserve(getNumOperands());
  for (Value operand : llvm::reverse(getOperands())) {
    auto tensorType = cast<RankedTensorType>(operand.getType());
    auto currShape = llvm::to_vector<4>(llvm::map_range(
        llvm::seq<int64_t>(0, tensorType.getRank()),
        [&](int64_t dim) -> OpFoldResult {
          return tensorType.isDynamicDim(dim)
                     ? static_cast<OpFoldResult>(
                           builder.createOrFold<tensor::DimOp>(loc, operand,
                                                               dim))
                     : static_cast<OpFoldResult>(
                           builder.getIndexAttr(tensorType.getDimSize(dim)));
        }));
    shapes.emplace_back(std::move(currShape));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SideEffectOp
//===----------------------------------------------------------------------===//

namespace {
/// A test resource for side effects.
struct TestResource : public SideEffects::Resource::Base<TestResource> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestResource)

  StringRef getName() final { return "<Test>"; }
};
} // namespace

void SideEffectOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // Check for an effects attribute on the op instance.
  ArrayAttr effectsAttr = (*this)->getAttrOfType<ArrayAttr>("effects");
  if (!effectsAttr)
    return;

  for (Attribute element : effectsAttr) {
    DictionaryAttr effectElement = cast<DictionaryAttr>(element);

    // Get the specific memory effect.
    MemoryEffects::Effect *effect =
        StringSwitch<MemoryEffects::Effect *>(
            cast<StringAttr>(effectElement.get("effect")).getValue())
            .Case("allocate", MemoryEffects::Allocate::get())
            .Case("free", MemoryEffects::Free::get())
            .Case("read", MemoryEffects::Read::get())
            .Case("write", MemoryEffects::Write::get());

    // Check for a non-default resource to use.
    SideEffects::Resource *resource = SideEffects::DefaultResource::get();
    if (effectElement.get("test_resource"))
      resource = TestResource::get();

    // Check for a result to affect.
    if (effectElement.get("on_result"))
      effects.emplace_back(effect, getOperation()->getOpResults()[0], resource);
    else if (Attribute ref = effectElement.get("on_reference"))
      effects.emplace_back(effect, cast<SymbolRefAttr>(ref), resource);
    else
      effects.emplace_back(effect, resource);
  }
}

void SideEffectOp::getEffects(
    SmallVectorImpl<TestEffects::EffectInstance> &effects) {
  testSideEffectOpGetEffect(getOperation(), effects);
}

void SideEffectWithRegionOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // Check for an effects attribute on the op instance.
  ArrayAttr effectsAttr = (*this)->getAttrOfType<ArrayAttr>("effects");
  if (!effectsAttr)
    return;

  for (Attribute element : effectsAttr) {
    DictionaryAttr effectElement = cast<DictionaryAttr>(element);

    // Get the specific memory effect.
    MemoryEffects::Effect *effect =
        StringSwitch<MemoryEffects::Effect *>(
            cast<StringAttr>(effectElement.get("effect")).getValue())
            .Case("allocate", MemoryEffects::Allocate::get())
            .Case("free", MemoryEffects::Free::get())
            .Case("read", MemoryEffects::Read::get())
            .Case("write", MemoryEffects::Write::get());

    // Check for a non-default resource to use.
    SideEffects::Resource *resource = SideEffects::DefaultResource::get();
    if (effectElement.get("test_resource"))
      resource = TestResource::get();

    // Check for a result to affect.
    if (effectElement.get("on_result"))
      effects.emplace_back(effect, getOperation()->getOpResults()[0], resource);
    else if (effectElement.get("on_operand"))
      effects.emplace_back(effect, &getOperation()->getOpOperands()[0],
                           resource);
    else if (effectElement.get("on_argument"))
      effects.emplace_back(effect, getOperation()->getRegion(0).getArgument(0),
                           resource);
    else if (Attribute ref = effectElement.get("on_reference"))
      effects.emplace_back(effect, cast<SymbolRefAttr>(ref), resource);
    else
      effects.emplace_back(effect, resource);
  }
}

void SideEffectWithRegionOp::getEffects(
    SmallVectorImpl<TestEffects::EffectInstance> &effects) {
  testSideEffectOpGetEffect(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// StringAttrPrettyNameOp
//===----------------------------------------------------------------------===//

// This op has fancy handling of its SSA result name.
ParseResult StringAttrPrettyNameOp::parse(OpAsmParser &parser,
                                          OperationState &result) {
  // Add the result types.
  for (size_t i = 0, e = parser.getNumResults(); i != e; ++i)
    result.addTypes(parser.getBuilder().getIntegerType(32));

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // If the attribute dictionary contains no 'names' attribute, infer it from
  // the SSA name (if specified).
  bool hadNames = llvm::any_of(result.attributes, [](NamedAttribute attr) {
    return attr.getName() == "names";
  });

  // If there was no name specified, check to see if there was a useful name
  // specified in the asm file.
  if (hadNames || parser.getNumResults() == 0)
    return success();

  SmallVector<StringRef, 4> names;
  auto *context = result.getContext();

  for (size_t i = 0, e = parser.getNumResults(); i != e; ++i) {
    auto resultName = parser.getResultName(i);
    StringRef nameStr;
    if (!resultName.first.empty() && !isdigit(resultName.first[0]))
      nameStr = resultName.first;

    names.push_back(nameStr);
  }

  auto namesAttr = parser.getBuilder().getStrArrayAttr(names);
  result.attributes.push_back({StringAttr::get(context, "names"), namesAttr});
  return success();
}

void StringAttrPrettyNameOp::print(OpAsmPrinter &p) {
  // Note that we only need to print the "name" attribute if the asmprinter
  // result name disagrees with it.  This can happen in strange cases, e.g.
  // when there are conflicts.
  bool namesDisagree = getNames().size() != getNumResults();

  SmallString<32> resultNameStr;
  for (size_t i = 0, e = getNumResults(); i != e && !namesDisagree; ++i) {
    resultNameStr.clear();
    llvm::raw_svector_ostream tmpStream(resultNameStr);
    p.printOperand(getResult(i), tmpStream);

    auto expectedName = dyn_cast<StringAttr>(getNames()[i]);
    if (!expectedName ||
        tmpStream.str().drop_front() != expectedName.getValue()) {
      namesDisagree = true;
    }
  }

  if (namesDisagree)
    p.printOptionalAttrDictWithKeyword((*this)->getAttrs());
  else
    p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), {"names"});
}

// We set the SSA name in the asm syntax to the contents of the name
// attribute.
void StringAttrPrettyNameOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {

  auto value = getNames();
  for (size_t i = 0, e = value.size(); i != e; ++i)
    if (auto str = dyn_cast<StringAttr>(value[i]))
      if (!str.getValue().empty())
        setNameFn(getResult(i), str.getValue());
}

//===----------------------------------------------------------------------===//
// CustomResultsNameOp
//===----------------------------------------------------------------------===//

void CustomResultsNameOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  ArrayAttr value = getNames();
  for (size_t i = 0, e = value.size(); i != e; ++i)
    if (auto str = dyn_cast<StringAttr>(value[i]))
      if (!str.empty())
        setNameFn(getResult(i), str.getValue());
}

//===----------------------------------------------------------------------===//
// ResultNameFromTypeOp
//===----------------------------------------------------------------------===//

void ResultNameFromTypeOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  auto result = getResult();
  auto setResultNameFn = [&](::llvm::StringRef name) {
    setNameFn(result, name);
  };
  auto opAsmTypeInterface =
      ::mlir::cast<::mlir::OpAsmTypeInterface>(result.getType());
  opAsmTypeInterface.getAsmName(setResultNameFn);
}

//===----------------------------------------------------------------------===//
// BlockArgumentNameFromTypeOp
//===----------------------------------------------------------------------===//

void BlockArgumentNameFromTypeOp::getAsmBlockArgumentNames(
    ::mlir::Region &region, ::mlir::OpAsmSetValueNameFn setNameFn) {
  for (auto &block : region) {
    for (auto arg : block.getArguments()) {
      if (auto opAsmTypeInterface =
              ::mlir::dyn_cast<::mlir::OpAsmTypeInterface>(arg.getType())) {
        auto setArgNameFn = [&](StringRef name) { setNameFn(arg, name); };
        opAsmTypeInterface.getAsmName(setArgNameFn);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// ResultTypeWithTraitOp
//===----------------------------------------------------------------------===//

LogicalResult ResultTypeWithTraitOp::verify() {
  if ((*this)->getResultTypes()[0].hasTrait<TypeTrait::TestTypeTrait>())
    return success();
  return emitError("result type should have trait 'TestTypeTrait'");
}

//===----------------------------------------------------------------------===//
// AttrWithTraitOp
//===----------------------------------------------------------------------===//

LogicalResult AttrWithTraitOp::verify() {
  if (getAttr().hasTrait<AttributeTrait::TestAttrTrait>())
    return success();
  return emitError("'attr' attribute should have trait 'TestAttrTrait'");
}

//===----------------------------------------------------------------------===//
// RegionIfOp
//===----------------------------------------------------------------------===//

void RegionIfOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printOperands(getOperands());
  p << ": " << getOperandTypes();
  p.printArrowTypeList(getResultTypes());
  p << " then ";
  p.printRegion(getThenRegion(),
                /*printEntryBlockArgs=*/true,
                /*printBlockTerminators=*/true);
  p << " else ";
  p.printRegion(getElseRegion(),
                /*printEntryBlockArgs=*/true,
                /*printBlockTerminators=*/true);
  p << " join ";
  p.printRegion(getJoinRegion(),
                /*printEntryBlockArgs=*/true,
                /*printBlockTerminators=*/true);
}

ParseResult RegionIfOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 2> operandInfos;
  SmallVector<Type, 2> operandTypes;

  result.regions.reserve(3);
  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();
  Region *joinRegion = result.addRegion();

  // Parse operand, type and arrow type lists.
  if (parser.parseOperandList(operandInfos) ||
      parser.parseColonTypeList(operandTypes) ||
      parser.parseArrowTypeList(result.types))
    return failure();

  // Parse all attached regions.
  if (parser.parseKeyword("then") || parser.parseRegion(*thenRegion, {}, {}) ||
      parser.parseKeyword("else") || parser.parseRegion(*elseRegion, {}, {}) ||
      parser.parseKeyword("join") || parser.parseRegion(*joinRegion, {}, {}))
    return failure();

  return parser.resolveOperands(operandInfos, operandTypes,
                                parser.getCurrentLocation(), result.operands);
}

OperandRange RegionIfOp::getEntrySuccessorOperands(RegionBranchPoint point) {
  assert(llvm::is_contained({&getThenRegion(), &getElseRegion()}, point) &&
         "invalid region index");
  return getOperands();
}

void RegionIfOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  // We always branch to the join region.
  if (!point.isParent()) {
    if (point != getJoinRegion())
      regions.push_back(RegionSuccessor(&getJoinRegion(), getJoinArgs()));
    else
      regions.push_back(RegionSuccessor(getResults()));
    return;
  }

  // The then and else regions are the entry regions of this op.
  regions.push_back(RegionSuccessor(&getThenRegion(), getThenArgs()));
  regions.push_back(RegionSuccessor(&getElseRegion(), getElseArgs()));
}

void RegionIfOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<InvocationBounds> &invocationBounds) {
  // Each region is invoked at most once.
  invocationBounds.assign(/*NumElts=*/3, /*Elt=*/{0, 1});
}

//===----------------------------------------------------------------------===//
// AnyCondOp
//===----------------------------------------------------------------------===//

void AnyCondOp::getSuccessorRegions(RegionBranchPoint point,
                                    SmallVectorImpl<RegionSuccessor> &regions) {
  // The parent op branches into the only region, and the region branches back
  // to the parent op.
  if (point.isParent())
    regions.emplace_back(&getRegion());
  else
    regions.emplace_back(getResults());
}

void AnyCondOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<InvocationBounds> &invocationBounds) {
  invocationBounds.emplace_back(1, 1);
}

//===----------------------------------------------------------------------===//
// SingleBlockImplicitTerminatorOp
//===----------------------------------------------------------------------===//

/// Testing the correctness of some traits.
static_assert(
    llvm::is_detected<OpTrait::has_implicit_terminator_t,
                      SingleBlockImplicitTerminatorOp>::value,
    "has_implicit_terminator_t does not match SingleBlockImplicitTerminatorOp");
static_assert(OpTrait::hasSingleBlockImplicitTerminator<
                  SingleBlockImplicitTerminatorOp>::value,
              "hasSingleBlockImplicitTerminator does not match "
              "SingleBlockImplicitTerminatorOp");

//===----------------------------------------------------------------------===//
// SingleNoTerminatorCustomAsmOp
//===----------------------------------------------------------------------===//

ParseResult SingleNoTerminatorCustomAsmOp::parse(OpAsmParser &parser,
                                                 OperationState &state) {
  Region *body = state.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();
  return success();
}

void SingleNoTerminatorCustomAsmOp::print(OpAsmPrinter &printer) {
  printer.printRegion(
      getRegion(), /*printEntryBlockArgs=*/false,
      // This op has a single block without terminators. But explicitly mark
      // as not printing block terminators for testing.
      /*printBlockTerminators=*/false);
}

//===----------------------------------------------------------------------===//
// TestVerifiersOp
//===----------------------------------------------------------------------===//

LogicalResult TestVerifiersOp::verify() {
  if (!getRegion().hasOneBlock())
    return emitOpError("`hasOneBlock` trait hasn't been verified");

  Operation *definingOp = getInput().getDefiningOp();
  if (definingOp && failed(mlir::verify(definingOp)))
    return emitOpError("operand hasn't been verified");

  // Avoid using `emitRemark(msg)` since that will trigger an infinite verifier
  // loop.
  mlir::emitRemark(getLoc(), "success run of verifier");

  return success();
}

LogicalResult TestVerifiersOp::verifyRegions() {
  if (!getRegion().hasOneBlock())
    return emitOpError("`hasOneBlock` trait hasn't been verified");

  for (Block &block : getRegion())
    for (Operation &op : block)
      if (failed(mlir::verify(&op)))
        return emitOpError("nested op hasn't been verified");

  // Avoid using `emitRemark(msg)` since that will trigger an infinite verifier
  // loop.
  mlir::emitRemark(getLoc(), "success run of region verifier");

  return success();
}

//===----------------------------------------------------------------------===//
// Test InferIntRangeInterface
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TestWithBoundsOp
//===----------------------------------------------------------------------===//

void TestWithBoundsOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                         SetIntRangeFn setResultRanges) {
  setResultRanges(getResult(), {getUmin(), getUmax(), getSmin(), getSmax()});
}

//===----------------------------------------------------------------------===//
// TestWithBoundsRegionOp
//===----------------------------------------------------------------------===//

ParseResult TestWithBoundsRegionOp::parse(OpAsmParser &parser,
                                          OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse the input argument
  OpAsmParser::Argument argInfo;
  if (failed(parser.parseArgument(argInfo, true)))
    return failure();

  // Parse the body region, and reuse the operand info as the argument info.
  Region *body = result.addRegion();
  return parser.parseRegion(*body, argInfo, /*enableNameShadowing=*/false);
}

void TestWithBoundsRegionOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs());
  p << ' ';
  p.printRegionArgument(getRegion().getArgument(0), /*argAttrs=*/{},
                        /*omitType=*/false);
  p << ' ';
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);
}

void TestWithBoundsRegionOp::inferResultRanges(
    ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRanges) {
  Value arg = getRegion().getArgument(0);
  setResultRanges(arg, {getUmin(), getUmax(), getSmin(), getSmax()});
}

//===----------------------------------------------------------------------===//
// TestIncrementOp
//===----------------------------------------------------------------------===//

void TestIncrementOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                        SetIntRangeFn setResultRanges) {
  const ConstantIntRanges &range = argRanges[0];
  APInt one(range.umin().getBitWidth(), 1);
  setResultRanges(getResult(),
                  {range.umin().uadd_sat(one), range.umax().uadd_sat(one),
                   range.smin().sadd_sat(one), range.smax().sadd_sat(one)});
}

//===----------------------------------------------------------------------===//
// TestReflectBoundsOp
//===----------------------------------------------------------------------===//

void TestReflectBoundsOp::inferResultRanges(
    ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRanges) {
  const ConstantIntRanges &range = argRanges[0];
  MLIRContext *ctx = getContext();
  Builder b(ctx);
  Type sIntTy, uIntTy;
  // For plain `IntegerType`s, we can derive the appropriate signed and unsigned
  // Types for the Attributes.
  Type type = getElementTypeOrSelf(getType());
  if (auto intTy = llvm::dyn_cast<IntegerType>(type)) {
    unsigned bitwidth = intTy.getWidth();
    sIntTy = b.getIntegerType(bitwidth, /*isSigned=*/true);
    uIntTy = b.getIntegerType(bitwidth, /*isSigned=*/false);
  } else {
    sIntTy = uIntTy = type;
  }

  setUminAttr(b.getIntegerAttr(uIntTy, range.umin()));
  setUmaxAttr(b.getIntegerAttr(uIntTy, range.umax()));
  setSminAttr(b.getIntegerAttr(sIntTy, range.smin()));
  setSmaxAttr(b.getIntegerAttr(sIntTy, range.smax()));
  setResultRanges(getResult(), range);
}

//===----------------------------------------------------------------------===//
// ConversionFuncOp
//===----------------------------------------------------------------------===//

ParseResult ConversionFuncOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void ConversionFuncOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// TestValueWithBoundsOp
//===----------------------------------------------------------------------===//

void TestValueWithBoundsOp::populateBoundsForIndexValue(
    Value v, ValueBoundsConstraintSet &cstr) {
  cstr.bound(v) >= getMin().getSExtValue();
  cstr.bound(v) <= getMax().getSExtValue();
}

//===----------------------------------------------------------------------===//
// ReifyBoundOp
//===----------------------------------------------------------------------===//

mlir::presburger::BoundType ReifyBoundOp::getBoundType() {
  if (getType() == "EQ")
    return mlir::presburger::BoundType::EQ;
  if (getType() == "LB")
    return mlir::presburger::BoundType::LB;
  if (getType() == "UB")
    return mlir::presburger::BoundType::UB;
  llvm_unreachable("invalid bound type");
}

LogicalResult ReifyBoundOp::verify() {
  if (isa<ShapedType>(getVar().getType())) {
    if (!getDim().has_value())
      return emitOpError("expected 'dim' attribute for shaped type variable");
  } else if (getVar().getType().isIndex()) {
    if (getDim().has_value())
      return emitOpError("unexpected 'dim' attribute for index variable");
  } else {
    return emitOpError("expected index-typed variable or shape type variable");
  }
  if (getConstant() && getScalable())
    return emitOpError("'scalable' and 'constant' are mutually exlusive");
  if (getScalable() != getVscaleMin().has_value())
    return emitOpError("expected 'vscale_min' if and only if 'scalable'");
  if (getScalable() != getVscaleMax().has_value())
    return emitOpError("expected 'vscale_min' if and only if 'scalable'");
  return success();
}

ValueBoundsConstraintSet::Variable ReifyBoundOp::getVariable() {
  if (getDim().has_value())
    return ValueBoundsConstraintSet::Variable(getVar(), *getDim());
  return ValueBoundsConstraintSet::Variable(getVar());
}

//===----------------------------------------------------------------------===//
// CompareOp
//===----------------------------------------------------------------------===//

ValueBoundsConstraintSet::ComparisonOperator
CompareOp::getComparisonOperator() {
  if (getCmp() == "EQ")
    return ValueBoundsConstraintSet::ComparisonOperator::EQ;
  if (getCmp() == "LT")
    return ValueBoundsConstraintSet::ComparisonOperator::LT;
  if (getCmp() == "LE")
    return ValueBoundsConstraintSet::ComparisonOperator::LE;
  if (getCmp() == "GT")
    return ValueBoundsConstraintSet::ComparisonOperator::GT;
  if (getCmp() == "GE")
    return ValueBoundsConstraintSet::ComparisonOperator::GE;
  llvm_unreachable("invalid comparison operator");
}

mlir::ValueBoundsConstraintSet::Variable CompareOp::getLhs() {
  if (!getLhsMap())
    return ValueBoundsConstraintSet::Variable(getVarOperands()[0]);
  SmallVector<Value> mapOperands(
      getVarOperands().slice(0, getLhsMap()->getNumInputs()));
  return ValueBoundsConstraintSet::Variable(*getLhsMap(), mapOperands);
}

mlir::ValueBoundsConstraintSet::Variable CompareOp::getRhs() {
  int64_t rhsOperandsBegin = getLhsMap() ? getLhsMap()->getNumInputs() : 1;
  if (!getRhsMap())
    return ValueBoundsConstraintSet::Variable(
        getVarOperands()[rhsOperandsBegin]);
  SmallVector<Value> mapOperands(
      getVarOperands().slice(rhsOperandsBegin, getRhsMap()->getNumInputs()));
  return ValueBoundsConstraintSet::Variable(*getRhsMap(), mapOperands);
}

LogicalResult CompareOp::verify() {
  if (getCompose() && (getLhsMap() || getRhsMap()))
    return emitOpError(
        "'compose' not supported when 'lhs_map' or 'rhs_map' is present");
  int64_t expectedNumOperands = getLhsMap() ? getLhsMap()->getNumInputs() : 1;
  expectedNumOperands += getRhsMap() ? getRhsMap()->getNumInputs() : 1;
  if (getVarOperands().size() != size_t(expectedNumOperands))
    return emitOpError("expected ")
           << expectedNumOperands << " operands, but got "
           << getVarOperands().size();
  return success();
}

//===----------------------------------------------------------------------===//
// TestOpInPlaceSelfFold
//===----------------------------------------------------------------------===//

OpFoldResult TestOpInPlaceSelfFold::fold(FoldAdaptor adaptor) {
  if (!getFolded()) {
    // The folder adds the "folded" if not present.
    setFolded(true);
    return getResult();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// TestOpFoldWithFoldAdaptor
//===----------------------------------------------------------------------===//

OpFoldResult TestOpFoldWithFoldAdaptor::fold(FoldAdaptor adaptor) {
  int64_t sum = 0;
  if (auto value = dyn_cast_or_null<IntegerAttr>(adaptor.getOp()))
    sum += value.getValue().getSExtValue();

  for (Attribute attr : adaptor.getVariadic())
    if (auto value = dyn_cast_or_null<IntegerAttr>(attr))
      sum += 2 * value.getValue().getSExtValue();

  for (ArrayRef<Attribute> attrs : adaptor.getVarOfVar())
    for (Attribute attr : attrs)
      if (auto value = dyn_cast_or_null<IntegerAttr>(attr))
        sum += 3 * value.getValue().getSExtValue();

  sum += 4 * std::distance(adaptor.getBody().begin(), adaptor.getBody().end());

  return IntegerAttr::get(getType(), sum);
}

//===----------------------------------------------------------------------===//
// OpWithInferTypeAdaptorInterfaceOp
//===----------------------------------------------------------------------===//

LogicalResult OpWithInferTypeAdaptorInterfaceOp::inferReturnTypes(
    MLIRContext *, std::optional<Location> location,
    OpWithInferTypeAdaptorInterfaceOp::Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  if (adaptor.getX().getType() != adaptor.getY().getType()) {
    return emitOptionalError(location, "operand type mismatch ",
                             adaptor.getX().getType(), " vs ",
                             adaptor.getY().getType());
  }
  inferredReturnTypes.assign({adaptor.getX().getType()});
  return success();
}

//===----------------------------------------------------------------------===//
// OpWithRefineTypeInterfaceOp
//===----------------------------------------------------------------------===//

// TODO: We should be able to only define either inferReturnType or
// refineReturnType, currently only refineReturnType can be omitted.
LogicalResult OpWithRefineTypeInterfaceOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &returnTypes) {
  returnTypes.clear();
  return OpWithRefineTypeInterfaceOp::refineReturnTypes(
      context, location, operands, attributes, properties, regions,
      returnTypes);
}

LogicalResult OpWithRefineTypeInterfaceOp::refineReturnTypes(
    MLIRContext *, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &returnTypes) {
  if (operands[0].getType() != operands[1].getType()) {
    return emitOptionalError(location, "operand type mismatch ",
                             operands[0].getType(), " vs ",
                             operands[1].getType());
  }
  // TODO: Add helper to make this more concise to write.
  if (returnTypes.empty())
    returnTypes.resize(1, nullptr);
  if (returnTypes[0] && returnTypes[0] != operands[0].getType())
    return emitOptionalError(location,
                             "required first operand and result to match");
  returnTypes[0] = operands[0].getType();
  return success();
}

//===----------------------------------------------------------------------===//
// OpWithShapedTypeInferTypeAdaptorInterfaceOp
//===----------------------------------------------------------------------===//

LogicalResult
OpWithShapedTypeInferTypeAdaptorInterfaceOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    OpWithShapedTypeInferTypeAdaptorInterfaceOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  // Create return type consisting of the last element of the first operand.
  auto operandType = adaptor.getOperand1().getType();
  auto sval = dyn_cast<ShapedType>(operandType);
  if (!sval)
    return emitOptionalError(location, "only shaped type operands allowed");
  int64_t dim = sval.hasRank() ? sval.getShape().front() : ShapedType::kDynamic;
  auto type = IntegerType::get(context, 17);

  Attribute encoding;
  if (auto rankedTy = dyn_cast<RankedTensorType>(sval))
    encoding = rankedTy.getEncoding();
  inferredReturnShapes.push_back(ShapedTypeComponents({dim}, type, encoding));
  return success();
}

LogicalResult
OpWithShapedTypeInferTypeAdaptorInterfaceOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    llvm::SmallVectorImpl<Value> &shapes) {
  shapes = SmallVector<Value, 1>{
      builder.createOrFold<tensor::DimOp>(getLoc(), operands.front(), 0)};
  return success();
}

//===----------------------------------------------------------------------===//
// TestOpWithPropertiesAndInferredType
//===----------------------------------------------------------------------===//

LogicalResult TestOpWithPropertiesAndInferredType::inferReturnTypes(
    MLIRContext *context, std::optional<Location>, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {

  Adaptor adaptor(operands, attributes, properties, regions);
  inferredReturnTypes.push_back(IntegerType::get(
      context, adaptor.getLhs() + adaptor.getProperties().rhs));
  return success();
}

//===----------------------------------------------------------------------===//
// LoopBlockOp
//===----------------------------------------------------------------------===//

void LoopBlockOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  regions.emplace_back(&getBody(), getBody().getArguments());
  if (point.isParent())
    return;

  regions.emplace_back((*this)->getResults());
}

OperandRange LoopBlockOp::getEntrySuccessorOperands(RegionBranchPoint point) {
  assert(point == getBody());
  return MutableOperandRange(getInitMutable());
}

//===----------------------------------------------------------------------===//
// LoopBlockTerminatorOp
//===----------------------------------------------------------------------===//

MutableOperandRange
LoopBlockTerminatorOp::getMutableSuccessorOperands(RegionBranchPoint point) {
  if (point.isParent())
    return getExitArgMutable();
  return getNextIterArgMutable();
}

//===----------------------------------------------------------------------===//
// SwitchWithNoBreakOp
//===----------------------------------------------------------------------===//

void TestNoTerminatorOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {}

//===----------------------------------------------------------------------===//
// Test InferIntRangeInterface
//===----------------------------------------------------------------------===//

OpFoldResult ManualCppOpWithFold::fold(ArrayRef<Attribute> attributes) {
  // Just a simple fold for testing purposes that reads an operands constant
  // value and returns it.
  if (!attributes.empty())
    return attributes.front();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Tensor/Buffer Ops
//===----------------------------------------------------------------------===//

void ReadBufferOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // The buffer operand is read.
  effects.emplace_back(MemoryEffects::Read::get(), &getBufferMutable(),
                       SideEffects::DefaultResource::get());
  // The buffer contents are dumped.
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// Test Dataflow
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TestCallAndStoreOp
//===----------------------------------------------------------------------===//

CallInterfaceCallable TestCallAndStoreOp::getCallableForCallee() {
  return getCallee();
}

void TestCallAndStoreOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  setCalleeAttr(cast<SymbolRefAttr>(callee));
}

Operation::operand_range TestCallAndStoreOp::getArgOperands() {
  return getCalleeOperands();
}

MutableOperandRange TestCallAndStoreOp::getArgOperandsMutable() {
  return getCalleeOperandsMutable();
}

//===----------------------------------------------------------------------===//
// TestCallOnDeviceOp
//===----------------------------------------------------------------------===//

CallInterfaceCallable TestCallOnDeviceOp::getCallableForCallee() {
  return getCallee();
}

void TestCallOnDeviceOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  setCalleeAttr(cast<SymbolRefAttr>(callee));
}

Operation::operand_range TestCallOnDeviceOp::getArgOperands() {
  return getForwardedOperands();
}

MutableOperandRange TestCallOnDeviceOp::getArgOperandsMutable() {
  return getForwardedOperandsMutable();
}

//===----------------------------------------------------------------------===//
// TestStoreWithARegion
//===----------------------------------------------------------------------===//

void TestStoreWithARegion::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent())
    regions.emplace_back(&getBody(), getBody().front().getArguments());
  else
    regions.emplace_back();
}

//===----------------------------------------------------------------------===//
// TestStoreWithALoopRegion
//===----------------------------------------------------------------------===//

void TestStoreWithALoopRegion::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  // Both the operation itself and the region may be branching into the body or
  // back into the operation itself. It is possible for the operation not to
  // enter the body.
  regions.emplace_back(
      RegionSuccessor(&getBody(), getBody().front().getArguments()));
  regions.emplace_back();
}

//===----------------------------------------------------------------------===//
// TestVersionedOpA
//===----------------------------------------------------------------------===//

LogicalResult
TestVersionedOpA::readProperties(mlir::DialectBytecodeReader &reader,
                                 mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>();
  if (mlir::failed(reader.readAttribute(prop.dims)))
    return mlir::failure();

  // Check if we have a version. If not, assume we are parsing the current
  // version.
  auto maybeVersion = reader.getDialectVersion<test::TestDialect>();
  if (succeeded(maybeVersion)) {
    // If version is less than 2.0, there is no additional attribute to parse.
    // We can materialize missing properties post parsing before verification.
    const auto *version =
        reinterpret_cast<const TestDialectVersion *>(*maybeVersion);
    if ((version->major_ < 2)) {
      return success();
    }
  }

  if (mlir::failed(reader.readAttribute(prop.modifier)))
    return mlir::failure();
  return mlir::success();
}

void TestVersionedOpA::writeProperties(mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties();
  writer.writeAttribute(prop.dims);

  auto maybeVersion = writer.getDialectVersion<test::TestDialect>();
  if (succeeded(maybeVersion)) {
    // If version is less than 2.0, there is no additional attribute to write.
    const auto *version =
        reinterpret_cast<const TestDialectVersion *>(*maybeVersion);
    if ((version->major_ < 2)) {
      llvm::outs() << "downgrading op properties...\n";
      return;
    }
  }
  writer.writeAttribute(prop.modifier);
}

//===----------------------------------------------------------------------===//
// TestOpWithVersionedProperties
//===----------------------------------------------------------------------===//

llvm::LogicalResult TestOpWithVersionedProperties::readFromMlirBytecode(
    mlir::DialectBytecodeReader &reader, test::VersionedProperties &prop) {
  uint64_t value1, value2 = 0;
  if (failed(reader.readVarInt(value1)))
    return failure();

  // Check if we have a version. If not, assume we are parsing the current
  // version.
  auto maybeVersion = reader.getDialectVersion<test::TestDialect>();
  bool needToParseAnotherInt = true;
  if (succeeded(maybeVersion)) {
    // If version is less than 2.0, there is no additional attribute to parse.
    // We can materialize missing properties post parsing before verification.
    const auto *version =
        reinterpret_cast<const TestDialectVersion *>(*maybeVersion);
    if ((version->major_ < 2))
      needToParseAnotherInt = false;
  }
  if (needToParseAnotherInt && failed(reader.readVarInt(value2)))
    return failure();

  prop.value1 = value1;
  prop.value2 = value2;
  return success();
}

void TestOpWithVersionedProperties::writeToMlirBytecode(
    mlir::DialectBytecodeWriter &writer,
    const test::VersionedProperties &prop) {
  writer.writeVarInt(prop.value1);
  writer.writeVarInt(prop.value2);
}

//===----------------------------------------------------------------------===//
// TestMultiSlotAlloca
//===----------------------------------------------------------------------===//

llvm::SmallVector<MemorySlot> TestMultiSlotAlloca::getPromotableSlots() {
  SmallVector<MemorySlot> slots;
  for (Value result : getResults()) {
    slots.push_back(MemorySlot{
        result, cast<MemRefType>(result.getType()).getElementType()});
  }
  return slots;
}

Value TestMultiSlotAlloca::getDefaultValue(const MemorySlot &slot,
                                           OpBuilder &builder) {
  return builder.create<TestOpConstant>(getLoc(), slot.elemType,
                                        builder.getI32IntegerAttr(42));
}

void TestMultiSlotAlloca::handleBlockArgument(const MemorySlot &slot,
                                              BlockArgument argument,
                                              OpBuilder &builder) {
  // Not relevant for testing.
}

/// Creates a new TestMultiSlotAlloca operation, just without the `slot`.
static std::optional<TestMultiSlotAlloca>
createNewMultiAllocaWithoutSlot(const MemorySlot &slot, OpBuilder &builder,
                                TestMultiSlotAlloca oldOp) {

  if (oldOp.getNumResults() == 1) {
    oldOp.erase();
    return std::nullopt;
  }

  SmallVector<Type> newTypes;
  SmallVector<Value> remainingValues;

  for (Value oldResult : oldOp.getResults()) {
    if (oldResult == slot.ptr)
      continue;
    remainingValues.push_back(oldResult);
    newTypes.push_back(oldResult.getType());
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(oldOp);
  auto replacement =
      builder.create<TestMultiSlotAlloca>(oldOp->getLoc(), newTypes);
  for (auto [oldResult, newResult] :
       llvm::zip_equal(remainingValues, replacement.getResults()))
    oldResult.replaceAllUsesWith(newResult);

  oldOp.erase();
  return replacement;
}

std::optional<PromotableAllocationOpInterface>
TestMultiSlotAlloca::handlePromotionComplete(const MemorySlot &slot,
                                             Value defaultValue,
                                             OpBuilder &builder) {
  if (defaultValue && defaultValue.use_empty())
    defaultValue.getDefiningOp()->erase();
  return createNewMultiAllocaWithoutSlot(slot, builder, *this);
}

SmallVector<DestructurableMemorySlot>
TestMultiSlotAlloca::getDestructurableSlots() {
  SmallVector<DestructurableMemorySlot> slots;
  for (Value result : getResults()) {
    auto memrefType = cast<MemRefType>(result.getType());
    auto destructurable = dyn_cast<DestructurableTypeInterface>(memrefType);
    if (!destructurable)
      continue;

    std::optional<DenseMap<Attribute, Type>> destructuredType =
        destructurable.getSubelementIndexMap();
    if (!destructuredType)
      continue;
    slots.emplace_back(
        DestructurableMemorySlot{{result, memrefType}, *destructuredType});
  }
  return slots;
}

DenseMap<Attribute, MemorySlot> TestMultiSlotAlloca::destructure(
    const DestructurableMemorySlot &slot,
    const SmallPtrSetImpl<Attribute> &usedIndices, OpBuilder &builder,
    SmallVectorImpl<DestructurableAllocationOpInterface> &newAllocators) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(*this);

  DenseMap<Attribute, MemorySlot> slotMap;

  for (Attribute usedIndex : usedIndices) {
    Type elemType = slot.subelementTypes.lookup(usedIndex);
    MemRefType elemPtr = MemRefType::get({}, elemType);
    auto subAlloca = builder.create<TestMultiSlotAlloca>(getLoc(), elemPtr);
    newAllocators.push_back(subAlloca);
    slotMap.try_emplace<MemorySlot>(usedIndex,
                                    {subAlloca.getResult(0), elemType});
  }

  return slotMap;
}

std::optional<DestructurableAllocationOpInterface>
TestMultiSlotAlloca::handleDestructuringComplete(
    const DestructurableMemorySlot &slot, OpBuilder &builder) {
  return createNewMultiAllocaWithoutSlot(slot, builder, *this);
}

::mlir::LogicalResult test::TestDummyTensorOp::bufferize(
    ::mlir::RewriterBase &rewriter,
    const ::mlir::bufferization::BufferizationOptions &options,
    ::mlir::bufferization::BufferizationState &state) {
  auto buffer =
      mlir::bufferization::getBuffer(rewriter, getInput(), options, state);
  if (mlir::failed(buffer))
    return failure();

  const auto outType = getOutput().getType();
  const auto bufferizedOutType = test::TestMemrefType::get(
      getContext(), outType.getShape(), outType.getElementType(), nullptr);
  // replace op with memref analogy
  auto dummyMemrefOp = rewriter.create<test::TestDummyMemrefOp>(
      getLoc(), bufferizedOutType, *buffer);

  mlir::bufferization::replaceOpWithBufferizedValues(rewriter, getOperation(),
                                                     dummyMemrefOp.getResult());

  return mlir::success();
}

::mlir::LogicalResult test::TestCreateTensorOp::bufferize(
    ::mlir::RewriterBase &rewriter,
    const ::mlir::bufferization::BufferizationOptions &options,
    ::mlir::bufferization::BufferizationState &state) {
  // Note: mlir::bufferization::getBufferType() would internally call
  // TestCreateTensorOp::getBufferType()
  const auto bufferizedOutType =
      mlir::bufferization::getBufferType(getOutput(), options, state);
  if (mlir::failed(bufferizedOutType))
    return failure();

  // replace op with memref analogy
  auto createMemrefOp =
      rewriter.create<test::TestCreateMemrefOp>(getLoc(), *bufferizedOutType);

  mlir::bufferization::replaceOpWithBufferizedValues(
      rewriter, getOperation(), createMemrefOp.getResult());

  return mlir::success();
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
test::TestCreateTensorOp::getBufferType(
    mlir::Value value, const mlir::bufferization::BufferizationOptions &,
    const mlir::bufferization::BufferizationState &,
    llvm::SmallVector<::mlir::Value> &) {
  const auto type = dyn_cast<test::TestTensorType>(value.getType());
  if (type == nullptr)
    return failure();

  return cast<mlir::bufferization::BufferLikeType>(test::TestMemrefType::get(
      getContext(), type.getShape(), type.getElementType(), nullptr));
}
