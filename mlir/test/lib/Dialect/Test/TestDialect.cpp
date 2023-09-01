//===- TestDialect.cpp - MLIR Dialect for Testing -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "TestAttributes.h"
#include "TestInterfaces.h"
#include "TestTypes.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/ODSSupport.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Base64.h"

#include <cstdint>
#include <numeric>
#include <optional>

// Include this before the using namespace lines below to
// test that we don't have namespace dependencies.
#include "TestOpsDialect.cpp.inc"

using namespace mlir;
using namespace test;

Attribute MyPropStruct::asAttribute(MLIRContext *ctx) const {
  return StringAttr::get(ctx, content);
}
LogicalResult MyPropStruct::setFromAttr(MyPropStruct &prop, Attribute attr,
                                        InFlightDiagnostic *diag) {
  StringAttr strAttr = dyn_cast<StringAttr>(attr);
  if (!strAttr) {
    if (diag)
      *diag << "Expect StringAttr but got " << attr;
    return failure();
  }
  prop.content = strAttr.getValue();
  return success();
}
llvm::hash_code MyPropStruct::hash() const {
  return hash_value(StringRef(content));
}

static LogicalResult readFromMlirBytecode(DialectBytecodeReader &reader,
                                          MyPropStruct &prop) {
  StringRef str;
  if (failed(reader.readString(str)))
    return failure();
  prop.content = str.str();
  return success();
}

static void writeToMlirBytecode(::mlir::DialectBytecodeWriter &writer,
                                MyPropStruct &prop) {
  writer.writeOwnedString(prop.content);
}

static LogicalResult readFromMlirBytecode(DialectBytecodeReader &reader,
                                          MutableArrayRef<int64_t> prop) {
  uint64_t size;
  if (failed(reader.readVarInt(size)))
    return failure();
  if (size != prop.size())
    return reader.emitError("array size mismach when reading properties: ")
           << size << " vs expected " << prop.size();
  for (auto &elt : prop) {
    uint64_t value;
    if (failed(reader.readVarInt(value)))
      return failure();
    elt = value;
  }
  return success();
}

static void writeToMlirBytecode(::mlir::DialectBytecodeWriter &writer,
                                ArrayRef<int64_t> prop) {
  writer.writeVarInt(prop.size());
  for (auto elt : prop)
    writer.writeVarInt(elt);
}

static LogicalResult setPropertiesFromAttribute(PropertiesWithCustomPrint &prop,
                                                Attribute attr,
                                                InFlightDiagnostic *diagnostic);
static DictionaryAttr
getPropertiesAsAttribute(MLIRContext *ctx,
                         const PropertiesWithCustomPrint &prop);
static llvm::hash_code computeHash(const PropertiesWithCustomPrint &prop);
static void customPrintProperties(OpAsmPrinter &p,
                                  const PropertiesWithCustomPrint &prop);
static ParseResult customParseProperties(OpAsmParser &parser,
                                         PropertiesWithCustomPrint &prop);
static LogicalResult setPropertiesFromAttribute(VersionedProperties &prop,
                                                Attribute attr,
                                                InFlightDiagnostic *diagnostic);
static DictionaryAttr getPropertiesAsAttribute(MLIRContext *ctx,
                                               const VersionedProperties &prop);
static llvm::hash_code computeHash(const VersionedProperties &prop);
static void customPrintProperties(OpAsmPrinter &p,
                                  const VersionedProperties &prop);
static ParseResult customParseProperties(OpAsmParser &parser,
                                         VersionedProperties &prop);

void test::registerTestDialect(DialectRegistry &registry) {
  registry.insert<TestDialect>();
}

//===----------------------------------------------------------------------===//
// Dynamic operations
//===----------------------------------------------------------------------===//

std::unique_ptr<DynamicOpDefinition> getDynamicGenericOp(TestDialect *dialect) {
  return DynamicOpDefinition::get(
      "dynamic_generic", dialect, [](Operation *op) { return success(); },
      [](Operation *op) { return success(); });
}

std::unique_ptr<DynamicOpDefinition>
getDynamicOneOperandTwoResultsOp(TestDialect *dialect) {
  return DynamicOpDefinition::get(
      "dynamic_one_operand_two_results", dialect,
      [](Operation *op) {
        if (op->getNumOperands() != 1) {
          op->emitOpError()
              << "expected 1 operand, but had " << op->getNumOperands();
          return failure();
        }
        if (op->getNumResults() != 2) {
          op->emitOpError()
              << "expected 2 results, but had " << op->getNumResults();
          return failure();
        }
        return success();
      },
      [](Operation *op) { return success(); });
}

std::unique_ptr<DynamicOpDefinition>
getDynamicCustomParserPrinterOp(TestDialect *dialect) {
  auto verifier = [](Operation *op) {
    if (op->getNumOperands() == 0 && op->getNumResults() == 0)
      return success();
    op->emitError() << "operation should have no operands and no results";
    return failure();
  };
  auto regionVerifier = [](Operation *op) { return success(); };

  auto parser = [](OpAsmParser &parser, OperationState &state) {
    return parser.parseKeyword("custom_keyword");
  };

  auto printer = [](Operation *op, OpAsmPrinter &printer, llvm::StringRef) {
    printer << op->getName() << " custom_keyword";
  };

  return DynamicOpDefinition::get("dynamic_custom_parser_printer", dialect,
                                  verifier, regionVerifier, parser, printer);
}

//===----------------------------------------------------------------------===//
// TestDialect
//===----------------------------------------------------------------------===//

static void testSideEffectOpGetEffect(
    Operation *op,
    SmallVectorImpl<SideEffects::EffectInstance<TestEffects::Effect>> &effects);

// This is the implementation of a dialect fallback for `TestEffectOpInterface`.
struct TestOpEffectInterfaceFallback
    : public TestEffectOpInterface::FallbackModel<
          TestOpEffectInterfaceFallback> {
  static bool classof(Operation *op) {
    bool isSupportedOp =
        op->getName().getStringRef() == "test.unregistered_side_effect_op";
    assert(isSupportedOp && "Unexpected dispatch");
    return isSupportedOp;
  }

  void
  getEffects(Operation *op,
             SmallVectorImpl<SideEffects::EffectInstance<TestEffects::Effect>>
                 &effects) const {
    testSideEffectOpGetEffect(op, effects);
  }
};

void TestDialect::initialize() {
  registerAttributes();
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "TestOps.cpp.inc"
      >();
  registerOpsSyntax();
  addOperations<ManualCppOpWithFold>();
  registerDynamicOp(getDynamicGenericOp(this));
  registerDynamicOp(getDynamicOneOperandTwoResultsOp(this));
  registerDynamicOp(getDynamicCustomParserPrinterOp(this));
  registerInterfaces();
  allowUnknownOperations();

  // Instantiate our fallback op interface that we'll use on specific
  // unregistered op.
  fallbackEffectOpInterfaces = new TestOpEffectInterfaceFallback;
}
TestDialect::~TestDialect() {
  delete static_cast<TestOpEffectInterfaceFallback *>(
      fallbackEffectOpInterfaces);
}

Operation *TestDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  return builder.create<TestOpConstant>(loc, type, value);
}

void *TestDialect::getRegisteredInterfaceForOp(TypeID typeID,
                                               OperationName opName) {
  if (opName.getIdentifier() == "test.unregistered_side_effect_op" &&
      typeID == TypeID::get<TestEffectOpInterface>())
    return fallbackEffectOpInterfaces;
  return nullptr;
}

LogicalResult TestDialect::verifyOperationAttribute(Operation *op,
                                                    NamedAttribute namedAttr) {
  if (namedAttr.getName() == "test.invalid_attr")
    return op->emitError() << "invalid to use 'test.invalid_attr'";
  return success();
}

LogicalResult TestDialect::verifyRegionArgAttribute(Operation *op,
                                                    unsigned regionIndex,
                                                    unsigned argIndex,
                                                    NamedAttribute namedAttr) {
  if (namedAttr.getName() == "test.invalid_attr")
    return op->emitError() << "invalid to use 'test.invalid_attr'";
  return success();
}

LogicalResult
TestDialect::verifyRegionResultAttribute(Operation *op, unsigned regionIndex,
                                         unsigned resultIndex,
                                         NamedAttribute namedAttr) {
  if (namedAttr.getName() == "test.invalid_attr")
    return op->emitError() << "invalid to use 'test.invalid_attr'";
  return success();
}

std::optional<Dialect::ParseOpHook>
TestDialect::getParseOperationHook(StringRef opName) const {
  if (opName == "test.dialect_custom_printer") {
    return ParseOpHook{[](OpAsmParser &parser, OperationState &state) {
      return parser.parseKeyword("custom_format");
    }};
  }
  if (opName == "test.dialect_custom_format_fallback") {
    return ParseOpHook{[](OpAsmParser &parser, OperationState &state) {
      return parser.parseKeyword("custom_format_fallback");
    }};
  }
  if (opName == "test.dialect_custom_printer.with.dot") {
    return ParseOpHook{[](OpAsmParser &parser, OperationState &state) {
      return ParseResult::success();
    }};
  }
  return std::nullopt;
}

llvm::unique_function<void(Operation *, OpAsmPrinter &)>
TestDialect::getOperationPrinter(Operation *op) const {
  StringRef opName = op->getName().getStringRef();
  if (opName == "test.dialect_custom_printer") {
    return [](Operation *op, OpAsmPrinter &printer) {
      printer.getStream() << " custom_format";
    };
  }
  if (opName == "test.dialect_custom_format_fallback") {
    return [](Operation *op, OpAsmPrinter &printer) {
      printer.getStream() << " custom_format_fallback";
    };
  }
  return {};
}

//===----------------------------------------------------------------------===//
// TypedAttrOp
//===----------------------------------------------------------------------===//

/// Parse an attribute with a given type.
static ParseResult parseAttrElideType(AsmParser &parser, TypeAttr type,
                                      Attribute &attr) {
  return parser.parseAttribute(attr, type.getValue());
}

/// Print an attribute without its type.
static void printAttrElideType(AsmPrinter &printer, Operation *op,
                               TypeAttr type, Attribute attr) {
  printer.printAttributeWithoutType(attr);
}

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
// TestProducingBranchOp
//===----------------------------------------------------------------------===//

SuccessorOperands TestInternalBranchOp::getSuccessorOperands(unsigned index) {
  assert(index <= 1 && "invalid successor index");
  if (index == 0)
    return SuccessorOperands(0, getSuccessOperandsMutable());
  return SuccessorOperands(1, getErrorOperandsMutable());
}

//===----------------------------------------------------------------------===//
// TestDialectCanonicalizerOp
//===----------------------------------------------------------------------===//

static LogicalResult
dialectCanonicalizationPattern(TestDialectCanonicalizerOp op,
                               PatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<arith::ConstantOp>(
      op, rewriter.getI32IntegerAttr(42));
  return success();
}

void TestDialect::getCanonicalizationPatterns(
    RewritePatternSet &results) const {
  results.add(&dialectCanonicalizationPattern);
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
// TestFoldToCallOp
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
// Test IsolatedRegionOp - parse passthrough region arguments.
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
// Test SSACFGRegionOp
//===----------------------------------------------------------------------===//

RegionKind SSACFGRegionOp::getRegionKind(unsigned index) {
  return RegionKind::SSACFG;
}

//===----------------------------------------------------------------------===//
// Test GraphRegionOp
//===----------------------------------------------------------------------===//

RegionKind GraphRegionOp::getRegionKind(unsigned index) {
  return RegionKind::Graph;
}

//===----------------------------------------------------------------------===//
// Test AffineScopeOp
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
// Test removing op with inner ops.
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

void TestOpWithRegionPattern::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<TestRemoveOpWithInnerOps>(context);
}

OpFoldResult TestOpWithRegionFold::fold(FoldAdaptor adaptor) {
  return getOperand();
}

OpFoldResult TestOpConstant::fold(FoldAdaptor adaptor) { return getValue(); }

LogicalResult TestOpWithVariadicResultsAndFolder::fold(
    FoldAdaptor adaptor, SmallVectorImpl<OpFoldResult> &results) {
  for (Value input : this->getOperands()) {
    results.push_back(input);
  }
  return success();
}

OpFoldResult TestOpInPlaceFold::fold(FoldAdaptor adaptor) {
  if (adaptor.getOp() && !(*this)->getAttr("attr")) {
    // The folder adds "attr" if not present.
    (*this)->setAttr("attr", adaptor.getOp());
    return getResult();
  }
  return {};
}

OpFoldResult TestPassthroughFold::fold(FoldAdaptor adaptor) {
  return getOperand();
}

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
// Test SideEffect interfaces
//===----------------------------------------------------------------------===//

namespace {
/// A test resource for side effects.
struct TestResource : public SideEffects::Resource::Base<TestResource> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestResource)

  StringRef getName() final { return "<Test>"; }
};
} // namespace

static void testSideEffectOpGetEffect(
    Operation *op,
    SmallVectorImpl<SideEffects::EffectInstance<TestEffects::Effect>>
        &effects) {
  auto effectsAttr = op->getAttrOfType<AffineMapAttr>("effect_parameter");
  if (!effectsAttr)
    return;

  effects.emplace_back(TestEffects::Concrete::get(), effectsAttr);
}

void SideEffectOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // Check for an effects attribute on the op instance.
  ArrayAttr effectsAttr = (*this)->getAttrOfType<ArrayAttr>("effects");
  if (!effectsAttr)
    return;

  // If there is one, it is an array of dictionary attributes that hold
  // information on the effects of this operation.
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
      effects.emplace_back(effect, getResult(), resource);
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

void CustomResultsNameOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  ArrayAttr value = getNames();
  for (size_t i = 0, e = value.size(); i != e; ++i)
    if (auto str = dyn_cast<StringAttr>(value[i]))
      if (!str.getValue().empty())
        setNameFn(getResult(i), str.getValue());
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
  return getInitMutable();
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

void TestWithBoundsOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                         SetIntRangeFn setResultRanges) {
  setResultRanges(getResult(), {getUmin(), getUmax(), getSmin(), getSmax()});
}

ParseResult TestWithBoundsRegionOp::parse(OpAsmParser &parser,
                                          OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse the input argument
  OpAsmParser::Argument argInfo;
  argInfo.type = parser.getBuilder().getIndexType();
  if (failed(parser.parseArgument(argInfo)))
    return failure();

  // Parse the body region, and reuse the operand info as the argument info.
  Region *body = result.addRegion();
  return parser.parseRegion(*body, argInfo, /*enableNameShadowing=*/false);
}

void TestWithBoundsRegionOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs());
  p << ' ';
  p.printRegionArgument(getRegion().getArgument(0), /*argAttrs=*/{},
                        /*omitType=*/true);
  p << ' ';
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);
}

void TestWithBoundsRegionOp::inferResultRanges(
    ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRanges) {
  Value arg = getRegion().getArgument(0);
  setResultRanges(arg, {getUmin(), getUmax(), getSmin(), getSmax()});
}

void TestIncrementOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                        SetIntRangeFn setResultRanges) {
  const ConstantIntRanges &range = argRanges[0];
  APInt one(range.umin().getBitWidth(), 1);
  setResultRanges(getResult(),
                  {range.umin().uadd_sat(one), range.umax().uadd_sat(one),
                   range.smin().sadd_sat(one), range.smax().sadd_sat(one)});
}

void TestReflectBoundsOp::inferResultRanges(
    ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRanges) {
  const ConstantIntRanges &range = argRanges[0];
  MLIRContext *ctx = getContext();
  Builder b(ctx);
  setUminAttr(b.getIndexAttr(range.umin().getZExtValue()));
  setUmaxAttr(b.getIndexAttr(range.umax().getZExtValue()));
  setSminAttr(b.getIndexAttr(range.smin().getSExtValue()));
  setSmaxAttr(b.getIndexAttr(range.smax().getSExtValue()));
  setResultRanges(getResult(), range);
}

OpFoldResult ManualCppOpWithFold::fold(ArrayRef<Attribute> attributes) {
  // Just a simple fold for testing purposes that reads an operands constant
  // value and returns it.
  if (!attributes.empty())
    return attributes.front();
  return nullptr;
}

static LogicalResult
setPropertiesFromAttribute(PropertiesWithCustomPrint &prop, Attribute attr,
                           InFlightDiagnostic *diagnostic) {
  DictionaryAttr dict = dyn_cast<DictionaryAttr>(attr);
  if (!dict) {
    if (diagnostic)
      *diagnostic << "expected DictionaryAttr to set TestProperties";
    return failure();
  }
  auto label = dict.getAs<mlir::StringAttr>("label");
  if (!label) {
    if (diagnostic)
      *diagnostic << "expected StringAttr for key `label`";
    return failure();
  }
  auto valueAttr = dict.getAs<IntegerAttr>("value");
  if (!valueAttr) {
    if (diagnostic)
      *diagnostic << "expected IntegerAttr for key `value`";
    return failure();
  }

  prop.label = std::make_shared<std::string>(label.getValue());
  prop.value = valueAttr.getValue().getSExtValue();
  return success();
}
static DictionaryAttr
getPropertiesAsAttribute(MLIRContext *ctx,
                         const PropertiesWithCustomPrint &prop) {
  SmallVector<NamedAttribute> attrs;
  Builder b{ctx};
  attrs.push_back(b.getNamedAttr("label", b.getStringAttr(*prop.label)));
  attrs.push_back(b.getNamedAttr("value", b.getI32IntegerAttr(prop.value)));
  return b.getDictionaryAttr(attrs);
}
static llvm::hash_code computeHash(const PropertiesWithCustomPrint &prop) {
  return llvm::hash_combine(prop.value, StringRef(*prop.label));
}
static void customPrintProperties(OpAsmPrinter &p,
                                  const PropertiesWithCustomPrint &prop) {
  p.printKeywordOrString(*prop.label);
  p << " is " << prop.value;
}
static ParseResult customParseProperties(OpAsmParser &parser,
                                         PropertiesWithCustomPrint &prop) {
  std::string label;
  if (parser.parseKeywordOrString(&label) || parser.parseKeyword("is") ||
      parser.parseInteger(prop.value))
    return failure();
  prop.label = std::make_shared<std::string>(std::move(label));
  return success();
}
static LogicalResult
setPropertiesFromAttribute(VersionedProperties &prop, Attribute attr,
                           InFlightDiagnostic *diagnostic) {
  DictionaryAttr dict = dyn_cast<DictionaryAttr>(attr);
  if (!dict) {
    if (diagnostic)
      *diagnostic << "expected DictionaryAttr to set VersionedProperties";
    return failure();
  }
  auto value1Attr = dict.getAs<IntegerAttr>("value1");
  if (!value1Attr) {
    if (diagnostic)
      *diagnostic << "expected IntegerAttr for key `value1`";
    return failure();
  }
  auto value2Attr = dict.getAs<IntegerAttr>("value2");
  if (!value2Attr) {
    if (diagnostic)
      *diagnostic << "expected IntegerAttr for key `value2`";
    return failure();
  }

  prop.value1 = value1Attr.getValue().getSExtValue();
  prop.value2 = value2Attr.getValue().getSExtValue();
  return success();
}
static DictionaryAttr
getPropertiesAsAttribute(MLIRContext *ctx, const VersionedProperties &prop) {
  SmallVector<NamedAttribute> attrs;
  Builder b{ctx};
  attrs.push_back(b.getNamedAttr("value1", b.getI32IntegerAttr(prop.value1)));
  attrs.push_back(b.getNamedAttr("value2", b.getI32IntegerAttr(prop.value2)));
  return b.getDictionaryAttr(attrs);
}
static llvm::hash_code computeHash(const VersionedProperties &prop) {
  return llvm::hash_combine(prop.value1, prop.value2);
}
static void customPrintProperties(OpAsmPrinter &p,
                                  const VersionedProperties &prop) {
  p << prop.value1 << " | " << prop.value2;
}
static ParseResult customParseProperties(OpAsmParser &parser,
                                         VersionedProperties &prop) {
  if (parser.parseInteger(prop.value1) || parser.parseVerticalBar() ||
      parser.parseInteger(prop.value2))
    return failure();
  return success();
}

static bool parseUsingPropertyInCustom(OpAsmParser &parser, int64_t value[3]) {
  return parser.parseLSquare() || parser.parseInteger(value[0]) ||
         parser.parseComma() || parser.parseInteger(value[1]) ||
         parser.parseComma() || parser.parseInteger(value[2]) ||
         parser.parseRSquare();
}

static void printUsingPropertyInCustom(OpAsmPrinter &printer, Operation *op,
                                       ArrayRef<int64_t> value) {
  printer << '[' << value << ']';
}

static bool parseIntProperty(OpAsmParser &parser, int64_t &value) {
  return failed(parser.parseInteger(value));
}

static void printIntProperty(OpAsmPrinter &printer, Operation *op,
                             int64_t value) {
  printer << value;
}

static bool parseSumProperty(OpAsmParser &parser, int64_t &second,
                             int64_t first) {
  int64_t sum;
  auto loc = parser.getCurrentLocation();
  if (parser.parseInteger(second) || parser.parseEqual() ||
      parser.parseInteger(sum))
    return true;
  if (sum != second + first) {
    parser.emitError(loc, "Expected sum to equal first + second");
    return true;
  }
  return false;
}

static void printSumProperty(OpAsmPrinter &printer, Operation *op,
                             int64_t second, int64_t first) {
  printer << second << " = " << (second + first);
}

//===----------------------------------------------------------------------===//
// Test Dataflow
//===----------------------------------------------------------------------===//

CallInterfaceCallable TestCallAndStoreOp::getCallableForCallee() {
  return getCallee();
}

void TestCallAndStoreOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  setCalleeAttr(callee.get<SymbolRefAttr>());
}

Operation::operand_range TestCallAndStoreOp::getArgOperands() {
  return getCalleeOperands();
}

MutableOperandRange TestCallAndStoreOp::getArgOperandsMutable() {
  return getCalleeOperandsMutable();
}

CallInterfaceCallable TestCallOnDeviceOp::getCallableForCallee() {
  return getCallee();
}

void TestCallOnDeviceOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  setCalleeAttr(callee.get<SymbolRefAttr>());
}

Operation::operand_range TestCallOnDeviceOp::getArgOperands() {
  return getForwardedOperands();
}

MutableOperandRange TestCallOnDeviceOp::getArgOperandsMutable() {
  return getForwardedOperandsMutable();
}

void TestStoreWithARegion::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent())
    regions.emplace_back(&getBody(), getBody().front().getArguments());
  else
    regions.emplace_back();
}

LogicalResult
TestVersionedOpA::readProperties(::mlir::DialectBytecodeReader &reader,
                                 ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>();
  if (::mlir::failed(reader.readAttribute(prop.dims)))
    return ::mlir::failure();

  // Check if we have a version. If not, assume we are parsing the current
  // version.
  auto maybeVersion = reader.getDialectVersion("test");
  if (succeeded(maybeVersion)) {
    // If version is less than 2.0, there is no additional attribute to parse.
    // We can materialize missing properties post parsing before verification.
    const auto *version =
        reinterpret_cast<const TestDialectVersion *>(*maybeVersion);
    if ((version->major_ < 2)) {
      return success();
    }
  }

  if (::mlir::failed(reader.readAttribute(prop.modifier)))
    return ::mlir::failure();
  return ::mlir::success();
}

void TestVersionedOpA::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties();
  writer.writeAttribute(prop.dims);
  writer.writeAttribute(prop.modifier);
}

::mlir::LogicalResult TestOpWithVersionedProperties::readFromMlirBytecode(
    ::mlir::DialectBytecodeReader &reader, test::VersionedProperties &prop) {
  uint64_t value1, value2 = 0;
  if (failed(reader.readVarInt(value1)))
    return failure();

  // Check if we have a version. If not, assume we are parsing the current
  // version.
  auto maybeVersion = reader.getDialectVersion("test");
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
    ::mlir::DialectBytecodeWriter &writer,
    const test::VersionedProperties &prop) {
  writer.writeVarInt(prop.value1);
  writer.writeVarInt(prop.value2);
}

#include "TestOpEnums.cpp.inc"
#include "TestOpInterfaces.cpp.inc"
#include "TestTypeInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "TestOps.cpp.inc"
