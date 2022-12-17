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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Reducer/ReductionPatternInterface.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"

// Include this before the using namespace lines below to
// test that we don't have namespace dependencies.
#include "TestOpsDialect.cpp.inc"

using namespace mlir;
using namespace test;

void test::registerTestDialect(DialectRegistry &registry) {
  registry.insert<TestDialect>();
}

//===----------------------------------------------------------------------===//
// TestDialect Interfaces
//===----------------------------------------------------------------------===//

namespace {

/// Testing the correctness of some traits.
static_assert(
    llvm::is_detected<OpTrait::has_implicit_terminator_t,
                      SingleBlockImplicitTerminatorOp>::value,
    "has_implicit_terminator_t does not match SingleBlockImplicitTerminatorOp");
static_assert(OpTrait::hasSingleBlockImplicitTerminator<
                  SingleBlockImplicitTerminatorOp>::value,
              "hasSingleBlockImplicitTerminator does not match "
              "SingleBlockImplicitTerminatorOp");

struct TestResourceBlobManagerInterface
    : public ResourceBlobManagerDialectInterfaceBase<
          TestDialectResourceBlobHandle> {
  using ResourceBlobManagerDialectInterfaceBase<
      TestDialectResourceBlobHandle>::ResourceBlobManagerDialectInterfaceBase;
};

// Test support for interacting with the AsmPrinter.
struct TestOpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
  TestOpAsmInterface(Dialect *dialect, TestResourceBlobManagerInterface &mgr)
      : OpAsmDialectInterface(dialect), blobManager(mgr) {}

  //===------------------------------------------------------------------===//
  // Aliases
  //===------------------------------------------------------------------===//

  AliasResult getAlias(Attribute attr, raw_ostream &os) const final {
    StringAttr strAttr = attr.dyn_cast<StringAttr>();
    if (!strAttr)
      return AliasResult::NoAlias;

    // Check the contents of the string attribute to see what the test alias
    // should be named.
    Optional<StringRef> aliasName =
        StringSwitch<Optional<StringRef>>(strAttr.getValue())
            .Case("alias_test:dot_in_name", StringRef("test.alias"))
            .Case("alias_test:trailing_digit", StringRef("test_alias0"))
            .Case("alias_test:prefixed_digit", StringRef("0_test_alias"))
            .Case("alias_test:sanitize_conflict_a",
                  StringRef("test_alias_conflict0"))
            .Case("alias_test:sanitize_conflict_b",
                  StringRef("test_alias_conflict0_"))
            .Case("alias_test:tensor_encoding", StringRef("test_encoding"))
            .Default(std::nullopt);
    if (!aliasName)
      return AliasResult::NoAlias;

    os << *aliasName;
    return AliasResult::FinalAlias;
  }

  AliasResult getAlias(Type type, raw_ostream &os) const final {
    if (auto tupleType = type.dyn_cast<TupleType>()) {
      if (tupleType.size() > 0 &&
          llvm::all_of(tupleType.getTypes(), [](Type elemType) {
            return elemType.isa<SimpleAType>();
          })) {
        os << "test_tuple";
        return AliasResult::FinalAlias;
      }
    }
    if (auto intType = type.dyn_cast<TestIntegerType>()) {
      if (intType.getSignedness() ==
              TestIntegerType::SignednessSemantics::Unsigned &&
          intType.getWidth() == 8) {
        os << "test_ui8";
        return AliasResult::FinalAlias;
      }
    }
    if (auto recType = type.dyn_cast<TestRecursiveType>()) {
      if (recType.getName() == "type_to_alias") {
        // We only make alias for a specific recursive type.
        os << "testrec";
        return AliasResult::FinalAlias;
      }
    }
    return AliasResult::NoAlias;
  }

  //===------------------------------------------------------------------===//
  // Resources
  //===------------------------------------------------------------------===//

  std::string
  getResourceKey(const AsmDialectResourceHandle &handle) const override {
    return cast<TestDialectResourceBlobHandle>(handle).getKey().str();
  }

  FailureOr<AsmDialectResourceHandle>
  declareResource(StringRef key) const final {
    return blobManager.insert(key);
  }

  LogicalResult parseResource(AsmParsedResourceEntry &entry) const final {
    FailureOr<AsmResourceBlob> blob = entry.parseAsBlob();
    if (failed(blob))
      return failure();

    // Update the blob for this entry.
    blobManager.update(entry.getKey(), std::move(*blob));
    return success();
  }

  void
  buildResources(Operation *op,
                 const SetVector<AsmDialectResourceHandle> &referencedResources,
                 AsmResourceBuilder &provider) const final {
    blobManager.buildResources(provider, referencedResources.getArrayRef());
  }

private:
  /// The blob manager for the dialect.
  TestResourceBlobManagerInterface &blobManager;
};

struct TestDialectFoldInterface : public DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  /// Registered hook to check if the given region, which is attached to an
  /// operation that is *not* isolated from above, should be used when
  /// materializing constants.
  bool shouldMaterializeInto(Region *region) const final {
    // If this is a one region operation, then insert into it.
    return isa<OneRegionOp>(region->getParentOp());
  }
};

/// This class defines the interface for handling inlining with standard
/// operations.
struct TestInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    // Don't allow inlining calls that are marked `noinline`.
    return !call->hasAttr("noinline");
  }
  bool isLegalToInline(Region *, Region *, bool,
                       BlockAndValueMapping &) const final {
    // Inlining into test dialect regions is legal.
    return true;
  }
  bool isLegalToInline(Operation *, Region *, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }

  bool shouldAnalyzeRecursively(Operation *op) const final {
    // Analyze recursively if this is not a functional region operation, it
    // froms a separate functional scope.
    return !isa<FunctionalRegionOp>(op);
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToRepl) const final {
    // Only handle "test.return" here.
    auto returnOp = dyn_cast<TestReturnOp>(op);
    if (!returnOp)
      return;

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }

  /// Attempt to materialize a conversion for a type mismatch between a call
  /// from this dialect, and a callable region. This method should generate an
  /// operation that takes 'input' as the only operand, and produces a single
  /// result of 'resultType'. If a conversion can not be generated, nullptr
  /// should be returned.
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type resultType,
                                       Location conversionLoc) const final {
    // Only allow conversion for i16/i32 types.
    if (!(resultType.isSignlessInteger(16) ||
          resultType.isSignlessInteger(32)) ||
        !(input.getType().isSignlessInteger(16) ||
          input.getType().isSignlessInteger(32)))
      return nullptr;
    return builder.create<TestCastOp>(conversionLoc, resultType, input);
  }

  void processInlinedCallBlocks(
      Operation *call,
      iterator_range<Region::iterator> inlinedBlocks) const final {
    if (!isa<ConversionCallOp>(call))
      return;

    // Set attributed on all ops in the inlined blocks.
    for (Block &block : inlinedBlocks) {
      block.walk([&](Operation *op) {
        op->setAttr("inlined_conversion", UnitAttr::get(call->getContext()));
      });
    }
  }
};

struct TestReductionPatternInterface : public DialectReductionPatternInterface {
public:
  TestReductionPatternInterface(Dialect *dialect)
      : DialectReductionPatternInterface(dialect) {}

  void populateReductionPatterns(RewritePatternSet &patterns) const final {
    populateTestReductionPatterns(patterns);
  }
};

} // namespace

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
  registerDynamicOp(getDynamicGenericOp(this));
  registerDynamicOp(getDynamicOneOperandTwoResultsOp(this));
  registerDynamicOp(getDynamicCustomParserPrinterOp(this));

  auto &blobInterface = addInterface<TestResourceBlobManagerInterface>();
  addInterface<TestOpAsmInterface>(blobInterface);

  addInterfaces<TestDialectFoldInterface, TestInlinerInterface,
                TestReductionPatternInterface>();
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

::mlir::LogicalResult FormatInferType2Op::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  inferredReturnTypes.assign({::mlir::IntegerType::get(context, 16)});
  return ::mlir::success();
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

Optional<Dialect::ParseOpHook>
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
// Test Format* operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Parsing

static ParseResult parseCustomOptionalOperand(
    OpAsmParser &parser, Optional<OpAsmParser::UnresolvedOperand> &optOperand) {
  if (succeeded(parser.parseOptionalLParen())) {
    optOperand.emplace();
    if (parser.parseOperand(*optOperand) || parser.parseRParen())
      return failure();
  }
  return success();
}

static ParseResult parseCustomDirectiveOperands(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &operand,
    Optional<OpAsmParser::UnresolvedOperand> &optOperand,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &varOperands) {
  if (parser.parseOperand(operand))
    return failure();
  if (succeeded(parser.parseOptionalComma())) {
    optOperand.emplace();
    if (parser.parseOperand(*optOperand))
      return failure();
  }
  if (parser.parseArrow() || parser.parseLParen() ||
      parser.parseOperandList(varOperands) || parser.parseRParen())
    return failure();
  return success();
}
static ParseResult
parseCustomDirectiveResults(OpAsmParser &parser, Type &operandType,
                            Type &optOperandType,
                            SmallVectorImpl<Type> &varOperandTypes) {
  if (parser.parseColon())
    return failure();

  if (parser.parseType(operandType))
    return failure();
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseType(optOperandType))
      return failure();
  }
  if (parser.parseArrow() || parser.parseLParen() ||
      parser.parseTypeList(varOperandTypes) || parser.parseRParen())
    return failure();
  return success();
}
static ParseResult
parseCustomDirectiveWithTypeRefs(OpAsmParser &parser, Type operandType,
                                 Type optOperandType,
                                 const SmallVectorImpl<Type> &varOperandTypes) {
  if (parser.parseKeyword("type_refs_capture"))
    return failure();

  Type operandType2, optOperandType2;
  SmallVector<Type, 1> varOperandTypes2;
  if (parseCustomDirectiveResults(parser, operandType2, optOperandType2,
                                  varOperandTypes2))
    return failure();

  if (operandType != operandType2 || optOperandType != optOperandType2 ||
      varOperandTypes != varOperandTypes2)
    return failure();

  return success();
}
static ParseResult parseCustomDirectiveOperandsAndTypes(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &operand,
    Optional<OpAsmParser::UnresolvedOperand> &optOperand,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &varOperands,
    Type &operandType, Type &optOperandType,
    SmallVectorImpl<Type> &varOperandTypes) {
  if (parseCustomDirectiveOperands(parser, operand, optOperand, varOperands) ||
      parseCustomDirectiveResults(parser, operandType, optOperandType,
                                  varOperandTypes))
    return failure();
  return success();
}
static ParseResult parseCustomDirectiveRegions(
    OpAsmParser &parser, Region &region,
    SmallVectorImpl<std::unique_ptr<Region>> &varRegions) {
  if (parser.parseRegion(region))
    return failure();
  if (failed(parser.parseOptionalComma()))
    return success();
  std::unique_ptr<Region> varRegion = std::make_unique<Region>();
  if (parser.parseRegion(*varRegion))
    return failure();
  varRegions.emplace_back(std::move(varRegion));
  return success();
}
static ParseResult
parseCustomDirectiveSuccessors(OpAsmParser &parser, Block *&successor,
                               SmallVectorImpl<Block *> &varSuccessors) {
  if (parser.parseSuccessor(successor))
    return failure();
  if (failed(parser.parseOptionalComma()))
    return success();
  Block *varSuccessor;
  if (parser.parseSuccessor(varSuccessor))
    return failure();
  varSuccessors.append(2, varSuccessor);
  return success();
}
static ParseResult parseCustomDirectiveAttributes(OpAsmParser &parser,
                                                  IntegerAttr &attr,
                                                  IntegerAttr &optAttr) {
  if (parser.parseAttribute(attr))
    return failure();
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseAttribute(optAttr))
      return failure();
  }
  return success();
}
static ParseResult parseCustomDirectiveSpacing(OpAsmParser &parser,
                                               mlir::StringAttr &attr) {
  return parser.parseAttribute(attr);
}
static ParseResult parseCustomDirectiveAttrDict(OpAsmParser &parser,
                                                NamedAttrList &attrs) {
  return parser.parseOptionalAttrDict(attrs);
}
static ParseResult parseCustomDirectiveOptionalOperandRef(
    OpAsmParser &parser, Optional<OpAsmParser::UnresolvedOperand> &optOperand) {
  int64_t operandCount = 0;
  if (parser.parseInteger(operandCount))
    return failure();
  bool expectedOptionalOperand = operandCount == 0;
  return success(expectedOptionalOperand != optOperand.has_value());
}

//===----------------------------------------------------------------------===//
// Printing

static void printCustomOptionalOperand(OpAsmPrinter &printer, Operation *,
                                       Value optOperand) {
  if (optOperand)
    printer << "(" << optOperand << ") ";
}

static void printCustomDirectiveOperands(OpAsmPrinter &printer, Operation *,
                                         Value operand, Value optOperand,
                                         OperandRange varOperands) {
  printer << operand;
  if (optOperand)
    printer << ", " << optOperand;
  printer << " -> (" << varOperands << ")";
}
static void printCustomDirectiveResults(OpAsmPrinter &printer, Operation *,
                                        Type operandType, Type optOperandType,
                                        TypeRange varOperandTypes) {
  printer << " : " << operandType;
  if (optOperandType)
    printer << ", " << optOperandType;
  printer << " -> (" << varOperandTypes << ")";
}
static void printCustomDirectiveWithTypeRefs(OpAsmPrinter &printer,
                                             Operation *op, Type operandType,
                                             Type optOperandType,
                                             TypeRange varOperandTypes) {
  printer << " type_refs_capture ";
  printCustomDirectiveResults(printer, op, operandType, optOperandType,
                              varOperandTypes);
}
static void printCustomDirectiveOperandsAndTypes(
    OpAsmPrinter &printer, Operation *op, Value operand, Value optOperand,
    OperandRange varOperands, Type operandType, Type optOperandType,
    TypeRange varOperandTypes) {
  printCustomDirectiveOperands(printer, op, operand, optOperand, varOperands);
  printCustomDirectiveResults(printer, op, operandType, optOperandType,
                              varOperandTypes);
}
static void printCustomDirectiveRegions(OpAsmPrinter &printer, Operation *,
                                        Region &region,
                                        MutableArrayRef<Region> varRegions) {
  printer.printRegion(region);
  if (!varRegions.empty()) {
    printer << ", ";
    for (Region &region : varRegions)
      printer.printRegion(region);
  }
}
static void printCustomDirectiveSuccessors(OpAsmPrinter &printer, Operation *,
                                           Block *successor,
                                           SuccessorRange varSuccessors) {
  printer << successor;
  if (!varSuccessors.empty())
    printer << ", " << varSuccessors.front();
}
static void printCustomDirectiveAttributes(OpAsmPrinter &printer, Operation *,
                                           Attribute attribute,
                                           Attribute optAttribute) {
  printer << attribute;
  if (optAttribute)
    printer << ", " << optAttribute;
}
static void printCustomDirectiveSpacing(OpAsmPrinter &printer, Operation *op,
                                        Attribute attribute) {
  printer << attribute;
}
static void printCustomDirectiveAttrDict(OpAsmPrinter &printer, Operation *op,
                                         DictionaryAttr attrs) {
  printer.printOptionalAttrDict(attrs.getValue());
}

static void printCustomDirectiveOptionalOperandRef(OpAsmPrinter &printer,
                                                   Operation *op,
                                                   Value optOperand) {
  printer << (optOperand ? "1" : "0");
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
  p << "test.isolated_region ";
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
  p << "test.affine_scope ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// Test parser.
//===----------------------------------------------------------------------===//

ParseResult ParseIntegerLiteralOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  if (parser.parseOptionalColon())
    return success();
  uint64_t numResults;
  if (parser.parseInteger(numResults))
    return failure();

  IndexType type = parser.getBuilder().getIndexType();
  for (unsigned i = 0; i < numResults; ++i)
    result.addTypes(type);
  return success();
}

void ParseIntegerLiteralOp::print(OpAsmPrinter &p) {
  if (unsigned numResults = getNumResults())
    p << " : " << numResults;
}

ParseResult ParseWrappedKeywordOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return failure();
  result.addAttribute("keyword", parser.getBuilder().getStringAttr(keyword));
  return success();
}

void ParseWrappedKeywordOp::print(OpAsmPrinter &p) { p << " " << getKeyword(); }

ParseResult ParseB64BytesOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  std::vector<char> bytes;
  if (parser.parseBase64Bytes(&bytes))
    return failure();
  result.addAttribute("b64", parser.getBuilder().getStringAttr(
                                 StringRef(&bytes.front(), bytes.size())));
  return success();
}

void ParseB64BytesOp::print(OpAsmPrinter &p) {
  // Don't print the base64 version to check that we decoded it correctly.
  p << " \"" << getB64() << "\"";
}

//===----------------------------------------------------------------------===//
// Test WrapRegionOp - wrapping op exercising `parseGenericOperation()`.

ParseResult WrappingRegionOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  if (parser.parseKeyword("wraps"))
    return failure();

  // Parse the wrapped op in a region
  Region &body = *result.addRegion();
  body.push_back(new Block);
  Block &block = body.back();
  Operation *wrappedOp = parser.parseGenericOperation(&block, block.begin());
  if (!wrappedOp)
    return failure();

  // Create a return terminator in the inner region, pass as operand to the
  // terminator the returned values from the wrapped operation.
  SmallVector<Value, 8> returnOperands(wrappedOp->getResults());
  OpBuilder builder(parser.getContext());
  builder.setInsertionPointToEnd(&block);
  builder.create<TestReturnOp>(wrappedOp->getLoc(), returnOperands);

  // Get the results type for the wrapping op from the terminator operands.
  Operation &returnOp = body.back().back();
  result.types.append(returnOp.operand_type_begin(),
                      returnOp.operand_type_end());

  // Use the location of the wrapped op for the "test.wrapping_region" op.
  result.location = wrappedOp->getLoc();

  return success();
}

void WrappingRegionOp::print(OpAsmPrinter &p) {
  p << " wraps ";
  p.printGenericOp(&getRegion().front().front());
}

//===----------------------------------------------------------------------===//
// Test PrettyPrintedRegionOp -  exercising the following parser APIs
//   parseGenericOperationAfterOpName
//   parseCustomOperationName
//===----------------------------------------------------------------------===//

ParseResult PrettyPrintedRegionOp::parse(OpAsmParser &parser,
                                         OperationState &result) {

  SMLoc loc = parser.getCurrentLocation();
  Location currLocation = parser.getEncodedSourceLoc(loc);

  // Parse the operands.
  SmallVector<OpAsmParser::UnresolvedOperand, 2> operands;
  if (parser.parseOperandList(operands))
    return failure();

  // Check if we are parsing the pretty-printed version
  //  test.pretty_printed_region start <inner-op> end : <functional-type>
  // Else fallback to parsing the "non pretty-printed" version.
  if (!succeeded(parser.parseOptionalKeyword("start")))
    return parser.parseGenericOperationAfterOpName(
        result, llvm::makeArrayRef(operands));

  FailureOr<OperationName> parseOpNameInfo = parser.parseCustomOperationName();
  if (failed(parseOpNameInfo))
    return failure();

  StringAttr innerOpName = parseOpNameInfo->getIdentifier();

  FunctionType opFntype;
  Optional<Location> explicitLoc;
  if (parser.parseKeyword("end") || parser.parseColon() ||
      parser.parseType(opFntype) ||
      parser.parseOptionalLocationSpecifier(explicitLoc))
    return failure();

  // If location of the op is explicitly provided, then use it; Else use
  // the parser's current location.
  Location opLoc = explicitLoc.value_or(currLocation);

  // Derive the SSA-values for op's operands.
  if (parser.resolveOperands(operands, opFntype.getInputs(), loc,
                             result.operands))
    return failure();

  // Add a region for op.
  Region &region = *result.addRegion();

  // Create a basic-block inside op's region.
  Block &block = region.emplaceBlock();

  // Create and insert an "inner-op" operation in the block.
  // Just for testing purposes, we can assume that inner op is a binary op with
  // result and operand types all same as the test-op's first operand.
  Type innerOpType = opFntype.getInput(0);
  Value lhs = block.addArgument(innerOpType, opLoc);
  Value rhs = block.addArgument(innerOpType, opLoc);

  OpBuilder builder(parser.getBuilder().getContext());
  builder.setInsertionPointToStart(&block);

  Operation *innerOp =
      builder.create(opLoc, innerOpName, /*operands=*/{lhs, rhs}, innerOpType);

  // Insert a return statement in the block returning the inner-op's result.
  builder.create<TestReturnOp>(innerOp->getLoc(), innerOp->getResults());

  // Populate the op operation-state with result-type and location.
  result.addTypes(opFntype.getResults());
  result.location = innerOp->getLoc();

  return success();
}

void PrettyPrintedRegionOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printOperands(getOperands());

  Operation &innerOp = getRegion().front().front();
  // Assuming that region has a single non-terminator inner-op, if the inner-op
  // meets some criteria (which in this case is a simple one  based on the name
  // of inner-op), then we can print the entire region in a succinct way.
  // Here we assume that the prototype of "special.op" can be trivially derived
  // while parsing it back.
  if (innerOp.getName().getStringRef().equals("special.op")) {
    p << " start special.op end";
  } else {
    p << " (";
    p.printRegion(getRegion());
    p << ")";
  }

  p << " : ";
  p.printFunctionalType(*this);
}

//===----------------------------------------------------------------------===//
// Test PolyForOp - parse list of region arguments.
//===----------------------------------------------------------------------===//

ParseResult PolyForOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument, 4> ivsInfo;
  // Parse list of region arguments without a delimiter.
  if (parser.parseArgumentList(ivsInfo, OpAsmParser::Delimiter::None))
    return failure();

  // Parse the body region.
  Region *body = result.addRegion();
  for (auto &iv : ivsInfo)
    iv.type = parser.getBuilder().getIndexType();
  return parser.parseRegion(*body, ivsInfo);
}

void PolyForOp::print(OpAsmPrinter &p) { p.printGenericOp(*this); }

void PolyForOp::getAsmBlockArgumentNames(Region &region,
                                         OpAsmSetValueNameFn setNameFn) {
  auto arrayAttr = getOperation()->getAttrOfType<ArrayAttr>("arg_names");
  if (!arrayAttr)
    return;
  auto args = getRegion().front().getArguments();
  auto e = std::min(arrayAttr.size(), args.size());
  for (unsigned i = 0; i < e; ++i) {
    if (auto strAttr = arrayAttr[i].dyn_cast<StringAttr>())
      setNameFn(args[i], strAttr.getValue());
  }
}

//===----------------------------------------------------------------------===//
// TestAttrWithLoc - parse/printOptionalLocationSpecifier
//===----------------------------------------------------------------------===//

static ParseResult parseOptionalLoc(OpAsmParser &p, Attribute &loc) {
  Optional<Location> result;
  SMLoc sourceLoc = p.getCurrentLocation();
  if (p.parseOptionalLocationSpecifier(result))
    return failure();
  if (result)
    loc = *result;
  else
    loc = p.getEncodedSourceLoc(sourceLoc);
  return success();
}

static void printOptionalLoc(OpAsmPrinter &p, Operation *op, Attribute loc) {
  p.printOptionalLocationSpecifier(loc.cast<LocationAttr>());
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

OpFoldResult TestOpWithRegionFold::fold(ArrayRef<Attribute> operands) {
  return getOperand();
}

OpFoldResult TestOpConstant::fold(ArrayRef<Attribute> operands) {
  return getValue();
}

LogicalResult TestOpWithVariadicResultsAndFolder::fold(
    ArrayRef<Attribute> operands, SmallVectorImpl<OpFoldResult> &results) {
  for (Value input : this->getOperands()) {
    results.push_back(input);
  }
  return success();
}

OpFoldResult TestOpInPlaceFold::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1);
  if (operands.front()) {
    (*this)->setAttr("attr", operands.front());
    return getResult();
  }
  return {};
}

OpFoldResult TestPassthroughFold::fold(ArrayRef<Attribute> operands) {
  return getOperand();
}

LogicalResult OpWithInferTypeInterfaceOp::inferReturnTypes(
    MLIRContext *, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  if (operands[0].getType() != operands[1].getType()) {
    return emitOptionalError(location, "operand type mismatch ",
                             operands[0].getType(), " vs ",
                             operands[1].getType());
  }
  inferredReturnTypes.assign({operands[0].getType()});
  return success();
}

// TODO: We should be able to only define either inferReturnType or
// refineReturnType, currently only refineReturnType can be omitted.
LogicalResult OpWithRefineTypeInterfaceOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &returnTypes) {
  returnTypes.clear();
  return OpWithRefineTypeInterfaceOp::refineReturnTypes(
      context, location, operands, attributes, regions, returnTypes);
}

LogicalResult OpWithRefineTypeInterfaceOp::refineReturnTypes(
    MLIRContext *, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
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
    MLIRContext *context, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  // Create return type consisting of the last element of the first operand.
  auto operandType = operands.front().getType();
  auto sval = operandType.dyn_cast<ShapedType>();
  if (!sval) {
    return emitOptionalError(location, "only shaped type operands allowed");
  }
  int64_t dim =
      sval.hasRank() ? sval.getShape().front() : ShapedType::kDynamic;
  auto type = IntegerType::get(context, 17);

  Attribute encoding;
  if (auto ranked_ty = sval.dyn_cast<RankedTensorType>())
    encoding = ranked_ty.getEncoding();
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

LogicalResult OpWithResultShapeInterfaceOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    llvm::SmallVectorImpl<Value> &shapes) {
  Location loc = getLoc();
  shapes.reserve(operands.size());
  for (Value operand : llvm::reverse(operands)) {
    auto rank = operand.getType().cast<RankedTensorType>().getRank();
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
    auto currShape = llvm::to_vector<4>(llvm::map_range(
        llvm::seq<int64_t>(
            0, operand.getType().cast<RankedTensorType>().getRank()),
        [&](int64_t dim) -> Value {
          return builder.createOrFold<tensor::DimOp>(loc, operand, dim);
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
    DictionaryAttr effectElement = element.cast<DictionaryAttr>();

    // Get the specific memory effect.
    MemoryEffects::Effect *effect =
        StringSwitch<MemoryEffects::Effect *>(
            effectElement.get("effect").cast<StringAttr>().getValue())
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
      effects.emplace_back(effect, ref.cast<SymbolRefAttr>(), resource);
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

    auto expectedName = getNames()[i].dyn_cast<StringAttr>();
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
    if (auto str = value[i].dyn_cast<StringAttr>())
      if (!str.getValue().empty())
        setNameFn(getResult(i), str.getValue());
}

void CustomResultsNameOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  ArrayAttr value = getNames();
  for (size_t i = 0, e = value.size(); i != e; ++i)
    if (auto str = value[i].dyn_cast<StringAttr>())
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

OperandRange RegionIfOp::getSuccessorEntryOperands(Optional<unsigned> index) {
  assert(index && *index < 2 && "invalid region index");
  return getOperands();
}

void RegionIfOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  // We always branch to the join region.
  if (index.has_value()) {
    if (index.value() < 2)
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

void AnyCondOp::getSuccessorRegions(Optional<unsigned> index,
                                    ArrayRef<Attribute> operands,
                                    SmallVectorImpl<RegionSuccessor> &regions) {
  // The parent op branches into the only region, and the region branches back
  // to the parent op.
  if (!index)
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

  emitRemark("success run of verifier");

  return success();
}

LogicalResult TestVerifiersOp::verifyRegions() {
  if (!getRegion().hasOneBlock())
    return emitOpError("`hasOneBlock` trait hasn't been verified");

  for (Block &block : getRegion())
    for (Operation &op : block)
      if (failed(mlir::verify(&op)))
        return emitOpError("nested op hasn't been verified");

  emitRemark("success run of region verifier");

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

#include "TestOpEnums.cpp.inc"
#include "TestOpInterfaces.cpp.inc"
#include "TestTypeInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "TestOps.cpp.inc"
