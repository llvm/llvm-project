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
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/ODSSupport.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Reducer/ReductionPatternInterface.h"
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

void test::registerTestDialect(DialectRegistry &registry) {
  registry.insert<TestDialect>();
}

//===----------------------------------------------------------------------===//
// TestDialect version utilities
//===----------------------------------------------------------------------===//

struct TestDialectVersion : public DialectVersion {
  uint32_t major = 2;
  uint32_t minor = 0;
};

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

namespace {
enum test_encoding { k_attr_params = 0 };
}

// Test support for interacting with the Bytecode reader/writer.
struct TestBytecodeDialectInterface : public BytecodeDialectInterface {
  using BytecodeDialectInterface::BytecodeDialectInterface;
  TestBytecodeDialectInterface(Dialect *dialect)
      : BytecodeDialectInterface(dialect) {}

  LogicalResult writeAttribute(Attribute attr,
                               DialectBytecodeWriter &writer) const final {
    if (auto concreteAttr = llvm::dyn_cast<TestAttrParamsAttr>(attr)) {
      writer.writeVarInt(test_encoding::k_attr_params);
      writer.writeVarInt(concreteAttr.getV0());
      writer.writeVarInt(concreteAttr.getV1());
      return success();
    }
    return failure();
  }

  Attribute readAttribute(DialectBytecodeReader &reader,
                          const DialectVersion &version_) const final {
    const auto &version = static_cast<const TestDialectVersion &>(version_);
    if (version.major < 2)
      return readAttrOldEncoding(reader);
    if (version.major == 2 && version.minor == 0)
      return readAttrNewEncoding(reader);
    // Forbid reading future versions by returning nullptr.
    return Attribute();
  }

  // Emit a specific version of the dialect.
  void writeVersion(DialectBytecodeWriter &writer) const final {
    auto version = TestDialectVersion();
    writer.writeVarInt(version.major); // major
    writer.writeVarInt(version.minor); // minor
  }

  std::unique_ptr<DialectVersion>
  readVersion(DialectBytecodeReader &reader) const final {
    uint64_t major, minor;
    if (failed(reader.readVarInt(major)) || failed(reader.readVarInt(minor)))
      return nullptr;
    auto version = std::make_unique<TestDialectVersion>();
    version->major = major;
    version->minor = minor;
    return version;
  }

  LogicalResult upgradeFromVersion(Operation *topLevelOp,
                                   const DialectVersion &version_) const final {
    const auto &version = static_cast<const TestDialectVersion &>(version_);
    if ((version.major == 2) && (version.minor == 0))
      return success();
    if (version.major > 2 || (version.major == 2 && version.minor > 0)) {
      return topLevelOp->emitError()
             << "current test dialect version is 2.0, can't parse version: "
             << version.major << "." << version.minor;
    }
    // Prior version 2.0, the old op supported only a single attribute called
    // "dimensions". We can perform the upgrade.
    topLevelOp->walk([](TestVersionedOpA op) {
      if (auto dims = op->getAttr("dimensions")) {
        op->removeAttr("dimensions");
        op->setAttr("dims", dims);
      }
      op->setAttr("modifier", BoolAttr::get(op->getContext(), false));
    });
    return success();
  }

private:
  Attribute readAttrNewEncoding(DialectBytecodeReader &reader) const {
    uint64_t encoding;
    if (failed(reader.readVarInt(encoding)) ||
        encoding != test_encoding::k_attr_params)
      return Attribute();
    // The new encoding has v0 first, v1 second.
    uint64_t v0, v1;
    if (failed(reader.readVarInt(v0)) || failed(reader.readVarInt(v1)))
      return Attribute();
    return TestAttrParamsAttr::get(getContext(), static_cast<int>(v0),
                                   static_cast<int>(v1));
  }

  Attribute readAttrOldEncoding(DialectBytecodeReader &reader) const {
    uint64_t encoding;
    if (failed(reader.readVarInt(encoding)) ||
        encoding != test_encoding::k_attr_params)
      return Attribute();
    // The old encoding has v1 first, v0 second.
    uint64_t v0, v1;
    if (failed(reader.readVarInt(v1)) || failed(reader.readVarInt(v0)))
      return Attribute();
    return TestAttrParamsAttr::get(getContext(), static_cast<int>(v0),
                                   static_cast<int>(v1));
  }
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
    StringAttr strAttr = dyn_cast<StringAttr>(attr);
    if (!strAttr)
      return AliasResult::NoAlias;

    // Check the contents of the string attribute to see what the test alias
    // should be named.
    std::optional<StringRef> aliasName =
        StringSwitch<std::optional<StringRef>>(strAttr.getValue())
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
    if (auto tupleType = dyn_cast<TupleType>(type)) {
      if (tupleType.size() > 0 &&
          llvm::all_of(tupleType.getTypes(), [](Type elemType) {
            return isa<SimpleAType>(elemType);
          })) {
        os << "test_tuple";
        return AliasResult::FinalAlias;
      }
    }
    if (auto intType = dyn_cast<TestIntegerType>(type)) {
      if (intType.getSignedness() ==
              TestIntegerType::SignednessSemantics::Unsigned &&
          intType.getWidth() == 8) {
        os << "test_ui8";
        return AliasResult::FinalAlias;
      }
    }
    if (auto recType = dyn_cast<TestRecursiveType>(type)) {
      if (recType.getName() == "type_to_alias") {
        // We only make alias for a specific recursive type.
        os << "testrec";
        return AliasResult::FinalAlias;
      }
    }
    if (auto recAliasType = dyn_cast<TestRecursiveAliasType>(type)) {
      os << recAliasType.getName();
      return AliasResult::FinalAlias;
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
  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    // Inlining into test dialect regions is legal.
    return true;
  }
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
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

  Value handleArgument(OpBuilder &builder, Operation *call, Operation *callable,
                       Value argument,
                       DictionaryAttr argumentAttrs) const final {
    if (!argumentAttrs.contains("test.handle_argument"))
      return argument;
    return builder.create<TestTypeChangerOp>(call->getLoc(), argument.getType(),
                                             argument);
  }

  Value handleResult(OpBuilder &builder, Operation *call, Operation *callable,
                     Value result, DictionaryAttr resultAttrs) const final {
    if (!resultAttrs.contains("test.handle_result"))
      return result;
    return builder.create<TestTypeChangerOp>(call->getLoc(), result.getType(),
                                             result);
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
  addOperations<ManualCppOpWithFold>();
  registerDynamicOp(getDynamicGenericOp(this));
  registerDynamicOp(getDynamicOneOperandTwoResultsOp(this));
  registerDynamicOp(getDynamicCustomParserPrinterOp(this));

  auto &blobInterface = addInterface<TestResourceBlobManagerInterface>();
  addInterface<TestOpAsmInterface>(blobInterface);

  addInterfaces<TestDialectFoldInterface, TestInlinerInterface,
                TestReductionPatternInterface, TestBytecodeDialectInterface>();
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
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    OpaqueProperties properties, ::mlir::RegionRange regions,
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
// Test Format* operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Parsing

static ParseResult parseCustomOptionalOperand(
    OpAsmParser &parser,
    std::optional<OpAsmParser::UnresolvedOperand> &optOperand) {
  if (succeeded(parser.parseOptionalLParen())) {
    optOperand.emplace();
    if (parser.parseOperand(*optOperand) || parser.parseRParen())
      return failure();
  }
  return success();
}

static ParseResult parseCustomDirectiveOperands(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &operand,
    std::optional<OpAsmParser::UnresolvedOperand> &optOperand,
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
    std::optional<OpAsmParser::UnresolvedOperand> &optOperand,
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
    OpAsmParser &parser,
    std::optional<OpAsmParser::UnresolvedOperand> &optOperand) {
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
  p << " \"" << llvm::encodeBase64(getB64()) << "\"";
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
    return parser.parseGenericOperationAfterOpName(result,
                                                   llvm::ArrayRef(operands));

  FailureOr<OperationName> parseOpNameInfo = parser.parseCustomOperationName();
  if (failed(parseOpNameInfo))
    return failure();

  StringAttr innerOpName = parseOpNameInfo->getIdentifier();

  FunctionType opFntype;
  std::optional<Location> explicitLoc;
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
  // Here we assume that the prototype of "test.special.op" can be trivially
  // derived while parsing it back.
  if (innerOp.getName().getStringRef().equals("test.special.op")) {
    p << " start test.special.op end";
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

void PolyForOp::print(OpAsmPrinter &p) {
  p << " ";
  llvm::interleaveComma(getRegion().getArguments(), p, [&](auto arg) {
    p.printRegionArgument(arg, /*argAttrs =*/{}, /*omitType=*/true);
  });
  p << " ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);
}

void PolyForOp::getAsmBlockArgumentNames(Region &region,
                                         OpAsmSetValueNameFn setNameFn) {
  auto arrayAttr = getOperation()->getAttrOfType<ArrayAttr>("arg_names");
  if (!arrayAttr)
    return;
  auto args = getRegion().front().getArguments();
  auto e = std::min(arrayAttr.size(), args.size());
  for (unsigned i = 0; i < e; ++i) {
    if (auto strAttr = dyn_cast<StringAttr>(arrayAttr[i]))
      setNameFn(args[i], strAttr.getValue());
  }
}

//===----------------------------------------------------------------------===//
// TestAttrWithLoc - parse/printOptionalLocationSpecifier
//===----------------------------------------------------------------------===//

static ParseResult parseOptionalLoc(OpAsmParser &p, Attribute &loc) {
  std::optional<Location> result;
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
  p.printOptionalLocationSpecifier(cast<LocationAttr>(loc));
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

OperandRange
RegionIfOp::getSuccessorEntryOperands(std::optional<unsigned> index) {
  assert(index && *index < 2 && "invalid region index");
  return getOperands();
}

void RegionIfOp::getSuccessorRegions(
    std::optional<unsigned> index, ArrayRef<Attribute> operands,
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

void AnyCondOp::getSuccessorRegions(std::optional<unsigned> index,
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

void TestStoreWithARegion::getSuccessorRegions(
    std::optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  if (!index) {
    regions.emplace_back(&getBody(), getBody().front().getArguments());
  } else {
    regions.emplace_back();
  }
}

MutableOperandRange TestStoreWithARegionTerminator::getMutableSuccessorOperands(
    std::optional<unsigned> index) {
  return MutableOperandRange(getOperation());
}

#include "TestOpEnums.cpp.inc"
#include "TestOpInterfaces.cpp.inc"
#include "TestTypeInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "TestOps.cpp.inc"
