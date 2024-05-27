//===- TestDialect.cpp - MLIR Dialect for Testing -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "TestOps.h"
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
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/Casting.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Reducer/ReductionPatternInterface.h"
#include "mlir/Transforms/InliningUtils.h"
#include <cstdint>
#include <numeric>
#include <optional>

// Include this before the using namespace lines below to test that we don't
// have namespace dependencies.
#include "TestOpsDialect.cpp.inc"

using namespace mlir;
using namespace test;

//===----------------------------------------------------------------------===//
// PropertiesWithCustomPrint
//===----------------------------------------------------------------------===//

LogicalResult
test::setPropertiesFromAttribute(PropertiesWithCustomPrint &prop,
                                 Attribute attr,
                                 function_ref<InFlightDiagnostic()> emitError) {
  DictionaryAttr dict = dyn_cast<DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set TestProperties";
    return failure();
  }
  auto label = dict.getAs<mlir::StringAttr>("label");
  if (!label) {
    emitError() << "expected StringAttr for key `label`";
    return failure();
  }
  auto valueAttr = dict.getAs<IntegerAttr>("value");
  if (!valueAttr) {
    emitError() << "expected IntegerAttr for key `value`";
    return failure();
  }

  prop.label = std::make_shared<std::string>(label.getValue());
  prop.value = valueAttr.getValue().getSExtValue();
  return success();
}

DictionaryAttr
test::getPropertiesAsAttribute(MLIRContext *ctx,
                               const PropertiesWithCustomPrint &prop) {
  SmallVector<NamedAttribute> attrs;
  Builder b{ctx};
  attrs.push_back(b.getNamedAttr("label", b.getStringAttr(*prop.label)));
  attrs.push_back(b.getNamedAttr("value", b.getI32IntegerAttr(prop.value)));
  return b.getDictionaryAttr(attrs);
}

llvm::hash_code test::computeHash(const PropertiesWithCustomPrint &prop) {
  return llvm::hash_combine(prop.value, StringRef(*prop.label));
}

void test::customPrintProperties(OpAsmPrinter &p,
                                 const PropertiesWithCustomPrint &prop) {
  p.printKeywordOrString(*prop.label);
  p << " is " << prop.value;
}

ParseResult test::customParseProperties(OpAsmParser &parser,
                                        PropertiesWithCustomPrint &prop) {
  std::string label;
  if (parser.parseKeywordOrString(&label) || parser.parseKeyword("is") ||
      parser.parseInteger(prop.value))
    return failure();
  prop.label = std::make_shared<std::string>(std::move(label));
  return success();
}

//===----------------------------------------------------------------------===//
// MyPropStruct
//===----------------------------------------------------------------------===//

Attribute MyPropStruct::asAttribute(MLIRContext *ctx) const {
  return StringAttr::get(ctx, content);
}

LogicalResult
MyPropStruct::setFromAttr(MyPropStruct &prop, Attribute attr,
                          function_ref<InFlightDiagnostic()> emitError) {
  StringAttr strAttr = dyn_cast<StringAttr>(attr);
  if (!strAttr) {
    emitError() << "Expect StringAttr but got " << attr;
    return failure();
  }
  prop.content = strAttr.getValue();
  return success();
}

llvm::hash_code MyPropStruct::hash() const {
  return hash_value(StringRef(content));
}

LogicalResult test::readFromMlirBytecode(DialectBytecodeReader &reader,
                                         MyPropStruct &prop) {
  StringRef str;
  if (failed(reader.readString(str)))
    return failure();
  prop.content = str.str();
  return success();
}

void test::writeToMlirBytecode(DialectBytecodeWriter &writer,
                               MyPropStruct &prop) {
  writer.writeOwnedString(prop.content);
}

//===----------------------------------------------------------------------===//
// VersionedProperties
//===----------------------------------------------------------------------===//

LogicalResult
test::setPropertiesFromAttribute(VersionedProperties &prop, Attribute attr,
                                 function_ref<InFlightDiagnostic()> emitError) {
  DictionaryAttr dict = dyn_cast<DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set VersionedProperties";
    return failure();
  }
  auto value1Attr = dict.getAs<IntegerAttr>("value1");
  if (!value1Attr) {
    emitError() << "expected IntegerAttr for key `value1`";
    return failure();
  }
  auto value2Attr = dict.getAs<IntegerAttr>("value2");
  if (!value2Attr) {
    emitError() << "expected IntegerAttr for key `value2`";
    return failure();
  }

  prop.value1 = value1Attr.getValue().getSExtValue();
  prop.value2 = value2Attr.getValue().getSExtValue();
  return success();
}

DictionaryAttr test::getPropertiesAsAttribute(MLIRContext *ctx,
                                              const VersionedProperties &prop) {
  SmallVector<NamedAttribute> attrs;
  Builder b{ctx};
  attrs.push_back(b.getNamedAttr("value1", b.getI32IntegerAttr(prop.value1)));
  attrs.push_back(b.getNamedAttr("value2", b.getI32IntegerAttr(prop.value2)));
  return b.getDictionaryAttr(attrs);
}

llvm::hash_code test::computeHash(const VersionedProperties &prop) {
  return llvm::hash_combine(prop.value1, prop.value2);
}

void test::customPrintProperties(OpAsmPrinter &p,
                                 const VersionedProperties &prop) {
  p << prop.value1 << " | " << prop.value2;
}

ParseResult test::customParseProperties(OpAsmParser &parser,
                                        VersionedProperties &prop) {
  if (parser.parseInteger(prop.value1) || parser.parseVerticalBar() ||
      parser.parseInteger(prop.value2))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// Bytecode Support
//===----------------------------------------------------------------------===//

LogicalResult test::readFromMlirBytecode(DialectBytecodeReader &reader,
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

void test::writeToMlirBytecode(DialectBytecodeWriter &writer,
                               ArrayRef<int64_t> prop) {
  writer.writeVarInt(prop.size());
  for (auto elt : prop)
    writer.writeVarInt(elt);
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

void test::registerTestDialect(DialectRegistry &registry) {
  registry.insert<TestDialect>();
}

void test::testSideEffectOpGetEffect(
    Operation *op,
    SmallVectorImpl<SideEffects::EffectInstance<TestEffects::Effect>>
        &effects) {
  auto effectsAttr = op->getAttrOfType<AffineMapAttr>("effect_parameter");
  if (!effectsAttr)
    return;

  effects.emplace_back(TestEffects::Concrete::get(), effectsAttr);
}

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
  registerOpsSyntax();
  addOperations<ManualCppOpWithFold>();
  registerTestDialectOperations(this);
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
