//===- TestOpProperties.cpp - Test all properties-related APIs ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/OpDefinition.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"
#include <optional>

using namespace mlir;

namespace {
/// Simple structure definining a struct to define "properties" for a given
/// operation. Default values are honored when creating an operation.
struct TestProperties {
  int a = -1;
  float b = -1.;
  std::vector<int64_t> array = {-33};
  /// A shared_ptr to a const object is safe: it is equivalent to a value-based
  /// member. Here the label will be deallocated when the last operation
  /// referring to it is destroyed. However there is no pool-allocation: this is
  /// offloaded to the client.
  std::shared_ptr<const std::string> label;
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestProperties)
};

/// Convert a DictionaryAttr to a TestProperties struct, optionally emit errors
/// through the provided diagnostic if any. This is used for example during
/// parsing with the generic format.
static LogicalResult
setPropertiesFromAttribute(TestProperties &prop, Attribute attr,
                           InFlightDiagnostic *diagnostic) {
  DictionaryAttr dict = dyn_cast<DictionaryAttr>(attr);
  if (!dict) {
    if (diagnostic)
      *diagnostic << "expected DictionaryAttr to set TestProperties";
    return failure();
  }
  auto aAttr = dict.getAs<IntegerAttr>("a");
  if (!aAttr) {
    if (diagnostic)
      *diagnostic << "expected IntegerAttr for key `a`";
    return failure();
  }
  auto bAttr = dict.getAs<FloatAttr>("b");
  if (!bAttr ||
      &bAttr.getValue().getSemantics() != &llvm::APFloatBase::IEEEsingle()) {
    if (diagnostic)
      *diagnostic << "expected FloatAttr for key `b`";
    return failure();
  }

  auto arrayAttr = dict.getAs<DenseI64ArrayAttr>("array");
  if (!arrayAttr) {
    if (diagnostic)
      *diagnostic << "expected DenseI64ArrayAttr for key `array`";
    return failure();
  }

  auto label = dict.getAs<mlir::StringAttr>("label");
  if (!label) {
    if (diagnostic)
      *diagnostic << "expected StringAttr for key `label`";
    return failure();
  }

  prop.a = aAttr.getValue().getSExtValue();
  prop.b = bAttr.getValue().convertToFloat();
  prop.array.assign(arrayAttr.asArrayRef().begin(),
                    arrayAttr.asArrayRef().end());
  prop.label = std::make_shared<std::string>(label.getValue());
  return success();
}

/// Convert a TestProperties struct to a DictionaryAttr, this is used for
/// example during printing with the generic format.
static Attribute getPropertiesAsAttribute(MLIRContext *ctx,
                                          const TestProperties &prop) {
  SmallVector<NamedAttribute> attrs;
  Builder b{ctx};
  attrs.push_back(b.getNamedAttr("a", b.getI32IntegerAttr(prop.a)));
  attrs.push_back(b.getNamedAttr("b", b.getF32FloatAttr(prop.b)));
  attrs.push_back(b.getNamedAttr("array", b.getDenseI64ArrayAttr(prop.array)));
  attrs.push_back(b.getNamedAttr(
      "label", b.getStringAttr(prop.label ? *prop.label : "<nullptr>")));
  return b.getDictionaryAttr(attrs);
}

inline llvm::hash_code computeHash(const TestProperties &prop) {
  // We hash `b` which is a float using its underlying array of char:
  unsigned char const *p = reinterpret_cast<unsigned char const *>(&prop.b);
  ArrayRef<unsigned char> bBytes{p, sizeof(prop.b)};
  return llvm::hash_combine(
      prop.a, llvm::hash_combine_range(bBytes.begin(), bBytes.end()),
      llvm::hash_combine_range(prop.array.begin(), prop.array.end()),
      StringRef(*prop.label));
}

/// A custom operation for the purpose of showcasing how to use "properties".
class OpWithProperties : public Op<OpWithProperties> {
public:
  // Begin boilerplate
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpWithProperties)
  using Op::Op;
  static ArrayRef<StringRef> getAttributeNames() { return {}; }
  static StringRef getOperationName() {
    return "test_op_properties.op_with_properties";
  }
  // End boilerplate

  // This alias is the only definition needed for enabling "properties" for this
  // operation.
  using Properties = TestProperties;
  static std::optional<mlir::Attribute> getInherentAttr(MLIRContext *context,
                                                        const Properties &prop,
                                                        StringRef name) {
    return std::nullopt;
  }
  static void setInherentAttr(Properties &prop, StringRef name,
                              mlir::Attribute value) {}
  static void populateInherentAttrs(MLIRContext *context,
                                    const Properties &prop,
                                    NamedAttrList &attrs) {}
  static LogicalResult
  verifyInherentAttrs(OperationName opName, NamedAttrList &attrs,
                      function_ref<InFlightDiagnostic()> getDiag) {
    return success();
  }
};

// A trivial supporting dialect to register the above operation.
class TestOpPropertiesDialect : public Dialect {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestOpPropertiesDialect)
  static constexpr StringLiteral getDialectNamespace() {
    return StringLiteral("test_op_properties");
  }
  explicit TestOpPropertiesDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context,
                TypeID::get<TestOpPropertiesDialect>()) {
    addOperations<OpWithProperties>();
  }
};

constexpr StringLiteral mlirSrc = R"mlir(
    "test_op_properties.op_with_properties"()
      <{a = -42 : i32,
        b = -4.200000e+01 : f32,
        array = array<i64: 40, 41>,
        label = "bar foo"}> : () -> ()
)mlir";

TEST(OpPropertiesTest, Properties) {
  MLIRContext context;
  context.getOrLoadDialect<TestOpPropertiesDialect>();
  ParserConfig config(&context);
  // Parse the operation with some properties.
  OwningOpRef<Operation *> op = parseSourceString(mlirSrc, config);
  ASSERT_TRUE(op.get() != nullptr);
  auto opWithProp = dyn_cast<OpWithProperties>(op.get());
  ASSERT_TRUE(opWithProp);
  {
    std::string output;
    llvm::raw_string_ostream os(output);
    opWithProp.print(os);
    ASSERT_STREQ("\"test_op_properties.op_with_properties\"() "
                 "<{a = -42 : i32, "
                 "array = array<i64: 40, 41>, "
                 "b = -4.200000e+01 : f32, "
                 "label = \"bar foo\"}> : () -> ()\n",
                 os.str().c_str());
  }
  // Get a mutable reference to the properties for this operation and modify it
  // in place one member at a time.
  TestProperties &prop = opWithProp.getProperties();
  prop.a = 42;
  {
    std::string output;
    llvm::raw_string_ostream os(output);
    opWithProp.print(os);
    EXPECT_TRUE(StringRef(os.str()).contains("a = 42"));
    EXPECT_TRUE(StringRef(os.str()).contains("b = -4.200000e+01"));
    EXPECT_TRUE(StringRef(os.str()).contains("array = array<i64: 40, 41>"));
    EXPECT_TRUE(StringRef(os.str()).contains("label = \"bar foo\""));
  }
  prop.b = 42.;
  {
    std::string output;
    llvm::raw_string_ostream os(output);
    opWithProp.print(os);
    EXPECT_TRUE(StringRef(os.str()).contains("a = 42"));
    EXPECT_TRUE(StringRef(os.str()).contains("b = 4.200000e+01"));
    EXPECT_TRUE(StringRef(os.str()).contains("array = array<i64: 40, 41>"));
    EXPECT_TRUE(StringRef(os.str()).contains("label = \"bar foo\""));
  }
  prop.array.push_back(42);
  {
    std::string output;
    llvm::raw_string_ostream os(output);
    opWithProp.print(os);
    EXPECT_TRUE(StringRef(os.str()).contains("a = 42"));
    EXPECT_TRUE(StringRef(os.str()).contains("b = 4.200000e+01"));
    EXPECT_TRUE(StringRef(os.str()).contains("array = array<i64: 40, 41, 42>"));
    EXPECT_TRUE(StringRef(os.str()).contains("label = \"bar foo\""));
  }
  prop.label = std::make_shared<std::string>("foo bar");
  {
    std::string output;
    llvm::raw_string_ostream os(output);
    opWithProp.print(os);
    EXPECT_TRUE(StringRef(os.str()).contains("a = 42"));
    EXPECT_TRUE(StringRef(os.str()).contains("b = 4.200000e+01"));
    EXPECT_TRUE(StringRef(os.str()).contains("array = array<i64: 40, 41, 42>"));
    EXPECT_TRUE(StringRef(os.str()).contains("label = \"foo bar\""));
  }
}

// Test diagnostic emission when using invalid dictionary.
TEST(OpPropertiesTest, FailedProperties) {
  MLIRContext context;
  context.getOrLoadDialect<TestOpPropertiesDialect>();
  std::string diagnosticStr;
  context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    diagnosticStr += diag.str();
    return success();
  });

  // Parse the operation with some properties.
  ParserConfig config(&context);

  // Parse an operation with invalid (incomplete) properties.
  OwningOpRef<Operation *> owningOp =
      parseSourceString("\"test_op_properties.op_with_properties\"() "
                        "<{a = -42 : i32}> : () -> ()\n",
                        config);
  ASSERT_EQ(owningOp.get(), nullptr);
  EXPECT_STREQ(
      "invalid properties {a = -42 : i32} for op "
      "test_op_properties.op_with_properties: expected FloatAttr for key `b`",
      diagnosticStr.c_str());
  diagnosticStr.clear();

  owningOp = parseSourceString(mlirSrc, config);
  Operation *op = owningOp.get();
  ASSERT_TRUE(op != nullptr);
  Location loc = op->getLoc();
  auto opWithProp = dyn_cast<OpWithProperties>(op);
  ASSERT_TRUE(opWithProp);

  OperationState state(loc, op->getName());
  Builder b{&context};
  NamedAttrList attrs;
  attrs.push_back(b.getNamedAttr("a", b.getStringAttr("foo")));
  state.propertiesAttr = attrs.getDictionary(&context);
  {
    auto diag = op->emitError("setting properties failed: ");
    auto result = state.setProperties(op, &diag);
    EXPECT_TRUE(result.failed());
  }
  EXPECT_STREQ("setting properties failed: expected IntegerAttr for key `a`",
               diagnosticStr.c_str());
}

TEST(OpPropertiesTest, DefaultValues) {
  MLIRContext context;
  context.getOrLoadDialect<TestOpPropertiesDialect>();
  OperationState state(UnknownLoc::get(&context),
                       "test_op_properties.op_with_properties");
  Operation *op = Operation::create(state);
  ASSERT_TRUE(op != nullptr);
  {
    std::string output;
    llvm::raw_string_ostream os(output);
    op->print(os);
    EXPECT_TRUE(StringRef(os.str()).contains("a = -1"));
    EXPECT_TRUE(StringRef(os.str()).contains("b = -1"));
    EXPECT_TRUE(StringRef(os.str()).contains("array = array<i64: -33>"));
  }
  op->erase();
}

TEST(OpPropertiesTest, Cloning) {
  MLIRContext context;
  context.getOrLoadDialect<TestOpPropertiesDialect>();
  ParserConfig config(&context);
  // Parse the operation with some properties.
  OwningOpRef<Operation *> op = parseSourceString(mlirSrc, config);
  ASSERT_TRUE(op.get() != nullptr);
  auto opWithProp = dyn_cast<OpWithProperties>(op.get());
  ASSERT_TRUE(opWithProp);
  Operation *clone = opWithProp->clone();

  // Check that op and its clone prints equally
  std::string opStr;
  std::string cloneStr;
  {
    llvm::raw_string_ostream os(opStr);
    op.get()->print(os);
  }
  {
    llvm::raw_string_ostream os(cloneStr);
    clone->print(os);
  }
  clone->erase();
  EXPECT_STREQ(opStr.c_str(), cloneStr.c_str());
}

TEST(OpPropertiesTest, Equivalence) {
  MLIRContext context;
  context.getOrLoadDialect<TestOpPropertiesDialect>();
  ParserConfig config(&context);
  // Parse the operation with some properties.
  OwningOpRef<Operation *> op = parseSourceString(mlirSrc, config);
  ASSERT_TRUE(op.get() != nullptr);
  auto opWithProp = dyn_cast<OpWithProperties>(op.get());
  ASSERT_TRUE(opWithProp);
  llvm::hash_code reference = OperationEquivalence::computeHash(opWithProp);
  TestProperties &prop = opWithProp.getProperties();
  prop.a = 42;
  EXPECT_NE(reference, OperationEquivalence::computeHash(opWithProp));
  prop.a = -42;
  EXPECT_EQ(reference, OperationEquivalence::computeHash(opWithProp));
  prop.b = 42.;
  EXPECT_NE(reference, OperationEquivalence::computeHash(opWithProp));
  prop.b = -42.;
  EXPECT_EQ(reference, OperationEquivalence::computeHash(opWithProp));
  prop.array.push_back(42);
  EXPECT_NE(reference, OperationEquivalence::computeHash(opWithProp));
  prop.array.pop_back();
  EXPECT_EQ(reference, OperationEquivalence::computeHash(opWithProp));
}

TEST(OpPropertiesTest, getOrAddProperties) {
  MLIRContext context;
  context.getOrLoadDialect<TestOpPropertiesDialect>();
  OperationState state(UnknownLoc::get(&context),
                       "test_op_properties.op_with_properties");
  // Test `getOrAddProperties` API on OperationState.
  TestProperties &prop = state.getOrAddProperties<TestProperties>();
  prop.a = 1;
  prop.b = 2;
  prop.array = {3, 4, 5};
  Operation *op = Operation::create(state);
  ASSERT_TRUE(op != nullptr);
  {
    std::string output;
    llvm::raw_string_ostream os(output);
    op->print(os);
    EXPECT_TRUE(StringRef(os.str()).contains("a = 1"));
    EXPECT_TRUE(StringRef(os.str()).contains("b = 2"));
    EXPECT_TRUE(StringRef(os.str()).contains("array = array<i64: 3, 4, 5>"));
  }
  op->erase();
}

} // namespace
