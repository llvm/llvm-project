//===- AdaptorTest.cpp - Adaptor unit tests -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace mlir;

StringLiteral irWithResources = R"(
module @TestDialectResources attributes {
  bytecode.test = dense_resource<resource> : tensor<4xi32>
} {}
{-#
  dialect_resources: {
    builtin: {
      resource: "0x2000000001000000020000000300000004000000",
      resource_2: "0x2000000001000000020000000300000004000000"
    }
  }
#-}
)";

TEST(Bytecode, MultiModuleWithResource) {
  MLIRContext context;
  Builder builder(&context);
  ParserConfig parseConfig(&context);
  OwningOpRef<Operation *> module =
      parseSourceString<Operation *>(irWithResources, parseConfig);
  ASSERT_TRUE(module);

  // Write the module to bytecode
  std::string buffer;
  llvm::raw_string_ostream ostream(buffer);
  ASSERT_TRUE(succeeded(writeBytecodeToFile(module.get(), ostream)));

  // Create copy of buffer which is aligned to requested resource alignment.
  constexpr size_t kAlignment = 0x20;
  size_t bufferSize = buffer.size();
  buffer.reserve(bufferSize + kAlignment - 1);
  size_t pad = (~(uintptr_t)buffer.data() + 1) & (kAlignment - 1);
  buffer.insert(0, pad, ' ');
  StringRef alignedBuffer(buffer.data() + pad, bufferSize);

  // Parse it back
  OwningOpRef<Operation *> roundTripModule =
      parseSourceString<Operation *>(alignedBuffer, parseConfig);
  ASSERT_TRUE(roundTripModule);

  // FIXME: Parsing external resources does not work on big-endian
  // platforms currently.
  if (llvm::endianness::native == llvm::endianness::big)
    GTEST_SKIP();

  // Try to see if we have a valid resource in the parsed module.
  auto checkResourceAttribute = [](Operation *parsedModule) {
    Attribute attr = parsedModule->getDiscardableAttr("bytecode.test");
    ASSERT_TRUE(attr);
    auto denseResourceAttr = dyn_cast<DenseI32ResourceElementsAttr>(attr);
    ASSERT_TRUE(denseResourceAttr);
    std::optional<ArrayRef<int32_t>> attrData =
        denseResourceAttr.tryGetAsArrayRef();
    ASSERT_TRUE(attrData.has_value());
    ASSERT_EQ(attrData->size(), static_cast<size_t>(4));
    EXPECT_EQ((*attrData)[0], 1);
    EXPECT_EQ((*attrData)[1], 2);
    EXPECT_EQ((*attrData)[2], 3);
    EXPECT_EQ((*attrData)[3], 4);
  };

  checkResourceAttribute(*module);
  checkResourceAttribute(*roundTripModule);
}

namespace {
/// A custom operation for the purpose of showcasing how discardable attributes
/// are handled in absence of properties.
class OpWithoutProperties : public Op<OpWithoutProperties> {
public:
  // Begin boilerplate.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpWithoutProperties)
  using Op::Op;
  static ArrayRef<StringRef> getAttributeNames() {
    static StringRef attributeNames[] = {StringRef("inherent_attr")};
    return ArrayRef(attributeNames);
  };
  static StringRef getOperationName() {
    return "test_op_properties.op_without_properties";
  }
  // End boilerplate.
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
    addOperations<OpWithoutProperties>();
  }
};
} // namespace

constexpr StringLiteral withoutPropertiesAttrsSrc = R"mlir(
    "test_op_properties.op_without_properties"()
      {inherent_attr = 42, other_attr = 56} : () -> ()
)mlir";

TEST(Bytecode, OpWithoutProperties) {
  MLIRContext context;
  context.getOrLoadDialect<TestOpPropertiesDialect>();
  ParserConfig config(&context);
  OwningOpRef<Operation *> op =
      parseSourceString(withoutPropertiesAttrsSrc, config);

  std::string bytecode;
  llvm::raw_string_ostream os(bytecode);
  ASSERT_TRUE(succeeded(writeBytecodeToFile(op.get(), os)));
  std::unique_ptr<Block> block = std::make_unique<Block>();
  ASSERT_TRUE(succeeded(readBytecodeFile(
      llvm::MemoryBufferRef(bytecode, "string-buffer"), block.get(), config)));
  Operation *roundtripped = &block->front();
  EXPECT_EQ(roundtripped->getAttrs().size(), 2u);
  EXPECT_TRUE(roundtripped->getInherentAttr("inherent_attr") != std::nullopt);
  EXPECT_TRUE(roundtripped->getDiscardableAttr("other_attr") != Attribute());

  EXPECT_TRUE(OperationEquivalence::computeHash(op.get()) ==
              OperationEquivalence::computeHash(roundtripped));
}
