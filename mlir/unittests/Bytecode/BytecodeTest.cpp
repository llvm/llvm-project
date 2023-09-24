//===- AdaptorTest.cpp - Adaptor unit tests -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Endian.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace mlir;

using ::testing::StartsWith;

StringLiteral IRWithResources = R"(
module @TestDialectResources attributes {
  bytecode.test = dense_resource<resource> : tensor<4xi32>
} {}
{-#
  dialect_resources: {
    builtin: {
      resource: "0x2000000001000000020000000300000004000000"
    }
  }
#-}
)";

TEST(Bytecode, MultiModuleWithResource) {
  MLIRContext context;
  Builder builder(&context);
  ParserConfig parseConfig(&context);
  OwningOpRef<Operation *> module =
      parseSourceString<Operation *>(IRWithResources, parseConfig);
  ASSERT_TRUE(module);

  // Write the module to bytecode
  std::string buffer;
  llvm::raw_string_ostream ostream(buffer);
  ASSERT_TRUE(succeeded(writeBytecodeToFile(module.get(), ostream)));
  ostream.flush();

  // Create copy of buffer which is aligned to requested resource alignment.
  constexpr size_t kAlignment = 0x20;
  size_t buffer_size = buffer.size();
  buffer.reserve(buffer_size + kAlignment - 1);
  size_t pad = ~(uintptr_t)buffer.data() + 1 & kAlignment - 1;
  buffer.insert(0, pad, ' ');
  StringRef aligned_buffer(buffer.data() + pad, buffer_size);

  // Parse it back
  OwningOpRef<Operation *> roundTripModule =
      parseSourceString<Operation *>(aligned_buffer, parseConfig);
  ASSERT_TRUE(roundTripModule);

  // FIXME: Parsing external resources does not work on big-endian
  // platforms currently.
  if (llvm::support::endian::system_endianness() ==
      llvm::support::endianness::big)
    GTEST_SKIP();

  // Try to see if we have a valid resource in the parsed module.
  auto checkResourceAttribute = [&](Operation *op) {
    Attribute attr = roundTripModule->getDiscardableAttr("bytecode.test");
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

TEST(Bytecode, InsufficientAlignmentFailure) {
  MLIRContext context;
  Builder builder(&context);
  ParserConfig parseConfig(&context);
  OwningOpRef<Operation *> module =
      parseSourceString<Operation *>(IRWithResources, parseConfig);
  ASSERT_TRUE(module);

  // Write the module to bytecode
  std::string buffer;
  llvm::raw_string_ostream ostream(buffer);
  ASSERT_TRUE(succeeded(writeBytecodeToFile(module.get(), ostream)));
  ostream.flush();

  // Create copy of buffer which is insufficiently aligned.
  constexpr size_t kAlignment = 0x20;
  size_t buffer_size = buffer.size();
  buffer.reserve(buffer_size + kAlignment - 1);
  size_t pad = ~(uintptr_t)buffer.data() + kAlignment / 2 + 1 & kAlignment - 1;
  buffer.insert(0, pad, ' ');
  StringRef misaligned_buffer(buffer.data() + pad, buffer_size);

  std::unique_ptr<Diagnostic> diagnostic;
  context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    diagnostic = std::make_unique<Diagnostic>(std::move(diag));
  });

  // Try to parse it back and check for alignment error.
  OwningOpRef<Operation *> roundTripModule =
      parseSourceString<Operation *>(misaligned_buffer, parseConfig);
  EXPECT_FALSE(roundTripModule);
  ASSERT_TRUE(diagnostic);
  EXPECT_THAT(diagnostic->str(),
              StartsWith("expected bytecode buffer to be aligned to 32"));
}
