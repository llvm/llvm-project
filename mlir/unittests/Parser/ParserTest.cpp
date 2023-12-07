//===- ParserTest.cpp -----------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Parser/Parser.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"

#include "gmock/gmock.h"

using namespace mlir;

namespace {
TEST(MLIRParser, ParseInvalidIR) {
  std::string moduleStr = R"mlir(
    module attributes {bad} {}
  )mlir";

  MLIRContext context;
  ParserConfig config(&context, /*verifyAfterParse=*/false);

  // Check that we properly parse the op, but it fails the verifier.
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, config);
  ASSERT_TRUE(module);
  ASSERT_TRUE(failed(verify(*module)));
}

TEST(MLIRParser, ParseAtEnd) {
  std::string firstModuleStr = R"mlir(
    "test.first"() : () -> ()
  )mlir";
  std::string secondModuleStr = R"mlir(
    "test.second"() : () -> ()
  )mlir";

  MLIRContext context;
  context.allowUnregisteredDialects();
  Block block;

  // Parse the first module string.
  LogicalResult firstParse =
      parseSourceString(firstModuleStr, &block, &context);
  EXPECT_TRUE(succeeded(firstParse));

  // Parse the second module string.
  LogicalResult secondParse =
      parseSourceString(secondModuleStr, &block, &context);
  EXPECT_TRUE(succeeded(secondParse));

  // Check the we parse at the end.
  EXPECT_EQ(block.front().getName().getStringRef(), "test.first");
  EXPECT_EQ(block.back().getName().getStringRef(), "test.second");
}

TEST(MLIRParser, ParseAttr) {
  using namespace testing;
  MLIRContext context;
  Builder b(&context);
  { // Successful parse
    StringLiteral attrAsm = "array<i64: 1, 2, 3>";
    size_t numRead = 0;
    Attribute attr = parseAttribute(attrAsm, &context, Type(), &numRead);
    EXPECT_EQ(attr, b.getDenseI64ArrayAttr({1, 2, 3}));
    EXPECT_EQ(numRead, attrAsm.size());
  }
  { // Failed parse
    std::vector<std::string> diagnostics;
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic &d) {
      llvm::raw_string_ostream(diagnostics.emplace_back())
          << d.getLocation() << ": " << d;
    });
    size_t numRead = 0;
    EXPECT_FALSE(parseAttribute("dense<>", &context, Type(), &numRead));
    EXPECT_THAT(diagnostics, ElementsAre("loc(\"dense<>\":1:7): expected ':'"));
    EXPECT_EQ(numRead, size_t(0));
  }
  { // Parse with trailing characters
    std::vector<std::string> diagnostics;
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic &d) {
      llvm::raw_string_ostream(diagnostics.emplace_back())
          << d.getLocation() << ": " << d;
    });
    EXPECT_FALSE(parseAttribute("10  foo", &context));
    EXPECT_THAT(
        diagnostics,
        ElementsAre("loc(\"10  foo\":1:5): found trailing characters: 'foo'"));

    size_t numRead = 0;
    EXPECT_EQ(parseAttribute("10  foo", &context, Type(), &numRead),
              b.getI64IntegerAttr(10));
    EXPECT_EQ(numRead, size_t(4)); // includes trailing whitespace
  }
  { // Parse without null-terminator
    StringRef attrAsm("999", 1);
    Attribute attr = parseAttribute(attrAsm, &context);
    EXPECT_EQ(attr, b.getI64IntegerAttr(9));
  }
}
} // namespace
