//===- ParserTest.cpp -----------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Parser/Parser.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/AsmParser/AsmParserState.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"

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

TEST(MLIRParser, ParseUnterminatedString) {
  // The source text ends inside a token and is followed in memory by bytes
  // that are not part of it; a lexer that reads past the end of the StringRef
  // lexes `i32garbage` instead of `i32`.
  StringRef text = R"mlir("test.first"() : () -> i32)mlir";
  std::string storage = (text + "garbage").str();
  StringRef source(storage.data(), text.size());

  MLIRContext context;
  context.allowUnregisteredDialects();
  Block block;

  LogicalResult parse = parseSourceString(source, &block, &context);
  EXPECT_TRUE(succeeded(parse));
  ASSERT_EQ(block.getOperations().size(), 1u);
  EXPECT_EQ(block.front().getName().getStringRef(), "test.first");
  ASSERT_EQ(block.front().getNumResults(), 1u);
  EXPECT_TRUE(block.front().getResult(0).getType().isInteger(32));
}

/// The outcome of parsing a source string: whether the parse succeeded, the
/// diagnostics it emitted, and the operations it produced.
static std::string parseOutcome(StringRef source) {
  MLIRContext context;
  context.allowUnregisteredDialects();
  std::string outcome;
  llvm::raw_string_ostream os(outcome);
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diag) {
    os << diag.getLocation() << ": " << diag << "\n";
  });
  Block block;
  os << (succeeded(parseSourceString(source, &block, &context)) ? "success"
                                                                : "failure")
     << "\n";
  for (Operation &op : block)
    os << op << "\n";
  return outcome;
}

TEST(MLIRParser, ParseTokenAtUnterminatedEnd) {
  // Each source ends inside a token and is followed in memory by bytes that
  // could continue that token. Parsing it must not look past the end of the
  // StringRef: the outcome must equal the outcome for the same text alone.
  StringRef locAlias = "\"test.op\"() : () -> () loc(#l)\n#l = ";
  std::string sources[][2] = {
      // A bare identifier.
      {"foo", "bar }"},
      // A keyword.
      {"\"test.op\"() : () -> index", "x }"},
      // A symbol reference.
      {(locAlias + "@s").str(), "ym }"},
      // A value name.
      {"%x", "y }"},
      // A type alias.
      {"\"test.op\"() : () -> !t", "y }"},
      // An integer.
      {(locAlias + "4").str(), "2 }"},
      // A float with an exponent.
      {(locAlias + "1.5e3").str(), "7 }"},
      // A hexadecimal integer.
      {(locAlias + "0x1F").str(), "F }"},
      // A string literal ending in an escape.
      {(locAlias + "\"ab\\").str(), "\"\" }"},
      // A line comment.
      {"\"test.op\"() : () -> ()\n// comment", "\n}"},
  };
  for (auto &[text, following] : sources) {
    std::string storage = text + following;
    StringRef unterminated(storage.data(), text.size());
    EXPECT_EQ(parseOutcome(unterminated), parseOutcome(text))
        << "source: " << text;
  }
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

TEST(MLIRParser, AsmParserLocations) {
  std::string moduleStr = R"mlir(
func.func @foo() -> !llvm.array<2 x f32> {
  %0 = llvm.mlir.undef : !llvm.array<2 x f32>
  func.return %0 : !llvm.array<2 x f32>
}
  )mlir";

  DialectRegistry registry;
  registry.insert<func::FuncDialect, LLVM::LLVMDialect>();
  MLIRContext context(registry);

  auto memBuffer =
      llvm::MemoryBuffer::getMemBuffer(moduleStr, "AsmParserTest.mlir",
                                       /*RequiresNullTerminator=*/false);
  ASSERT_TRUE(memBuffer);

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), llvm::SMLoc());

  Block block;
  AsmParserState parseState;
  const LogicalResult parseResult =
      parseAsmSourceFile(sourceMgr, &block, &context, &parseState);
  ASSERT_TRUE(parseResult.succeeded());

  auto funcOp = *block.getOps<func::FuncOp>().begin();
  const AsmParserState::OperationDefinition *funcOpDefinition =
      parseState.getOpDef(funcOp);
  ASSERT_TRUE(funcOpDefinition);

  const std::pair expectedStartFunc{2u, 1u};
  const std::pair expectedEndFunc{2u, 10u};
  const std::pair expectedScopeEndFunc{5u, 2u};
  ASSERT_EQ(sourceMgr.getLineAndColumn(funcOpDefinition->loc.Start),
            expectedStartFunc);
  ASSERT_EQ(sourceMgr.getLineAndColumn(funcOpDefinition->loc.End),
            expectedEndFunc);
  ASSERT_EQ(funcOpDefinition->loc.Start, funcOpDefinition->scopeLoc.Start);
  ASSERT_EQ(sourceMgr.getLineAndColumn(funcOpDefinition->scopeLoc.End),
            expectedScopeEndFunc);

  auto llvmUndef = *funcOp.getOps<LLVM::UndefOp>().begin();
  const AsmParserState::OperationDefinition *llvmUndefDefinition =
      parseState.getOpDef(llvmUndef);
  ASSERT_TRUE(llvmUndefDefinition);

  const std::pair expectedStartUndef{3u, 8u};
  const std::pair expectedEndUndef{3u, 23u};
  const std::pair expectedScopeEndUndef{3u, 46u};
  ASSERT_EQ(sourceMgr.getLineAndColumn(llvmUndefDefinition->loc.Start),
            expectedStartUndef);
  ASSERT_EQ(sourceMgr.getLineAndColumn(llvmUndefDefinition->loc.End),
            expectedEndUndef);
  ASSERT_EQ(llvmUndefDefinition->loc.Start,
            llvmUndefDefinition->scopeLoc.Start);
  ASSERT_EQ(sourceMgr.getLineAndColumn(llvmUndefDefinition->scopeLoc.End),
            expectedScopeEndUndef);

  auto funcReturn = *funcOp.getOps<func::ReturnOp>().begin();
  const AsmParserState::OperationDefinition *funcReturnDefinition =
      parseState.getOpDef(funcReturn);
  ASSERT_TRUE(funcReturnDefinition);

  const std::pair expectedStartReturn{4u, 3u};
  const std::pair expectedEndReturn{4u, 14u};
  const std::pair expectedScopeEndReturn{4u, 40u};
  ASSERT_EQ(sourceMgr.getLineAndColumn(funcReturnDefinition->loc.Start),
            expectedStartReturn);
  ASSERT_EQ(sourceMgr.getLineAndColumn(funcReturnDefinition->loc.End),
            expectedEndReturn);
  ASSERT_EQ(funcReturnDefinition->loc.Start,
            funcReturnDefinition->scopeLoc.Start);
  ASSERT_EQ(sourceMgr.getLineAndColumn(funcReturnDefinition->scopeLoc.End),
            expectedScopeEndReturn);
}
} // namespace
