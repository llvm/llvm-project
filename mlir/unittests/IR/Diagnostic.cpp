//===- Diagnostic.cpp - Dialect unit tests -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/TypeID.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::detail;

namespace {

static std::string emitSourceMgrDiagnostic(Location loc,
                                           llvm::SourceMgr &sourceMgr) {
  std::string output;
  llvm::raw_string_ostream os(output);
  SourceMgrDiagnosticHandler handler(sourceMgr, loc->getContext(), os,
                                     [](Location) { return true; });
  emitError(loc, "message");
  return output;
}

static std::string emitSourceMgrDiagnostic(Location loc, StringRef source,
                                           StringRef filename = "test.mlir") {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(source, filename), SMLoc());
  return emitSourceMgrDiagnostic(loc, sourceMgr);
}

TEST(DiagnosticLifetime, TestCopiesConstCharStar) {
  const auto *expectedMessage = "Error 1, don't mutate this";

  // Copy expected message into a mutable container, and call the constructor.
  std::string myStr(expectedMessage);

  mlir::MLIRContext context;
  Diagnostic diagnostic(mlir::UnknownLoc::get(&context),
                        DiagnosticSeverity::Note);
  diagnostic << myStr.c_str();

  // Mutate underlying pointer, but ensure diagnostic still has orig. message
  myStr[0] = '^';

  std::string resultMessage;
  llvm::raw_string_ostream stringStream(resultMessage);
  diagnostic.print(stringStream);
  ASSERT_STREQ(expectedMessage, resultMessage.c_str());
}

TEST(DiagnosticLifetime, TestLazyCopyStringLiteral) {
  char charArr[21] = "Error 1, mutate this";
  mlir::MLIRContext context;
  Diagnostic diagnostic(mlir::UnknownLoc::get(&context),
                        DiagnosticSeverity::Note);

  // Diagnostic contains optimization which assumes string literals are
  // represented by `const char[]` type. This is imperfect as we can sometimes
  // trick the type system as seen below.
  //
  // Still we use this to check the diagnostic is lazily storing the pointer.
  auto addToDiagnosticAsConst = [&diagnostic](const char(&charArr)[21]) {
    diagnostic << charArr;
  };
  addToDiagnosticAsConst(charArr);

  // Mutate the underlying pointer and ensure the string does change
  charArr[0] = '^';

  std::string resultMessage;
  llvm::raw_string_ostream stringStream(resultMessage);
  diagnostic.print(stringStream);
  ASSERT_STREQ("^rror 1, mutate this", resultMessage.c_str());
}

TEST(SourceMgrDiagnosticHandler, PreservesFileLineColLocDiagnostic) {
  MLIRContext context;
  Location loc = FileLineColLoc::get(&context, "test.mlir", 1, 3);

  EXPECT_EQ(emitSourceMgrDiagnostic(loc, "abcdef\n"),
            "test.mlir:1:3: error: message\n"
            "abcdef\n"
            "  ^\n");
}

TEST(SourceMgrDiagnosticHandler, HighlightsSameLineFileLineColRange) {
  MLIRContext context;
  auto filename = StringAttr::get(&context, "test.mlir");
  Location loc = FileLineColRange::get(filename, 1, 2, 1, 5);

  EXPECT_EQ(emitSourceMgrDiagnostic(loc, "abcdef\n"),
            "test.mlir:1:2: error: message\n"
            "abcdef\n"
            " ^~~\n");
}

TEST(SourceMgrDiagnosticHandler, HighlightsFirstLineOfMultilineRange) {
  MLIRContext context;
  auto filename = StringAttr::get(&context, "test.mlir");
  Location loc = FileLineColRange::get(filename, 1, 2, 2, 4);

  EXPECT_EQ(emitSourceMgrDiagnostic(loc, "abcdef\nsecond\n"),
            "test.mlir:1:2: error: message\n"
            "abcdef\n"
            " ^~~~~\n");
}

TEST(SourceMgrDiagnosticHandler, FindsNestedFileLineColRange) {
  MLIRContext context;
  auto filename = StringAttr::get(&context, "test.mlir");
  Location range = FileLineColRange::get(filename, 1, 2, 1, 5);
  Location loc = NameLoc::get(StringAttr::get(&context, "nested"), range);

  EXPECT_EQ(emitSourceMgrDiagnostic(loc, "abcdef\n"),
            "test.mlir:1:2: error: message\n"
            "abcdef\n"
            " ^~~\n");
}

TEST(SourceMgrDiagnosticHandler, CrossFileNestedLocationsRemainSeparate) {
  MLIRContext context;
  auto firstFilename = StringAttr::get(&context, "first.mlir");
  auto secondFilename = StringAttr::get(&context, "second.mlir");
  Location first = FileLineColRange::get(firstFilename, 1, 2, 1, 5);
  Location second = FileLineColRange::get(secondFilename, 1, 3, 1, 6);
  Location loc = FusedLoc::get(&context, {first, second});

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer("abcdef\n", "first.mlir"), SMLoc());
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer("ghijkl\n", "second.mlir"), SMLoc());

  EXPECT_EQ(emitSourceMgrDiagnostic(loc, sourceMgr),
            "first.mlir:1:2: error: message\n"
            "abcdef\n"
            " ^~~\n");
}

TEST(SourceMgrDiagnosticHandler, InvalidRangeFallsBackToStartPoint) {
  MLIRContext context;
  auto filename = StringAttr::get(&context, "test.mlir");
  Location loc = FileLineColRange::get(filename, 1, 5, 1, 2);

  EXPECT_EQ(emitSourceMgrDiagnostic(loc, "abcdef\n"),
            "test.mlir:1:5: error: message\n"
            "abcdef\n"
            "    ^\n");
}

TEST(SourceMgrDiagnosticHandler, IncompleteRangeFallsBackToStartPoint) {
  MLIRContext context;
  auto filename = StringAttr::get(&context, "test.mlir");
  Location loc = FileLineColRange::get(filename, 1, 3, 0, 0);

  EXPECT_EQ(emitSourceMgrDiagnostic(loc, "abcdef\n"),
            "test.mlir:1:3: error: message\n"
            "abcdef\n"
            "  ^\n");
}

TEST(SourceMgrDiagnosticHandler, ZeroStartFallsBackToLocation) {
  MLIRContext context;
  auto filename = StringAttr::get(&context, "test.mlir");
  Location loc = FileLineColRange::get(filename, 0, 0, 1, 5);

  EXPECT_EQ(emitSourceMgrDiagnostic(loc, "abcdef\n"),
            "test.mlir:0:0: error: message\n");
}

TEST(SourceMgrDiagnosticHandler, UnresolvableEndFallsBackToStartPoint) {
  MLIRContext context;
  auto filename = StringAttr::get(&context, "test.mlir");
  Location loc = FileLineColRange::get(filename, 1, 2, 1, 99);

  EXPECT_EQ(emitSourceMgrDiagnostic(loc, "abcdef\n"),
            "test.mlir:1:2: error: message\n"
            "abcdef\n"
            " ^\n");
}

TEST(SourceMgrDiagnosticHandler, UnresolvableRangeFallsBackToLocation) {
  MLIRContext context;
  auto filename = StringAttr::get(&context, "missing.mlir");
  Location loc = FileLineColRange::get(filename, 1, 2, 1, 5);

  EXPECT_EQ(emitSourceMgrDiagnostic(loc, "abcdef\n"),
            "missing.mlir:1:2: error: message\n");
}

} // namespace
