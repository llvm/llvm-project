//===- Diagnostic.cpp - Dialect unit tests -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/TypeID.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <memory>

using namespace mlir;
using namespace mlir::detail;

namespace {

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

// Register counting handlers first so they receive diagnostics only after
// the verifier's registration is erased.

TEST(SourceMgrDiagnosticVerifierHandler, ScopedRegistration) {
  MLIRContext own, other;
  unsigned seen = 0;
  ScopedDiagnosticHandler counter(&other, [&](Diagnostic &) { ++seen; });
  llvm::SourceMgr sourceMgr;
  llvm::raw_null_ostream out;
  SourceMgrDiagnosticVerifierHandler verifier(
      sourceMgr, &own, out,
      SourceMgrDiagnosticVerifierHandler::Level::OnlyExpected);
  {
    std::unique_ptr<ScopedDiagnosticHandler> registration =
        verifier.registerInContext(&other);
    emitRemark(UnknownLoc::get(&other), "while registered");
    EXPECT_EQ(seen, 0u);
  }
  emitRemark(UnknownLoc::get(&other), "after the handle");
  EXPECT_EQ(seen, 1u);
  EXPECT_TRUE(succeeded(verifier.verify()));
}

TEST(SourceMgrDiagnosticVerifierHandler, ErasesOwnRegistration) {
  MLIRContext own;
  unsigned seen = 0;
  ScopedDiagnosticHandler counter(&own, [&](Diagnostic &) { ++seen; });
  {
    llvm::SourceMgr sourceMgr;
    llvm::raw_null_ostream out;
    SourceMgrDiagnosticVerifierHandler verifier(
        sourceMgr, &own, out,
        SourceMgrDiagnosticVerifierHandler::Level::OnlyExpected);
    emitRemark(UnknownLoc::get(&own), "while alive");
    EXPECT_EQ(seen, 0u);
  }
  emitRemark(UnknownLoc::get(&own), "after the verifier");
  EXPECT_EQ(seen, 1u);
}

TEST(SourceMgrDiagnosticVerifierHandler, OutlivesRegisteredContext) {
  // Match mlir-opt's per-buffer context lifetime.
  MLIRContext context;
  llvm::SourceMgr sourceMgr;
  llvm::raw_null_ostream out;
  SourceMgrDiagnosticVerifierHandler verifier(
      sourceMgr, &context, out,
      SourceMgrDiagnosticVerifierHandler::Level::OnlyExpected);
  {
    MLIRContext perBuffer;
    std::unique_ptr<ScopedDiagnosticHandler> registration =
        verifier.registerInContext(&perBuffer);
    emitRemark(UnknownLoc::get(&perBuffer), "in the temporary context");
  }
  emitRemark(UnknownLoc::get(&context), "verifier still usable");
  EXPECT_TRUE(succeeded(verifier.verify()));
}

} // namespace
