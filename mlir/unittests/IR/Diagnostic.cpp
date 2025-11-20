//===- Diagnostic.cpp - Dialect unit tests -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/TypeID.h"
#include "gtest/gtest.h"

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

} // namespace
