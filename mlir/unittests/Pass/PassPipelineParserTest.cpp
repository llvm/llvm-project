//===- PassPipelineParserTest.cpp - Pass Parser unit tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <memory>

using namespace mlir;
using namespace mlir::detail;

namespace {
TEST(PassPipelineParserTest, InvalidOpAnchor) {
  // Helper functor used to parse a pipeline and check that it results in the
  // provided error message.
  auto checkParseFailure = [](StringRef pipeline, StringRef expectedErrorMsg) {
    std::string errorMsg;
    {
      llvm::raw_string_ostream os(errorMsg);
      FailureOr<OpPassManager> result = parsePassPipeline(pipeline, os);
      EXPECT_TRUE(failed(result));
    }
    EXPECT_TRUE(StringRef(errorMsg).contains(expectedErrorMsg));
  };

  // Handle parse errors when the anchor is incorrectly structured.
  StringRef anchorErrorMsg =
      "expected pass pipeline to be wrapped with the anchor operation type";
  checkParseFailure("module", anchorErrorMsg);
  checkParseFailure("()", anchorErrorMsg);
  checkParseFailure("module(", anchorErrorMsg);
  checkParseFailure("module)", anchorErrorMsg);
}

} // namespace
