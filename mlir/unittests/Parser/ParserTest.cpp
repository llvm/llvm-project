//===- ParserTest.cpp -----------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Parser/Parser.h"
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
} // namespace
