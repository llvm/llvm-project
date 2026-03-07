//===- BuiltinDialectVersionTest.cpp - Test builtin dialect versioning ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;

TEST(BuiltinDialectVersionTest, RejectFutureVersion) {
  MLIRContext ctx;
  auto module = ModuleOp::create(UnknownLoc::get(&ctx));

  auto writeBytecode = [&](std::string &out,
                           std::optional<int64_t> version =
                               std::nullopt) -> LogicalResult {
    BytecodeWriterConfig config;
    if (version)
      config.setDialectVersion<BuiltinDialect>(
          std::make_unique<BuiltinDialectVersion>(*version));
    llvm::raw_string_ostream os(out);
    return writeBytecodeToFile(module, os, config);
  };

  std::string bytecode;
  ASSERT_TRUE(succeeded(writeBytecode(bytecode)));
  std::string bytecodeWithFutureVersion;
  ASSERT_TRUE(succeeded(writeBytecode(bytecodeWithFutureVersion, 99)));

  EXPECT_NE(bytecode, bytecodeWithFutureVersion);

  std::string warning;
  ScopedDiagnosticHandler handler(&ctx, [&](Diagnostic &diag) {
    if (diag.getSeverity() == DiagnosticSeverity::Warning)
      warning = diag.str();
    return success();
  });

  auto parsed =
      parseSourceString(StringRef(bytecodeWithFutureVersion),
                        ParserConfig(&ctx, /*verifyAfterParse=*/true));
  EXPECT_TRUE(warning.find("reading newer builtin dialect version") !=
              std::string::npos);

  module->erase();
}
