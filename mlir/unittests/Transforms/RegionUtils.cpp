//===- RegionUtils.cpp - Region utility unit tests ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;

namespace {
TEST(RegionUtilsTest, ReportsDeadBlockArgumentRemoval) {
  MLIRContext context;
  context.getOrLoadDialect<cf::ControlFlowDialect>();
  context.getOrLoadDialect<func::FuncDialect>();

  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @f(%arg0: i32) {
      cf.br ^bb1(%arg0 : i32)
    ^bb1(%unused: i32):
      return
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);

  func::FuncOp func = module->lookupSymbol<func::FuncOp>("f");
  ASSERT_TRUE(func);
  Region &region = func.getBody();
  ASSERT_EQ(region.getBlocks().size(), 2u);
  ASSERT_EQ(region.back().getNumArguments(), 1u);

  IRRewriter rewriter(&context);
  EXPECT_TRUE(
      succeeded(runRegionDCE(rewriter, func.getOperation()->getRegions())));
  EXPECT_EQ(region.back().getNumArguments(), 0u);
  EXPECT_EQ(region.front().getTerminator()->getNumOperands(), 0u);
}
} // namespace
