//===- LivenessTest.cpp - Liveness analysis unit tests --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

using namespace mlir;

namespace {

const StringLiteral moduleStr = R"mlir(
"test.func"() ({
^bb0:
  %0 = "test.def"() : () -> i32
  "test.br"()[^bb1] : () -> ()
^bb1:
  "test.use"(%0) : (i32) -> ()
  "test.ret"() : () -> ()
}) : () -> ()
)mlir";

TEST(LivenessTest, CoveredBlocks) {
  MLIRContext context;
  context.allowUnregisteredDialects();
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(moduleStr, &context);
  ASSERT_TRUE(module);

  Region &region = module->getBody()->getOperations().front().getRegion(0);
  Block &entryBlock = region.front();
  Block &secondBlock = region.back();

  Liveness liveness(module.get());

  // Blocks that existed when the analysis was constructed are covered.
  EXPECT_NE(liveness.getLiveness(&entryBlock), nullptr);
  EXPECT_NE(liveness.getLiveness(&secondBlock), nullptr);

  Value def = entryBlock.front().getResult(0);
  EXPECT_TRUE(liveness.getLiveOut(&entryBlock).contains(def));
  EXPECT_TRUE(liveness.getLiveIn(&secondBlock).contains(def));
  EXPECT_FALSE(liveness.isDeadAfter(def, &entryBlock.front()));
  EXPECT_TRUE(liveness.isDeadAfter(def, &secondBlock.front()));
  EXPECT_EQ(liveness.resolveLiveness(def).size(), 3u);
}

TEST(LivenessTest, BlockCreatedAfterConstructionIsNotCovered) {
  MLIRContext context;
  context.allowUnregisteredDialects();
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(moduleStr, &context);
  ASSERT_TRUE(module);

  Region &region = module->getBody()->getOperations().front().getRegion(0);
  Block &secondBlock = region.back();

  Liveness liveness(module.get());

  // Blocks created after the analysis was constructed (e.g. by a rewrite)
  // are not covered; `getLiveness` returns nullptr for them.
  Block *newBlock = secondBlock.splitBlock(&secondBlock.front());
  EXPECT_EQ(liveness.getLiveness(newBlock), nullptr);
  EXPECT_NE(liveness.getLiveness(&secondBlock), nullptr);
}

} // namespace
