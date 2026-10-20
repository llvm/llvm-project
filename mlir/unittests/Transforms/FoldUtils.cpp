//===- FoldUtils.cpp - Operation folding tests --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/FoldUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "gtest/gtest.h"

#include "../../test/lib/Dialect/Test/TestDialect.h"

using namespace mlir;

namespace {
static std::string region(StringRef body) {
  return ("%r = \"test.one_region_with_recursive_memory_effects\"() ({\n" +
          body + "\n\"test.return\"() : () -> ()\n}) : () -> i32\n")
      .str();
}

class OperationFolderTest : public testing::Test {
protected:
  OperationFolderTest() { context.loadDialect<test::TestDialect>(); }
  MLIRContext context;
};

TEST_F(OperationFolderTest, ConstantHoistingNotifiesListener) {
  // Constant CSE must report moves and replacements through the driver's
  // listener chain, including changes to users of the replaced constant.
  auto moduleOp = parseSourceString<ModuleOp>(region(R"mlir(
    %a = "test.constant"() {value = 1 : i32} : () -> i32
    %b = "test.constant"() {value = 1 : i32} : () -> i32
    "test.use"(%a, %b) : (i32, i32) -> ()
  )mlir"),
                                              &context);
  ASSERT_TRUE(moduleOp);
  struct Listener : RewriterBase::Listener {
    unsigned moved = 0, erased = 0, modified = 0;
    void notifyOperationInserted(Operation *op,
                                 OpBuilder::InsertPoint previous) override {
      if (op->hasTrait<OpTrait::ConstantLike>()) {
        EXPECT_TRUE(previous.isSet());
        EXPECT_NE(previous.getBlock(), op->getBlock());
        ++moved;
      }
    }
    void notifyOperationErased(Operation *op) override {
      if (op->hasTrait<OpTrait::ConstantLike>())
        ++erased;
    }
    void notifyOperationModified(Operation *op) override {
      if (op->getName().getStringRef() == "test.use")
        ++modified;
    }
  } listener;
  GreedyRewriteConfig config;
  config.setListener(&listener)
      .enableFolding(false)
      .enableConstantCSE(true)
      .setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
  ASSERT_TRUE(succeeded(
      applyPatternsGreedily(*moduleOp, RewritePatternSet(&context), config)));
  EXPECT_EQ(listener.moved, 1u);
  EXPECT_EQ(listener.erased, 1u);
  EXPECT_EQ(listener.modified, 1u);
  EXPECT_TRUE(succeeded(verify(*moduleOp)));
}

TEST_F(OperationFolderTest, ConstantRehoistingNotifiesListener) {
  // A constant already owned by the folder must report each move, whether
  // rehoisting is requested by constant insertion or by folding.
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    %a = "test.constant"() {value = 1 : i32} : () -> i32
    "test.use"(%a) : (i32) -> ()
  )mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);
  struct Listener : RewriterBase::Listener {
    unsigned moved = 0;
    void notifyOperationInserted(Operation *op,
                                 OpBuilder::InsertPoint previous) override {
      EXPECT_TRUE(previous.isSet());
      EXPECT_EQ(previous.getBlock(), op->getBlock());
      ++moved;
    }
  } listener;
  Operation *constant = &moduleOp->getBody()->front();
  Operation *user = constant->getNextNode();
  OperationFolder folder(&context, &listener);
  EXPECT_EQ(folder.insertKnownConstant(constant),
            OperationFolder::ConstantInsertionResult::Unchanged);
  // Inserting an operation ahead of the constant requires rehoisting
  // without changing the constant's users or value.
  OpBuilder builder(&context);
  builder.setInsertionPoint(constant);
  OperationState state(builder.getUnknownLoc(), "test.side_effect_op");
  state.addTypes(builder.getI32Type());
  Operation *inserted = builder.create(state);
  EXPECT_EQ(folder.insertKnownConstant(constant),
            OperationFolder::ConstantInsertionResult::Hoisted);
  EXPECT_EQ(listener.moved, 1u);
  inserted->moveBefore(constant);
  EXPECT_TRUE(failed(folder.tryToFold(constant)));
  EXPECT_EQ(listener.moved, 2u);
  EXPECT_EQ(user->getOperand(0), constant->getResult(0));
  EXPECT_TRUE(succeeded(verify(*moduleOp)));
}
} // namespace
