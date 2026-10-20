//===- GreedyPatternRewriteDriver.cpp - Greedy rewrite tests -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/FoldUtils.h"
#include "gtest/gtest.h"

#include "../../test/lib/Dialect/Test/TestDialect.h"

using namespace mlir;

namespace {
static std::string region(StringRef body) {
  return ("%r = \"test.one_region_with_recursive_memory_effects\"() ({\n" +
          body + "\n\"test.return\"() : () -> ()\n}) : () -> i32\n")
      .str();
}

constexpr StringLiteral effect =
    "%e = \"test.side_effect_op\"() {effects = [{effect = \"write\"}]} "
    ": () -> i32\n";

class RewriteRoot : public RewritePattern {
public:
  RewriteRoot(MLIRContext *context, Operation *root,
              function_ref<void(PatternRewriter &)> rewrite)
      : RewritePattern(root->getName().getStringRef(), 1, context), root(root),
        rewrite(rewrite) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op != root || applied)
      return failure();
    applied = true;
    rewrite(rewriter);
    return success();
  }

private:
  Operation *root;
  function_ref<void(PatternRewriter &)> rewrite;
  mutable bool applied = false;
};

class GreedyDeadnessTest : public testing::Test {
protected:
  GreedyDeadnessTest() { context.loadDialect<test::TestDialect>(); }

  void run(ModuleOp moduleOp, function_ref<void(PatternRewriter &)> rewrite) {
    RewritePatternSet patterns(&context);
    patterns.add<RewriteRoot>(&context, &moduleOp.getBody()->front(), rewrite);
    GreedyRewriteConfig config;
    // Query the outer operation before the rewrite changes its nested IR.
    // This populates the cache so the rewrite must invalidate stale entries.
    config.setUseTopDownTraversal(true)
        .enableFolding(false)
        .enableConstantCSE(false)
        .setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    ASSERT_TRUE(succeeded(
        applyPatternsGreedily(moduleOp, std::move(patterns), config)));
    EXPECT_TRUE(succeeded(verify(moduleOp)));
  }

  MLIRContext context;
};

TEST_F(GreedyDeadnessTest, EraseLastEffect) {
  // Erasing the only write must invalidate cached live effects in both
  // enclosing operations, so the driver can delete them.
  auto moduleOp = parseSourceString<ModuleOp>(region(region(effect)), &context);
  ASSERT_TRUE(moduleOp);
  Operation *outer = &moduleOp->getBody()->front();
  Operation *inner = &outer->getRegion(0).front().front();
  Operation *write = &inner->getRegion(0).front().front();
  run(*moduleOp, [&](PatternRewriter &rewriter) { rewriter.eraseOp(write); });
  EXPECT_TRUE(moduleOp->getBody()->empty());
}

TEST_F(GreedyDeadnessTest, ModifyLastEffect) {
  // Changing the write to an effect-free operation must make both enclosing
  // operations dead, even though no operation was erased by the pattern.
  auto moduleOp = parseSourceString<ModuleOp>(region(region(effect)), &context);
  ASSERT_TRUE(moduleOp);
  Operation *outer = &moduleOp->getBody()->front();
  Operation *inner = &outer->getRegion(0).front().front();
  Operation *write = &inner->getRegion(0).front().front();
  run(*moduleOp, [&](PatternRewriter &rewriter) {
    rewriter.modifyOpInPlace(
        write, [&] { write->setAttr("effects", rewriter.getArrayAttr({})); });
  });
  EXPECT_TRUE(moduleOp->getBody()->empty());
}

TEST_F(GreedyDeadnessTest, UsedInnerResultDoesNotPreventOuterDeletion) {
  // Internal uses do not keep the unused outer operation live after its last
  // write effect is removed. Cached summaries must describe effects, not uses.
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    %outer = "test.one_region_with_recursive_memory_effects"() ({
      %inner = "test.one_region_with_recursive_memory_effects"() ({
        %v = "test.side_effect_op"() {effects = [{effect = "write"}]} : () -> i32
        "test.return"(%v) : (i32) -> ()
      }) : () -> i32
      "test.return"(%inner) : (i32) -> ()
    }) : () -> i32
  )mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);
  Operation *outer = &moduleOp->getBody()->front();
  Operation *inner = &outer->getRegion(0).front().front();
  Operation *write = &inner->getRegion(0).front().front();
  ASSERT_FALSE(inner->use_empty());
  run(*moduleOp, [&](PatternRewriter &rewriter) {
    rewriter.modifyOpInPlace(
        write, [&] { write->setAttr("effects", rewriter.getArrayAttr({})); });
  });
  EXPECT_TRUE(moduleOp->getBody()->empty());
}

TEST_F(GreedyDeadnessTest, InsertEffectIntoDeadRegion) {
  // The first query caches the empty inner operation as dead before finding
  // the write beside it. Inserting a write into it must invalidate that result;
  // erasing the original write must not cause the new write to be deleted.
  auto moduleOp =
      parseSourceString<ModuleOp>(region(region("") + effect.str()), &context);
  ASSERT_TRUE(moduleOp);
  Operation *outer = &moduleOp->getBody()->front();
  Operation *inner = &outer->getRegion(0).front().front();
  Operation *write = inner->getNextNode();
  run(*moduleOp, [&](PatternRewriter &rewriter) {
    rewriter.setInsertionPointToStart(&inner->getRegion(0).front());
    rewriter.clone(*write);
    rewriter.eraseOp(write);
  });
  ASSERT_FALSE(moduleOp->getBody()->empty());
  ASSERT_FALSE(outer->getRegion(0).front().empty());
  EXPECT_EQ(&outer->getRegion(0).front().front(), inner);
  EXPECT_EQ(inner->getRegion(0).front().front().getName().getStringRef(),
            "test.side_effect_op");
}

TEST_F(GreedyDeadnessTest, MoveEffectBetweenRegions) {
  // Moving the only write must invalidate both parents: the destination must
  // survive, and the now-empty source must be deleted.
  std::string empty = region("");
  std::string live = region(effect);
  live.replace(0, 2, "%s");
  auto moduleOp = parseSourceString<ModuleOp>(region(empty + live), &context);
  ASSERT_TRUE(moduleOp);
  Operation *outer = &moduleOp->getBody()->front();
  Operation *target = &outer->getRegion(0).front().front();
  Operation *source = target->getNextNode();
  Operation *write = &source->getRegion(0).front().front();
  run(*moduleOp, [&](PatternRewriter &rewriter) {
    rewriter.moveOpBefore(write, &target->getRegion(0).front(),
                          target->getRegion(0).front().begin());
  });
  ASSERT_FALSE(moduleOp->getBody()->empty());
  EXPECT_EQ(outer->getRegion(0).front().getOperations().size(), 2u);
  EXPECT_EQ(&outer->getRegion(0).front().front(), target);
  EXPECT_EQ(&target->getRegion(0).front().front(), write);
}

TEST_F(GreedyDeadnessTest, MoveBlocksBetweenRegions) {
  // Moving whole blocks must update the effects of both enclosing operations,
  // just as moving a write does. Keep the new owner of the live block and
  // delete the operation that receives the empty block.
  std::string empty = region("");
  std::string live = region(effect);
  live.replace(0, 2, "%s");
  auto moduleOp = parseSourceString<ModuleOp>(region(empty + live), &context);
  ASSERT_TRUE(moduleOp);
  Operation *outer = &moduleOp->getBody()->front();
  Operation *target = &outer->getRegion(0).front().front();
  Operation *source = target->getNextNode();
  Block *emptyBlock = &target->getRegion(0).front();
  Block *liveBlock = &source->getRegion(0).front();
  run(*moduleOp, [&](PatternRewriter &rewriter) {
    rewriter.moveBlockBefore(liveBlock, &target->getRegion(0),
                             target->getRegion(0).begin());
    rewriter.moveBlockBefore(emptyBlock, &source->getRegion(0),
                             source->getRegion(0).begin());
  });
  ASSERT_FALSE(moduleOp->getBody()->empty());
  EXPECT_EQ(outer->getRegion(0).front().getOperations().size(), 2u);
  EXPECT_EQ(&outer->getRegion(0).front().front(), target);
  EXPECT_EQ(&target->getRegion(0).front(), liveBlock);
}

TEST_F(GreedyDeadnessTest, ConstantHoistingNotifiesListener) {
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

TEST_F(GreedyDeadnessTest, ConstantRehoistingNotifiesListener) {
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

TEST_F(GreedyDeadnessTest, UsesAreNotCached) {
  // Change uses without invalidating the effect cache. Each query must check
  // current uses, including a new use added after caching a dead result.
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    %v = "test.side_effect_op"() : () -> i32
    "test.use"(%v) : (i32) -> ()
  )mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);
  Operation *value = &moduleOp->getBody()->front();
  DeadnessCache cache;
  EXPECT_FALSE(cache.isOpTriviallyDead(value));
  value->getNextNode()->erase();
  EXPECT_TRUE(cache.isOpTriviallyDead(value));
  OpBuilder builder(&context);
  builder.setInsertionPointAfter(value);
  OperationState state(builder.getUnknownLoc(), "test.use");
  state.addOperands(value->getResult(0));
  builder.create(state);
  EXPECT_FALSE(cache.isOpTriviallyDead(value));
}

TEST_F(GreedyDeadnessTest, TerminatorsRemainLiveAfterRecursiveQuery) {
  // The outer query caches the terminator's effects as dead. A direct query
  // must still reject deleting the terminator itself.
  auto moduleOp = parseSourceString<ModuleOp>(region(""), &context);
  ASSERT_TRUE(moduleOp);
  Operation *outer = &moduleOp->getBody()->front();
  Operation *terminator = &outer->getRegion(0).front().back();
  DeadnessCache cache;
  EXPECT_TRUE(cache.isOpTriviallyDead(outer));
  EXPECT_FALSE(cache.isOpTriviallyDead(terminator));
  EXPECT_FALSE(cache.isOpTriviallyDead(*moduleOp));
}
} // namespace
