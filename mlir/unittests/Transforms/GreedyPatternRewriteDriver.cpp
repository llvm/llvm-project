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
#include "mlir/Interfaces/Utils/MemorySlotUtils.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"
#include <new> // placement new in ReusePendingBlockAddress

#include "../../test/lib/Dialect/Test/TestDialect.h"
#include "../../test/lib/Dialect/Test/TestOps.h"

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

struct WriteToMemRef
    : MemoryEffectOpInterface::ExternalModel<WriteToMemRef,
                                             test::TestTokenAnyTypeOp> {
  void
  getEffects(Operation *op,
             SmallVectorImpl<MemoryEffects::EffectInstance> &effects) const {
    if (isa<MemRefType>(op->getOperand(0).getType()))
      effects.emplace_back(MemoryEffects::Write::get(), &op->getOpOperand(0));
  }
};

TEST_F(GreedyDeadnessTest, ReplacementOwnerChangesBlockArgumentType) {
  test::TestTokenAnyTypeOp::attachInterface<WriteToMemRef>(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
    %outer = "test.one_region_with_recursive_memory_effects"() ({
    ^bb0(%arg: tensor<4xi32>):
      "test.token.any_type"(%arg) : (tensor<4xi32>) -> ()
      %keep = "test.side_effect_op"() {effects = [{effect = "write"}]} : () -> i32
      "test.return"() : () -> ()
    }) : () -> i32
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  Operation *outer = &module->getBody()->front();
  Block &body = outer->getRegion(0).front();
  Operation *consumer = &body.front();
  Operation *originalWrite = consumer->getNextNode();
  run(*module, [&](PatternRewriter &rewriter) {
    // The consumer gains a write effect when its block argument changes type.
    // Only the owner receives a modification notification.
    outer = memoryslot::replaceWithNewResults(rewriter, outer,
                                              outer->getResultTypes());
    rewriter.modifyOpInPlace(outer, [&] {
      body.getArgument(0).setType(MemRefType::get({4}, rewriter.getI32Type()));
    });
    rewriter.eraseOp(originalWrite);
    EXPECT_FALSE(mlir::isOpTriviallyDead(outer));
  });
  ASSERT_FALSE(module->getBody()->empty());
  ASSERT_EQ(&module->getBody()->front(), outer);
  EXPECT_EQ(&body.front(), consumer);
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

TEST_F(GreedyDeadnessTest, InPlaceUpdateOfNestedIR) {
  // The first query caches `inner` as dead while the sibling write keeps
  // `outer` live. Changing the outer block argument's type makes the nested
  // consumer write. Invalidating only `outer` and its ancestors would leave
  // `inner` cached as dead and allow the driver to erase the live `outer`.
  test::TestTokenAnyTypeOp::attachInterface<WriteToMemRef>(context);
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    %outer = "test.one_region_with_recursive_memory_effects"() ({
    ^bb0(%arg: tensor<4xi32>):
      %inner = "test.one_region_with_recursive_memory_effects"() ({
        "test.token.any_type"(%arg) : (tensor<4xi32>) -> ()
        "test.return"() : () -> ()
      }) : () -> i32
      %keep = "test.side_effect_op"() {effects = [{effect = "write"}]} : () -> i32
      "test.return"() : () -> ()
    }) : () -> i32
  )mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);
  Operation *outer = &moduleOp->getBody()->front();
  Operation *inner = &outer->getRegion(0).front().front();
  Operation *nested = &inner->getRegion(0).front().front();
  Operation *write = inner->getNextNode();
  run(*moduleOp, [&](PatternRewriter &rewriter) {
    rewriter.eraseOp(write);
    // Modify the owner's block argument, rather than directly changing a
    // nested op's attributes without notifying the rewriter about that op.
    rewriter.modifyOpInPlace(outer, [&] {
      outer->getRegion(0).front().getArgument(0).setType(
          MemRefType::get({4}, rewriter.getI32Type()));
    });
  });
  // The nested write is live, so nothing may be deleted.
  ASSERT_FALSE(moduleOp->getBody()->empty());
  ASSERT_EQ(&moduleOp->getBody()->front(), outer);
  ASSERT_EQ(&outer->getRegion(0).front().front(), inner);
  EXPECT_EQ(&inner->getRegion(0).front().front(), nested);
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

TEST_F(GreedyDeadnessTest, MoveBlockFromDetachedRegion) {
  auto module =
      parseSourceString<ModuleOp>(region(region("") + effect.str()), &context);
  ASSERT_TRUE(module);
  Block &body = module->getBody()->front().getRegion(0).front();
  Operation *target = &body.front();
  Operation *originalWrite = target->getNextNode();
  run(*module, [&](PatternRewriter &rewriter) {
    Region staged;
    Block *block = rewriter.createBlock(&staged);
    rewriter.clone(*originalWrite);
    rewriter.clone(target->getRegion(0).front().back());
    Region &destination = target->getRegion(0);
    Block *oldBlock = &destination.front();
    rewriter.moveBlockBefore(block, &destination, destination.begin());
    rewriter.eraseBlock(oldBlock);
    rewriter.eraseOp(originalWrite);
  });
  ASSERT_FALSE(module->getBody()->empty());
  ASSERT_EQ(&body.front(), target);
  EXPECT_EQ(target->getRegion(0).front().front().getName().getStringRef(),
            "test.side_effect_op");
}

TEST_F(GreedyDeadnessTest, MultipleBlocksAgreeWithUncachedQuery) {
  // A write in either block must keep the parent live on repeated queries.
  // An effect-free parent becomes dead only after both blocks are checked.
  for (int writeBlock : {-1, 0, 1}) {
    std::string body = "\"test.any_cond\"() ({\n";
    for (int block = 0; block < 2; ++block) {
      body += "^bb" + std::to_string(block) + ":\n";
      if (block == writeBlock)
        body += effect.str();
      body += "\"test.return\"() : () -> ()\n";
    }
    body += "}) : () -> ()\n";
    auto moduleOp = parseSourceString<ModuleOp>(body, &context);
    ASSERT_TRUE(moduleOp);
    Operation *root = &moduleOp->getBody()->front();
    DeadnessCache cache;
    EXPECT_EQ(wouldOpBeTriviallyDead(root), writeBlock == -1);
    EXPECT_EQ(cache.isOpTriviallyDead(root), writeBlock == -1);
    EXPECT_EQ(cache.isOpTriviallyDead(root), writeBlock == -1);
  }
}

TEST_F(GreedyDeadnessTest, EmptyRegionAgreesWithUncachedQuery) {
  // A recursive-effect operation with no blocks has no nested live effects.
  auto moduleOp = parseSourceString<ModuleOp>(
      "\"test.any_cond\"() ({}) : () -> ()", &context);
  ASSERT_TRUE(moduleOp);
  Operation *root = &moduleOp->getBody()->front();
  DeadnessCache cache;
  EXPECT_TRUE(wouldOpBeTriviallyDead(root));
  EXPECT_TRUE(cache.isOpTriviallyDead(root));
  EXPECT_TRUE(cache.isOpTriviallyDead(root));
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

TEST_F(GreedyDeadnessTest, SubtreeInvalidationSurvivesAncestorRequest) {
  for (bool fullFirst : {false, true}) {
    auto module = parseSourceString<ModuleOp>(region(region(effect)), &context);
    ASSERT_TRUE(module);
    Operation *outer = &module->getBody()->front();
    Operation *inner = &outer->getRegion(0).front().front();
    Operation *write = &inner->getRegion(0).front().front();
    DeadnessCache cache;
    ASSERT_FALSE(cache.isOpTriviallyDead(outer));
    write->removeAttr("effects");
    if (fullFirst)
      cache.scheduleInvalidation(inner);
    cache.scheduleAncestorInvalidation(inner);
    if (!fullFirst)
      cache.scheduleInvalidation(inner);
    cache.scheduleAncestorInvalidation(outer);
    EXPECT_TRUE(cache.isOpTriviallyDead(outer));
  }
}

TEST_F(GreedyDeadnessTest, BlockInvalidationPromotesEarlierAncestorRequest) {
  auto module = parseSourceString<ModuleOp>(region(effect), &context);
  ASSERT_TRUE(module);
  Block *block = module->getBody();
  Operation *owner = &block->front();
  Operation *leaf = &owner->getRegion(0).front().front();
  Attribute effects = leaf->removeAttr("effects");
  DeadnessCache cache;
  ASSERT_TRUE(cache.isOpTriviallyDead(leaf));

  // Expanding the block must invalidate the leaf even though the flush has
  // already processed the ancestor request for its owner.
  cache.scheduleAncestorInvalidation(owner);
  leaf->setAttr("effects", effects);
  cache.scheduleBlockInvalidation(block);
  ASSERT_FALSE(isOpTriviallyDead(leaf));
  EXPECT_FALSE(cache.isOpTriviallyDead(leaf));
}

TEST_F(GreedyDeadnessTest, EraseBlocksAndRegionOwnerBeforeDeadnessQuery) {
  auto module = parseSourceString<ModuleOp>(region(R"mlir(
        "test.two_region_op"() ({
          "test.return"() : () -> ()
        }, {
          "test.return"() : () -> ()
        }) : () -> ()
      )mlir" + effect.str()),
                                            &context);
  ASSERT_TRUE(module);
  Operation *outer = &module->getBody()->front();
  Operation *inner = &outer->getRegion(0).front().front();
  Operation *keep = inner->getNextNode();
  run(*module, [&](PatternRewriter &rewriter) {
    for (Region &region : inner->getRegions()) {
      Block &body = region.front();
      Block *replacement = rewriter.createBlock(&region);
      rewriter.moveOpBefore(&body.back(), replacement, replacement->end());
      rewriter.eraseBlock(&body);
    }
    rewriter.eraseOp(inner);
  });
  ASSERT_FALSE(module->getBody()->empty());
  EXPECT_EQ(&outer->getRegion(0).front().front(), keep);
}

TEST_F(GreedyDeadnessTest, MoveWhileCacheIsEmpty) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    %src = "test.one_region_with_recursive_memory_effects"() ({
      %first = "test.side_effect_op"() : () -> i32
      %write = "test.side_effect_op"() {effects = [{effect = "write"}]} : () -> i32
      "test.return"() : () -> ()
    }) : () -> i32
    %dst = "test.one_region_with_recursive_memory_effects"() ({
      "test.return"() : () -> ()
    }) : () -> i32
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  Operation *source = &module->getBody()->front();
  Operation *dest = source->getNextNode();
  Operation *first = &source->getRegion(0).front().front();
  Operation *write = first->getNextNode();
  DeadnessCache cache;
  ASSERT_TRUE(cache.isOpTriviallyDead(first));
  cache.notifyOperationErased(first);
  first->erase();

  Operation *previous = source->getParentOp();
  Block &body = dest->getRegion(0).front();
  source->moveBefore(&body, body.begin());
  cache.scheduleAncestorInvalidation(previous);
  cache.scheduleInvalidation(source);

  // A query below the moved operation must register its new ancestors.
  ASSERT_FALSE(cache.isOpTriviallyDead(write));
  write->removeAttr("effects");
  cache.scheduleInvalidation(dest);
  EXPECT_TRUE(cache.isOpTriviallyDead(write));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(GreedyDeadnessTest, ReusePendingBlockAddress) {
  auto module = parseSourceString<ModuleOp>(region(effect), &context);
  ASSERT_TRUE(module);
  Operation *owner = &module->getBody()->front();
  Operation *write = &owner->getRegion(0).front().front();
  DeadnessCache cache;
  ASSERT_FALSE(cache.isOpTriviallyDead(write));

  // Keep the owner requests pending while one block replaces another.
  cache.scheduleAncestorInvalidation(owner);
  cache.scheduleAncestorInvalidation(owner->getParentOp());

  Block block;
  cache.scheduleBlockInvalidation(&block);
  cache.notifyBlockErased(&block);
  // Construct the replacement at the cancelled block's address so the test
  // exercises address reuse on every run.
  block.~Block();
  new (&block) Block;

  write->moveBefore(&block, block.end());
  write->removeAttr("effects");
  cache.scheduleBlockInvalidation(&block);
  EXPECT_TRUE(cache.isOpTriviallyDead(write));
}

TEST_F(GreedyDeadnessTest, CloneEffectIntoAnotherParent) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    %src = "test.one_region_with_recursive_memory_effects"() ({
      %old = "test.side_effect_op"() {effects = [{effect = "write"}]} : () -> i32
      %keep = "test.side_effect_op"() {effects = [{effect = "write"}]} : () -> i32
      "test.return"() : () -> ()
    }) : () -> i32
    %dst = "test.one_region_with_recursive_memory_effects"() ({
      "test.return"() : () -> ()
    }) : () -> i32
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  Operation *source = &module->getBody()->front();
  Operation *dest = source->getNextNode();
  Operation *oldWrite = &source->getRegion(0).front().front();
  Operation *keep = oldWrite->getNextNode();
  DeadnessCache cache;
  ASSERT_FALSE(cache.isOpTriviallyDead(source));
  ASSERT_TRUE(cache.isOpTriviallyDead(dest));
  cache.scheduleInvalidation(source);
  cache.notifyOperationErased(oldWrite);
  oldWrite->erase();

  OpBuilder builder(&context);
  builder.setInsertionPointToStart(&dest->getRegion(0).front());
  cache.scheduleInvalidation(builder.clone(*keep));
  EXPECT_FALSE(cache.isOpTriviallyDead(dest));
  EXPECT_TRUE(succeeded(verify(*module)));
}
} // namespace
