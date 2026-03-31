//===- MemorySlotUtilsTest.cpp - MemorySlot utility tests -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/Utils/MemorySlotUtils.h"
#include "../../test/lib/Dialect/Test/TestDialect.h"
#include "../../test/lib/Dialect/Test/TestOps.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;

static Operation *createOp(MLIRContext &ctx, StringRef name,
                           TypeRange resultTypes = {},
                           unsigned numRegions = 0) {
  return Operation::create(UnknownLoc::get(&ctx), OperationName(name, &ctx),
                           resultTypes, /*operands=*/{}, NamedAttrList(),
                           /*properties=*/nullptr, /*successors=*/{},
                           numRegions);
}

//===----------------------------------------------------------------------===//
// updateTerminator
//===----------------------------------------------------------------------===//

TEST(MemorySlotUtilsTest, UpdateTerminatorAppendsReachingDef) {
  MLIRContext context;
  context.allowUnregisteredDialects();
  auto i32Ty = IntegerType::get(&context, 32);

  Operation *outerOp = createOp(context, "foo.outer", {}, /*numRegions=*/1);
  Block *block = new Block();
  outerOp->getRegion(0).push_back(block);

  Operation *defOp = createOp(context, "foo.def", {i32Ty});
  block->push_back(defOp);
  Operation *otherDefOp = createOp(context, "foo.other_def", {i32Ty});
  block->push_back(otherDefOp);
  Operation *terminator = createOp(context, "foo.terminator");
  block->push_back(terminator);

  DenseMap<Block *, Value> reachingAtBlockEnd;
  reachingAtBlockEnd[block] = defOp->getResult(0);

  // Pass otherDefOp as default, which should not be used since the map has an
  // entry for this block.
  memoryslot::updateTerminator(block, otherDefOp->getResult(0),
                               reachingAtBlockEnd);

  EXPECT_EQ(terminator->getNumOperands(), 1u);
  EXPECT_EQ(terminator->getOperand(0), defOp->getResult(0));

  outerOp->destroy();
}

TEST(MemorySlotUtilsTest, UpdateTerminatorUsesDefaultForMissingBlock) {
  MLIRContext context;
  context.allowUnregisteredDialects();
  auto i32Ty = IntegerType::get(&context, 32);

  Operation *outerOp = createOp(context, "foo.outer", {}, /*numRegions=*/1);
  Block *block = new Block();
  outerOp->getRegion(0).push_back(block);

  Operation *defOp = createOp(context, "foo.def", {i32Ty});
  block->push_back(defOp);
  Operation *otherDefOp = createOp(context, "foo.other_def", {i32Ty});
  block->push_back(otherDefOp);
  Operation *terminator = createOp(context, "foo.terminator");
  block->push_back(terminator);

  // Empty map: the default (defOp) should be used.
  DenseMap<Block *, Value> reachingAtBlockEnd;

  memoryslot::updateTerminator(block, defOp->getResult(0), reachingAtBlockEnd);

  EXPECT_EQ(terminator->getNumOperands(), 1u);
  EXPECT_EQ(terminator->getOperand(0), defOp->getResult(0));

  outerOp->destroy();
}

//===----------------------------------------------------------------------===//
// replaceWithNewResults
//===----------------------------------------------------------------------===//

TEST(MemorySlotUtilsTest, ReplaceWithNewResultsAddsResults) {
  MLIRContext context;
  context.allowUnregisteredDialects();
  auto i32Ty = IntegerType::get(&context, 32);
  auto i64Ty = IntegerType::get(&context, 64);
  auto f32Ty = Float32Type::get(&context);

  Operation *parent = createOp(context, "foo.parent", {}, /*numRegions=*/1);
  Block *block = new Block();
  parent->getRegion(0).push_back(block);

  Operation *op = createOp(context, "foo.op", {i32Ty});
  block->push_back(op);
  Operation *terminator = createOp(context, "foo.terminator");
  block->push_back(terminator);

  // Add two new results (i64 and f32) on top of the original i32.
  IRRewriter rewriter(&context);
  Operation *newOp =
      memoryslot::replaceWithNewResults(rewriter, op, {i32Ty, i64Ty, f32Ty});

  EXPECT_EQ(newOp->getNumResults(), 3u);
  EXPECT_EQ(newOp->getResult(0).getType(), i32Ty);
  EXPECT_EQ(newOp->getResult(1).getType(), i64Ty);
  EXPECT_EQ(newOp->getResult(2).getType(), f32Ty);
  EXPECT_EQ(newOp->getName().getStringRef(), "foo.op");

  parent->destroy();
}

TEST(MemorySlotUtilsTest, ReplaceWithNewResultsPreservesRegions) {
  MLIRContext context;
  context.allowUnregisteredDialects();
  auto i32Ty = IntegerType::get(&context, 32);

  Operation *parent = createOp(context, "foo.parent", {}, /*numRegions=*/1);
  Block *block = new Block();
  parent->getRegion(0).push_back(block);

  Operation *op = createOp(context, "foo.region_op", {}, /*numRegions=*/1);
  block->push_back(op);

  Block *innerBlock = new Block();
  op->getRegion(0).push_back(innerBlock);
  Operation *innerOp = createOp(context, "foo.inner");
  innerBlock->push_back(innerOp);

  Operation *terminator = createOp(context, "foo.terminator");
  block->push_back(terminator);

  IRRewriter rewriter(&context);
  Operation *newOp = memoryslot::replaceWithNewResults(rewriter, op, {i32Ty});

  EXPECT_EQ(newOp->getNumRegions(), 1u);
  EXPECT_FALSE(newOp->getRegion(0).empty());
  Operation &movedInnerOp = newOp->getRegion(0).front().front();
  EXPECT_EQ(movedInnerOp.getName().getStringRef(), "foo.inner");

  parent->destroy();
}

TEST(MemorySlotUtilsTest, ReplaceWithNewResultsPreservesProperties) {
  MLIRContext context;
  context.loadDialect<test::TestDialect>();

  const char *src = R"mlir(
    "builtin.module"() ({
      test.with_properties a = 42, b = "hello", c = "world",
                           flag = true, array = [1, 2, 3], array32 = [4, 5]
    }) : () -> ()
  )mlir";

  auto module = parseSourceString<ModuleOp>(src, &context);
  ASSERT_TRUE(!!module);

  auto &opInModule = module->getBody()->front();
  ASSERT_EQ(opInModule.getName().getStringRef(), "test.with_properties");

  auto i32Ty = IntegerType::get(&context, 32);
  IRRewriter rewriter(&context);
  Operation *newOp =
      memoryslot::replaceWithNewResults(rewriter, &opInModule, {i32Ty});

  std::string newStr;
  {
    llvm::raw_string_ostream os(newStr);
    newOp->print(os);
  }

  EXPECT_EQ(newOp->getNumResults(), 1u);
  StringRef view(newStr);
  EXPECT_TRUE(view.contains("a = 42"));
  EXPECT_TRUE(view.contains("b = \"hello\""));
  EXPECT_TRUE(view.contains("c = \"world\""));
  EXPECT_TRUE(view.contains("flag = true"));
  EXPECT_TRUE(view.contains("array<i64: 1, 2, 3>"));
  EXPECT_TRUE(view.contains("array<i32: 4, 5>"));
}
