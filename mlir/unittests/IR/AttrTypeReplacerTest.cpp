//===- AttrTypeReplacerTest.cpp - Sub-element replacer unit tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "gtest/gtest.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// CyclicAttrTypeReplacer
//===----------------------------------------------------------------------===//

TEST(CyclicAttrTypeReplacerTest, testNoRecursion) {
  MLIRContext ctx;

  CyclicAttrTypeReplacer replacer;
  replacer.addReplacement([&](BoolAttr b) {
    return StringAttr::get(&ctx, b.getValue() ? "true" : "false");
  });

  EXPECT_EQ(replacer.replace(BoolAttr::get(&ctx, true)),
            StringAttr::get(&ctx, "true"));
  EXPECT_EQ(replacer.replace(BoolAttr::get(&ctx, false)),
            StringAttr::get(&ctx, "false"));
  EXPECT_EQ(replacer.replace(mlir::UnitAttr::get(&ctx)),
            mlir::UnitAttr::get(&ctx));
}

TEST(CyclicAttrTypeReplacerTest, testInPlaceRecursionPruneAnywhere) {
  MLIRContext ctx;
  Builder b(&ctx);

  CyclicAttrTypeReplacer replacer;
  // Replacer cycles through integer attrs 0 -> 1 -> 2 -> 0 -> ...
  replacer.addReplacement([&](IntegerAttr attr) {
    return replacer.replace(b.getI8IntegerAttr((attr.getInt() + 1) % 3));
  });
  // The first repeat of any integer attr is pruned into a unit attr.
  replacer.addCycleBreaker([&](IntegerAttr attr) { return b.getUnitAttr(); });

  // No recursion case.
  EXPECT_EQ(replacer.replace(mlir::UnitAttr::get(&ctx)),
            mlir::UnitAttr::get(&ctx));
  // Starting at 0.
  EXPECT_EQ(replacer.replace(b.getI8IntegerAttr(0)), mlir::UnitAttr::get(&ctx));
  // Starting at 2.
  EXPECT_EQ(replacer.replace(b.getI8IntegerAttr(2)), mlir::UnitAttr::get(&ctx));
}

//===----------------------------------------------------------------------===//
// CyclicAttrTypeReplacerTest: ChainRecursion
//===----------------------------------------------------------------------===//

class CyclicAttrTypeReplacerChainRecursionPruningTest : public ::testing::Test {
public:
  CyclicAttrTypeReplacerChainRecursionPruningTest() : b(&ctx) {
    // IntegerType<width = N>
    // ==> FunctionType<() => IntegerType< width = (N+1) % 3>>.
    // This will create a chain of infinite length without recursion pruning.
    replacer.addReplacement([&](mlir::IntegerType intType) {
      ++invokeCount;
      return b.getFunctionType(
          {}, {mlir::IntegerType::get(&ctx, (intType.getWidth() + 1) % 3)});
    });
  }

  void setBaseCase(std::optional<unsigned> pruneAt) {
    replacer.addCycleBreaker([&, pruneAt](mlir::IntegerType intType) {
      return (!pruneAt || intType.getWidth() == *pruneAt)
                 ? std::make_optional(b.getIndexType())
                 : std::nullopt;
    });
  }

  Type getFunctionTypeChain(unsigned N) {
    Type type = b.getIndexType();
    for (unsigned i = 0; i < N; i++)
      type = b.getFunctionType({}, type);
    return type;
  };

  MLIRContext ctx;
  Builder b;
  CyclicAttrTypeReplacer replacer;
  int invokeCount = 0;
};

TEST_F(CyclicAttrTypeReplacerChainRecursionPruningTest, testPruneAnywhere0) {
  setBaseCase(std::nullopt);

  // No recursion case.
  EXPECT_EQ(replacer.replace(b.getIndexType()), b.getIndexType());
  EXPECT_EQ(invokeCount, 0);

  // Starting at 0. Cycle length is 3.
  invokeCount = 0;
  EXPECT_EQ(replacer.replace(mlir::IntegerType::get(&ctx, 0)),
            getFunctionTypeChain(3));
  EXPECT_EQ(invokeCount, 3);

  // Starting at 1. Cycle length is 5 now because of a cached replacement at 0.
  invokeCount = 0;
  EXPECT_EQ(replacer.replace(mlir::IntegerType::get(&ctx, 1)),
            getFunctionTypeChain(5));
  EXPECT_EQ(invokeCount, 2);
}

TEST_F(CyclicAttrTypeReplacerChainRecursionPruningTest, testPruneAnywhere1) {
  setBaseCase(std::nullopt);

  // Starting at 1. Cycle length is 3.
  EXPECT_EQ(replacer.replace(mlir::IntegerType::get(&ctx, 1)),
            getFunctionTypeChain(3));
  EXPECT_EQ(invokeCount, 3);
}

TEST_F(CyclicAttrTypeReplacerChainRecursionPruningTest, testPruneSpecific0) {
  setBaseCase(0);

  // Starting at 0. Cycle length is 3.
  EXPECT_EQ(replacer.replace(mlir::IntegerType::get(&ctx, 0)),
            getFunctionTypeChain(3));
  EXPECT_EQ(invokeCount, 3);
}

TEST_F(CyclicAttrTypeReplacerChainRecursionPruningTest, testPruneSpecific1) {
  setBaseCase(0);

  // Starting at 1. Cycle length is 5 (1 -> 2 -> 0 -> 1 -> 2 -> Prune).
  EXPECT_EQ(replacer.replace(mlir::IntegerType::get(&ctx, 1)),
            getFunctionTypeChain(5));
  EXPECT_EQ(invokeCount, 5);
}

//===----------------------------------------------------------------------===//
// CyclicAttrTypeReplacerTest: BranchingRecusion
//===----------------------------------------------------------------------===//

class CyclicAttrTypeReplacerBranchingRecusionPruningTest
    : public ::testing::Test {
public:
  CyclicAttrTypeReplacerBranchingRecusionPruningTest() : b(&ctx) {
    // IntegerType<width = N>
    // ==> FunctionType<
    //       IntegerType< width = (N+1) % 3> =>
    //         IntegerType< width = (N+1) % 3>>.
    // This will create a binary tree of infinite depth without pruning.
    replacer.addReplacement([&](mlir::IntegerType intType) {
      ++invokeCount;
      Type child = mlir::IntegerType::get(&ctx, (intType.getWidth() + 1) % 3);
      return b.getFunctionType({child}, {child});
    });
  }

  void setBaseCase(std::optional<unsigned> pruneAt) {
    replacer.addCycleBreaker([&, pruneAt](mlir::IntegerType intType) {
      return (!pruneAt || intType.getWidth() == *pruneAt)
                 ? std::make_optional(b.getIndexType())
                 : std::nullopt;
    });
  }

  Type getFunctionTypeTree(unsigned N) {
    Type type = b.getIndexType();
    for (unsigned i = 0; i < N; i++)
      type = b.getFunctionType(type, type);
    return type;
  };

  MLIRContext ctx;
  Builder b;
  CyclicAttrTypeReplacer replacer;
  int invokeCount = 0;
};

TEST_F(CyclicAttrTypeReplacerBranchingRecusionPruningTest, testPruneAnywhere0) {
  setBaseCase(std::nullopt);

  // No recursion case.
  EXPECT_EQ(replacer.replace(b.getIndexType()), b.getIndexType());
  EXPECT_EQ(invokeCount, 0);

  // Starting at 0. Cycle length is 3.
  invokeCount = 0;
  EXPECT_EQ(replacer.replace(mlir::IntegerType::get(&ctx, 0)),
            getFunctionTypeTree(3));
  // Since both branches are identical, this should incur linear invocations
  // of the replacement function instead of exponential.
  EXPECT_EQ(invokeCount, 3);

  // Starting at 1. Cycle length is 5 now because of a cached replacement at 0.
  invokeCount = 0;
  EXPECT_EQ(replacer.replace(mlir::IntegerType::get(&ctx, 1)),
            getFunctionTypeTree(5));
  EXPECT_EQ(invokeCount, 2);
}

TEST_F(CyclicAttrTypeReplacerBranchingRecusionPruningTest, testPruneAnywhere1) {
  setBaseCase(std::nullopt);

  // Starting at 1. Cycle length is 3.
  EXPECT_EQ(replacer.replace(mlir::IntegerType::get(&ctx, 1)),
            getFunctionTypeTree(3));
  EXPECT_EQ(invokeCount, 3);
}

TEST_F(CyclicAttrTypeReplacerBranchingRecusionPruningTest, testPruneSpecific0) {
  setBaseCase(0);

  // Starting at 0. Cycle length is 3.
  EXPECT_EQ(replacer.replace(mlir::IntegerType::get(&ctx, 0)),
            getFunctionTypeTree(3));
  EXPECT_EQ(invokeCount, 3);
}

TEST_F(CyclicAttrTypeReplacerBranchingRecusionPruningTest, testPruneSpecific1) {
  setBaseCase(0);

  // Starting at 1. Cycle length is 5 (1 -> 2 -> 0 -> 1 -> 2 -> Prune).
  EXPECT_EQ(replacer.replace(mlir::IntegerType::get(&ctx, 1)),
            getFunctionTypeTree(5));
  EXPECT_EQ(invokeCount, 5);
}
