//===- ControlFlowInterfacesTest.cpp - Unit Tests for Control Flow Interf. ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/DebugLog.h"

#include <gtest/gtest.h>

using namespace mlir;

/// A dummy op that is also a terminator.
struct DummyOp : public Op<DummyOp, OpTrait::IsTerminator, OpTrait::ZeroResults,
                           OpTrait::ZeroSuccessors,
                           RegionBranchTerminatorOpInterface::Trait> {
  using Op::Op;
  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static StringRef getOperationName() { return "cftest.dummy_op"; }

  MutableOperandRange getMutableSuccessorOperands(RegionSuccessor point) {
    return MutableOperandRange(getOperation(), 0, 0);
  }
};

/// All regions of this op are mutually exclusive.
struct MutuallyExclusiveRegionsOp
    : public Op<MutuallyExclusiveRegionsOp, RegionBranchOpInterface::Trait> {
  using Op::Op;
  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static StringRef getOperationName() {
    return "cftest.mutually_exclusive_regions_op";
  }

  // Regions have no successors.
  void getSuccessorRegions(RegionBranchPoint point,
                           SmallVectorImpl<RegionSuccessor> &regions) {}
  using RegionBranchOpInterface::Trait<
      MutuallyExclusiveRegionsOp>::getSuccessorRegions;
};

/// All regions of this op call each other in a large circle.
struct LoopRegionsOp
    : public Op<LoopRegionsOp, RegionBranchOpInterface::Trait> {
  using Op::Op;
  static const unsigned kNumRegions = 3;

  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static StringRef getOperationName() { return "cftest.loop_regions_op"; }

  void getSuccessorRegions(RegionBranchPoint point,
                           SmallVectorImpl<RegionSuccessor> &regions) {
    if (point.getTerminatorPredecessorOrNull()) {
      Region *region =
          point.getTerminatorPredecessorOrNull()->getParentRegion();
      if (region == &(*this)->getRegion(1))
        // This region also branches back to this operation.
        regions.push_back(RegionSuccessor(getOperation()));
      regions.push_back(RegionSuccessor(region));
    }
  }

  ValueRange getSuccessorInputs(RegionSuccessor successor) {
    return successor.isOperation() ? ValueRange(getOperation()->getResults())
                                   : ValueRange();
  }

  using RegionBranchOpInterface::Trait<LoopRegionsOp>::getSuccessorRegions;
};

/// Each region branches back it itself or the parent.
struct DoubleLoopRegionsOp
    : public Op<DoubleLoopRegionsOp, RegionBranchOpInterface::Trait> {
  using Op::Op;

  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static StringRef getOperationName() {
    return "cftest.double_loop_regions_op";
  }

  void getSuccessorRegions(RegionBranchPoint point,
                           SmallVectorImpl<RegionSuccessor> &regions) {
    if (point.getTerminatorPredecessorOrNull()) {
      Region *region =
          point.getTerminatorPredecessorOrNull()->getParentRegion();
      regions.push_back(RegionSuccessor(getOperation()));
      regions.push_back(RegionSuccessor(region));
    }
  }

  ValueRange getSuccessorInputs(RegionSuccessor successor) {
    return successor.isOperation() ? ValueRange(getOperation()->getResults())
                                   : ValueRange();
  }

  using RegionBranchOpInterface::Trait<
      DoubleLoopRegionsOp>::getSuccessorRegions;
};

/// Regions are executed sequentially.
struct SequentialRegionsOp
    : public Op<SequentialRegionsOp, RegionBranchOpInterface::Trait> {
  using Op::Op;
  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static StringRef getOperationName() { return "cftest.sequential_regions_op"; }

  // Region 0 has Region 1 as a successor.
  void getSuccessorRegions(RegionBranchPoint point,
                           SmallVectorImpl<RegionSuccessor> &regions) {
    if (point.getTerminatorPredecessorOrNull() &&
        point.getTerminatorPredecessorOrNull()->getParentRegion() ==
            &(*this)->getRegion(0)) {
      Operation *thisOp = this->getOperation();
      regions.push_back(RegionSuccessor(&thisOp->getRegion(1)));
    }
  }
  using RegionBranchOpInterface::Trait<
      SequentialRegionsOp>::getSuccessorRegions;
};

/// Two attributes define the region graph of this op. `entries` lists the
/// regions that the op enters. `successors[i]` lists the regions that each
/// terminator of region i branches to. Every terminator can also branch back to
/// the op.
struct RegionGraphOp
    : public Op<RegionGraphOp, RegionBranchOpInterface::Trait> {
  using Op::Op;
  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static StringRef getOperationName() { return "cftest.region_graph_op"; }

  void getSuccessorRegions(RegionBranchPoint point,
                           SmallVectorImpl<RegionSuccessor> &regions) {
    Operation *op = getOperation();
    DenseI64ArrayAttr targets;
    if (point.isParent()) {
      targets = cast<DenseI64ArrayAttr>(op->getAttr("entries"));
    } else {
      // Branching back to the op adds no edge between regions. It only lets
      // every region exit.
      regions.push_back(RegionSuccessor(op));
      Region *from = point.getTerminatorPredecessorOrNull()->getParentRegion();
      targets = cast<DenseI64ArrayAttr>(
          cast<ArrayAttr>(op->getAttr("successors"))[from->getRegionNumber()]);
    }
    for (int64_t target : targets.asArrayRef())
      regions.push_back(RegionSuccessor(&op->getRegion(target)));
  }

  ValueRange getSuccessorInputs(RegionSuccessor successor) {
    return successor.isOperation() ? ValueRange(getOperation()->getResults())
                                   : ValueRange();
  }

  using RegionBranchOpInterface::Trait<RegionGraphOp>::getSuccessorRegions;
};

/// A dialect putting all the above together.
struct CFTestDialect : Dialect {
  explicit CFTestDialect(MLIRContext *ctx)
      : Dialect(getDialectNamespace(), ctx, TypeID::get<CFTestDialect>()) {
    addOperations<DummyOp, MutuallyExclusiveRegionsOp, LoopRegionsOp,
                  DoubleLoopRegionsOp, SequentialRegionsOp, RegionGraphOp>();
  }
  static StringRef getDialectNamespace() { return "cftest"; }
};

TEST(RegionBranchOpInterface, MutuallyExclusiveOps) {
  const char *ir = R"MLIR(
"cftest.mutually_exclusive_regions_op"() (
      {"cftest.dummy_op"() : () -> ()},  // op1
      {"cftest.dummy_op"() : () -> ()}   // op2
  ) : () -> ()
  )MLIR";

  DialectRegistry registry;
  registry.insert<CFTestDialect>();
  MLIRContext ctx(registry);

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &ctx);
  Operation *testOp = &module->getBody()->getOperations().front();
  Operation *op1 = &testOp->getRegion(0).front().front();
  Operation *op2 = &testOp->getRegion(1).front().front();

  EXPECT_TRUE(insideMutuallyExclusiveRegions(op1, op2));
  EXPECT_TRUE(insideMutuallyExclusiveRegions(op2, op1));
}

TEST(RegionBranchOpInterface, MutuallyExclusiveOps2) {
  const char *ir = R"MLIR(
"cftest.double_loop_regions_op"() (
      {"cftest.dummy_op"() : () -> ()},  // op1
      {"cftest.dummy_op"() : () -> ()}   // op2
  ) : () -> ()
  )MLIR";

  DialectRegistry registry;
  registry.insert<CFTestDialect>();
  MLIRContext ctx(registry);

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &ctx);
  Operation *testOp = &module->getBody()->getOperations().front();
  Operation *op1 = &testOp->getRegion(0).front().front();
  Operation *op2 = &testOp->getRegion(1).front().front();

  EXPECT_TRUE(insideMutuallyExclusiveRegions(op1, op2));
  EXPECT_TRUE(insideMutuallyExclusiveRegions(op2, op1));
}

TEST(RegionBranchOpInterface, NotMutuallyExclusiveOps) {
  const char *ir = R"MLIR(
"cftest.sequential_regions_op"() (
      {"cftest.dummy_op"() : () -> ()},  // op1
      {"cftest.dummy_op"() : () -> ()}   // op2
  ) : () -> ()
  )MLIR";

  DialectRegistry registry;
  registry.insert<CFTestDialect>();
  MLIRContext ctx(registry);

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &ctx);
  Operation *testOp = &module->getBody()->getOperations().front();
  Operation *op1 = &testOp->getRegion(0).front().front();
  Operation *op2 = &testOp->getRegion(1).front().front();

  EXPECT_FALSE(insideMutuallyExclusiveRegions(op1, op2));
  EXPECT_FALSE(insideMutuallyExclusiveRegions(op2, op1));
}

TEST(RegionBranchOpInterface, NestedMutuallyExclusiveOps) {
  const char *ir = R"MLIR(
"cftest.mutually_exclusive_regions_op"() (
      {
        "cftest.sequential_regions_op"() (
              {"cftest.dummy_op"() : () -> ()},  // op1
              {"cftest.dummy_op"() : () -> ()}   // op3
          ) : () -> ()
        "cftest.dummy_op"() : () -> ()
      },
      {"cftest.dummy_op"() : () -> ()}           // op2
  ) : () -> ()
  )MLIR";

  DialectRegistry registry;
  registry.insert<CFTestDialect>();
  MLIRContext ctx(registry);

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &ctx);
  Operation *testOp = &module->getBody()->getOperations().front();
  Operation *op1 =
      &testOp->getRegion(0).front().front().getRegion(0).front().front();
  Operation *op2 = &testOp->getRegion(1).front().front();
  Operation *op3 =
      &testOp->getRegion(0).front().front().getRegion(1).front().front();

  EXPECT_TRUE(insideMutuallyExclusiveRegions(op1, op2));
  EXPECT_TRUE(insideMutuallyExclusiveRegions(op3, op2));
  EXPECT_FALSE(insideMutuallyExclusiveRegions(op1, op3));
}

TEST(RegionBranchOpInterface, RecursiveRegions) {
  const char *ir = R"MLIR(
"cftest.loop_regions_op"() (
      {"cftest.dummy_op"() : () -> ()},  // op1
      {"cftest.dummy_op"() : () -> ()},  // op2
      {"cftest.dummy_op"() : () -> ()}   // op3
  ) : () -> ()
  )MLIR";

  DialectRegistry registry;
  registry.insert<CFTestDialect>();
  MLIRContext ctx(registry);

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &ctx);
  Operation *testOp = &module->getBody()->getOperations().front();
  auto regionOp = cast<RegionBranchOpInterface>(testOp);
  Operation *op1 = &testOp->getRegion(0).front().front();
  Operation *op2 = &testOp->getRegion(1).front().front();
  Operation *op3 = &testOp->getRegion(2).front().front();

  EXPECT_TRUE(regionOp.isRepetitiveRegion(0));
  EXPECT_TRUE(regionOp.isRepetitiveRegion(1));
  EXPECT_TRUE(regionOp.isRepetitiveRegion(2));
  EXPECT_NE(getEnclosingRepetitiveRegion(op1), nullptr);
  EXPECT_NE(getEnclosingRepetitiveRegion(op2), nullptr);
  EXPECT_NE(getEnclosingRepetitiveRegion(op3), nullptr);
}

TEST(RegionBranchOpInterface, NotRecursiveRegions) {
  const char *ir = R"MLIR(
"cftest.sequential_regions_op"() (
      {"cftest.dummy_op"() : () -> ()},  // op1
      {"cftest.dummy_op"() : () -> ()}   // op2
  ) : () -> ()
  )MLIR";

  DialectRegistry registry;
  registry.insert<CFTestDialect>();
  MLIRContext ctx(registry);

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &ctx);
  Operation *testOp = &module->getBody()->getOperations().front();
  Operation *op1 = &testOp->getRegion(0).front().front();
  Operation *op2 = &testOp->getRegion(1).front().front();

  EXPECT_EQ(getEnclosingRepetitiveRegion(op1), nullptr);
  EXPECT_EQ(getEnclosingRepetitiveRegion(op2), nullptr);
}

/// Builds `cftest.region_graph_op` ops and keeps them alive for the test.
class RegionGraphTest : public ::testing::Test {
protected:
  RegionGraphTest() : builder(&ctx) {
    ctx.loadDialect<CFTestDialect>();
    module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module->getBody());
  }

  /// Create a `cftest.region_graph_op` with one region per entry of
  /// `successors`. Each region has one block, whose terminator branches to the
  /// regions listed for it.
  RegionBranchOpInterface
  createRegionGraphOp(ArrayRef<int64_t> entries,
                      ArrayRef<SmallVector<int64_t>> successors) {
    OperationState state(builder.getUnknownLoc(),
                         RegionGraphOp::getOperationName());
    SmallVector<Attribute> successorAttrs = llvm::map_to_vector(
        successors, [&](ArrayRef<int64_t> regions) -> Attribute {
          return builder.getDenseI64ArrayAttr(regions);
        });
    state.addAttribute("entries", builder.getDenseI64ArrayAttr(entries));
    state.addAttribute("successors", builder.getArrayAttr(successorAttrs));
    for (size_t i = 0, e = successors.size(); i < e; ++i)
      state.addRegion();
    Operation *op = builder.create(state);
    for (Region &region : op->getRegions())
      addTerminatedBlock(region);
    return cast<RegionBranchOpInterface>(op);
  }

  /// Append a block to `region` that holds only a region branch terminator.
  void addTerminatedBlock(Region &region) {
    OpBuilder::InsertionGuard guard(builder);
    builder.createBlock(&region);
    builder.create(
        OperationState(builder.getUnknownLoc(), DummyOp::getOperationName()));
  }

  MLIRContext ctx;
  OpBuilder builder;
  OwningOpRef<ModuleOp> module;
};

TEST_F(RegionGraphTest, MultipleBranchesToSameRegionIsNotALoop) {
  RegionBranchOpInterface op =
      createRegionGraphOp(/*entries=*/{0}, /*successors=*/{{1}, {}});
  // Give region 0 a second terminator. Both terminators branch to region 1.
  addTerminatedBlock(op->getRegion(0));
  EXPECT_FALSE(op.isRepetitiveRegion(0));
  EXPECT_FALSE(op.isRepetitiveRegion(1));
  EXPECT_FALSE(op.hasLoop());
}

TEST_F(RegionGraphTest, DiamondIsNotALoop) {
  RegionBranchOpInterface op = createRegionGraphOp(
      /*entries=*/{0}, /*successors=*/{{1, 2}, {3}, {3}, {}});
  EXPECT_FALSE(op.hasLoop());
}

TEST_F(RegionGraphTest, LoopAfterEntryRegion) {
  RegionBranchOpInterface op =
      createRegionGraphOp(/*entries=*/{0}, /*successors=*/{{1}, {1}});
  EXPECT_FALSE(op.isRepetitiveRegion(0));
  EXPECT_TRUE(op.isRepetitiveRegion(1));
  EXPECT_TRUE(op.hasLoop());
}

TEST_F(RegionGraphTest, LoopThroughEntryRegion) {
  RegionBranchOpInterface op =
      createRegionGraphOp(/*entries=*/{0}, /*successors=*/{{1}, {0}});
  EXPECT_TRUE(op.hasLoop());
}

TEST_F(RegionGraphTest, UnreachableLoopIsIgnored) {
  RegionBranchOpInterface op =
      createRegionGraphOp(/*entries=*/{0}, /*successors=*/{{1}, {}, {2}});
  EXPECT_TRUE(op.isRepetitiveRegion(2));
  EXPECT_FALSE(op.hasLoop());
}

TEST_F(RegionGraphTest, OverlappingEntriesAreNotALoop) {
  // Region 1 is an entry region and is also reachable from entry region 0.
  // The search must not depend on the order of the entry regions.
  EXPECT_FALSE(createRegionGraphOp(/*entries=*/{0, 1}, /*successors=*/{{1}, {}})
                   .hasLoop());
  EXPECT_FALSE(createRegionGraphOp(/*entries=*/{1, 0}, /*successors=*/{{1}, {}})
                   .hasLoop());
}
