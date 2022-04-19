//===- ControlFlowInterfacesTest.cpp - Unit Tests for Control Flow Interf. ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser/Parser.h"

#include <gtest/gtest.h>

using namespace mlir;

/// A dummy op that is also a terminator.
struct DummyOp : public Op<DummyOp, OpTrait::IsTerminator> {
  using Op::Op;
  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static StringRef getOperationName() { return "cftest.dummy_op"; }
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
  void getSuccessorRegions(Optional<unsigned> index,
                           ArrayRef<Attribute> operands,
                           SmallVectorImpl<RegionSuccessor> &regions) {}
};

/// All regions of this op call each other in a large circle.
struct LoopRegionsOp
    : public Op<LoopRegionsOp, RegionBranchOpInterface::Trait> {
  using Op::Op;
  static const unsigned kNumRegions = 3;

  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static StringRef getOperationName() { return "cftest.loop_regions_op"; }

  void getSuccessorRegions(Optional<unsigned> index,
                           ArrayRef<Attribute> operands,
                           SmallVectorImpl<RegionSuccessor> &regions) {
    if (index) {
      if (*index == 1)
        // This region also branches back to the parent.
        regions.push_back(RegionSuccessor());
      regions.push_back(
          RegionSuccessor(&getOperation()->getRegion(*index % kNumRegions)));
    }
  }
};

/// Regions are executed sequentially.
struct SequentialRegionsOp
    : public Op<SequentialRegionsOp, RegionBranchOpInterface::Trait> {
  using Op::Op;
  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static StringRef getOperationName() { return "cftest.sequential_regions_op"; }

  // Region 0 has Region 1 as a successor.
  void getSuccessorRegions(Optional<unsigned> index,
                           ArrayRef<Attribute> operands,
                           SmallVectorImpl<RegionSuccessor> &regions) {
    if (index == 0u) {
      Operation *thisOp = this->getOperation();
      regions.push_back(RegionSuccessor(&thisOp->getRegion(1)));
    }
  }
};

/// A dialect putting all the above together.
struct CFTestDialect : Dialect {
  explicit CFTestDialect(MLIRContext *ctx)
      : Dialect(getDialectNamespace(), ctx, TypeID::get<CFTestDialect>()) {
    addOperations<DummyOp, MutuallyExclusiveRegionsOp, LoopRegionsOp,
                  SequentialRegionsOp>();
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
