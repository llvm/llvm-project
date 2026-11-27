//===- SideEffectInterfacesTest.cpp - Unit tests for Resource hierarchy ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Analysis/AnalysisManager.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/TypeID.h"
#include "llvm/Support/Casting.h"
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::SideEffects;

namespace {

/// Custom resource hierarchy (root -> child -> grandchild) for testing.
struct TestRootResource : public Resource::Base<TestRootResource> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestRootResource)
  TestRootResource() = default;
  StringRef getName() const override { return "TestRoot"; }

protected:
  TestRootResource(TypeID id) : Base(id) {}
};

struct TestChildResource
    : public Resource::Base<TestChildResource, TestRootResource> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestChildResource)
  TestChildResource() = default;
  StringRef getName() const override { return "TestChild"; }
  Resource *getParent() const override { return TestRootResource::get(); }

protected:
  TestChildResource(TypeID id) : Base(id) {}
};

struct TestGrandchildResource
    : public Resource::Base<TestGrandchildResource, TestChildResource> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestGrandchildResource)
  StringRef getName() const override { return "TestGrandchild"; }
  Resource *getParent() const override { return TestChildResource::get(); }
};

} // namespace

TEST(SideEffectResourceTest, BuiltInHierarchyIsaCast) {
  EXPECT_TRUE(isa<DefaultResource>(DefaultResource::get()));
  EXPECT_TRUE(isa<DefaultResource>(AutomaticAllocationScopeResource::get()));
  EXPECT_FALSE(isa<AutomaticAllocationScopeResource>(DefaultResource::get()));

  // Each type has its own singleton; cast yields the child object as parent*
  EXPECT_NE(cast<DefaultResource>(AutomaticAllocationScopeResource::get()),
            nullptr);
  EXPECT_TRUE(isa<DefaultResource>(
      cast<DefaultResource>(AutomaticAllocationScopeResource::get())));

  EXPECT_NE(dyn_cast<DefaultResource>(AutomaticAllocationScopeResource::get()),
            nullptr);
  EXPECT_EQ(dyn_cast<AutomaticAllocationScopeResource>(DefaultResource::get()),
            nullptr);
}

TEST(SideEffectResourceTest, CustomHierarchyIsaCast) {
  // Root and child
  EXPECT_TRUE(isa<TestRootResource>(TestRootResource::get()));
  EXPECT_TRUE(isa<TestRootResource>(TestChildResource::get()));
  EXPECT_FALSE(isa<TestChildResource>(TestRootResource::get()));
  EXPECT_NE(cast<TestRootResource>(TestChildResource::get()), nullptr);

  // Grandchild isa root, child, and self
  EXPECT_TRUE(isa<TestRootResource>(TestGrandchildResource::get()));
  EXPECT_TRUE(isa<TestChildResource>(TestGrandchildResource::get()));
  EXPECT_TRUE(isa<TestGrandchildResource>(TestGrandchildResource::get()));

  // Root/child are not isa grandchild
  EXPECT_FALSE(isa<TestGrandchildResource>(TestRootResource::get()));
  EXPECT_FALSE(isa<TestGrandchildResource>(TestChildResource::get()));

  // Cast grandchild to root and child (each type has its own singleton)
  EXPECT_NE(cast<TestRootResource>(TestGrandchildResource::get()), nullptr);
  EXPECT_NE(cast<TestChildResource>(TestGrandchildResource::get()), nullptr);

  // dyn_cast
  EXPECT_EQ(dyn_cast<TestGrandchildResource>(TestRootResource::get()), nullptr);
  EXPECT_NE(dyn_cast<TestRootResource>(TestGrandchildResource::get()), nullptr);

  // getParent chain
  EXPECT_EQ(TestGrandchildResource::get()->getParent(),
            TestChildResource::get());
  EXPECT_EQ(TestChildResource::get()->getParent(), TestRootResource::get());
  EXPECT_EQ(TestRootResource::get()->getParent(), DefaultResource::get());
  EXPECT_EQ(DefaultResource::get()->getParent(), nullptr);

  // isSubresourceOf
  EXPECT_TRUE(
      TestGrandchildResource::get()->isSubresourceOf(TestRootResource::get()));
  EXPECT_TRUE(
      TestGrandchildResource::get()->isSubresourceOf(TestChildResource::get()));
  EXPECT_FALSE(
      TestRootResource::get()->isSubresourceOf(TestGrandchildResource::get()));

  // Custom hierarchy is a subresource of DefaultResource (the root).
  EXPECT_FALSE(TestRootResource::get()->isDisjointFrom(DefaultResource::get()));
  EXPECT_TRUE(TestRootResource::get()->isSubresourceOf(DefaultResource::get()));
  EXPECT_TRUE(
      TestGrandchildResource::get()->isSubresourceOf(DefaultResource::get()));
  EXPECT_FALSE(
      DefaultResource::get()->isSubresourceOf(TestRootResource::get()));
  EXPECT_FALSE(isa<TestRootResource>(DefaultResource::get()));
  EXPECT_EQ(dyn_cast<TestRootResource>(DefaultResource::get()), nullptr);
}

TEST(SideEffectResourceTest, DisjointnessAndGetParent) {
  EXPECT_EQ(DefaultResource::get()->getParent(), nullptr);
  EXPECT_EQ(AutomaticAllocationScopeResource::get()->getParent(),
            DefaultResource::get());
  // TestRootResource is a subresource of DefaultResource (the root),
  // so they are not disjoint.
  EXPECT_FALSE(DefaultResource::get()->isDisjointFrom(TestRootResource::get()));
  EXPECT_TRUE(TestRootResource::get()->isSubresourceOf(DefaultResource::get()));
  EXPECT_TRUE(
      TestChildResource::get()->isSubresourceOf(TestRootResource::get()));
  EXPECT_FALSE(
      TestRootResource::get()->isSubresourceOf(TestChildResource::get()));
}

TEST(Speculatability, LazyPassAnalysis) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, DLTIDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>} {
      %i = arith.constant 2 : i32
      %x = arith.constant 4294967296 : index
      %a = arith.divui %i, %i : i32
      %b = arith.divui %x, %x : index
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  struct LayoutObserver : AnalysisInstrumentation {
    unsigned starts = 0, finishes = 0;

    void runBeforeAnalysis(StringRef, TypeID id, Operation *) override {
      EXPECT_EQ(id, TypeID::get<DataLayoutAnalysis>());
      ++starts;
    }
    void runAfterAnalysis(StringRef, TypeID id, Operation *) override {
      EXPECT_EQ(id, TypeID::get<DataLayoutAnalysis>());
      ++finishes;
    }
  } observer;
  ModuleAnalysisManager owner(*module, &observer);
  AnalysisManager actx = owner;
  auto divs = module->getOps<arith::DivUIOp>();
  auto it = divs.begin();
  arith::DivUIOp fixed = *it++;
  arith::DivUIOp index = *it;

  EXPECT_TRUE(isPure(fixed, &actx));
  EXPECT_FALSE(actx.getCachedAnalysis<DataLayoutAnalysis>());
  // The divisor can truncate to zero when the caller supplies no layout.
  EXPECT_FALSE(isPure(index));
  EXPECT_TRUE(isPure(index, &actx));
  auto cached = actx.getCachedAnalysis<DataLayoutAnalysis>();
  ASSERT_TRUE(cached);
  AnalysisManager samePass = owner;
  EXPECT_TRUE(isPure(index, &samePass));
  EXPECT_EQ(&cached->get(), &samePass.getAnalysis<DataLayoutAnalysis>());
  EXPECT_EQ(observer.starts, 1u);
  EXPECT_EQ(observer.finishes, 1u);
}
