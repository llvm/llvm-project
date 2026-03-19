//===- DialectTest.cpp - Dialect unit tests -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/Support/TypeID.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::detail;

namespace {
struct TestDialect : public Dialect {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDialect)

  static StringRef getDialectNamespace() { return "test"; };
  TestDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context, TypeID::get<TestDialect>()) {}
};
struct AnotherTestDialect : public Dialect {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnotherTestDialect)

  static StringRef getDialectNamespace() { return "test"; };
  AnotherTestDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context,
                TypeID::get<AnotherTestDialect>()) {}
};

TEST(DialectDeathTest, MultipleDialectsWithSameNamespace) {
  MLIRContext context;

  // Registering a dialect with the same namespace twice should result in a
  // failure.
  context.loadDialect<TestDialect>();
  ASSERT_DEATH(context.loadDialect<AnotherTestDialect>(), "");
}

struct SecondTestDialect : public Dialect {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SecondTestDialect)

  static StringRef getDialectNamespace() { return "test2"; }
  SecondTestDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context,
                TypeID::get<SecondTestDialect>()) {}
};

struct TestDialectInterfaceBase
    : public DialectInterface::Base<TestDialectInterfaceBase> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDialectInterfaceBase)

  TestDialectInterfaceBase(Dialect *dialect) : Base(dialect) {}
  virtual int function() const { return 42; }
};

struct TestDialectInterface : public TestDialectInterfaceBase {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDialectInterface)

  using TestDialectInterfaceBase::TestDialectInterfaceBase;
  int function() const final { return 56; }
};

struct SecondTestDialectInterface : public TestDialectInterfaceBase {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SecondTestDialectInterface)

  using TestDialectInterfaceBase::TestDialectInterfaceBase;
  int function() const final { return 78; }
};

TEST(Dialect, DelayedInterfaceRegistration) {
  DialectRegistry registry;
  registry.insert<TestDialect, SecondTestDialect>();

  // Delayed registration of an interface for TestDialect.
  registry.addExtension(+[](MLIRContext *ctx, TestDialect *dialect) {
    dialect->addInterfaces<TestDialectInterface>();
  });

  MLIRContext context(registry);

  // Load the TestDialect and check that the interface got registered for it.
  Dialect *testDialect = context.getOrLoadDialect<TestDialect>();
  ASSERT_TRUE(testDialect != nullptr);
  auto *testDialectInterface = dyn_cast<TestDialectInterfaceBase>(testDialect);
  EXPECT_TRUE(testDialectInterface != nullptr);

  // Load the SecondTestDialect and check that the interface is not registered
  // for it.
  Dialect *secondTestDialect = context.getOrLoadDialect<SecondTestDialect>();
  ASSERT_TRUE(secondTestDialect != nullptr);
  auto *secondTestDialectInterface =
      dyn_cast<SecondTestDialectInterface>(secondTestDialect);
  EXPECT_TRUE(secondTestDialectInterface == nullptr);

  // Use the same mechanism as for delayed registration but for an already
  // loaded dialect and check that the interface is now registered.
  DialectRegistry secondRegistry;
  secondRegistry.insert<SecondTestDialect>();
  secondRegistry.addExtension(
      +[](MLIRContext *ctx, SecondTestDialect *dialect) {
        dialect->addInterfaces<SecondTestDialectInterface>();
      });
  context.appendDialectRegistry(secondRegistry);
  secondTestDialectInterface =
      dyn_cast<SecondTestDialectInterface>(secondTestDialect);
  EXPECT_TRUE(secondTestDialectInterface != nullptr);
}

TEST(Dialect, RepeatedDelayedRegistration) {
  // Set up the delayed registration.
  DialectRegistry registry;
  registry.insert<TestDialect>();
  registry.addExtension(+[](MLIRContext *ctx, TestDialect *dialect) {
    dialect->addInterfaces<TestDialectInterface>();
  });
  MLIRContext context(registry);

  // Load the TestDialect and check that the interface got registered for it.
  Dialect *testDialect = context.getOrLoadDialect<TestDialect>();
  ASSERT_TRUE(testDialect != nullptr);
  auto *testDialectInterface = dyn_cast<TestDialectInterfaceBase>(testDialect);
  EXPECT_TRUE(testDialectInterface != nullptr);

  // Try adding the same dialect interface again and check that we don't crash
  // on repeated interface registration.
  DialectRegistry secondRegistry;
  secondRegistry.insert<TestDialect>();
  secondRegistry.addExtension(+[](MLIRContext *ctx, TestDialect *dialect) {
    dialect->addInterfaces<TestDialectInterface>();
  });
  context.appendDialectRegistry(secondRegistry);
  testDialectInterface = dyn_cast<TestDialectInterfaceBase>(testDialect);
  EXPECT_TRUE(testDialectInterface != nullptr);
}

namespace {
/// A dummy extension that increases a counter when being applied and
/// recursively adds additional extensions.
struct DummyExtension : DialectExtension<DummyExtension, TestDialect> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DummyExtension)

  DummyExtension(int *counter, int numRecursive)
      : DialectExtension(), counter(counter), numRecursive(numRecursive) {}

  void apply(MLIRContext *ctx, TestDialect *dialect) const final {
    ++(*counter);
    DialectRegistry nestedRegistry;
    for (int i = 0; i < numRecursive; ++i) {
      // Create unique TypeIDs for these recursive extensions so they don't get
      // de-duplicated.
      auto extension =
          std::make_unique<DummyExtension>(counter, /*numRecursive=*/0);
      auto typeID = TypeID::getFromOpaquePointer(extension.get());
      nestedRegistry.addExtension(typeID, std::move(extension));
    }
    // Adding additional extensions may trigger a reallocation of the
    // `extensions` vector in the dialect registry.
    ctx->appendDialectRegistry(nestedRegistry);
  }

private:
  int *counter;
  int numRecursive;
};
} // namespace

TEST(Dialect, NestedDialectExtension) {
  DialectRegistry registry;
  registry.insert<TestDialect>();

  // Add an extension that adds 100 more extensions.
  int counter1 = 0;
  registry.addExtension(TypeID::get<DummyExtension>(),
                        std::make_unique<DummyExtension>(&counter1, 100));
  // Add one more extension. This should not crash.
  int counter2 = 0;
  registry.addExtension(TypeID::getFromOpaquePointer(&counter2),
                        std::make_unique<DummyExtension>(&counter2, 0));

  // Load dialect and apply extensions.
  MLIRContext context(registry);
  Dialect *testDialect = context.getOrLoadDialect<TestDialect>();
  ASSERT_TRUE(testDialect != nullptr);

  // Extensions are de-duplicated by typeID. Make sure that each expected
  // extension was applied at least once.
  EXPECT_GE(counter1, 101);
  EXPECT_GE(counter2, 1);
}

TEST(Dialect, SubsetWithExtensions) {
  DialectRegistry registry1, registry2;
  registry1.insert<TestDialect>();
  registry2.insert<TestDialect>();

  // Validate that the registries are equivalent.
  ASSERT_TRUE(registry1.isSubsetOf(registry2));
  ASSERT_TRUE(registry2.isSubsetOf(registry1));

  // Add extensions to registry2.
  int counter = 0;
  registry2.addExtension(TypeID::get<DummyExtension>(),
                         std::make_unique<DummyExtension>(&counter, 0));

  // Expect that (1) is a subset of (2) but not the other way around.
  ASSERT_TRUE(registry1.isSubsetOf(registry2));
  ASSERT_FALSE(registry2.isSubsetOf(registry1));

  // Add extensions to registry1.
  registry1.addExtension(TypeID::get<DummyExtension>(),
                         std::make_unique<DummyExtension>(&counter, 0));

  // Expect that (1) and (2) are equivalent.
  ASSERT_TRUE(registry1.isSubsetOf(registry2));
  ASSERT_TRUE(registry2.isSubsetOf(registry1));

  // Load dialect and apply extensions.
  MLIRContext context(registry1);
  context.getOrLoadDialect<TestDialect>();
  context.appendDialectRegistry(registry2);
  // Expect that the extension as only invoked once.
  ASSERT_EQ(counter, 1);
}

} // namespace
