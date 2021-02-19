//===- DialectTest.cpp - Dialect unit tests -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectInterface.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::detail;

namespace {
struct TestDialect : public Dialect {
  static StringRef getDialectNamespace() { return "test"; };
  TestDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context, TypeID::get<TestDialect>()) {}
};
struct AnotherTestDialect : public Dialect {
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
  static StringRef getDialectNamespace() { return "test2"; }
  SecondTestDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context,
                TypeID::get<SecondTestDialect>()) {}
};

struct TestDialectInterfaceBase
    : public DialectInterface::Base<TestDialectInterfaceBase> {
  TestDialectInterfaceBase(Dialect *dialect) : Base(dialect) {}
  virtual int function() const { return 42; }
};

struct TestDialectInterface : public TestDialectInterfaceBase {
  using TestDialectInterfaceBase::TestDialectInterfaceBase;
  int function() const final { return 56; }
};

struct SecondTestDialectInterface : public TestDialectInterfaceBase {
  using TestDialectInterfaceBase::TestDialectInterfaceBase;
  int function() const final { return 78; }
};

TEST(Dialect, DelayedInterfaceRegistration) {
  DialectRegistry registry;
  registry.insert<TestDialect, SecondTestDialect>();

  // Delayed registration of an interface for TestDialect.
  registry.addDialectInterface<TestDialect, TestDialectInterface>();

  MLIRContext context(registry);

  // Load the TestDialect and check that the interface got registered for it.
  auto *testDialect = context.getOrLoadDialect<TestDialect>();
  ASSERT_TRUE(testDialect != nullptr);
  auto *testDialectInterface =
      testDialect->getRegisteredInterface<TestDialectInterfaceBase>();
  EXPECT_TRUE(testDialectInterface != nullptr);

  // Load the SecondTestDialect and check that the interface is not registered
  // for it.
  auto *secondTestDialect = context.getOrLoadDialect<SecondTestDialect>();
  ASSERT_TRUE(secondTestDialect != nullptr);
  auto *secondTestDialectInterface =
      secondTestDialect->getRegisteredInterface<SecondTestDialectInterface>();
  EXPECT_TRUE(secondTestDialectInterface == nullptr);

  // Use the same mechanism as for delayed registration but for an already
  // loaded dialect and check that the interface is now registered.
  DialectRegistry secondRegistry;
  secondRegistry.insert<SecondTestDialect>();
  secondRegistry
      .addDialectInterface<SecondTestDialect, SecondTestDialectInterface>();
  context.appendDialectRegistry(secondRegistry);
  secondTestDialectInterface =
      secondTestDialect->getRegisteredInterface<SecondTestDialectInterface>();
  EXPECT_TRUE(secondTestDialectInterface != nullptr);
}

TEST(Dialect, RepeatedDelayedRegistration) {
  // Set up the delayed registration.
  DialectRegistry registry;
  registry.insert<TestDialect>();
  registry.addDialectInterface<TestDialect, TestDialectInterface>();
  MLIRContext context(registry);

  // Load the TestDialect and check that the interface got registered for it.
  auto *testDialect = context.getOrLoadDialect<TestDialect>();
  ASSERT_TRUE(testDialect != nullptr);
  auto *testDialectInterface =
      testDialect->getRegisteredInterface<TestDialectInterfaceBase>();
  EXPECT_TRUE(testDialectInterface != nullptr);

  // Try adding the same dialect interface again and check that we don't crash
  // on repeated interface registration.
  DialectRegistry secondRegistry;
  secondRegistry.insert<TestDialect>();
  secondRegistry.addDialectInterface<TestDialect, TestDialectInterface>();
  context.appendDialectRegistry(secondRegistry);
  testDialectInterface =
      testDialect->getRegisteredInterface<TestDialectInterfaceBase>();
  EXPECT_TRUE(testDialectInterface != nullptr);
}

// A dialect that registers two interfaces with the same InterfaceID, triggering
// an assertion failure.
struct RepeatedRegistrationDialect : public Dialect {
  static StringRef getDialectNamespace() { return "repeatedreg"; }
  RepeatedRegistrationDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context,
                TypeID::get<RepeatedRegistrationDialect>()) {
    addInterfaces<TestDialectInterface>();
    addInterfaces<SecondTestDialectInterface>();
  }
};

TEST(Dialect, RepeatedInterfaceRegistrationDeath) {
  MLIRContext context;
  (void)context;

  // This triggers an assertion in debug mode.
#ifndef NDEBUG
  ASSERT_DEATH(context.loadDialect<RepeatedRegistrationDialect>(),
               "interface kind has already been registered");
#endif
}

} // end namespace
