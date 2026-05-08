//===- OpenACCTypeInterfacesTest.cpp - Tests for OpenACC type interfaces -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::acc;

namespace {

/// Test model that attaches ReducibleType to IntegerType for testing purposes.
/// This only implements a small subset of reduction operators to exercise the
/// interface - a real implementation would handle all valid operators.
struct TestReducibleIntegerModel
    : public ReducibleType::ExternalModel<TestReducibleIntegerModel,
                                          IntegerType> {
  std::optional<arith::AtomicRMWKind>
  getAtomicRMWKind(Type type, ReductionOperator redOp) const {
    switch (redOp) {
    case ReductionOperator::AccAdd:
      return arith::AtomicRMWKind::addi;
    default:
      return std::nullopt;
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

class OpenACCTypeInterfacesTest : public ::testing::Test {
protected:
  OpenACCTypeInterfacesTest() : context() {
    // Register the test external model before loading dialects.
    DialectRegistry registry;
    registry.addExtension(+[](MLIRContext *ctx, BuiltinDialect *dialect) {
      IntegerType::attachInterface<TestReducibleIntegerModel>(*ctx);
    });
    context.appendDialectRegistry(registry);
    context.loadDialect<acc::OpenACCDialect, arith::ArithDialect>();
  }

  MLIRContext context;
};

//===----------------------------------------------------------------------===//
// ReducibleType Interface Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCTypeInterfacesTest, ReducibleTypeGetAtomicRMWKindAdd) {
  Type i32Type = IntegerType::get(&context, 32);
  auto reducible = dyn_cast<ReducibleType>(i32Type);
  ASSERT_TRUE(reducible != nullptr);

  auto kind = reducible.getAtomicRMWKind(ReductionOperator::AccAdd);
  ASSERT_TRUE(kind.has_value());
  EXPECT_EQ(*kind, arith::AtomicRMWKind::addi);
}

TEST_F(OpenACCTypeInterfacesTest, ReducibleTypeGetAtomicRMWKindUnsupported) {
  // Test that unsupported reduction operators return nullopt.
  Type i32Type = IntegerType::get(&context, 32);
  auto reducible = dyn_cast<ReducibleType>(i32Type);
  ASSERT_TRUE(reducible != nullptr);

  // The test model only implements AccAdd, so other operators return nullopt.
  auto mulKind = reducible.getAtomicRMWKind(ReductionOperator::AccMul);
  EXPECT_FALSE(mulKind.has_value());

  auto noneKind = reducible.getAtomicRMWKind(ReductionOperator::AccNone);
  EXPECT_FALSE(noneKind.has_value());
}

TEST_F(OpenACCTypeInterfacesTest, NonReducibleTypeReturnsNull) {
  // Test that a type without the interface attached returns nullptr.
  Type f32Type = Float32Type::get(&context);
  auto reducible = dyn_cast<ReducibleType>(f32Type);
  EXPECT_TRUE(reducible == nullptr);
}
