//===- ArithUtilsTest.cpp - Unit tests for Arith dialect utils ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <optional>

using namespace mlir;

namespace {

class ArithUtilsTest : public ::testing::Test {
protected:
  void SetUp() override {
    registry.insert<arith::ArithDialect>();
    ctx = std::make_unique<MLIRContext>(registry);
    ctx->loadAllAvailableDialects();
  }

  DialectRegistry registry;
  std::unique_ptr<MLIRContext> ctx;
};

TEST_F(ArithUtilsTest, CreateSignedScalarOrSplatConst) {
  OpBuilder builder(ctx.get());
  auto loc = builder.getUnknownLoc();
  auto i8Ty = builder.getIntegerType(8);

  auto createConst = [&](int64_t value) -> std::optional<APInt> {
    auto constValue = createScalarOrSplatConstant(builder, loc, i8Ty, value);

    APInt matchedValue;
    auto match = matchPattern(constValue, m_ConstantInt(&matchedValue));
    if (!match)
      return std::nullopt;

    return matchedValue;
  };

  auto negativeValue = createConst(-1);
  auto positiveValue = createConst(1);

  ASSERT_TRUE(negativeValue.has_value());
  ASSERT_TRUE(negativeValue->isNegative());

  ASSERT_TRUE(positiveValue.has_value());
  ASSERT_TRUE(positiveValue->isNonNegative());
}
} // namespace
