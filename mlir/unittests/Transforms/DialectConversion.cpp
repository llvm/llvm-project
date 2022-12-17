//===- DialectConversion.cpp - Dialect conversion unit tests --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/DialectConversion.h"
#include "gtest/gtest.h"

using namespace mlir;

static Operation *createOp(MLIRContext *context) {
  context->allowUnregisteredDialects();
  return Operation::create(UnknownLoc::get(context),
                           OperationName("foo.bar", context), std::nullopt,
                           std::nullopt, std::nullopt, std::nullopt, 0);
}

namespace {
struct DummyOp {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DummyOp)

  static StringRef getOperationName() { return "foo.bar"; }
};

TEST(DialectConversionTest, DynamicallyLegalOpCallbackOrder) {
  MLIRContext context;
  ConversionTarget target(context);

  int index = 0;
  int callbackCalled1 = 0;
  target.addDynamicallyLegalOp<DummyOp>([&](Operation *) {
    callbackCalled1 = ++index;
    return true;
  });

  int callbackCalled2 = 0;
  target.addDynamicallyLegalOp<DummyOp>([&](Operation *) -> Optional<bool> {
    callbackCalled2 = ++index;
    return std::nullopt;
  });

  auto *op = createOp(&context);
  EXPECT_TRUE(target.isLegal(op));
  EXPECT_EQ(2, callbackCalled1);
  EXPECT_EQ(1, callbackCalled2);
  EXPECT_FALSE(target.isIllegal(op));
  EXPECT_EQ(4, callbackCalled1);
  EXPECT_EQ(3, callbackCalled2);
  op->destroy();
}

TEST(DialectConversionTest, DynamicallyLegalOpCallbackSkip) {
  MLIRContext context;
  ConversionTarget target(context);

  int index = 0;
  int callbackCalled = 0;
  target.addDynamicallyLegalOp<DummyOp>([&](Operation *) -> Optional<bool> {
    callbackCalled = ++index;
    return std::nullopt;
  });

  auto *op = createOp(&context);
  EXPECT_FALSE(target.isLegal(op));
  EXPECT_EQ(1, callbackCalled);
  EXPECT_FALSE(target.isIllegal(op));
  EXPECT_EQ(2, callbackCalled);
  op->destroy();
}

TEST(DialectConversionTest, DynamicallyLegalUnknownOpCallbackOrder) {
  MLIRContext context;
  ConversionTarget target(context);

  int index = 0;
  int callbackCalled1 = 0;
  target.markUnknownOpDynamicallyLegal([&](Operation *) {
    callbackCalled1 = ++index;
    return true;
  });

  int callbackCalled2 = 0;
  target.markUnknownOpDynamicallyLegal([&](Operation *) -> Optional<bool> {
    callbackCalled2 = ++index;
    return std::nullopt;
  });

  auto *op = createOp(&context);
  EXPECT_TRUE(target.isLegal(op));
  EXPECT_EQ(2, callbackCalled1);
  EXPECT_EQ(1, callbackCalled2);
  EXPECT_FALSE(target.isIllegal(op));
  EXPECT_EQ(4, callbackCalled1);
  EXPECT_EQ(3, callbackCalled2);
  op->destroy();
}

TEST(DialectConversionTest, DynamicallyLegalReturnNone) {
  MLIRContext context;
  ConversionTarget target(context);

  target.addDynamicallyLegalOp<DummyOp>(
      [&](Operation *) -> Optional<bool> { return std::nullopt; });

  auto *op = createOp(&context);
  EXPECT_FALSE(target.isLegal(op));
  EXPECT_FALSE(target.isIllegal(op));

  EXPECT_TRUE(succeeded(applyPartialConversion(op, target, {})));
  EXPECT_TRUE(failed(applyFullConversion(op, target, {})));

  op->destroy();
}

TEST(DialectConversionTest, DynamicallyLegalUnknownReturnNone) {
  MLIRContext context;
  ConversionTarget target(context);

  target.markUnknownOpDynamicallyLegal(
      [&](Operation *) -> Optional<bool> { return std::nullopt; });

  auto *op = createOp(&context);
  EXPECT_FALSE(target.isLegal(op));
  EXPECT_FALSE(target.isIllegal(op));

  EXPECT_TRUE(succeeded(applyPartialConversion(op, target, {})));
  EXPECT_TRUE(failed(applyFullConversion(op, target, {})));

  op->destroy();
}
} // namespace
