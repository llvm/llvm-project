//===- ABIRewriteContextTest.cpp - Unit tests for ABIRewriteContext -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/ABI/ABIRewriteContext.h"
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::abi;

namespace {

class MockRewriteContext : public ABIRewriteContext {
public:
  LogicalResult rewriteFunctionDefinition(FunctionOpInterface,
                                          const FunctionClassification &,
                                          OpBuilder &) override {
    return success();
  }

  LogicalResult rewriteCallSite(Operation *, const FunctionClassification &,
                                OpBuilder &) override {
    return success();
  }

  StringRef getDialectNamespace() const override { return "mock"; }
};

TEST(ABIRewriteContextTest, MockCanBeConstructedAndDestroyed) {
  MockRewriteContext ctx;
  EXPECT_EQ(ctx.getDialectNamespace(), "mock");
}

TEST(ABIRewriteContextTest, ArgClassificationDirect) {
  auto c = ArgClassification::getDirect();
  EXPECT_EQ(c.kind, ArgKind::Direct);
  EXPECT_EQ(c.coercedType, nullptr);
  EXPECT_TRUE(c.canFlatten);
}

TEST(ABIRewriteContextTest, ArgClassificationDirectWithType) {
  MLIRContext mlirCtx;
  auto i32 = IntegerType::get(&mlirCtx, 32);
  auto c = ArgClassification::getDirect(i32);
  EXPECT_EQ(c.kind, ArgKind::Direct);
  EXPECT_EQ(c.coercedType, i32);
}

TEST(ABIRewriteContextTest, ArgClassificationIgnore) {
  auto c = ArgClassification::getIgnore();
  EXPECT_EQ(c.kind, ArgKind::Ignore);
}

TEST(ABIRewriteContextTest, ArgClassificationIndirect) {
  auto c = ArgClassification::getIndirect(llvm::Align(8), true);
  EXPECT_EQ(c.kind, ArgKind::Indirect);
  EXPECT_EQ(c.indirectAlign, llvm::Align(8));
  EXPECT_TRUE(c.byVal);
}

TEST(ABIRewriteContextTest, ArgClassificationIndirectNoByVal) {
  auto c = ArgClassification::getIndirect(llvm::Align(16), false);
  EXPECT_EQ(c.kind, ArgKind::Indirect);
  EXPECT_EQ(c.indirectAlign, llvm::Align(16));
  EXPECT_FALSE(c.byVal);
}

TEST(ABIRewriteContextTest, ArgClassificationExtend) {
  MLIRContext mlirCtx;
  auto i8 = IntegerType::get(&mlirCtx, 8);

  auto signExt = ArgClassification::getExtend(i8, true);
  EXPECT_EQ(signExt.kind, ArgKind::Extend);
  EXPECT_TRUE(signExt.signExtend);

  auto zeroExt = ArgClassification::getExtend(i8, false);
  EXPECT_EQ(zeroExt.kind, ArgKind::Extend);
  EXPECT_FALSE(zeroExt.signExtend);
}

TEST(ABIRewriteContextTest, FunctionClassificationHoldsReturnAndArgs) {
  FunctionClassification fc;
  fc.returnInfo = ArgClassification::getDirect();
  fc.argInfos.push_back(ArgClassification::getDirect());
  fc.argInfos.push_back(ArgClassification::getIndirect(llvm::Align(8), true));
  fc.argInfos.push_back(ArgClassification::getIgnore());

  EXPECT_EQ(fc.returnInfo.kind, ArgKind::Direct);
  EXPECT_EQ(fc.argInfos.size(), 3u);
  EXPECT_EQ(fc.argInfos[0].kind, ArgKind::Direct);
  EXPECT_EQ(fc.argInfos[1].kind, ArgKind::Indirect);
  EXPECT_EQ(fc.argInfos[2].kind, ArgKind::Ignore);
}

} // namespace
