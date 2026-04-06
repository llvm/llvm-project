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
  EXPECT_EQ(c.Kind, ArgKind::Direct);
  EXPECT_EQ(c.CoercedType, nullptr);
  EXPECT_TRUE(c.CanFlatten);
}

TEST(ABIRewriteContextTest, ArgClassificationDirectWithType) {
  MLIRContext mlirCtx;
  auto i32 = IntegerType::get(&mlirCtx, 32);
  auto c = ArgClassification::getDirect(i32);
  EXPECT_EQ(c.Kind, ArgKind::Direct);
  EXPECT_EQ(c.CoercedType, i32);
}

TEST(ABIRewriteContextTest, ArgClassificationIgnore) {
  auto c = ArgClassification::getIgnore();
  EXPECT_EQ(c.Kind, ArgKind::Ignore);
}

TEST(ABIRewriteContextTest, ArgClassificationIndirect) {
  auto c = ArgClassification::getIndirect(llvm::Align(8), true);
  EXPECT_EQ(c.Kind, ArgKind::Indirect);
  EXPECT_EQ(c.IndirectAlign, llvm::Align(8));
  EXPECT_TRUE(c.ByVal);
}

TEST(ABIRewriteContextTest, ArgClassificationIndirectNoByVal) {
  auto c = ArgClassification::getIndirect(llvm::Align(16), false);
  EXPECT_EQ(c.Kind, ArgKind::Indirect);
  EXPECT_EQ(c.IndirectAlign, llvm::Align(16));
  EXPECT_FALSE(c.ByVal);
}

TEST(ABIRewriteContextTest, ArgClassificationExtend) {
  MLIRContext mlirCtx;
  auto i8 = IntegerType::get(&mlirCtx, 8);

  auto signExt = ArgClassification::getExtend(i8, true);
  EXPECT_EQ(signExt.Kind, ArgKind::Extend);
  EXPECT_TRUE(signExt.SignExtend);

  auto zeroExt = ArgClassification::getExtend(i8, false);
  EXPECT_EQ(zeroExt.Kind, ArgKind::Extend);
  EXPECT_FALSE(zeroExt.SignExtend);
}

TEST(ABIRewriteContextTest, FunctionClassificationHoldsReturnAndArgs) {
  FunctionClassification fc;
  fc.ReturnInfo = ArgClassification::getDirect();
  fc.ArgInfos.push_back(ArgClassification::getDirect());
  fc.ArgInfos.push_back(ArgClassification::getIndirect(llvm::Align(8), true));
  fc.ArgInfos.push_back(ArgClassification::getIgnore());

  EXPECT_EQ(fc.ReturnInfo.Kind, ArgKind::Direct);
  EXPECT_EQ(fc.ArgInfos.size(), 3u);
  EXPECT_EQ(fc.ArgInfos[0].Kind, ArgKind::Direct);
  EXPECT_EQ(fc.ArgInfos[1].Kind, ArgKind::Indirect);
  EXPECT_EQ(fc.ArgInfos[2].Kind, ArgKind::Ignore);
}

} // namespace
