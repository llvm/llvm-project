//===- SCFOps.cpp - SCF Op Unit Tests -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;

namespace {
class SCFOpsTest : public testing::Test {
public:
  SCFOpsTest() {
    context.getOrLoadDialect<scf::SCFDialect>();
    context.getOrLoadDialect<StandardOpsDialect>();
  }

protected:
  MLIRContext context;
};

TEST_F(SCFOpsTest, IfOpNumRegionInvocations) {
  const char *const code = R"mlir(
func @test(%cond : i1) -> () {
  scf.if %cond {
    scf.yield
  } else {
    scf.yield
  }
  return
}
)mlir";
  Builder builder(&context);

  auto module = parseSourceString(code, &context);
  ASSERT_TRUE(module);
  scf::IfOp op;
  module->walk([&](scf::IfOp ifOp) { op = ifOp; });
  ASSERT_TRUE(op);

  SmallVector<int64_t> countPerRegion;
  op.getNumRegionInvocations({Attribute()}, countPerRegion);
  EXPECT_EQ(countPerRegion.size(), 2u);
  EXPECT_EQ(countPerRegion[0], kUnknownNumRegionInvocations);
  EXPECT_EQ(countPerRegion[1], kUnknownNumRegionInvocations);

  countPerRegion.clear();
  op.getNumRegionInvocations(
      {builder.getIntegerAttr(builder.getI1Type(), true)}, countPerRegion);
  EXPECT_EQ(countPerRegion.size(), 2u);
  EXPECT_EQ(countPerRegion[0], 1);
  EXPECT_EQ(countPerRegion[1], 0);

  countPerRegion.clear();
  op.getNumRegionInvocations(
      {builder.getIntegerAttr(builder.getI1Type(), false)}, countPerRegion);
  EXPECT_EQ(countPerRegion.size(), 2u);
  EXPECT_EQ(countPerRegion[0], 0);
  EXPECT_EQ(countPerRegion[1], 1);
}
} // end anonymous namespace
