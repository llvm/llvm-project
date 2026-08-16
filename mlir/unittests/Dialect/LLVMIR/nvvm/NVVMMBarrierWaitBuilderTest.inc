//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Builder tests for NVVM MBarrierTestWaitOp and MBarrierTryWaitOp.
///
//===----------------------------------------------------------------------===//

class NVVMMBarrierWaitTest : public ::testing::Test {
protected:
  NVVMMBarrierWaitTest() {
    context.loadDialect<NVVMDialect>();
    context.loadDialect<LLVMDialect>();
  }

  // Creates a block, sets the builder insertion point, and adds addr
  // (!llvm.ptr<3>) and stateOrPhase (stateOrPhaseTy) as block arguments.
  // Returns {addr, stateOrPhase}.
  std::pair<Value, Value> makeArgs(Block &block, Type stateOrPhaseTy) {
    OpBuilder builder(&context);
    builder.setInsertionPointToStart(&block);
    auto loc = builder.getUnknownLoc();
    Value addr = block.addArgument(LLVMPointerType::get(&context, 3), loc);
    Value stateOrPhase = block.addArgument(stateOrPhaseTy, loc);
    return {addr, stateOrPhase};
  }

  MLIRContext context;
};

TEST_F(NVVMMBarrierWaitTest, MBarrierWait) {
  auto loc = UnknownLoc::get(&context);
  auto i1Ty = IntegerType::get(&context, 1);
  auto i32Ty = IntegerType::get(&context, 32);
  auto i64Ty = IntegerType::get(&context, 64);

  Block block;
  auto [addr, state] = makeArgs(block, i64Ty);
  Value ticks = block.addArgument(i32Ty, loc);
  OpBuilder builder(&context);
  builder.setInsertionPointToEnd(&block);

  // MBarrierTestWaitOp.
  auto testWait = MBarrierTestWaitOp::create(builder, loc, i1Ty, addr, state);
  EXPECT_EQ(testWait.getRes().getType(), i1Ty);
  EXPECT_EQ(testWait.getAddr(), addr);
  EXPECT_EQ(testWait.getStateOrPhase(), state);

  // MBarrierTryWaitOp without ticks.
  auto tryWaitNoTicks =
      MBarrierTryWaitOp::create(builder, loc, i1Ty, addr, state, Value{});
  EXPECT_EQ(tryWaitNoTicks.getRes().getType(), i1Ty);
  EXPECT_EQ(tryWaitNoTicks.getAddr(), addr);
  EXPECT_EQ(tryWaitNoTicks.getStateOrPhase(), state);
  EXPECT_FALSE(tryWaitNoTicks.getTicks());

  // MBarrierTryWaitOp with ticks.
  auto tryWaitWithTicks =
      MBarrierTryWaitOp::create(builder, loc, i1Ty, addr, state, ticks);
  EXPECT_EQ(tryWaitWithTicks.getRes().getType(), i1Ty);
  EXPECT_TRUE(tryWaitWithTicks.getTicks());
  EXPECT_EQ(tryWaitWithTicks.getTicks(), ticks);
}
