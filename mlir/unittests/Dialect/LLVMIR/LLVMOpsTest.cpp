//===- LLVMOpsTest.cpp - Tests for LLVM operations ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLVMTestBase.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;
using namespace mlir::LLVM;

TEST_F(LLVMIRTest, CallIntrinsicMixedArguments) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module->getBody());

  auto ptrType = LLVMPointerType::get(&context);
  auto functionType =
      LLVMFunctionType::get(LLVMVoidType::get(&context), {ptrType, ptrType});
  auto function = LLVMFuncOp::create(builder, loc, "test", functionType);
  Block *entry = function.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry);

  SmallVector<OpFoldResult> args = {entry->getArgument(0),
                                    builder.getI64IntegerAttr(-1),
                                    entry->getArgument(1)};
  auto call = CallIntrinsicOp::create(
      builder, loc, TypeRange{}, builder.getStringAttr("llvm.invariant.end"),
      ArrayRef<OpFoldResult>(args));
  ReturnOp::create(builder, loc, ValueRange{});

  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_EQ(call.getArgs().size(), 2u);
  EXPECT_EQ(call.getArgs()[0], entry->getArgument(0));
  EXPECT_EQ(call.getArgs()[1], entry->getArgument(1));
  EXPECT_EQ(call.getMixedArgs(), args);
}
