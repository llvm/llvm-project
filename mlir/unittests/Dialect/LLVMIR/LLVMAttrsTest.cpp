//===- LLVMAttrsTest.cpp - Tests for LLVM attributes ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLVMTestBase.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::LLVM;

TEST_F(LLVMIRTest, TargetFeaturesAtLLVMFuncOp) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module->getBody());

  auto functionType =
      LLVMFunctionType::get(LLVMVoidType::get(&context), /*params=*/{});
  LLVMFuncOp function = LLVMFuncOp::create(builder, loc, "test", functionType);
  TargetFeaturesAttr targetFeatures = TargetFeaturesAttr::get(&context, "+sve");
  function.setTargetFeaturesAttr(targetFeatures);

  Block *entry = function.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry);
  ReturnOp returnOp = ReturnOp::create(builder, loc, ValueRange{});

  EXPECT_EQ(TargetFeaturesAttr::featuresAt(returnOp), targetFeatures);
}

TEST_F(LLVMIRTest, TargetFeaturesAtFuncOpDiscardableAttr) {
  context.loadDialect<func::FuncDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module->getBody());

  auto functionType = builder.getFunctionType(/*inputs=*/{}, /*results=*/{});
  func::FuncOp function =
      func::FuncOp::create(builder, loc, "test", functionType);
  TargetFeaturesAttr targetFeatures = TargetFeaturesAttr::get(&context, "+sve");
  function->setDiscardableAttr(TargetFeaturesAttr::getAttributeName(),
                               targetFeatures);

  Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  func::ReturnOp returnOp = func::ReturnOp::create(builder, loc);

  EXPECT_EQ(TargetFeaturesAttr::featuresAt(returnOp), targetFeatures);
}
