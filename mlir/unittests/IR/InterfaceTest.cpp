//===- InterfaceTest.cpp - Test interfaces --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OwningOpRef.h"
#include "gtest/gtest.h"

#include "../../test/lib/Dialect/Test/TestAttributes.h"
#include "../../test/lib/Dialect/Test/TestDialect.h"
#include "../../test/lib/Dialect/Test/TestTypes.h"

using namespace mlir;
using namespace test;

TEST(InterfaceTest, OpInterfaceDenseMapKey) {
  MLIRContext context;
  context.loadDialect<test::TestDialect>();

  OwningOpRef<ModuleOp> module = ModuleOp::create(UnknownLoc::get(&context));
  OpBuilder builder(module->getBody(), module->getBody()->begin());
  auto op1 = builder.create<test::SideEffectOp>(builder.getUnknownLoc(),
                                                builder.getI32Type());
  auto op2 = builder.create<test::SideEffectOp>(builder.getUnknownLoc(),
                                                builder.getI32Type());
  auto op3 = builder.create<test::SideEffectOp>(builder.getUnknownLoc(),
                                                builder.getI32Type());
  DenseSet<MemoryEffectOpInterface> opSet;
  opSet.insert(op1);
  opSet.insert(op2);
  opSet.erase(op1);
  EXPECT_FALSE(opSet.contains(op1));
  EXPECT_TRUE(opSet.contains(op2));
  EXPECT_FALSE(opSet.contains(op3));
}

TEST(InterfaceTest, TypeInterfaceDenseMapKey) {
  MLIRContext context;
  context.loadDialect<test::TestDialect>();

  OpBuilder builder(&context);
  DenseSet<DataLayoutTypeInterface> typeSet;
  auto type1 = builder.getType<test::TestTypeWithLayoutType>(1);
  auto type2 = builder.getType<test::TestTypeWithLayoutType>(2);
  auto type3 = builder.getType<test::TestTypeWithLayoutType>(3);
  typeSet.insert(type1);
  typeSet.insert(type2);
  typeSet.erase(type1);
  EXPECT_FALSE(typeSet.contains(type1));
  EXPECT_TRUE(typeSet.contains(type2));
  EXPECT_FALSE(typeSet.contains(type3));
}

TEST(InterfaceTest, TestCustomClassOf) {
  MLIRContext context;
  context.loadDialect<test::TestDialect>();

  OpBuilder builder(&context);
  auto op = builder.create<TestOpOptionallyImplementingInterface>(
      builder.getUnknownLoc(), /*implementsInterface=*/true);
  EXPECT_TRUE(isa<TestOptionallyImplementedOpInterface>(*op));
  op.setImplementsInterface(false);
  EXPECT_FALSE(isa<TestOptionallyImplementedOpInterface>(*op));
}
