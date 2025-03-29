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
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OwningOpRef.h"
#include "gtest/gtest.h"

#include "../../test/lib/Dialect/Test/TestAttributes.h"
#include "../../test/lib/Dialect/Test/TestDialect.h"
#include "../../test/lib/Dialect/Test/TestOps.h"
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
  op.erase();
}

TEST(InterfaceTest, TestImplicitConversion) {
  MLIRContext context;
  context.loadDialect<test::TestDialect>();

  TestBaseTypeInterfacePrintTypeB typeB;
  TestBaseTypeInterfacePrintTypeA typeA = typeB;
  EXPECT_EQ(typeA, nullptr);

  typeB = TestType::get(&context);
  typeA = typeB;
  EXPECT_EQ(typeA, typeB);
}

TEST(InterfaceTest, TestCustomTensorIsTensorType) {
  MLIRContext context;
  context.loadDialect<test::TestDialect>();

  auto customTensorType = test::TestTensorType::get(
      &context, {1, 2, 3}, mlir::IntegerType::get(&context, 32));
  EXPECT_TRUE(mlir::isa<mlir::TensorType>(customTensorType));

  auto customCloneType = customTensorType.clone({3, 4, 5});
  EXPECT_EQ(customTensorType.getElementType(),
            customCloneType.getElementType());
  EXPECT_TRUE(mlir::isa<mlir::TensorType>(customCloneType));
  EXPECT_TRUE(mlir::isa<test::TestTensorType>(customCloneType));

  // user-specified conversions
  TensorType baseCopy = customTensorType;
  std::ignore = baseCopy;

  ShapedType shapedBaseCopy = customTensorType;
  std::ignore = shapedBaseCopy;
}

TEST(InterfaceTest, TestCustomMemrefIsBaseMemref) {
  MLIRContext context;
  context.loadDialect<test::TestDialect>();

  auto customMemrefType = test::TestMemrefType::get(
      &context, {1, 2, 3}, mlir::IntegerType::get(&context, 32),
      mlir::StringAttr::get(&context, "some_memspace"));
  EXPECT_TRUE(mlir::isa<mlir::BaseMemRefType>(customMemrefType));

  auto customCloneType = customMemrefType.clone({3, 4, 5});
  EXPECT_EQ(customMemrefType.getElementType(),
            customCloneType.getElementType());
  EXPECT_TRUE(mlir::isa<mlir::BaseMemRefType>(customCloneType));
  EXPECT_TRUE(mlir::isa<test::TestMemrefType>(customCloneType));
  EXPECT_EQ(customMemrefType.getMemorySpace(),
            mlir::cast<test::TestMemrefType>(customCloneType).getMemorySpace());

  // user-specified conversions
  BaseMemRefType baseCopy = customMemrefType;
  std::ignore = baseCopy;

  ShapedType shapedBaseCopy = customMemrefType;
  std::ignore = shapedBaseCopy;
}
