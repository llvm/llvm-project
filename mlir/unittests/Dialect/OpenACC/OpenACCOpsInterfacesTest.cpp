//===- OpenACCOpsInterfacesTest.cpp - Unit tests for OpenACC interfaces --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::acc;

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

class OpenACCOpsInterfacesTest : public ::testing::Test {
protected:
  OpenACCOpsInterfacesTest()
      : context(), builder(&context), loc(UnknownLoc::get(&context)) {
    context.loadDialect<acc::OpenACCDialect, memref::MemRefDialect>();
  }

  MLIRContext context;
  OpBuilder builder;
  Location loc;
};

//===----------------------------------------------------------------------===//
// GlobalVariableOpInterface Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCOpsInterfacesTest, GlobalVariableOpInterfaceNonConstant) {
  // Test that a non-constant global returns false for isConstant()

  auto memrefType = MemRefType::get({10}, builder.getF32Type());
  OwningOpRef<memref::GlobalOp> globalOp = memref::GlobalOp::create(
      builder, loc,
      /*sym_name=*/builder.getStringAttr("mutable_global"),
      /*sym_visibility=*/builder.getStringAttr("private"),
      /*type=*/TypeAttr::get(memrefType),
      /*initial_value=*/Attribute(),
      /*constant=*/UnitAttr(),
      /*alignment=*/IntegerAttr());

  auto globalVarIface =
      dyn_cast<GlobalVariableOpInterface>(globalOp->getOperation());
  ASSERT_TRUE(globalVarIface != nullptr);
  EXPECT_FALSE(globalVarIface.isConstant());
}

TEST_F(OpenACCOpsInterfacesTest, GlobalVariableOpInterfaceConstant) {
  // Test that a constant global returns true for isConstant()

  auto memrefType = MemRefType::get({5}, builder.getI32Type());
  OwningOpRef<memref::GlobalOp> constantGlobalOp = memref::GlobalOp::create(
      builder, loc,
      /*sym_name=*/builder.getStringAttr("constant_global"),
      /*sym_visibility=*/builder.getStringAttr("public"),
      /*type=*/TypeAttr::get(memrefType),
      /*initial_value=*/Attribute(),
      /*constant=*/builder.getUnitAttr(),
      /*alignment=*/IntegerAttr());

  auto globalVarIface =
      dyn_cast<GlobalVariableOpInterface>(constantGlobalOp->getOperation());
  ASSERT_TRUE(globalVarIface != nullptr);
  EXPECT_TRUE(globalVarIface.isConstant());
}

//===----------------------------------------------------------------------===//
// AddressOfGlobalOpInterface Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCOpsInterfacesTest, AddressOfGlobalOpInterfaceGetSymbol) {
  // Test that getSymbol() returns the correct symbol reference

  auto memrefType = MemRefType::get({5}, builder.getI32Type());
  const auto *symbolName = "test_global_symbol";

  OwningOpRef<memref::GetGlobalOp> getGlobalOp = memref::GetGlobalOp::create(
      builder, loc, memrefType, FlatSymbolRefAttr::get(&context, symbolName));

  auto addrOfGlobalIface =
      dyn_cast<AddressOfGlobalOpInterface>(getGlobalOp->getOperation());
  ASSERT_TRUE(addrOfGlobalIface != nullptr);
  EXPECT_EQ(addrOfGlobalIface.getSymbol().getLeafReference(), symbolName);
}
