//===- OpenACCOpsInterfacesTest.cpp - Unit tests for OpenACC interfaces --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMConstantOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenACC/Extensions/AllExtensions.h"
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
    DialectRegistry extensions;
    acc::registerAllExtensions(extensions);
    context.appendDialectRegistry(extensions);
    context.loadDialect<acc::OpenACCDialect, memref::MemRefDialect,
                        LLVM::LLVMDialect>();
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

TEST_F(OpenACCOpsInterfacesTest, GlobalVariableOpInterfaceInitRegion) {
  // Test that memref::GlobalOp returns nullptr for getInitRegion()
  // since it uses attributes for initialization, not regions

  auto memrefType = MemRefType::get({10}, builder.getF32Type());
  OwningOpRef<memref::GlobalOp> globalOp = memref::GlobalOp::create(
      builder, loc,
      /*sym_name=*/builder.getStringAttr("test_global"),
      /*sym_visibility=*/builder.getStringAttr("private"),
      /*type=*/TypeAttr::get(memrefType),
      /*initial_value=*/Attribute(),
      /*constant=*/UnitAttr(),
      /*alignment=*/IntegerAttr());

  auto globalVarIface =
      dyn_cast<GlobalVariableOpInterface>(globalOp->getOperation());
  ASSERT_TRUE(globalVarIface != nullptr);

  // memref::GlobalOp doesn't have regions for initialization
  EXPECT_EQ(globalVarIface.getInitRegion(), nullptr);
}

TEST_F(OpenACCOpsInterfacesTest, GlobalVariableOpInterfaceLocalLinkage) {
  auto memrefType = MemRefType::get({10}, builder.getF32Type());
  auto makeGlobal = [&](StringRef name, StringAttr visibility) {
    return memref::GlobalOp::create(builder, loc,
                                    /*sym_name=*/builder.getStringAttr(name),
                                    /*sym_visibility=*/visibility,
                                    /*type=*/TypeAttr::get(memrefType),
                                    /*initial_value=*/Attribute(),
                                    /*constant=*/UnitAttr(),
                                    /*alignment=*/IntegerAttr());
  };

  // memref.global "private" @private_global : memref<10xf32>
  // Lowers to llvm.mlir.global private. LLVM hasLocalLinkage is true.
  OwningOpRef<memref::GlobalOp> privateGlobal =
      makeGlobal("private_global", builder.getStringAttr("private"));
  auto privateIface =
      dyn_cast<GlobalVariableOpInterface>(privateGlobal->getOperation());
  ASSERT_TRUE(privateIface != nullptr);
  EXPECT_TRUE(privateIface.hasLocalLinkage());

  // memref.global "nested" @nested_global : memref<10xf32>
  // Nested is not public, so this lowers to llvm.mlir.global private.
  // LLVM hasLocalLinkage is true.
  OwningOpRef<memref::GlobalOp> nestedGlobal =
      makeGlobal("nested_global", builder.getStringAttr("nested"));
  auto nestedIface =
      dyn_cast<GlobalVariableOpInterface>(nestedGlobal->getOperation());
  ASSERT_TRUE(nestedIface != nullptr);
  EXPECT_TRUE(nestedIface.hasLocalLinkage());

  // memref.global "public" @public_global : memref<10xf32>
  // Lowers to llvm.mlir.global external. LLVM hasLocalLinkage is false.
  OwningOpRef<memref::GlobalOp> publicGlobal =
      makeGlobal("public_global", builder.getStringAttr("public"));
  auto publicIface =
      dyn_cast<GlobalVariableOpInterface>(publicGlobal->getOperation());
  ASSERT_TRUE(publicIface != nullptr);
  EXPECT_FALSE(publicIface.hasLocalLinkage());

  // memref.global @default_global : memref<10xf32>
  // Omitted visibility is public, so this lowers to llvm.mlir.global external.
  // LLVM hasLocalLinkage is false.
  OwningOpRef<memref::GlobalOp> defaultGlobal =
      makeGlobal("default_global", StringAttr());
  auto defaultIface =
      dyn_cast<GlobalVariableOpInterface>(defaultGlobal->getOperation());
  ASSERT_TRUE(defaultIface != nullptr);
  EXPECT_FALSE(defaultIface.hasLocalLinkage());
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

TEST_F(OpenACCOpsInterfacesTest, AddressOfGlobalOpInterfaceLLVMGetSymbol) {
  // Test that getSymbol() returns the correct symbol reference for the LLVM
  // dialect address-of operation.

  const auto *symbolName = "test_llvm_global_symbol";

  OwningOpRef<LLVM::AddressOfOp> addressOfOp = LLVM::AddressOfOp::create(
      builder, loc, LLVM::LLVMPointerType::get(&context),
      FlatSymbolRefAttr::get(&context, symbolName));

  auto addrOfGlobalIface =
      dyn_cast<AddressOfGlobalOpInterface>(addressOfOp->getOperation());
  ASSERT_TRUE(addrOfGlobalIface != nullptr);
  EXPECT_EQ(addrOfGlobalIface.getSymbol().getLeafReference(), symbolName);
}
