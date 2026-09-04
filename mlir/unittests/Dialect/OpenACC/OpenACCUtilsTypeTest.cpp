//===- OpenACCUtilsTypeTest.cpp - Unit tests for OpenACC type utilities ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/OpenACCUtilsType.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::acc;

class OpenACCUtilsTypeTest : public ::testing::Test {
protected:
  OpenACCUtilsTypeTest() : b(&context), loc(UnknownLoc::get(&context)) {
    context.loadDialect<acc::OpenACCDialect, DLTIDialect, func::FuncDialect,
                        memref::MemRefDialect, LLVM::LLVMDialect>();
  }

  MLIRContext context;
  OpBuilder b;
  Location loc;
};

TEST_F(OpenACCUtilsTypeTest, IntegerTypeSizeAndAlignment) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);
  auto i32 = b.getI32Type();
  auto result = getTypeSizeAndAlignment(i32, *module);
  ASSERT_TRUE(result.has_value());
  DataLayout dl(*module);
  EXPECT_EQ(result->first, dl.getTypeSize(i32));
  EXPECT_EQ(result->second.getFixedValue(), dl.getTypeABIAlignment(i32));
}

TEST_F(OpenACCUtilsTypeTest, StaticMemRefTypeSizeAndAlignment) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);
  auto f64 = b.getF64Type();
  auto memrefTy = MemRefType::get({4, 3}, f64);
  auto result = getTypeSizeAndAlignment(memrefTy, *module);
  ASSERT_TRUE(result.has_value());
  DataLayout dl(*module);
  EXPECT_EQ(result->first,
            llvm::TypeSize::getFixed(dl.getTypeSize(f64).getFixedValue() * 12));
  EXPECT_EQ(result->second.getFixedValue(), dl.getTypeABIAlignment(f64));
}

TEST_F(OpenACCUtilsTypeTest, VectorTypeUsesDataLayout) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);
  auto vectorTy = VectorType::get({3}, b.getF32Type());
  auto result = getTypeSizeAndAlignment(vectorTy, *module);
  ASSERT_TRUE(result.has_value());
  DataLayout dl(*module);
  EXPECT_EQ(result->first, dl.getTypeSize(vectorTy));
  EXPECT_EQ(result->second.getFixedValue(), dl.getTypeABIAlignment(vectorTy));
}

TEST_F(OpenACCUtilsTypeTest, FunctionTypeUsesPointerSize) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);
  auto funcTy = b.getFunctionType({}, {});
  auto ptrResult = getTypeSizeAndAlignment(funcTy, *module);
  auto ptrTy = mlir::LLVM::LLVMPointerType::get(&context);
  auto directPtrResult = getTypeSizeAndAlignment(ptrTy, *module);
  ASSERT_TRUE(ptrResult.has_value());
  ASSERT_TRUE(directPtrResult.has_value());
  EXPECT_EQ(ptrResult->first, directPtrResult->first);
}

TEST_F(OpenACCUtilsTypeTest, DynamicMemRefReturnsNullopt) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);
  auto memrefTy = MemRefType::get({ShapedType::kDynamic}, b.getI32Type());
  EXPECT_FALSE(getTypeSizeAndAlignment(memrefTy, *module).has_value());
}

//===----------------------------------------------------------------------===//
// castPointerLikeTypeIfNeeded Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCUtilsTypeTest, CastPointerLikeTypeMatchingTypeIsNoOp) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);
  b.setInsertionPointToStart(module->getBody());

  auto memrefTy = MemRefType::get({10}, b.getF32Type());
  Value alloca = memref::AllocaOp::create(b, loc, memrefTy);

  EXPECT_EQ(castPointerLikeTypeIfNeeded(b, loc, alloca, memrefTy), alloca);
}

TEST_F(OpenACCUtilsTypeTest, CastPointerLikeTypeGeneratesMemRefCast) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);
  b.setInsertionPointToStart(module->getBody());

  auto memrefTy = MemRefType::get({10}, b.getF32Type());
  auto dynamicTy = MemRefType::get({ShapedType::kDynamic}, b.getF32Type());
  Value alloca = memref::AllocaOp::create(b, loc, memrefTy);

  Value casted = castPointerLikeTypeIfNeeded(b, loc, alloca, dynamicTy);

  EXPECT_EQ(casted.getType(), dynamicTy);
  ASSERT_TRUE(casted.getDefiningOp<memref::CastOp>());
}

TEST_F(OpenACCUtilsTypeTest, CastPointerLikeTypeGeneratesAddrSpaceCast) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);
  b.setInsertionPointToStart(module->getBody());

  auto ptrTy = LLVM::LLVMPointerType::get(&context);
  auto devicePtrTy = LLVM::LLVMPointerType::get(&context, /*addressSpace=*/1);
  Value one =
      LLVM::ConstantOp::create(b, loc, b.getI64Type(), b.getI64IntegerAttr(1));
  Value alloca = LLVM::AllocaOp::create(b, loc, ptrTy, b.getF32Type(), one);

  Value casted = castPointerLikeTypeIfNeeded(b, loc, alloca, devicePtrTy);

  EXPECT_EQ(casted.getType(), devicePtrTy);
  ASSERT_TRUE(casted.getDefiningOp<LLVM::AddrSpaceCastOp>());
}

TEST_F(OpenACCUtilsTypeTest, CastPointerLikeTypeUnsupportedEmitsError) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);
  b.setInsertionPointToStart(module->getBody());

  Value value =
      LLVM::ConstantOp::create(b, loc, b.getI32Type(), b.getI32IntegerAttr(0));

  std::string diagnostic;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diag) {
    diagnostic = diag.str();
    return success();
  });

  Value result = castPointerLikeTypeIfNeeded(b, loc, value, b.getF32Type());

  EXPECT_EQ(result, value);
  EXPECT_EQ(diagnostic, "unsupported pointer-like type cast from 'i32' to "
                        "'f32'");
}
