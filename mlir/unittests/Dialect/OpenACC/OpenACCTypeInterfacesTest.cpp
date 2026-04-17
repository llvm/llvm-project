//===- OpenACCTypeInterfacesTest.cpp - Tests for OpenACC type interfaces -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::acc;
using namespace mlir::LLVM;

namespace {

/// Test model that attaches ReducibleType to IntegerType for testing purposes.
/// This only implements a small subset of reduction operators to exercise the
/// interface - a real implementation would handle all valid operators.
struct TestReducibleIntegerModel
    : public ReducibleType::ExternalModel<TestReducibleIntegerModel,
                                          IntegerType> {
  std::optional<arith::AtomicRMWKind>
  getAtomicRMWKind(Type type, ReductionOperator redOp) const {
    switch (redOp) {
    case ReductionOperator::AccAdd:
      return arith::AtomicRMWKind::addi;
    default:
      return std::nullopt;
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

class OpenACCTypeInterfacesTest : public ::testing::Test {
protected:
  OpenACCTypeInterfacesTest() : context() {
    // Register the test external model before loading dialects.
    DialectRegistry registry;
    registry.addExtension(+[](MLIRContext *ctx, BuiltinDialect *dialect) {
      IntegerType::attachInterface<TestReducibleIntegerModel>(*ctx);
    });
    context.appendDialectRegistry(registry);
    context
        .loadDialect<acc::OpenACCDialect, arith::ArithDialect,
                     memref::MemRefDialect, func::FuncDialect, LLVMDialect>();
  }

  MLIRContext context;
};

//===----------------------------------------------------------------------===//
// ReducibleType Interface Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCTypeInterfacesTest, ReducibleTypeGetAtomicRMWKindAdd) {
  Type i32Type = IntegerType::get(&context, 32);
  auto reducible = dyn_cast<ReducibleType>(i32Type);
  ASSERT_TRUE(reducible != nullptr);

  auto kind = reducible.getAtomicRMWKind(ReductionOperator::AccAdd);
  ASSERT_TRUE(kind.has_value());
  EXPECT_EQ(*kind, arith::AtomicRMWKind::addi);
}

TEST_F(OpenACCTypeInterfacesTest, ReducibleTypeGetAtomicRMWKindUnsupported) {
  // Test that unsupported reduction operators return nullopt.
  Type i32Type = IntegerType::get(&context, 32);
  auto reducible = dyn_cast<ReducibleType>(i32Type);
  ASSERT_TRUE(reducible != nullptr);

  // The test model only implements AccAdd, so other operators return nullopt.
  auto mulKind = reducible.getAtomicRMWKind(ReductionOperator::AccMul);
  EXPECT_FALSE(mulKind.has_value());

  auto noneKind = reducible.getAtomicRMWKind(ReductionOperator::AccNone);
  EXPECT_FALSE(noneKind.has_value());
}

TEST_F(OpenACCTypeInterfacesTest, NonReducibleTypeReturnsNull) {
  // Test that a type without the interface attached returns nullptr.
  Type f32Type = Float32Type::get(&context);
  auto reducible = dyn_cast<ReducibleType>(f32Type);
  EXPECT_TRUE(reducible == nullptr);
}

//===----------------------------------------------------------------------===//
// PointerLikeType::genCast tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCTypeInterfacesTest, PointerLikeGenCastMemrefIdentity) {
  Location loc = UnknownLoc::get(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  OpBuilder builder(module->getBodyRegion());
  func::FuncOp fn = func::FuncOp::create(builder, loc, "cast_identity",
                                         builder.getFunctionType({}, {}));
  Block *block = fn.addEntryBlock();
  builder.setInsertionPointToStart(block);

  auto memTy = MemRefType::get({}, builder.getF32Type());
  memref::AllocaOp alloca = memref::AllocaOp::create(builder, loc, memTy);
  Value v = alloca.getResult();
  auto ptrLike = cast<PointerLikeType>(v.getType());
  Value out = ptrLike.genCast(builder, loc, v, memTy);
  ASSERT_TRUE(out);
  EXPECT_EQ(out, v);
}

TEST_F(OpenACCTypeInterfacesTest, PointerLikeGenCastMemrefStaticToDynamic) {
  Location loc = UnknownLoc::get(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  OpBuilder builder(module->getBodyRegion());
  func::FuncOp fn = func::FuncOp::create(builder, loc, "cast_memref",
                                         builder.getFunctionType({}, {}));
  Block *block = fn.addEntryBlock();
  builder.setInsertionPointToStart(block);

  auto srcTy = MemRefType::get({4}, builder.getF32Type());
  auto dstTy = MemRefType::get({ShapedType::kDynamic}, builder.getF32Type());
  memref::AllocaOp alloca = memref::AllocaOp::create(builder, loc, srcTy);
  Value v = alloca.getResult();
  auto ptrLike = cast<PointerLikeType>(v.getType());
  Value out = ptrLike.genCast(builder, loc, v, dstTy);
  ASSERT_TRUE(out);
  EXPECT_EQ(out.getType(), dstTy);
  ASSERT_TRUE(isa<memref::CastOp>(out.getDefiningOp()));
}

TEST_F(OpenACCTypeInterfacesTest, PointerLikeGenCastMemrefMemorySpace) {
  Location loc = UnknownLoc::get(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  OpBuilder builder(module->getBodyRegion());
  func::FuncOp fn = func::FuncOp::create(builder, loc, "cast_memref_memspace",
                                         builder.getFunctionType({}, {}));
  Block *block = fn.addEntryBlock();
  builder.setInsertionPointToStart(block);

  Attribute msHost = builder.getI32IntegerAttr(0);
  Attribute msDev = builder.getI32IntegerAttr(1);
  auto srcTy = MemRefType::get({4}, builder.getF32Type(), AffineMap(), msHost);
  auto dstTy = MemRefType::get({4}, builder.getF32Type(), AffineMap(), msDev);
  memref::AllocaOp alloca = memref::AllocaOp::create(builder, loc, srcTy);
  Value v = alloca.getResult();
  auto ptrLike = cast<PointerLikeType>(v.getType());
  Value out = ptrLike.genCast(builder, loc, v, dstTy);
  ASSERT_TRUE(out);
  EXPECT_EQ(out.getType(), dstTy);
  ASSERT_TRUE(isa<memref::MemorySpaceCastOp>(out.getDefiningOp()));
}

TEST_F(OpenACCTypeInterfacesTest, PointerLikeGenCastLLVMPtrAddrSpace) {
  Location loc = UnknownLoc::get(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  OpBuilder builder(module->getBodyRegion());
  func::FuncOp fn = func::FuncOp::create(builder, loc, "cast_llvm_addrspace",
                                         builder.getFunctionType({}, {}));
  Block *block = fn.addEntryBlock();
  builder.setInsertionPointToStart(block);

  Type ptrAs0 = LLVMPointerType::get(&context, 0);
  Type ptrAs3 = LLVMPointerType::get(&context, 3);
  Value v = UndefOp::create(builder, loc, ptrAs0);
  auto ptrLike = cast<PointerLikeType>(ptrAs0);
  Value out = ptrLike.genCast(builder, loc, v, ptrAs3);
  ASSERT_TRUE(out);
  EXPECT_EQ(out.getType(), ptrAs3);
  ASSERT_TRUE(isa<AddrSpaceCastOp>(out.getDefiningOp()));
}

TEST_F(OpenACCTypeInterfacesTest, PointerLikeGenCastLLVMPtrSameAddrSpaceNoOp) {
  Location loc = UnknownLoc::get(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  OpBuilder builder(module->getBodyRegion());
  func::FuncOp fn = func::FuncOp::create(builder, loc, "cast_llvm_same_as",
                                         builder.getFunctionType({}, {}));
  Block *block = fn.addEntryBlock();
  builder.setInsertionPointToStart(block);

  Type ptrTy = LLVMPointerType::get(&context, 2);
  Value v = UndefOp::create(builder, loc, ptrTy);
  auto ptrLike = cast<PointerLikeType>(ptrTy);
  Value out = ptrLike.genCast(builder, loc, v, ptrTy);
  ASSERT_TRUE(out);
  EXPECT_EQ(out, v);
}

TEST_F(OpenACCTypeInterfacesTest, PointerLikeGenCastLLVMPtrToI64) {
  Location loc = UnknownLoc::get(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  OpBuilder builder(module->getBodyRegion());
  func::FuncOp fn = func::FuncOp::create(builder, loc, "cast_llvm_ptrtoint",
                                         builder.getFunctionType({}, {}));
  Block *block = fn.addEntryBlock();
  builder.setInsertionPointToStart(block);

  Type ptrTy = LLVMPointerType::get(&context, 0);
  Type i64Ty = builder.getI64Type();
  Value v = UndefOp::create(builder, loc, ptrTy);
  auto ptrLike = cast<PointerLikeType>(ptrTy);
  Value out = ptrLike.genCast(builder, loc, v, i64Ty);
  ASSERT_TRUE(out);
  EXPECT_EQ(out.getType(), i64Ty);
  ASSERT_TRUE(isa<PtrToIntOp>(out.getDefiningOp()));
}

TEST_F(OpenACCTypeInterfacesTest, PointerLikeGenCastLLVMIntToPtrFromI64) {
  Location loc = UnknownLoc::get(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  OpBuilder builder(module->getBodyRegion());
  func::FuncOp fn = func::FuncOp::create(builder, loc, "cast_llvm_inttoptr",
                                         builder.getFunctionType({}, {}));
  Block *block = fn.addEntryBlock();
  builder.setInsertionPointToStart(block);

  Type ptrTy = LLVMPointerType::get(&context, 0);
  Value v = arith::ConstantIntOp::create(builder, loc, builder.getI64Type(), 0);
  auto ptrLike = cast<PointerLikeType>(ptrTy);
  Value out = ptrLike.genCast(builder, loc, v, ptrTy);
  ASSERT_TRUE(out);
  EXPECT_EQ(out.getType(), ptrTy);
  ASSERT_TRUE(isa<IntToPtrOp>(out.getDefiningOp()));
}

TEST_F(OpenACCTypeInterfacesTest, PointerLikeGenCastLLVMIntToPtrFromIndex) {
  Location loc = UnknownLoc::get(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  OpBuilder builder(module->getBodyRegion());
  func::FuncOp fn =
      func::FuncOp::create(builder, loc, "cast_llvm_index_inttoptr",
                           builder.getFunctionType({}, {}));
  Block *block = fn.addEntryBlock();
  builder.setInsertionPointToStart(block);

  Type ptrTy = LLVMPointerType::get(&context, 0);
  Value v = arith::ConstantIndexOp::create(builder, loc, 0);
  auto ptrLike = cast<PointerLikeType>(ptrTy);
  Value out = ptrLike.genCast(builder, loc, v, ptrTy);
  ASSERT_TRUE(out);
  EXPECT_EQ(out.getType(), ptrTy);
  auto intToPtr = dyn_cast<IntToPtrOp>(out.getDefiningOp());
  ASSERT_TRUE(intToPtr);
  EXPECT_TRUE(isa<arith::IndexCastUIOp>(intToPtr.getArg().getDefiningOp()));
}
