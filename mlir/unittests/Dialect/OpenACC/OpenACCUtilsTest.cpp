//===- OpenACCUtilsTest.cpp - Unit tests for OpenACC utils ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/Utils/OpenACCUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/SmallVector.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::acc;

class OpenACCUtilsTest : public ::testing::Test {
protected:
  OpenACCUtilsTest() : b(&context), loc(UnknownLoc::get(&context)) {
    context.loadDialect<acc::OpenACCDialect, arith::ArithDialect,
                        memref::MemRefDialect>();
  }

  MLIRContext context;
  OpBuilder b;
  Location loc;
};

template <typename Op>
void testDataOpVarPtr(OpBuilder &b, MLIRContext &context, Location loc) {
  auto memrefTy = MemRefType::get({}, b.getI32Type());
  OwningOpRef<memref::AllocaOp> varPtrOp =
      b.create<memref::AllocaOp>(loc, memrefTy);
  OwningOpRef<GetDevicePtrOp> accPtrOp =
      b.create<GetDevicePtrOp>(loc, varPtrOp->getResult(), true, true);
  auto memrefTy2 = MemRefType::get({}, b.getF64Type());
  OwningOpRef<memref::AllocaOp> varPtrOp2 =
      b.create<memref::AllocaOp>(loc, memrefTy2);

  OwningOpRef<Op> op;
  if constexpr (std::is_same<Op, CopyoutOp>() ||
                std::is_same<Op, UpdateHostOp>()) {
    op = b.create<Op>(loc, /*accPtr=*/accPtrOp->getResult(),
                      varPtrOp->getResult(),
                      /*structured=*/true, /*implicit=*/true);
  } else {
    op = b.create<Op>(loc, varPtrOp->getResult(),
                      /*structured=*/true, /*implicit=*/true);
  }
  EXPECT_EQ(varPtrOp->getResult(), getVarPtr(op.get()));
  EXPECT_EQ(op->getVarPtr(), getVarPtr(op.get()));
  setVarPtr(op.get(), varPtrOp2->getResult());
  EXPECT_EQ(varPtrOp2->getResult(), getVarPtr(op.get()));
  EXPECT_EQ(op->getVarPtr(), getVarPtr(op.get()));
}

TEST_F(OpenACCUtilsTest, dataOpVarPtr) {
  testDataOpVarPtr<PrivateOp>(b, context, loc);
  testDataOpVarPtr<FirstprivateOp>(b, context, loc);
  testDataOpVarPtr<ReductionOp>(b, context, loc);
  testDataOpVarPtr<DevicePtrOp>(b, context, loc);
  testDataOpVarPtr<PresentOp>(b, context, loc);
  testDataOpVarPtr<CopyinOp>(b, context, loc);
  testDataOpVarPtr<CreateOp>(b, context, loc);
  testDataOpVarPtr<NoCreateOp>(b, context, loc);
  testDataOpVarPtr<AttachOp>(b, context, loc);
  testDataOpVarPtr<GetDevicePtrOp>(b, context, loc);
  testDataOpVarPtr<UpdateDeviceOp>(b, context, loc);
  testDataOpVarPtr<UseDeviceOp>(b, context, loc);
  testDataOpVarPtr<DeclareDeviceResidentOp>(b, context, loc);
  testDataOpVarPtr<DeclareLinkOp>(b, context, loc);
  testDataOpVarPtr<CacheOp>(b, context, loc);
  testDataOpVarPtr<CopyoutOp>(b, context, loc);
  testDataOpVarPtr<UpdateHostOp>(b, context, loc);
}

template <typename Op>
void testDataOpAccPtr(OpBuilder &b, MLIRContext &context, Location loc) {
  auto memrefTy = MemRefType::get({}, b.getI32Type());
  OwningOpRef<memref::AllocaOp> varPtrOp =
      b.create<memref::AllocaOp>(loc, memrefTy);
  OwningOpRef<GetDevicePtrOp> accPtrOp =
      b.create<GetDevicePtrOp>(loc, varPtrOp->getResult(), true, true);
  auto memrefTy2 = MemRefType::get({}, b.getF64Type());
  OwningOpRef<memref::AllocaOp> varPtrOp2 =
      b.create<memref::AllocaOp>(loc, memrefTy2);
  OwningOpRef<GetDevicePtrOp> accPtrOp2 =
      b.create<GetDevicePtrOp>(loc, varPtrOp2->getResult(), true, true);

  OwningOpRef<Op> op;
  if constexpr (std::is_same<Op, CopyoutOp>() ||
                std::is_same<Op, UpdateHostOp>()) {
    op = b.create<Op>(loc, /*accPtr=*/accPtrOp->getResult(),
                      varPtrOp->getResult(),
                      /*structured=*/true, /*implicit=*/true);
    EXPECT_EQ(op->getAccPtr(), getAccPtr(op.get()));
    EXPECT_EQ(op->getAccPtr(), accPtrOp->getResult());
    setAccPtr(op.get(), accPtrOp2->getResult());
    EXPECT_EQ(op->getAccPtr(), getAccPtr(op.get()));
    EXPECT_EQ(op->getAccPtr(), accPtrOp2->getResult());
  } else if constexpr (std::is_same<Op, DeleteOp>() ||
                       std::is_same<Op, DetachOp>()) {
    op = b.create<Op>(loc, /*accPtr=*/accPtrOp->getResult(),
                      /*structured=*/true, /*implicit=*/true);
    EXPECT_EQ(op->getAccPtr(), getAccPtr(op.get()));
    EXPECT_EQ(op->getAccPtr(), accPtrOp->getResult());
    setAccPtr(op.get(), accPtrOp2->getResult());
    EXPECT_EQ(op->getAccPtr(), getAccPtr(op.get()));
    EXPECT_EQ(op->getAccPtr(), accPtrOp2->getResult());
  } else {
    op = b.create<Op>(loc, varPtrOp->getResult(),
                      /*structured=*/true, /*implicit=*/true);
    EXPECT_EQ(op->getAccPtr(), op->getResult());
    EXPECT_EQ(op->getAccPtr(), getAccPtr(op.get()));
  }
}

TEST_F(OpenACCUtilsTest, dataOpAccPtr) {
  testDataOpAccPtr<PrivateOp>(b, context, loc);
  testDataOpAccPtr<FirstprivateOp>(b, context, loc);
  testDataOpAccPtr<ReductionOp>(b, context, loc);
  testDataOpAccPtr<DevicePtrOp>(b, context, loc);
  testDataOpAccPtr<PresentOp>(b, context, loc);
  testDataOpAccPtr<CopyinOp>(b, context, loc);
  testDataOpAccPtr<CreateOp>(b, context, loc);
  testDataOpAccPtr<NoCreateOp>(b, context, loc);
  testDataOpAccPtr<AttachOp>(b, context, loc);
  testDataOpAccPtr<GetDevicePtrOp>(b, context, loc);
  testDataOpAccPtr<UpdateDeviceOp>(b, context, loc);
  testDataOpAccPtr<UseDeviceOp>(b, context, loc);
  testDataOpAccPtr<DeclareDeviceResidentOp>(b, context, loc);
  testDataOpAccPtr<DeclareLinkOp>(b, context, loc);
  testDataOpAccPtr<CacheOp>(b, context, loc);
  testDataOpAccPtr<CopyoutOp>(b, context, loc);
  testDataOpAccPtr<UpdateHostOp>(b, context, loc);
  testDataOpAccPtr<DeleteOp>(b, context, loc);
  testDataOpAccPtr<DetachOp>(b, context, loc);
}

template <typename Op>
void testDataOpVarPtrPtr(OpBuilder &b, MLIRContext &context, Location loc) {
  auto memrefTy = MemRefType::get({}, b.getI32Type());
  OwningOpRef<memref::AllocaOp> varPtrOp =
      b.create<memref::AllocaOp>(loc, memrefTy);

  auto memrefTy2 = MemRefType::get({}, memrefTy);
  OwningOpRef<memref::AllocaOp> varPtrPtr =
      b.create<memref::AllocaOp>(loc, memrefTy2);

  OwningOpRef<Op> op = b.create<Op>(loc, varPtrOp->getResult(),
                                    /*structured=*/true, /*implicit=*/true);

  EXPECT_EQ(op->getVarPtrPtr(), getVarPtrPtr(op.get()));
  EXPECT_EQ(op->getVarPtrPtr(), Value());
  setVarPtrPtr(op.get(), varPtrPtr->getResult());
  EXPECT_EQ(op->getVarPtrPtr(), getVarPtrPtr(op.get()));
  EXPECT_EQ(op->getVarPtrPtr(), varPtrPtr->getResult());
}

TEST_F(OpenACCUtilsTest, dataOpVarPtrPtr) {
  testDataOpVarPtr<PrivateOp>(b, context, loc);
  testDataOpVarPtr<FirstprivateOp>(b, context, loc);
  testDataOpVarPtr<ReductionOp>(b, context, loc);
  testDataOpVarPtr<DevicePtrOp>(b, context, loc);
  testDataOpVarPtr<PresentOp>(b, context, loc);
  testDataOpVarPtr<CopyinOp>(b, context, loc);
  testDataOpVarPtr<CreateOp>(b, context, loc);
  testDataOpVarPtr<NoCreateOp>(b, context, loc);
  testDataOpVarPtr<AttachOp>(b, context, loc);
  testDataOpVarPtr<GetDevicePtrOp>(b, context, loc);
  testDataOpVarPtr<UpdateDeviceOp>(b, context, loc);
  testDataOpVarPtr<UseDeviceOp>(b, context, loc);
  testDataOpVarPtr<DeclareDeviceResidentOp>(b, context, loc);
  testDataOpVarPtr<DeclareLinkOp>(b, context, loc);
  testDataOpVarPtr<CacheOp>(b, context, loc);
}

template <typename Op>
void testDataOpBounds(OpBuilder &b, MLIRContext &context, Location loc) {
  auto memrefTy = MemRefType::get({}, b.getI32Type());
  OwningOpRef<memref::AllocaOp> varPtrOp =
      b.create<memref::AllocaOp>(loc, memrefTy);
  OwningOpRef<GetDevicePtrOp> accPtrOp =
      b.create<GetDevicePtrOp>(loc, varPtrOp->getResult(), true, true);
  OwningOpRef<arith::ConstantIndexOp> extent =
      b.create<arith::ConstantIndexOp>(loc, 1);
  OwningOpRef<DataBoundsOp> bounds =
      b.create<DataBoundsOp>(loc, extent->getResult());

  OwningOpRef<Op> op;
  if constexpr (std::is_same<Op, CopyoutOp>() ||
                std::is_same<Op, UpdateHostOp>()) {
    op = b.create<Op>(loc, /*accPtr=*/accPtrOp->getResult(),
                      varPtrOp->getResult(),
                      /*structured=*/true, /*implicit=*/true);
  } else if constexpr (std::is_same<Op, DeleteOp>() ||
                       std::is_same<Op, DetachOp>()) {
    op = b.create<Op>(loc, /*accPtr=*/accPtrOp->getResult(),
                      /*structured=*/true, /*implicit=*/true);
  } else {
    op = b.create<Op>(loc, varPtrOp->getResult(),
                      /*structured=*/true, /*implicit=*/true);
  }

  EXPECT_EQ(op->getBounds().size(), getBounds(op.get()).size());
  for (auto [bound1, bound2] :
       llvm::zip(op->getBounds(), getBounds(op.get()))) {
    EXPECT_EQ(bound1, bound2);
  }
  setBounds(op.get(), bounds->getResult());
  EXPECT_EQ(op->getBounds().size(), getBounds(op.get()).size());
  for (auto [bound1, bound2] :
       llvm::zip(op->getBounds(), getBounds(op.get()))) {
    EXPECT_EQ(bound1, bound2);
    EXPECT_EQ(bound1, bounds->getResult());
  }
}

TEST_F(OpenACCUtilsTest, dataOpBounds) {
  testDataOpBounds<PrivateOp>(b, context, loc);
  testDataOpBounds<FirstprivateOp>(b, context, loc);
  testDataOpBounds<ReductionOp>(b, context, loc);
  testDataOpBounds<DevicePtrOp>(b, context, loc);
  testDataOpBounds<PresentOp>(b, context, loc);
  testDataOpBounds<CopyinOp>(b, context, loc);
  testDataOpBounds<CreateOp>(b, context, loc);
  testDataOpBounds<NoCreateOp>(b, context, loc);
  testDataOpBounds<AttachOp>(b, context, loc);
  testDataOpBounds<GetDevicePtrOp>(b, context, loc);
  testDataOpBounds<UpdateDeviceOp>(b, context, loc);
  testDataOpBounds<UseDeviceOp>(b, context, loc);
  testDataOpBounds<DeclareDeviceResidentOp>(b, context, loc);
  testDataOpBounds<DeclareLinkOp>(b, context, loc);
  testDataOpBounds<CacheOp>(b, context, loc);
  testDataOpBounds<CopyoutOp>(b, context, loc);
  testDataOpBounds<UpdateHostOp>(b, context, loc);
  testDataOpBounds<DeleteOp>(b, context, loc);
  testDataOpBounds<DetachOp>(b, context, loc);
}

template <typename Op>
void testDataOpName(OpBuilder &b, MLIRContext &context, Location loc) {
  auto memrefTy = MemRefType::get({}, b.getI32Type());
  OwningOpRef<memref::AllocaOp> varPtrOp =
      b.create<memref::AllocaOp>(loc, memrefTy);
  OwningOpRef<GetDevicePtrOp> accPtrOp =
      b.create<GetDevicePtrOp>(loc, varPtrOp->getResult(), true, true);

  OwningOpRef<Op> op;
  if constexpr (std::is_same<Op, CopyoutOp>() ||
                std::is_same<Op, UpdateHostOp>()) {
    op = b.create<Op>(loc, /*accPtr=*/accPtrOp->getResult(),
                      varPtrOp->getResult(),
                      /*structured=*/true, /*implicit=*/true, "varName");
  } else if constexpr (std::is_same<Op, DeleteOp>() ||
                       std::is_same<Op, DetachOp>()) {
    op = b.create<Op>(loc, /*accPtr=*/accPtrOp->getResult(),
                      /*structured=*/true, /*implicit=*/true, "varName");
  } else {
    op = b.create<Op>(loc, varPtrOp->getResult(),
                      /*structured=*/true, /*implicit=*/true, "varName");
  }

  EXPECT_EQ(op->getNameAttr().str(), "varName");
  EXPECT_EQ(getVarName(op.get()), "varName");
}

TEST_F(OpenACCUtilsTest, dataOpName) {
  testDataOpName<PrivateOp>(b, context, loc);
  testDataOpName<FirstprivateOp>(b, context, loc);
  testDataOpName<ReductionOp>(b, context, loc);
  testDataOpName<DevicePtrOp>(b, context, loc);
  testDataOpName<PresentOp>(b, context, loc);
  testDataOpName<CopyinOp>(b, context, loc);
  testDataOpName<CreateOp>(b, context, loc);
  testDataOpName<NoCreateOp>(b, context, loc);
  testDataOpName<AttachOp>(b, context, loc);
  testDataOpName<GetDevicePtrOp>(b, context, loc);
  testDataOpName<UpdateDeviceOp>(b, context, loc);
  testDataOpName<UseDeviceOp>(b, context, loc);
  testDataOpName<DeclareDeviceResidentOp>(b, context, loc);
  testDataOpName<DeclareLinkOp>(b, context, loc);
  testDataOpName<CacheOp>(b, context, loc);
  testDataOpName<CopyoutOp>(b, context, loc);
  testDataOpName<UpdateHostOp>(b, context, loc);
  testDataOpName<DeleteOp>(b, context, loc);
  testDataOpName<DetachOp>(b, context, loc);
}

template <typename Op>
void testDataOpStructured(OpBuilder &b, MLIRContext &context, Location loc) {
  auto memrefTy = MemRefType::get({}, b.getI32Type());
  OwningOpRef<memref::AllocaOp> varPtrOp =
      b.create<memref::AllocaOp>(loc, memrefTy);
  OwningOpRef<GetDevicePtrOp> accPtrOp =
      b.create<GetDevicePtrOp>(loc, varPtrOp->getResult(), true, true);

  OwningOpRef<Op> op;
  if constexpr (std::is_same<Op, CopyoutOp>() ||
                std::is_same<Op, UpdateHostOp>()) {
    op = b.create<Op>(loc, /*accPtr=*/accPtrOp->getResult(),
                      varPtrOp->getResult(),
                      /*structured=*/true, /*implicit=*/true);
  } else if constexpr (std::is_same<Op, DeleteOp>() ||
                       std::is_same<Op, DetachOp>()) {
    op = b.create<Op>(loc, /*accPtr=*/accPtrOp->getResult(),
                      /*structured=*/true, /*implicit=*/true);
  } else {
    op = b.create<Op>(loc, varPtrOp->getResult(),
                      /*structured=*/true, /*implicit=*/true);
  }

  EXPECT_EQ(op->getStructured(), getStructuredFlag(op.get()));
  EXPECT_EQ(op->getStructured(), true);
  setStructuredFlag(op.get(), false);
  EXPECT_EQ(op->getStructured(), getStructuredFlag(op.get()));
  EXPECT_EQ(op->getStructured(), false);
}

TEST_F(OpenACCUtilsTest, dataOpStructured) {
  testDataOpStructured<PrivateOp>(b, context, loc);
  testDataOpStructured<FirstprivateOp>(b, context, loc);
  testDataOpStructured<ReductionOp>(b, context, loc);
  testDataOpStructured<DevicePtrOp>(b, context, loc);
  testDataOpStructured<PresentOp>(b, context, loc);
  testDataOpStructured<CopyinOp>(b, context, loc);
  testDataOpStructured<CreateOp>(b, context, loc);
  testDataOpStructured<NoCreateOp>(b, context, loc);
  testDataOpStructured<AttachOp>(b, context, loc);
  testDataOpStructured<GetDevicePtrOp>(b, context, loc);
  testDataOpStructured<UpdateDeviceOp>(b, context, loc);
  testDataOpStructured<UseDeviceOp>(b, context, loc);
  testDataOpStructured<DeclareDeviceResidentOp>(b, context, loc);
  testDataOpStructured<DeclareLinkOp>(b, context, loc);
  testDataOpStructured<CacheOp>(b, context, loc);
  testDataOpStructured<CopyoutOp>(b, context, loc);
  testDataOpStructured<UpdateHostOp>(b, context, loc);
  testDataOpStructured<DeleteOp>(b, context, loc);
  testDataOpStructured<DetachOp>(b, context, loc);
}

template <typename Op>
void testDataOpImplicit(OpBuilder &b, MLIRContext &context, Location loc) {
  auto memrefTy = MemRefType::get({}, b.getI32Type());
  OwningOpRef<memref::AllocaOp> varPtrOp =
      b.create<memref::AllocaOp>(loc, memrefTy);
  OwningOpRef<GetDevicePtrOp> accPtrOp =
      b.create<GetDevicePtrOp>(loc, varPtrOp->getResult(), true, true);

  OwningOpRef<Op> op;
  if constexpr (std::is_same<Op, CopyoutOp>() ||
                std::is_same<Op, UpdateHostOp>()) {
    op = b.create<Op>(loc, /*accPtr=*/accPtrOp->getResult(),
                      varPtrOp->getResult(),
                      /*structured=*/true, /*implicit=*/true);
  } else if constexpr (std::is_same<Op, DeleteOp>() ||
                       std::is_same<Op, DetachOp>()) {
    op = b.create<Op>(loc, /*accPtr=*/accPtrOp->getResult(),
                      /*structured=*/true, /*implicit=*/true);
  } else {
    op = b.create<Op>(loc, varPtrOp->getResult(),
                      /*structured=*/true, /*implicit=*/true);
  }

  EXPECT_EQ(op->getImplicit(), getImplicitFlag(op.get()));
  EXPECT_EQ(op->getImplicit(), true);
  setImplicitFlag(op.get(), false);
  EXPECT_EQ(op->getImplicit(), getImplicitFlag(op.get()));
  EXPECT_EQ(op->getImplicit(), false);
}

TEST_F(OpenACCUtilsTest, dataOpImplicit) {
  testDataOpImplicit<PrivateOp>(b, context, loc);
  testDataOpImplicit<FirstprivateOp>(b, context, loc);
  testDataOpImplicit<ReductionOp>(b, context, loc);
  testDataOpImplicit<DevicePtrOp>(b, context, loc);
  testDataOpImplicit<PresentOp>(b, context, loc);
  testDataOpImplicit<CopyinOp>(b, context, loc);
  testDataOpImplicit<CreateOp>(b, context, loc);
  testDataOpImplicit<NoCreateOp>(b, context, loc);
  testDataOpImplicit<AttachOp>(b, context, loc);
  testDataOpImplicit<GetDevicePtrOp>(b, context, loc);
  testDataOpImplicit<UpdateDeviceOp>(b, context, loc);
  testDataOpImplicit<UseDeviceOp>(b, context, loc);
  testDataOpImplicit<DeclareDeviceResidentOp>(b, context, loc);
  testDataOpImplicit<DeclareLinkOp>(b, context, loc);
  testDataOpImplicit<CacheOp>(b, context, loc);
  testDataOpImplicit<CopyoutOp>(b, context, loc);
  testDataOpImplicit<UpdateHostOp>(b, context, loc);
  testDataOpImplicit<DeleteOp>(b, context, loc);
  testDataOpImplicit<DetachOp>(b, context, loc);
}

template <typename Op>
void testDataOpDataClause(OpBuilder &b, MLIRContext &context, Location loc,
                          DataClause dataClause) {
  auto memrefTy = MemRefType::get({}, b.getI32Type());
  OwningOpRef<memref::AllocaOp> varPtrOp =
      b.create<memref::AllocaOp>(loc, memrefTy);
  OwningOpRef<GetDevicePtrOp> accPtrOp =
      b.create<GetDevicePtrOp>(loc, varPtrOp->getResult(), true, true);

  OwningOpRef<Op> op;
  if constexpr (std::is_same<Op, CopyoutOp>() ||
                std::is_same<Op, UpdateHostOp>()) {
    op = b.create<Op>(loc, /*accPtr=*/accPtrOp->getResult(),
                      varPtrOp->getResult(),
                      /*structured=*/true, /*implicit=*/true);
  } else if constexpr (std::is_same<Op, DeleteOp>() ||
                       std::is_same<Op, DetachOp>()) {
    op = b.create<Op>(loc, /*accPtr=*/accPtrOp->getResult(),
                      /*structured=*/true, /*implicit=*/true);
  } else {
    op = b.create<Op>(loc, varPtrOp->getResult(),
                      /*structured=*/true, /*implicit=*/true);
  }

  EXPECT_EQ(op->getDataClause(), getDataClause(op.get()).value());
  EXPECT_EQ(op->getDataClause(), dataClause);
  setDataClause(op.get(), DataClause::acc_getdeviceptr);
  EXPECT_EQ(op->getDataClause(), getDataClause(op.get()).value());
  EXPECT_EQ(op->getDataClause(), DataClause::acc_getdeviceptr);
}

TEST_F(OpenACCUtilsTest, dataOpDataClause) {
  testDataOpDataClause<PrivateOp>(b, context, loc, DataClause::acc_private);
  testDataOpDataClause<FirstprivateOp>(b, context, loc,
                                       DataClause::acc_firstprivate);
  testDataOpDataClause<ReductionOp>(b, context, loc, DataClause::acc_reduction);
  testDataOpDataClause<DevicePtrOp>(b, context, loc, DataClause::acc_deviceptr);
  testDataOpDataClause<PresentOp>(b, context, loc, DataClause::acc_present);
  testDataOpDataClause<CopyinOp>(b, context, loc, DataClause::acc_copyin);
  testDataOpDataClause<CreateOp>(b, context, loc, DataClause::acc_create);
  testDataOpDataClause<NoCreateOp>(b, context, loc, DataClause::acc_no_create);
  testDataOpDataClause<AttachOp>(b, context, loc, DataClause::acc_attach);
  testDataOpDataClause<GetDevicePtrOp>(b, context, loc,
                                       DataClause::acc_getdeviceptr);
  testDataOpDataClause<UpdateDeviceOp>(b, context, loc,
                                       DataClause::acc_update_device);
  testDataOpDataClause<UseDeviceOp>(b, context, loc,
                                    DataClause::acc_use_device);
  testDataOpDataClause<DeclareDeviceResidentOp>(
      b, context, loc, DataClause::acc_declare_device_resident);
  testDataOpDataClause<DeclareLinkOp>(b, context, loc,
                                      DataClause::acc_declare_link);
  testDataOpDataClause<CacheOp>(b, context, loc, DataClause::acc_cache);
  testDataOpDataClause<CopyoutOp>(b, context, loc, DataClause::acc_copyout);
  testDataOpDataClause<UpdateHostOp>(b, context, loc,
                                     DataClause::acc_update_host);
  testDataOpDataClause<DeleteOp>(b, context, loc, DataClause::acc_delete);
  testDataOpDataClause<DetachOp>(b, context, loc, DataClause::acc_detach);
}
