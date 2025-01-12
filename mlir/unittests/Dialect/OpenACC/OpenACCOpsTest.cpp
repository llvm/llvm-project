//===- OpenACCOpsTest.cpp - Unit tests for OpenACC ops --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::acc;

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

class OpenACCOpsTest : public ::testing::Test {
protected:
  OpenACCOpsTest() : b(&context), loc(UnknownLoc::get(&context)) {
    context.loadDialect<acc::OpenACCDialect, arith::ArithDialect,
                        memref::MemRefDialect>();
  }

  MLIRContext context;
  OpBuilder b;
  Location loc;
  llvm::SmallVector<DeviceType> dtypes = {
      DeviceType::None,    DeviceType::Star, DeviceType::Multicore,
      DeviceType::Default, DeviceType::Host, DeviceType::Nvidia,
      DeviceType::Radeon};
  llvm::SmallVector<DeviceType> dtypesWithoutNone = {
      DeviceType::Star, DeviceType::Multicore, DeviceType::Default,
      DeviceType::Host, DeviceType::Nvidia,    DeviceType::Radeon};
};

template <typename Op>
void testAsyncOnly(OpBuilder &b, MLIRContext &context, Location loc,
                   llvm::SmallVector<DeviceType> &dtypes) {
  OwningOpRef<Op> op = b.create<Op>(loc, TypeRange{}, ValueRange{});
  EXPECT_FALSE(op->hasAsyncOnly());
  for (auto d : dtypes)
    EXPECT_FALSE(op->hasAsyncOnly(d));

  auto dtypeNone = DeviceTypeAttr::get(&context, DeviceType::None);
  op->setAsyncOnlyAttr(b.getArrayAttr({dtypeNone}));
  EXPECT_TRUE(op->hasAsyncOnly());
  EXPECT_TRUE(op->hasAsyncOnly(DeviceType::None));
  op->removeAsyncOnlyAttr();

  auto dtypeHost = DeviceTypeAttr::get(&context, DeviceType::Host);
  op->setAsyncOnlyAttr(b.getArrayAttr({dtypeHost}));
  EXPECT_TRUE(op->hasAsyncOnly(DeviceType::Host));
  EXPECT_FALSE(op->hasAsyncOnly());
  op->removeAsyncOnlyAttr();

  auto dtypeStar = DeviceTypeAttr::get(&context, DeviceType::Star);
  op->setAsyncOnlyAttr(b.getArrayAttr({dtypeHost, dtypeStar}));
  EXPECT_TRUE(op->hasAsyncOnly(DeviceType::Star));
  EXPECT_TRUE(op->hasAsyncOnly(DeviceType::Host));
  EXPECT_FALSE(op->hasAsyncOnly());

  op->removeAsyncOnlyAttr();
}

TEST_F(OpenACCOpsTest, asyncOnlyTest) {
  testAsyncOnly<ParallelOp>(b, context, loc, dtypes);
  testAsyncOnly<KernelsOp>(b, context, loc, dtypes);
  testAsyncOnly<SerialOp>(b, context, loc, dtypes);
}

template <typename Op>
void testAsyncValue(OpBuilder &b, MLIRContext &context, Location loc,
                    llvm::SmallVector<DeviceType> &dtypes) {
  OwningOpRef<Op> op = b.create<Op>(loc, TypeRange{}, ValueRange{});

  mlir::Value empty;
  EXPECT_EQ(op->getAsyncValue(), empty);
  for (auto d : dtypes)
    EXPECT_EQ(op->getAsyncValue(d), empty);

  OwningOpRef<arith::ConstantIndexOp> val =
      b.create<arith::ConstantIndexOp>(loc, 1);
  auto dtypeNvidia = DeviceTypeAttr::get(&context, DeviceType::Nvidia);
  op->setAsyncOperandsDeviceTypeAttr(b.getArrayAttr({dtypeNvidia}));
  op->getAsyncOperandsMutable().assign(val->getResult());
  EXPECT_EQ(op->getAsyncValue(), empty);
  EXPECT_EQ(op->getAsyncValue(DeviceType::Nvidia), val->getResult());

  op->getAsyncOperandsMutable().clear();
  op->removeAsyncOperandsDeviceTypeAttr();
}

TEST_F(OpenACCOpsTest, asyncValueTest) {
  testAsyncValue<ParallelOp>(b, context, loc, dtypes);
  testAsyncValue<KernelsOp>(b, context, loc, dtypes);
  testAsyncValue<SerialOp>(b, context, loc, dtypes);
}

template <typename Op>
void testNumGangsValues(OpBuilder &b, MLIRContext &context, Location loc,
                        llvm::SmallVector<DeviceType> &dtypes,
                        llvm::SmallVector<DeviceType> &dtypesWithoutNone) {
  OwningOpRef<Op> op = b.create<Op>(loc, TypeRange{}, ValueRange{});
  EXPECT_EQ(op->getNumGangsValues().begin(), op->getNumGangsValues().end());

  OwningOpRef<arith::ConstantIndexOp> val1 =
      b.create<arith::ConstantIndexOp>(loc, 1);
  OwningOpRef<arith::ConstantIndexOp> val2 =
      b.create<arith::ConstantIndexOp>(loc, 4);
  auto dtypeNone = DeviceTypeAttr::get(&context, DeviceType::None);
  op->getNumGangsMutable().assign(val1->getResult());
  op->setNumGangsDeviceTypeAttr(b.getArrayAttr({dtypeNone}));
  op->setNumGangsSegments(b.getDenseI32ArrayAttr({1}));
  EXPECT_EQ(op->getNumGangsValues().front(), val1->getResult());
  for (auto d : dtypesWithoutNone)
    EXPECT_EQ(op->getNumGangsValues(d).begin(), op->getNumGangsValues(d).end());

  op->getNumGangsMutable().clear();
  op->removeNumGangsDeviceTypeAttr();
  op->removeNumGangsSegmentsAttr();
  for (auto d : dtypes)
    EXPECT_EQ(op->getNumGangsValues(d).begin(), op->getNumGangsValues(d).end());

  op->getNumGangsMutable().append(val1->getResult());
  op->getNumGangsMutable().append(val2->getResult());
  op->setNumGangsDeviceTypeAttr(
      b.getArrayAttr({DeviceTypeAttr::get(&context, DeviceType::Host),
                      DeviceTypeAttr::get(&context, DeviceType::Star)}));
  op->setNumGangsSegments(b.getDenseI32ArrayAttr({1, 1}));
  EXPECT_EQ(op->getNumGangsValues(DeviceType::None).begin(),
            op->getNumGangsValues(DeviceType::None).end());
  EXPECT_EQ(op->getNumGangsValues(DeviceType::Host).front(), val1->getResult());
  EXPECT_EQ(op->getNumGangsValues(DeviceType::Star).front(), val2->getResult());

  op->getNumGangsMutable().clear();
  op->removeNumGangsDeviceTypeAttr();
  op->removeNumGangsSegmentsAttr();
  for (auto d : dtypes)
    EXPECT_EQ(op->getNumGangsValues(d).begin(), op->getNumGangsValues(d).end());

  op->getNumGangsMutable().append(val1->getResult());
  op->getNumGangsMutable().append(val2->getResult());
  op->getNumGangsMutable().append(val1->getResult());
  op->setNumGangsDeviceTypeAttr(
      b.getArrayAttr({DeviceTypeAttr::get(&context, DeviceType::Default),
                      DeviceTypeAttr::get(&context, DeviceType::Multicore)}));
  op->setNumGangsSegments(b.getDenseI32ArrayAttr({2, 1}));
  EXPECT_EQ(op->getNumGangsValues(DeviceType::None).begin(),
            op->getNumGangsValues(DeviceType::None).end());
  EXPECT_EQ(op->getNumGangsValues(DeviceType::Default).front(),
            val1->getResult());
  EXPECT_EQ(op->getNumGangsValues(DeviceType::Default).drop_front().front(),
            val2->getResult());
  EXPECT_EQ(op->getNumGangsValues(DeviceType::Multicore).front(),
            val1->getResult());

  op->getNumGangsMutable().clear();
  op->removeNumGangsDeviceTypeAttr();
  op->removeNumGangsSegmentsAttr();
}

TEST_F(OpenACCOpsTest, numGangsValuesTest) {
  testNumGangsValues<ParallelOp>(b, context, loc, dtypes, dtypesWithoutNone);
  testNumGangsValues<KernelsOp>(b, context, loc, dtypes, dtypesWithoutNone);
}

template <typename Op>
void testVectorLength(OpBuilder &b, MLIRContext &context, Location loc,
                      llvm::SmallVector<DeviceType> &dtypes) {
  OwningOpRef<Op> op = b.create<Op>(loc, TypeRange{}, ValueRange{});

  mlir::Value empty;
  EXPECT_EQ(op->getVectorLengthValue(), empty);
  for (auto d : dtypes)
    EXPECT_EQ(op->getVectorLengthValue(d), empty);

  OwningOpRef<arith::ConstantIndexOp> val =
      b.create<arith::ConstantIndexOp>(loc, 1);
  auto dtypeNvidia = DeviceTypeAttr::get(&context, DeviceType::Nvidia);
  op->setVectorLengthDeviceTypeAttr(b.getArrayAttr({dtypeNvidia}));
  op->getVectorLengthMutable().assign(val->getResult());
  EXPECT_EQ(op->getVectorLengthValue(), empty);
  EXPECT_EQ(op->getVectorLengthValue(DeviceType::Nvidia), val->getResult());

  op->getVectorLengthMutable().clear();
  op->removeVectorLengthDeviceTypeAttr();
}

TEST_F(OpenACCOpsTest, vectorLengthTest) {
  testVectorLength<ParallelOp>(b, context, loc, dtypes);
  testVectorLength<KernelsOp>(b, context, loc, dtypes);
}

template <typename Op>
void testWaitOnly(OpBuilder &b, MLIRContext &context, Location loc,
                  llvm::SmallVector<DeviceType> &dtypes,
                  llvm::SmallVector<DeviceType> &dtypesWithoutNone) {
  OwningOpRef<Op> op = b.create<Op>(loc, TypeRange{}, ValueRange{});
  EXPECT_FALSE(op->hasWaitOnly());
  for (auto d : dtypes)
    EXPECT_FALSE(op->hasWaitOnly(d));

  auto dtypeNone = DeviceTypeAttr::get(&context, DeviceType::None);
  op->setWaitOnlyAttr(b.getArrayAttr({dtypeNone}));
  EXPECT_TRUE(op->hasWaitOnly());
  EXPECT_TRUE(op->hasWaitOnly(DeviceType::None));
  for (auto d : dtypesWithoutNone)
    EXPECT_FALSE(op->hasWaitOnly(d));
  op->removeWaitOnlyAttr();

  auto dtypeHost = DeviceTypeAttr::get(&context, DeviceType::Host);
  op->setWaitOnlyAttr(b.getArrayAttr({dtypeHost}));
  EXPECT_TRUE(op->hasWaitOnly(DeviceType::Host));
  EXPECT_FALSE(op->hasWaitOnly());
  op->removeWaitOnlyAttr();

  auto dtypeStar = DeviceTypeAttr::get(&context, DeviceType::Star);
  op->setWaitOnlyAttr(b.getArrayAttr({dtypeHost, dtypeStar}));
  EXPECT_TRUE(op->hasWaitOnly(DeviceType::Star));
  EXPECT_TRUE(op->hasWaitOnly(DeviceType::Host));
  EXPECT_FALSE(op->hasWaitOnly());

  op->removeWaitOnlyAttr();
}

TEST_F(OpenACCOpsTest, waitOnlyTest) {
  testWaitOnly<ParallelOp>(b, context, loc, dtypes, dtypesWithoutNone);
  testWaitOnly<KernelsOp>(b, context, loc, dtypes, dtypesWithoutNone);
  testWaitOnly<SerialOp>(b, context, loc, dtypes, dtypesWithoutNone);
  testWaitOnly<UpdateOp>(b, context, loc, dtypes, dtypesWithoutNone);
  testWaitOnly<DataOp>(b, context, loc, dtypes, dtypesWithoutNone);
}

template <typename Op>
void testWaitValues(OpBuilder &b, MLIRContext &context, Location loc,
                    llvm::SmallVector<DeviceType> &dtypes,
                    llvm::SmallVector<DeviceType> &dtypesWithoutNone) {
  OwningOpRef<Op> op = b.create<Op>(loc, TypeRange{}, ValueRange{});
  EXPECT_EQ(op->getWaitValues().begin(), op->getWaitValues().end());

  OwningOpRef<arith::ConstantIndexOp> val1 =
      b.create<arith::ConstantIndexOp>(loc, 1);
  OwningOpRef<arith::ConstantIndexOp> val2 =
      b.create<arith::ConstantIndexOp>(loc, 4);
  OwningOpRef<arith::ConstantIndexOp> val3 =
      b.create<arith::ConstantIndexOp>(loc, 5);
  auto dtypeNone = DeviceTypeAttr::get(&context, DeviceType::None);
  op->getWaitOperandsMutable().assign(val1->getResult());
  op->setWaitOperandsDeviceTypeAttr(b.getArrayAttr({dtypeNone}));
  op->setWaitOperandsSegments(b.getDenseI32ArrayAttr({1}));
  op->setHasWaitDevnumAttr(b.getBoolArrayAttr({false}));
  EXPECT_EQ(op->getWaitValues().front(), val1->getResult());
  for (auto d : dtypesWithoutNone)
    EXPECT_TRUE(op->getWaitValues(d).empty());

  op->getWaitOperandsMutable().clear();
  op->removeWaitOperandsDeviceTypeAttr();
  op->removeWaitOperandsSegmentsAttr();
  op->removeHasWaitDevnumAttr();
  for (auto d : dtypes)
    EXPECT_TRUE(op->getWaitValues(d).empty());

  op->getWaitOperandsMutable().append(val1->getResult());
  op->getWaitOperandsMutable().append(val2->getResult());
  op->setWaitOperandsDeviceTypeAttr(
      b.getArrayAttr({DeviceTypeAttr::get(&context, DeviceType::Host),
                      DeviceTypeAttr::get(&context, DeviceType::Star)}));
  op->setWaitOperandsSegments(b.getDenseI32ArrayAttr({1, 1}));
  op->setHasWaitDevnumAttr(b.getBoolArrayAttr({false, false}));
  EXPECT_EQ(op->getWaitValues(DeviceType::None).begin(),
            op->getWaitValues(DeviceType::None).end());
  EXPECT_EQ(op->getWaitValues(DeviceType::Host).front(), val1->getResult());
  EXPECT_EQ(op->getWaitValues(DeviceType::Star).front(), val2->getResult());

  op->getWaitOperandsMutable().clear();
  op->removeWaitOperandsDeviceTypeAttr();
  op->removeWaitOperandsSegmentsAttr();
  op->removeHasWaitDevnumAttr();
  for (auto d : dtypes)
    EXPECT_TRUE(op->getWaitValues(d).empty());

  op->getWaitOperandsMutable().append(val1->getResult());
  op->getWaitOperandsMutable().append(val2->getResult());
  op->getWaitOperandsMutable().append(val1->getResult());
  op->setWaitOperandsDeviceTypeAttr(
      b.getArrayAttr({DeviceTypeAttr::get(&context, DeviceType::Default),
                      DeviceTypeAttr::get(&context, DeviceType::Multicore)}));
  op->setWaitOperandsSegments(b.getDenseI32ArrayAttr({2, 1}));
  op->setHasWaitDevnumAttr(b.getBoolArrayAttr({false, false}));
  EXPECT_EQ(op->getWaitValues(DeviceType::None).begin(),
            op->getWaitValues(DeviceType::None).end());
  EXPECT_EQ(op->getWaitValues(DeviceType::Default).front(), val1->getResult());
  EXPECT_EQ(op->getWaitValues(DeviceType::Default).drop_front().front(),
            val2->getResult());
  EXPECT_EQ(op->getWaitValues(DeviceType::Multicore).front(),
            val1->getResult());

  op->getWaitOperandsMutable().clear();
  op->removeWaitOperandsDeviceTypeAttr();
  op->removeWaitOperandsSegmentsAttr();

  op->getWaitOperandsMutable().append(val3->getResult());
  op->getWaitOperandsMutable().append(val2->getResult());
  op->getWaitOperandsMutable().append(val1->getResult());
  op->setWaitOperandsDeviceTypeAttr(
      b.getArrayAttr({DeviceTypeAttr::get(&context, DeviceType::Multicore)}));
  op->setHasWaitDevnumAttr(b.getBoolArrayAttr({true}));
  op->setWaitOperandsSegments(b.getDenseI32ArrayAttr({3}));
  EXPECT_EQ(op->getWaitValues(DeviceType::None).begin(),
            op->getWaitValues(DeviceType::None).end());
  EXPECT_FALSE(op->getWaitDevnum());

  EXPECT_EQ(op->getWaitDevnum(DeviceType::Multicore), val3->getResult());
  EXPECT_EQ(op->getWaitValues(DeviceType::Multicore).front(),
            val2->getResult());
  EXPECT_EQ(op->getWaitValues(DeviceType::Multicore).drop_front().front(),
            val1->getResult());

  op->getWaitOperandsMutable().clear();
  op->removeWaitOperandsDeviceTypeAttr();
  op->removeWaitOperandsSegmentsAttr();
  op->removeHasWaitDevnumAttr();
}

TEST_F(OpenACCOpsTest, waitValuesTest) {
  testWaitValues<KernelsOp>(b, context, loc, dtypes, dtypesWithoutNone);
  testWaitValues<ParallelOp>(b, context, loc, dtypes, dtypesWithoutNone);
  testWaitValues<SerialOp>(b, context, loc, dtypes, dtypesWithoutNone);
}

TEST_F(OpenACCOpsTest, loopOpGangVectorWorkerTest) {
  OwningOpRef<LoopOp> op = b.create<LoopOp>(loc, TypeRange{}, ValueRange{});
  EXPECT_FALSE(op->hasGang());
  EXPECT_FALSE(op->hasVector());
  EXPECT_FALSE(op->hasWorker());
  for (auto d : dtypes) {
    EXPECT_FALSE(op->hasGang(d));
    EXPECT_FALSE(op->hasVector(d));
    EXPECT_FALSE(op->hasWorker(d));
  }

  auto dtypeNone = DeviceTypeAttr::get(&context, DeviceType::None);
  op->setGangAttr(b.getArrayAttr({dtypeNone}));
  EXPECT_TRUE(op->hasGang());
  EXPECT_TRUE(op->hasGang(DeviceType::None));
  for (auto d : dtypesWithoutNone)
    EXPECT_FALSE(op->hasGang(d));
  for (auto d : dtypes) {
    EXPECT_FALSE(op->hasVector(d));
    EXPECT_FALSE(op->hasWorker(d));
  }
  op->removeGangAttr();

  op->setWorkerAttr(b.getArrayAttr({dtypeNone}));
  EXPECT_TRUE(op->hasWorker());
  EXPECT_TRUE(op->hasWorker(DeviceType::None));
  for (auto d : dtypesWithoutNone)
    EXPECT_FALSE(op->hasWorker(d));
  for (auto d : dtypes) {
    EXPECT_FALSE(op->hasGang(d));
    EXPECT_FALSE(op->hasVector(d));
  }
  op->removeWorkerAttr();

  op->setVectorAttr(b.getArrayAttr({dtypeNone}));
  EXPECT_TRUE(op->hasVector());
  EXPECT_TRUE(op->hasVector(DeviceType::None));
  for (auto d : dtypesWithoutNone)
    EXPECT_FALSE(op->hasVector(d));
  for (auto d : dtypes) {
    EXPECT_FALSE(op->hasGang(d));
    EXPECT_FALSE(op->hasWorker(d));
  }
  op->removeVectorAttr();
}

TEST_F(OpenACCOpsTest, routineOpTest) {
  OwningOpRef<RoutineOp> op =
      b.create<RoutineOp>(loc, TypeRange{}, ValueRange{});

  EXPECT_FALSE(op->hasSeq());
  EXPECT_FALSE(op->hasVector());
  EXPECT_FALSE(op->hasWorker());

  for (auto d : dtypes) {
    EXPECT_FALSE(op->hasSeq(d));
    EXPECT_FALSE(op->hasVector(d));
    EXPECT_FALSE(op->hasWorker(d));
  }

  auto dtypeNone = DeviceTypeAttr::get(&context, DeviceType::None);
  op->setSeqAttr(b.getArrayAttr({dtypeNone}));
  EXPECT_TRUE(op->hasSeq());
  for (auto d : dtypesWithoutNone)
    EXPECT_FALSE(op->hasSeq(d));
  op->removeSeqAttr();

  op->setVectorAttr(b.getArrayAttr({dtypeNone}));
  EXPECT_TRUE(op->hasVector());
  for (auto d : dtypesWithoutNone)
    EXPECT_FALSE(op->hasVector(d));
  op->removeVectorAttr();

  op->setWorkerAttr(b.getArrayAttr({dtypeNone}));
  EXPECT_TRUE(op->hasWorker());
  for (auto d : dtypesWithoutNone)
    EXPECT_FALSE(op->hasWorker(d));
  op->removeWorkerAttr();

  op->setGangAttr(b.getArrayAttr({dtypeNone}));
  EXPECT_TRUE(op->hasGang());
  for (auto d : dtypesWithoutNone)
    EXPECT_FALSE(op->hasGang(d));
  op->removeGangAttr();

  op->setGangDimDeviceTypeAttr(b.getArrayAttr({dtypeNone}));
  op->setGangDimAttr(b.getArrayAttr({b.getIntegerAttr(b.getI64Type(), 8)}));
  EXPECT_TRUE(op->getGangDimValue().has_value());
  EXPECT_EQ(op->getGangDimValue().value(), 8);
  for (auto d : dtypesWithoutNone)
    EXPECT_FALSE(op->getGangDimValue(d).has_value());
  op->removeGangDimDeviceTypeAttr();
  op->removeGangDimAttr();

  op->setBindNameDeviceTypeAttr(b.getArrayAttr({dtypeNone}));
  op->setBindNameAttr(b.getArrayAttr({b.getStringAttr("fname")}));
  EXPECT_TRUE(op->getBindNameValue().has_value());
  EXPECT_EQ(op->getBindNameValue().value(), "fname");
  for (auto d : dtypesWithoutNone)
    EXPECT_FALSE(op->getBindNameValue(d).has_value());
  op->removeBindNameDeviceTypeAttr();
  op->removeBindNameAttr();
}

template <typename Op>
void testShortDataEntryOpBuilders(OpBuilder &b, MLIRContext &context,
                                  Location loc, DataClause dataClause) {
  auto memrefTy = MemRefType::get({}, b.getI32Type());
  OwningOpRef<memref::AllocaOp> varPtrOp =
      b.create<memref::AllocaOp>(loc, memrefTy);

  TypedValue<PointerLikeType> varPtr =
      cast<TypedValue<PointerLikeType>>(varPtrOp->getResult());
  OwningOpRef<Op> op = b.create<Op>(loc, varPtr,
                                    /*structured=*/true, /*implicit=*/true);

  EXPECT_EQ(op->getVarPtr(), varPtr);
  EXPECT_EQ(op->getType(), memrefTy);
  EXPECT_EQ(op->getDataClause(), dataClause);
  EXPECT_TRUE(op->getImplicit());
  EXPECT_TRUE(op->getStructured());
  EXPECT_TRUE(op->getBounds().empty());
  EXPECT_FALSE(op->getVarPtrPtr());

  OwningOpRef<Op> op2 = b.create<Op>(loc, varPtr,
                                     /*structured=*/false, /*implicit=*/false);
  EXPECT_FALSE(op2->getImplicit());
  EXPECT_FALSE(op2->getStructured());

  OwningOpRef<arith::ConstantIndexOp> extent =
      b.create<arith::ConstantIndexOp>(loc, 1);
  OwningOpRef<DataBoundsOp> bounds =
      b.create<DataBoundsOp>(loc, extent->getResult());
  OwningOpRef<Op> opWithBounds =
      b.create<Op>(loc, varPtr,
                   /*structured=*/true, /*implicit=*/true, bounds->getResult());
  EXPECT_FALSE(opWithBounds->getBounds().empty());
  EXPECT_EQ(opWithBounds->getBounds().back(), bounds->getResult());

  OwningOpRef<Op> opWithName =
      b.create<Op>(loc, varPtr,
                   /*structured=*/true, /*implicit=*/true, "varName");
  EXPECT_EQ(opWithName->getNameAttr().str(), "varName");
}

TEST_F(OpenACCOpsTest, shortDataEntryOpBuilder) {
  testShortDataEntryOpBuilders<PrivateOp>(b, context, loc,
                                          DataClause::acc_private);
  testShortDataEntryOpBuilders<FirstprivateOp>(b, context, loc,
                                               DataClause::acc_firstprivate);
  testShortDataEntryOpBuilders<ReductionOp>(b, context, loc,
                                            DataClause::acc_reduction);
  testShortDataEntryOpBuilders<DevicePtrOp>(b, context, loc,
                                            DataClause::acc_deviceptr);
  testShortDataEntryOpBuilders<PresentOp>(b, context, loc,
                                          DataClause::acc_present);
  testShortDataEntryOpBuilders<CopyinOp>(b, context, loc,
                                         DataClause::acc_copyin);
  testShortDataEntryOpBuilders<CreateOp>(b, context, loc,
                                         DataClause::acc_create);
  testShortDataEntryOpBuilders<NoCreateOp>(b, context, loc,
                                           DataClause::acc_no_create);
  testShortDataEntryOpBuilders<AttachOp>(b, context, loc,
                                         DataClause::acc_attach);
  testShortDataEntryOpBuilders<GetDevicePtrOp>(b, context, loc,
                                               DataClause::acc_getdeviceptr);
  testShortDataEntryOpBuilders<UpdateDeviceOp>(b, context, loc,
                                               DataClause::acc_update_device);
  testShortDataEntryOpBuilders<UseDeviceOp>(b, context, loc,
                                            DataClause::acc_use_device);
  testShortDataEntryOpBuilders<DeclareDeviceResidentOp>(
      b, context, loc, DataClause::acc_declare_device_resident);
  testShortDataEntryOpBuilders<DeclareLinkOp>(b, context, loc,
                                              DataClause::acc_declare_link);
  testShortDataEntryOpBuilders<CacheOp>(b, context, loc, DataClause::acc_cache);
}

template <typename Op>
void testShortDataExitOpBuilders(OpBuilder &b, MLIRContext &context,
                                 Location loc, DataClause dataClause) {
  auto memrefTy = MemRefType::get({}, b.getI32Type());
  OwningOpRef<memref::AllocaOp> varPtrOp =
      b.create<memref::AllocaOp>(loc, memrefTy);
  TypedValue<PointerLikeType> varPtr =
      cast<TypedValue<PointerLikeType>>(varPtrOp->getResult());

  OwningOpRef<GetDevicePtrOp> accPtrOp = b.create<GetDevicePtrOp>(
      loc, varPtr, /*structured=*/true, /*implicit=*/true);
  TypedValue<PointerLikeType> accPtr =
      cast<TypedValue<PointerLikeType>>(accPtrOp->getResult());

  OwningOpRef<Op> op = b.create<Op>(loc, accPtr, varPtr,
                                    /*structured=*/true, /*implicit=*/true);

  EXPECT_EQ(op->getVarPtr(), varPtr);
  EXPECT_EQ(op->getAccPtr(), accPtr);
  EXPECT_EQ(op->getDataClause(), dataClause);
  EXPECT_TRUE(op->getImplicit());
  EXPECT_TRUE(op->getStructured());
  EXPECT_TRUE(op->getBounds().empty());

  OwningOpRef<Op> op2 = b.create<Op>(loc, accPtr, varPtr,
                                     /*structured=*/false, /*implicit=*/false);
  EXPECT_FALSE(op2->getImplicit());
  EXPECT_FALSE(op2->getStructured());

  OwningOpRef<arith::ConstantIndexOp> extent =
      b.create<arith::ConstantIndexOp>(loc, 1);
  OwningOpRef<DataBoundsOp> bounds =
      b.create<DataBoundsOp>(loc, extent->getResult());
  OwningOpRef<Op> opWithBounds =
      b.create<Op>(loc, accPtr, varPtr,
                   /*structured=*/true, /*implicit=*/true, bounds->getResult());
  EXPECT_FALSE(opWithBounds->getBounds().empty());
  EXPECT_EQ(opWithBounds->getBounds().back(), bounds->getResult());

  OwningOpRef<Op> opWithName =
      b.create<Op>(loc, accPtr, varPtr,
                   /*structured=*/true, /*implicit=*/true, "varName");
  EXPECT_EQ(opWithName->getNameAttr().str(), "varName");
}

TEST_F(OpenACCOpsTest, shortDataExitOpBuilder) {
  testShortDataExitOpBuilders<CopyoutOp>(b, context, loc,
                                         DataClause::acc_copyout);
  testShortDataExitOpBuilders<UpdateHostOp>(b, context, loc,
                                            DataClause::acc_update_host);
}

template <typename Op>
void testShortDataExitNoVarPtrOpBuilders(OpBuilder &b, MLIRContext &context,
                                         Location loc, DataClause dataClause) {
  auto memrefTy = MemRefType::get({}, b.getI32Type());
  OwningOpRef<memref::AllocaOp> varPtrOp =
      b.create<memref::AllocaOp>(loc, memrefTy);
  TypedValue<PointerLikeType> varPtr =
      cast<TypedValue<PointerLikeType>>(varPtrOp->getResult());

  OwningOpRef<GetDevicePtrOp> accPtrOp = b.create<GetDevicePtrOp>(
      loc, varPtr, /*structured=*/true, /*implicit=*/true);
  TypedValue<PointerLikeType> accPtr =
      cast<TypedValue<PointerLikeType>>(accPtrOp->getResult());

  OwningOpRef<Op> op = b.create<Op>(loc, accPtr,
                                    /*structured=*/true, /*implicit=*/true);

  EXPECT_EQ(op->getAccPtr(), accPtr);
  EXPECT_EQ(op->getDataClause(), dataClause);
  EXPECT_TRUE(op->getImplicit());
  EXPECT_TRUE(op->getStructured());
  EXPECT_TRUE(op->getBounds().empty());

  OwningOpRef<Op> op2 = b.create<Op>(loc, accPtr,
                                     /*structured=*/false, /*implicit=*/false);
  EXPECT_FALSE(op2->getImplicit());
  EXPECT_FALSE(op2->getStructured());

  OwningOpRef<arith::ConstantIndexOp> extent =
      b.create<arith::ConstantIndexOp>(loc, 1);
  OwningOpRef<DataBoundsOp> bounds =
      b.create<DataBoundsOp>(loc, extent->getResult());
  OwningOpRef<Op> opWithBounds =
      b.create<Op>(loc, accPtr,
                   /*structured=*/true, /*implicit=*/true, bounds->getResult());
  EXPECT_FALSE(opWithBounds->getBounds().empty());
  EXPECT_EQ(opWithBounds->getBounds().back(), bounds->getResult());

  OwningOpRef<Op> opWithName =
      b.create<Op>(loc, accPtr,
                   /*structured=*/true, /*implicit=*/true, "varName");
  EXPECT_EQ(opWithName->getNameAttr().str(), "varName");
}

TEST_F(OpenACCOpsTest, shortDataExitOpNoVarPtrBuilder) {
  testShortDataExitNoVarPtrOpBuilders<DeleteOp>(b, context, loc,
                                                DataClause::acc_delete);
  testShortDataExitNoVarPtrOpBuilders<DetachOp>(b, context, loc,
                                                DataClause::acc_detach);
}

template <typename Op>
void testShortDataEntryOpBuildersMappableVar(OpBuilder &b, MLIRContext &context,
                                             Location loc,
                                             DataClause dataClause) {
  auto int64Ty = b.getI64Type();
  auto memrefTy = MemRefType::get({}, int64Ty);
  OwningOpRef<memref::AllocaOp> varPtrOp =
      b.create<memref::AllocaOp>(loc, memrefTy);
  SmallVector<Value> indices;
  OwningOpRef<memref::LoadOp> loadVarOp =
      b.create<memref::LoadOp>(loc, int64Ty, varPtrOp->getResult(), indices);

  EXPECT_TRUE(isMappableType(loadVarOp->getResult().getType()));
  TypedValue<MappableType> var =
      cast<TypedValue<MappableType>>(loadVarOp->getResult());
  OwningOpRef<Op> op = b.create<Op>(loc, var,
                                    /*structured=*/true, /*implicit=*/true);

  EXPECT_EQ(op->getVar(), var);
  EXPECT_EQ(op->getVarPtr(), nullptr);
  EXPECT_EQ(op->getType(), int64Ty);
  EXPECT_EQ(op->getVarType(), int64Ty);
  EXPECT_EQ(op->getDataClause(), dataClause);
  EXPECT_TRUE(op->getImplicit());
  EXPECT_TRUE(op->getStructured());
  EXPECT_TRUE(op->getBounds().empty());
  EXPECT_FALSE(op->getVarPtrPtr());
}

struct IntegerOpenACCMappableModel
    : public mlir::acc::MappableType::ExternalModel<IntegerOpenACCMappableModel,
                                                    IntegerType> {};

TEST_F(OpenACCOpsTest, mappableTypeBuilderDataEntry) {
  // First, set up the test by attaching MappableInterface to IntegerType.
  IntegerType i64ty = IntegerType::get(&context, 8);
  ASSERT_FALSE(isMappableType(i64ty));
  IntegerType::attachInterface<IntegerOpenACCMappableModel>(context);
  ASSERT_TRUE(isMappableType(i64ty));

  testShortDataEntryOpBuildersMappableVar<PrivateOp>(b, context, loc,
                                                     DataClause::acc_private);
  testShortDataEntryOpBuildersMappableVar<FirstprivateOp>(
      b, context, loc, DataClause::acc_firstprivate);
  testShortDataEntryOpBuildersMappableVar<ReductionOp>(
      b, context, loc, DataClause::acc_reduction);
  testShortDataEntryOpBuildersMappableVar<DevicePtrOp>(
      b, context, loc, DataClause::acc_deviceptr);
  testShortDataEntryOpBuildersMappableVar<PresentOp>(b, context, loc,
                                                     DataClause::acc_present);
  testShortDataEntryOpBuildersMappableVar<CopyinOp>(b, context, loc,
                                                    DataClause::acc_copyin);
  testShortDataEntryOpBuildersMappableVar<CreateOp>(b, context, loc,
                                                    DataClause::acc_create);
  testShortDataEntryOpBuildersMappableVar<NoCreateOp>(
      b, context, loc, DataClause::acc_no_create);
  testShortDataEntryOpBuildersMappableVar<AttachOp>(b, context, loc,
                                                    DataClause::acc_attach);
  testShortDataEntryOpBuildersMappableVar<GetDevicePtrOp>(
      b, context, loc, DataClause::acc_getdeviceptr);
  testShortDataEntryOpBuildersMappableVar<UpdateDeviceOp>(
      b, context, loc, DataClause::acc_update_device);
  testShortDataEntryOpBuildersMappableVar<UseDeviceOp>(
      b, context, loc, DataClause::acc_use_device);
  testShortDataEntryOpBuildersMappableVar<DeclareDeviceResidentOp>(
      b, context, loc, DataClause::acc_declare_device_resident);
  testShortDataEntryOpBuildersMappableVar<DeclareLinkOp>(
      b, context, loc, DataClause::acc_declare_link);
  testShortDataEntryOpBuildersMappableVar<CacheOp>(b, context, loc,
                                                   DataClause::acc_cache);
}
