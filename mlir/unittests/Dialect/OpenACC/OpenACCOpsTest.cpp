//===- OpenACCOpsTest.cpp - OpenACC ops extra functiosn Tests -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::acc;

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

class OpenACCOpsTest : public ::testing::Test {
protected:
  OpenACCOpsTest() : b(&context), loc(UnknownLoc::get(&context)) {
    context.loadDialect<acc::OpenACCDialect, arith::ArithDialect>();
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
  op->setAsyncDeviceTypeAttr(b.getArrayAttr({dtypeNvidia}));
  op->getAsyncMutable().assign(val->getResult());
  EXPECT_EQ(op->getAsyncValue(), empty);
  EXPECT_EQ(op->getAsyncValue(DeviceType::Nvidia), val->getResult());

  op->getAsyncMutable().clear();
  op->removeAsyncDeviceTypeAttr();
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
  auto dtypeNone = DeviceTypeAttr::get(&context, DeviceType::None);
  op->getWaitOperandsMutable().assign(val1->getResult());
  op->setWaitOperandsDeviceTypeAttr(b.getArrayAttr({dtypeNone}));
  op->setWaitOperandsSegments(b.getDenseI32ArrayAttr({1}));
  EXPECT_EQ(op->getWaitValues().front(), val1->getResult());
  for (auto d : dtypesWithoutNone)
    EXPECT_EQ(op->getWaitValues(d).begin(), op->getWaitValues(d).end());

  op->getWaitOperandsMutable().clear();
  op->removeWaitOperandsDeviceTypeAttr();
  op->removeWaitOperandsSegmentsAttr();
  for (auto d : dtypes)
    EXPECT_EQ(op->getWaitValues(d).begin(), op->getWaitValues(d).end());

  op->getWaitOperandsMutable().append(val1->getResult());
  op->getWaitOperandsMutable().append(val2->getResult());
  op->setWaitOperandsDeviceTypeAttr(
      b.getArrayAttr({DeviceTypeAttr::get(&context, DeviceType::Host),
                      DeviceTypeAttr::get(&context, DeviceType::Star)}));
  op->setWaitOperandsSegments(b.getDenseI32ArrayAttr({1, 1}));
  EXPECT_EQ(op->getWaitValues(DeviceType::None).begin(),
            op->getWaitValues(DeviceType::None).end());
  EXPECT_EQ(op->getWaitValues(DeviceType::Host).front(), val1->getResult());
  EXPECT_EQ(op->getWaitValues(DeviceType::Star).front(), val2->getResult());

  op->getWaitOperandsMutable().clear();
  op->removeWaitOperandsDeviceTypeAttr();
  op->removeWaitOperandsSegmentsAttr();
  for (auto d : dtypes)
    EXPECT_EQ(op->getWaitValues(d).begin(), op->getWaitValues(d).end());

  op->getWaitOperandsMutable().append(val1->getResult());
  op->getWaitOperandsMutable().append(val2->getResult());
  op->getWaitOperandsMutable().append(val1->getResult());
  op->setWaitOperandsDeviceTypeAttr(
      b.getArrayAttr({DeviceTypeAttr::get(&context, DeviceType::Default),
                      DeviceTypeAttr::get(&context, DeviceType::Multicore)}));
  op->setWaitOperandsSegments(b.getDenseI32ArrayAttr({2, 1}));
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
