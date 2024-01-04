//===- LinalgInterfacesTest.cpp - LinalgInterfaces unit tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "gtest/gtest.h"

using namespace mlir;

class LinalgInterfacesTest : public ::testing::Test {
protected:
  LinalgInterfacesTest() {
    context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  }

  mlir::MLIRContext context;
};

TEST_F(LinalgInterfacesTest, ContractionOpOperandResultAccessor) {
  OpBuilder b(&context);
  SmallVector<int64_t> lhsShape = {1, 2};
  SmallVector<int64_t> rhsShape = {2, 4};
  SmallVector<int64_t> resShape = {1, 4};
  auto lhs = b.create<tensor::EmptyOp>(UnknownLoc::get(&context), lhsShape,
                                       b.getF32Type());
  auto rhs = b.create<tensor::EmptyOp>(UnknownLoc::get(&context), rhsShape,
                                       b.getF32Type());
  auto out = b.create<tensor::EmptyOp>(UnknownLoc::get(&context), resShape,
                                       b.getF32Type());
  Operation *op = b.create<linalg::MatmulOp>(
      UnknownLoc::get(&context), ValueRange{lhs, rhs}, ValueRange{out});
  auto contractOp = llvm::cast<linalg::ContractionOpInterface>(op);

  EXPECT_EQ(contractOp.lhs(), op->getOperand(0));
  EXPECT_EQ(contractOp.rhs(), op->getOperand(1));
  EXPECT_EQ(contractOp.res(), op->getResult(0));
}
