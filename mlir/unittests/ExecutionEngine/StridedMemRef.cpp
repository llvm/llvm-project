//===- StridedMemRef.cpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "llvm/ADT/SmallVector.h"

#include "gmock/gmock.h"

using namespace ::mlir;
using namespace ::testing;

TEST(StridedMemRef, rankOneWithOffset) {
  std::array<int, 15> data;

  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = i;
  }

  StridedMemRefType<int, 1> memRefA;
  memRefA.basePtr = data.data();
  memRefA.data = data.data();
  memRefA.offset = 0;
  memRefA.sizes[0] = 10;
  memRefA.strides[0] = 1;

  StridedMemRefType<int, 1> memRefB = memRefA;
  memRefB.offset = 5;

  llvm::SmallVector<int, 10> valuesA(memRefA.begin(), memRefA.end());
  llvm::SmallVector<int, 10> valuesB(memRefB.begin(), memRefB.end());

  for (int64_t i = 0; i < 10; ++i) {
    EXPECT_EQ(valuesA[i], i);
    EXPECT_EQ(valuesA[i] + 5, valuesB[i]);
  }
}
