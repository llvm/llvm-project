//===- DynamicMemRef.cpp ----------------------------------------*- C++ -*-===//
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

TEST(DynamicMemRef, rankZero) {
  int data = 57;

  StridedMemRefType<int, 0> memRef;
  memRef.basePtr = &data;
  memRef.data = &data;
  memRef.offset = 0;

  DynamicMemRefType<int> dynamicMemRef(memRef);

  llvm::SmallVector<int, 1> values(dynamicMemRef.begin(), dynamicMemRef.end());
  EXPECT_THAT(values, ElementsAre(57));
}

TEST(DynamicMemRef, rankOne) {
  std::array<int, 3> data;

  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = i;
  }

  StridedMemRefType<int, 1> memRef;
  memRef.basePtr = data.data();
  memRef.data = data.data();
  memRef.offset = 0;
  memRef.sizes[0] = 3;
  memRef.strides[0] = 1;

  DynamicMemRefType<int> dynamicMemRef(memRef);

  llvm::SmallVector<int, 3> values(dynamicMemRef.begin(), dynamicMemRef.end());
  EXPECT_THAT(values, ElementsAreArray(data));

  for (int64_t i = 0; i < 3; ++i) {
    EXPECT_EQ(*dynamicMemRef[i], data[i]);
  }
}

TEST(DynamicMemRef, rankTwo) {
  std::array<int, 6> data;

  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = i;
  }

  StridedMemRefType<int, 2> memRef;
  memRef.basePtr = data.data();
  memRef.data = data.data();
  memRef.offset = 0;
  memRef.sizes[0] = 2;
  memRef.sizes[1] = 3;
  memRef.strides[0] = 3;
  memRef.strides[1] = 1;

  DynamicMemRefType<int> dynamicMemRef(memRef);

  llvm::SmallVector<int, 6> values(dynamicMemRef.begin(), dynamicMemRef.end());
  EXPECT_THAT(values, ElementsAreArray(data));
}

TEST(DynamicMemRef, rankThree) {
  std::array<int, 24> data;

  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = i;
  }

  StridedMemRefType<int, 3> memRef;
  memRef.basePtr = data.data();
  memRef.data = data.data();
  memRef.offset = 0;
  memRef.sizes[0] = 2;
  memRef.sizes[1] = 3;
  memRef.sizes[2] = 4;
  memRef.strides[0] = 12;
  memRef.strides[1] = 4;
  memRef.strides[2] = 1;

  DynamicMemRefType<int> dynamicMemRef(memRef);

  llvm::SmallVector<int, 24> values(dynamicMemRef.begin(), dynamicMemRef.end());
  EXPECT_THAT(values, ElementsAreArray(data));
}