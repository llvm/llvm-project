//===- llvm/unittests/Target/DirectX/CBufferDataLayoutTests.cpp -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CBufferDataLayout.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::Contains;
using ::testing::Pair;

using namespace llvm;
using namespace llvm::dxil;

void checkLegacyLayout(CBufferDataLayout &CBDL, Type *T16, Type *T32,
                       Type *T64) {
  // Basic types.
  EXPECT_EQ(2ULL, CBDL.getTypeAllocSizeInBytes(T16).getFixedSize());
  EXPECT_EQ(4ULL, CBDL.getTypeAllocSizeInBytes(T32).getFixedSize());
  EXPECT_EQ(8ULL, CBDL.getTypeAllocSizeInBytes(T64).getFixedSize());
  // Vector types.
  Type *T16V2 = FixedVectorType::get(T16, 2);
  Type *T32V2 = FixedVectorType::get(T32, 2);
  Type *T64V2 = FixedVectorType::get(T64, 2);
  Type *T16V3 = FixedVectorType::get(T16, 3);
  Type *T32V3 = FixedVectorType::get(T32, 3);
  Type *T64V3 = FixedVectorType::get(T64, 3);
  Type *T16V4 = FixedVectorType::get(T16, 4);
  Type *T32V4 = FixedVectorType::get(T32, 4);
  Type *T64V4 = FixedVectorType::get(T64, 4);
  EXPECT_EQ(4ULL, CBDL.getTypeAllocSizeInBytes(T16V2).getFixedSize());
  EXPECT_EQ(8ULL, CBDL.getTypeAllocSizeInBytes(T32V2).getFixedSize());
  EXPECT_EQ(16ULL, CBDL.getTypeAllocSizeInBytes(T64V2).getFixedSize());
  EXPECT_EQ(6ULL, CBDL.getTypeAllocSizeInBytes(T16V3).getFixedSize());
  EXPECT_EQ(12ULL, CBDL.getTypeAllocSizeInBytes(T32V3).getFixedSize());
  EXPECT_EQ(24ULL, CBDL.getTypeAllocSizeInBytes(T64V3).getFixedSize());
  EXPECT_EQ(8ULL, CBDL.getTypeAllocSizeInBytes(T16V4).getFixedSize());
  EXPECT_EQ(16ULL, CBDL.getTypeAllocSizeInBytes(T32V4).getFixedSize());
  EXPECT_EQ(32ULL, CBDL.getTypeAllocSizeInBytes(T64V4).getFixedSize());

  // Array types.

  ArrayType *T16A3 = ArrayType::get(T16, 3);
  ArrayType *T32A3 = ArrayType::get(T32, 3);
  ArrayType *T64A3 = ArrayType::get(T64, 3);

  EXPECT_EQ(34ULL, CBDL.getTypeAllocSizeInBytes(T16A3).getFixedSize());
  EXPECT_EQ(36ULL, CBDL.getTypeAllocSizeInBytes(T32A3).getFixedSize());
  EXPECT_EQ(40ULL, CBDL.getTypeAllocSizeInBytes(T64A3).getFixedSize());

  ArrayType *T16V3A3 = ArrayType::get(T16V3, 3);
  ArrayType *T32V3A3 = ArrayType::get(T32V3, 3);
  ArrayType *T64V3A3 = ArrayType::get(T64V3, 3);

  EXPECT_EQ(38ULL, CBDL.getTypeAllocSizeInBytes(T16V3A3).getFixedSize());
  EXPECT_EQ(44ULL, CBDL.getTypeAllocSizeInBytes(T32V3A3).getFixedSize());
  EXPECT_EQ(88ULL, CBDL.getTypeAllocSizeInBytes(T64V3A3).getFixedSize());

  ArrayType *T16V3A3A3 = ArrayType::get(T16V3A3, 3);
  ArrayType *T32V3A3A3 = ArrayType::get(T32V3A3, 3);
  ArrayType *T64V3A3A3 = ArrayType::get(T64V3A3, 3);

  EXPECT_EQ((48 * 2 + 38ULL),
            CBDL.getTypeAllocSizeInBytes(T16V3A3A3).getFixedSize());
  EXPECT_EQ((48 * 2 + 44ULL),
            CBDL.getTypeAllocSizeInBytes(T32V3A3A3).getFixedSize());
  EXPECT_EQ((96 * 2 + 88ULL),
            CBDL.getTypeAllocSizeInBytes(T64V3A3A3).getFixedSize());

  // Struct types.
  StructType *BasicMix0 = StructType::get(T16, T32, T64);
  StructType *BasicMix1 = StructType::get(T16, T64, T32);
  StructType *BasicMix2 = StructType::get(T32, T64, T16);
  StructType *BasicMix3 = StructType::get(T32, T16, T64);
  StructType *BasicMix4 = StructType::get(T64, T16, T32);
  StructType *BasicMix5 = StructType::get(T64, T32, T16);

  EXPECT_EQ(16ULL, CBDL.getTypeAllocSizeInBytes(BasicMix0).getFixedSize());
  EXPECT_EQ(20ULL, CBDL.getTypeAllocSizeInBytes(BasicMix1).getFixedSize());
  EXPECT_EQ(18ULL, CBDL.getTypeAllocSizeInBytes(BasicMix2).getFixedSize());
  EXPECT_EQ(16ULL, CBDL.getTypeAllocSizeInBytes(BasicMix3).getFixedSize());
  EXPECT_EQ(16ULL, CBDL.getTypeAllocSizeInBytes(BasicMix4).getFixedSize());
  EXPECT_EQ(14ULL, CBDL.getTypeAllocSizeInBytes(BasicMix5).getFixedSize());

  StructType *VecMix0 = StructType::get(T16V3, T16, T32, T64V2);
  StructType *VecMix1 = StructType::get(T16V3, T32, T64V2, T16);
  StructType *VecMix2 = StructType::get(T16V3, T64, T32V2, T16);
  StructType *VecMix3 = StructType::get(T32V3, T64, T16V2, T32);
  StructType *VecMix4 = StructType::get(T32V3, T16, T16V2, T64);
  StructType *VecMix5 = StructType::get(T32V3, T64V3, T16V2, T64);

  EXPECT_EQ(32ULL, CBDL.getTypeAllocSizeInBytes(VecMix0).getFixedSize());
  EXPECT_EQ(34ULL, CBDL.getTypeAllocSizeInBytes(VecMix1).getFixedSize());
  EXPECT_EQ(26ULL, CBDL.getTypeAllocSizeInBytes(VecMix2).getFixedSize());
  EXPECT_EQ(32ULL, CBDL.getTypeAllocSizeInBytes(VecMix3).getFixedSize());
  EXPECT_EQ(32ULL, CBDL.getTypeAllocSizeInBytes(VecMix4).getFixedSize());
  EXPECT_EQ(56ULL, CBDL.getTypeAllocSizeInBytes(VecMix5).getFixedSize());

  StructType *ArrayMix0 = StructType::get(T16A3, T16, T32, T64A3);
  StructType *ArrayMix1 = StructType::get(T32A3, T16, T32, T16A3);
  StructType *ArrayMix2 = StructType::get(T16A3, T32, T64, T32A3);
  StructType *ArrayMix3 = StructType::get(T32A3, T32, T64, T16A3);
  StructType *ArrayMix4 = StructType::get(T16A3, T64, T16, T64A3);
  StructType *ArrayMix5 = StructType::get(T32A3, T64, T16, T32A3);

  EXPECT_EQ(88ULL, CBDL.getTypeAllocSizeInBytes(ArrayMix0).getFixedSize());
  EXPECT_EQ(82ULL, CBDL.getTypeAllocSizeInBytes(ArrayMix1).getFixedSize());
  EXPECT_EQ(84ULL, CBDL.getTypeAllocSizeInBytes(ArrayMix2).getFixedSize());
  EXPECT_EQ(82ULL, CBDL.getTypeAllocSizeInBytes(ArrayMix3).getFixedSize());
  EXPECT_EQ(104ULL, CBDL.getTypeAllocSizeInBytes(ArrayMix4).getFixedSize());
  EXPECT_EQ(100ULL, CBDL.getTypeAllocSizeInBytes(ArrayMix5).getFixedSize());

  StructType *StructMix0 = StructType::get(T16A3, T16, T32, ArrayMix0);
  StructType *StructMix1 = StructType::get(StructMix0, T16, T32, ArrayMix1);
  ArrayType *StructArray0 = ArrayType::get(StructMix1, 3);
  StructType *StructMix2 = StructType::get(StructArray0, T64);
  ArrayType *StructArray1 = ArrayType::get(StructMix2, 3);
  StructType *StructMix3 = StructType::get(StructArray1, T32);
  EXPECT_EQ(136ULL, CBDL.getTypeAllocSizeInBytes(StructMix0).getFixedSize());
  EXPECT_EQ(226ULL, CBDL.getTypeAllocSizeInBytes(StructMix1).getFixedSize());
  EXPECT_EQ(706ULL, CBDL.getTypeAllocSizeInBytes(StructArray0).getFixedSize());
  EXPECT_EQ(720ULL, CBDL.getTypeAllocSizeInBytes(StructMix2).getFixedSize());
  EXPECT_EQ(2160ULL, CBDL.getTypeAllocSizeInBytes(StructArray1).getFixedSize());
  EXPECT_EQ(2164ULL, CBDL.getTypeAllocSizeInBytes(StructMix3).getFixedSize());
}

TEST(CBufferDataLayout, LegacyLayout) {
  LLVMContext Context;
  llvm::DataLayout DL("e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-"
                      "f32:32-f64:64-n8:16:32:64");

  CBufferDataLayout CBDL(DL, true);

  Type *F16 = Type::getHalfTy(Context);
  Type *F32 = Type::getFloatTy(Context);
  Type *F64 = Type::getDoubleTy(Context);

  Type *I16 = Type::getInt16Ty(Context);
  Type *I32 = Type::getInt32Ty(Context);
  Type *I64 = Type::getInt64Ty(Context);

  checkLegacyLayout(CBDL, F16, F32, F64);
  checkLegacyLayout(CBDL, I16, I32, I64);
}
