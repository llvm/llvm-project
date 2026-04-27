//===- X86Test.cpp - Unit tests for X86_64 ABI classifier -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/FunctionInfo.h"
#include "llvm/ABI/TargetInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/Support/Allocator.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::abi;

namespace {

class X86_64ABITest : public ::testing::Test {
protected:
  BumpPtrAllocator Alloc;
  TypeBuilder TB{Alloc};
  Triple TargetTriple{"x86_64-unknown-linux-gnu"};

  std::unique_ptr<TargetInfo>
  createTarget(X86AVXABILevel AVX = X86AVXABILevel::None,
               bool Has64BitPtrs = true) {
    return createX8664TargetInfo(TB, TargetTriple, AVX, Has64BitPtrs,
                                 ABICompatInfo());
  }

  FunctionInfo *makeFI(const Type *RetTy,
                       ArrayRef<const Type *> ArgTypes = {}) {
    return FunctionInfo::create(CallingConv::C, RetTy, ArgTypes);
  }
};

TEST_F(X86_64ABITest, VoidReturn) {
  auto TI = createTarget();
  const VoidType *VoidTy = TB.getVoidType();
  std::unique_ptr<FunctionInfo> FI(makeFI(VoidTy));
  TI->computeInfo(*FI);
  EXPECT_TRUE(FI->getReturnInfo().isIgnore());
}

TEST_F(X86_64ABITest, IntegerReturnDirect) {
  auto TI = createTarget();
  const IntegerType *I32 = TB.getIntegerType(32, Align(4), true);
  std::unique_ptr<FunctionInfo> FI(makeFI(I32));
  TI->computeInfo(*FI);
  EXPECT_TRUE(FI->getReturnInfo().isDirect());
}

TEST_F(X86_64ABITest, Int64ReturnDirect) {
  auto TI = createTarget();
  const IntegerType *I64 = TB.getIntegerType(64, Align(8), true);
  std::unique_ptr<FunctionInfo> FI(makeFI(I64));
  TI->computeInfo(*FI);
  EXPECT_TRUE(FI->getReturnInfo().isDirect());
}

TEST_F(X86_64ABITest, FloatReturnSSE) {
  auto TI = createTarget();
  const FloatType *F32 = TB.getFloatType(APFloat::IEEEsingle(), Align(4));
  std::unique_ptr<FunctionInfo> FI(makeFI(F32));
  TI->computeInfo(*FI);
  EXPECT_TRUE(FI->getReturnInfo().isDirect());
}

TEST_F(X86_64ABITest, DoubleReturnSSE) {
  auto TI = createTarget();
  const FloatType *F64 = TB.getFloatType(APFloat::IEEEdouble(), Align(8));
  std::unique_ptr<FunctionInfo> FI(makeFI(F64));
  TI->computeInfo(*FI);
  EXPECT_TRUE(FI->getReturnInfo().isDirect());
}

TEST_F(X86_64ABITest, SmallStructDirect) {
  auto TI = createTarget();
  const IntegerType *I32 = TB.getIntegerType(32, Align(4), true);
  FieldInfo Fields[] = {FieldInfo(I32, 0), FieldInfo(I32, 32)};
  const RecordType *RT = TB.getRecordType(Fields, TypeSize::getFixed(64),
                                          Align(4), StructPacking::Default, {},
                                          {}, RecordFlags::CanPassInRegisters);
  std::unique_ptr<FunctionInfo> FI(makeFI(RT));
  TI->computeInfo(*FI);
  EXPECT_TRUE(FI->getReturnInfo().isDirect());
}

TEST_F(X86_64ABITest, LargeStructIndirect) {
  auto TI = createTarget();
  const IntegerType *I64 = TB.getIntegerType(64, Align(8), false);
  FieldInfo Fields[] = {FieldInfo(I64, 0), FieldInfo(I64, 64),
                        FieldInfo(I64, 128)};
  const RecordType *RT = TB.getRecordType(Fields, TypeSize::getFixed(192),
                                          Align(8), StructPacking::Default, {},
                                          {}, RecordFlags::CanPassInRegisters);
  std::unique_ptr<FunctionInfo> FI(makeFI(RT));
  TI->computeInfo(*FI);
  EXPECT_TRUE(FI->getReturnInfo().isIndirect());
}

TEST_F(X86_64ABITest, IntegerArgDirect) {
  auto TI = createTarget();
  const VoidType *VoidTy = TB.getVoidType();
  const IntegerType *I64 = TB.getIntegerType(64, Align(8), true);
  SmallVector<const Type *> Args = {I64};
  std::unique_ptr<FunctionInfo> FI(makeFI(VoidTy, Args));
  TI->computeInfo(*FI);
  EXPECT_TRUE(FI->getArgInfo(0).Info.isDirect());
}

TEST_F(X86_64ABITest, SmallIntegerArgExtend) {
  auto TI = createTarget();
  const VoidType *VoidTy = TB.getVoidType();
  const IntegerType *I16 = TB.getIntegerType(16, Align(2), true);
  SmallVector<const Type *> Args = {I16};
  std::unique_ptr<FunctionInfo> FI(makeFI(VoidTy, Args));
  TI->computeInfo(*FI);
  EXPECT_TRUE(FI->getArgInfo(0).Info.isExtend());
  EXPECT_TRUE(FI->getArgInfo(0).Info.isSignExt());
}

TEST_F(X86_64ABITest, UnsignedSmallIntegerArgZeroExt) {
  auto TI = createTarget();
  const VoidType *VoidTy = TB.getVoidType();
  const IntegerType *U8 = TB.getIntegerType(8, Align(1), false);
  SmallVector<const Type *> Args = {U8};
  std::unique_ptr<FunctionInfo> FI(makeFI(VoidTy, Args));
  TI->computeInfo(*FI);
  EXPECT_TRUE(FI->getArgInfo(0).Info.isExtend());
  EXPECT_TRUE(FI->getArgInfo(0).Info.isZeroExt());
}

TEST_F(X86_64ABITest, PointerArgDirect) {
  auto TI = createTarget();
  const VoidType *VoidTy = TB.getVoidType();
  const PointerType *PtrTy = TB.getPointerType(64, Align(8));
  SmallVector<const Type *> Args = {PtrTy};
  std::unique_ptr<FunctionInfo> FI(makeFI(VoidTy, Args));
  TI->computeInfo(*FI);
  EXPECT_TRUE(FI->getArgInfo(0).Info.isDirect());
}

TEST_F(X86_64ABITest, SmallStructArgDirect) {
  auto TI = createTarget();
  const VoidType *VoidTy = TB.getVoidType();
  const IntegerType *I32 = TB.getIntegerType(32, Align(4), true);
  FieldInfo Fields[] = {FieldInfo(I32, 0), FieldInfo(I32, 32)};
  const RecordType *RT = TB.getRecordType(Fields, TypeSize::getFixed(64),
                                          Align(4), StructPacking::Default, {},
                                          {}, RecordFlags::CanPassInRegisters);
  SmallVector<const Type *> Args = {RT};
  std::unique_ptr<FunctionInfo> FI(makeFI(VoidTy, Args));
  TI->computeInfo(*FI);
  EXPECT_TRUE(FI->getArgInfo(0).Info.isDirect());
}

TEST_F(X86_64ABITest, TwoEightbyteStructArgDirect) {
  auto TI = createTarget();
  const VoidType *VoidTy = TB.getVoidType();
  const IntegerType *I64 = TB.getIntegerType(64, Align(8), false);
  FieldInfo Fields[] = {FieldInfo(I64, 0), FieldInfo(I64, 64)};
  const RecordType *RT = TB.getRecordType(Fields, TypeSize::getFixed(128),
                                          Align(8), StructPacking::Default, {},
                                          {}, RecordFlags::CanPassInRegisters);
  SmallVector<const Type *> Args = {RT};
  std::unique_ptr<FunctionInfo> FI(makeFI(VoidTy, Args));
  TI->computeInfo(*FI);
  EXPECT_TRUE(FI->getArgInfo(0).Info.isDirect());
}

TEST_F(X86_64ABITest, LargeStructArgIndirect) {
  auto TI = createTarget();
  const VoidType *VoidTy = TB.getVoidType();
  const IntegerType *I64 = TB.getIntegerType(64, Align(8), false);
  FieldInfo Fields[] = {FieldInfo(I64, 0), FieldInfo(I64, 64),
                        FieldInfo(I64, 128)};
  const RecordType *RT = TB.getRecordType(Fields, TypeSize::getFixed(192),
                                          Align(8), StructPacking::Default, {},
                                          {}, RecordFlags::CanPassInRegisters);
  SmallVector<const Type *> Args = {RT};
  std::unique_ptr<FunctionInfo> FI(makeFI(VoidTy, Args));
  TI->computeInfo(*FI);
  EXPECT_TRUE(FI->getArgInfo(0).Info.isIndirect());
}

TEST_F(X86_64ABITest, NTCStructArgIndirect) {
  auto TI = createTarget();
  const VoidType *VoidTy = TB.getVoidType();
  const IntegerType *I32 = TB.getIntegerType(32, Align(4), true);
  FieldInfo Fields[] = {FieldInfo(I32, 0)};
  const RecordType *RT = TB.getRecordType(Fields, TypeSize::getFixed(32),
                                          Align(4), StructPacking::Default, {},
                                          {}, RecordFlags::IsCXXRecord);
  SmallVector<const Type *> Args = {RT};
  std::unique_ptr<FunctionInfo> FI(makeFI(VoidTy, Args));
  TI->computeInfo(*FI);
  EXPECT_TRUE(FI->getArgInfo(0).Info.isIndirect());
}

TEST_F(X86_64ABITest, SSEFloatArgDirect) {
  auto TI = createTarget();
  const VoidType *VoidTy = TB.getVoidType();
  const FloatType *F32 = TB.getFloatType(APFloat::IEEEsingle(), Align(4));
  SmallVector<const Type *> Args = {F32};
  std::unique_ptr<FunctionInfo> FI(makeFI(VoidTy, Args));
  TI->computeInfo(*FI);
  EXPECT_TRUE(FI->getArgInfo(0).Info.isDirect());
}

TEST_F(X86_64ABITest, X87LongDoubleReturn) {
  auto TI = createTarget();
  const FloatType *LDTy =
      TB.getFloatType(APFloat::x87DoubleExtended(), Align(16));
  std::unique_ptr<FunctionInfo> FI(makeFI(LDTy));
  TI->computeInfo(*FI);
  EXPECT_TRUE(FI->getReturnInfo().isDirect());
}

TEST_F(X86_64ABITest, VectorSSE128Direct) {
  auto TI = createTarget();
  const VoidType *VoidTy = TB.getVoidType();
  const FloatType *F32 = TB.getFloatType(APFloat::IEEEsingle(), Align(4));
  const VectorType *V4F32 =
      TB.getVectorType(F32, ElementCount::getFixed(4), Align(16));
  SmallVector<const Type *> Args = {V4F32};
  std::unique_ptr<FunctionInfo> FI(makeFI(VoidTy, Args));
  TI->computeInfo(*FI);
  EXPECT_TRUE(FI->getArgInfo(0).Info.isDirect());
}

TEST_F(X86_64ABITest, Vector256NoAVXIndirect) {
  auto TI = createTarget(X86AVXABILevel::None);
  const VoidType *VoidTy = TB.getVoidType();
  const FloatType *F32 = TB.getFloatType(APFloat::IEEEsingle(), Align(4));
  const VectorType *V8F32 =
      TB.getVectorType(F32, ElementCount::getFixed(8), Align(32));
  SmallVector<const Type *> Args = {V8F32};
  std::unique_ptr<FunctionInfo> FI(makeFI(VoidTy, Args));
  TI->computeInfo(*FI);
  EXPECT_TRUE(FI->getArgInfo(0).Info.isIndirect());
}

TEST_F(X86_64ABITest, Vector256WithAVXDirect) {
  auto TI = createTarget(X86AVXABILevel::AVX);
  const VoidType *VoidTy = TB.getVoidType();
  const FloatType *F32 = TB.getFloatType(APFloat::IEEEsingle(), Align(4));
  const VectorType *V8F32 =
      TB.getVectorType(F32, ElementCount::getFixed(8), Align(32));
  SmallVector<const Type *> Args = {V8F32};
  std::unique_ptr<FunctionInfo> FI(makeFI(VoidTy, Args));
  TI->computeInfo(*FI);
  EXPECT_TRUE(FI->getArgInfo(0).Info.isDirect());
}

TEST_F(X86_64ABITest, RegisterExhaustion) {
  auto TI = createTarget();
  const VoidType *VoidTy = TB.getVoidType();
  const IntegerType *I64 = TB.getIntegerType(64, Align(8), false);
  SmallVector<const Type *> Args(7, I64);
  std::unique_ptr<FunctionInfo> FI(makeFI(VoidTy, Args));
  TI->computeInfo(*FI);
  for (unsigned I = 0; I < 6; ++I)
    EXPECT_TRUE(FI->getArgInfo(I).Info.isDirect());
  EXPECT_TRUE(FI->getArgInfo(6).Info.isDirect());
}

TEST_F(X86_64ABITest, FloatStructCoercion) {
  auto TI = createTarget();
  const FloatType *F32 = TB.getFloatType(APFloat::IEEEsingle(), Align(4));
  FieldInfo Fields[] = {FieldInfo(F32, 0), FieldInfo(F32, 32)};
  const RecordType *RT = TB.getRecordType(Fields, TypeSize::getFixed(64),
                                          Align(4), StructPacking::Default, {},
                                          {}, RecordFlags::CanPassInRegisters);
  std::unique_ptr<FunctionInfo> FI(makeFI(RT));
  TI->computeInfo(*FI);
  EXPECT_TRUE(FI->getReturnInfo().isDirect());
  const Type *CoerceTy = FI->getReturnInfo().getCoerceToType();
  ASSERT_NE(CoerceTy, nullptr);
  EXPECT_TRUE(CoerceTy->isVector());
}

TEST_F(X86_64ABITest, EmptyStructIgnored) {
  auto TI = createTarget();
  const VoidType *VoidTy = TB.getVoidType();
  const RecordType *EmptyRT = TB.getRecordType(
      {}, TypeSize::getFixed(0), Align(1), StructPacking::Default, {}, {},
      RecordFlags::CanPassInRegisters);
  SmallVector<const Type *> Args = {EmptyRT};
  std::unique_ptr<FunctionInfo> FI(makeFI(VoidTy, Args));
  TI->computeInfo(*FI);
  EXPECT_TRUE(FI->getArgInfo(0).Info.isIgnore());
}

} // namespace
