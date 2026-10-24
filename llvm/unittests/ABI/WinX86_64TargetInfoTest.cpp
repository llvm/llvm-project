//===- WinX86_64TargetInfoTest.cpp - Windows x86-64 ABI unit tests --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/FunctionInfo.h"
#include "llvm/ABI/TargetInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Allocator.h"
#include "gtest/gtest.h"

namespace {

using namespace llvm;

using ABIType = llvm::abi::Type;
using llvm::abi::ABICompatInfo;
using llvm::abi::ArgInfo;
using llvm::abi::createWinX86_64TargetInfo;
using llvm::abi::FieldInfo;
using llvm::abi::FunctionInfo;
using llvm::abi::RecordFlags;
using llvm::abi::StructPacking;
using llvm::abi::TargetInfo;
using llvm::abi::TypeBuilder;
using llvm::abi::X86AVXABILevel;

class WinX86_64TargetInfoTest : public ::testing::Test {
protected:
  llvm::BumpPtrAllocator Alloc;
  TypeBuilder TB;
  const ABIType *Void;
  const ABIType *I1;
  const ABIType *I8;
  const ABIType *I16;
  const ABIType *I32;
  const ABIType *I64;
  const ABIType *F32;
  const ABIType *F64;

  WinX86_64TargetInfoTest()
      : TB(Alloc), Void(TB.getVoidType()),
        I1(TB.getIntegerType(1, llvm::Align(1), /*Signed=*/false)),
        I8(TB.getIntegerType(8, llvm::Align(1), /*Signed=*/true)),
        I16(TB.getIntegerType(16, llvm::Align(2), /*Signed=*/true)),
        I32(TB.getIntegerType(32, llvm::Align(4), /*Signed=*/true)),
        I64(TB.getIntegerType(64, llvm::Align(8), /*Signed=*/true)),
        F32(TB.getFloatType(llvm::APFloat::IEEEsingle(), llvm::Align(4))),
        F64(TB.getFloatType(llvm::APFloat::IEEEdouble(), llvm::Align(8))) {}

  std::unique_ptr<TargetInfo> target() const {
    return createWinX86_64TargetInfo(const_cast<TypeBuilder &>(TB),
                                     X86AVXABILevel::None, ABICompatInfo());
  }

  const ABIType *
  structOfBytes(uint64_t NumBytes,
                RecordFlags Flags = RecordFlags::CanPassInRegisters) {
    llvm::SmallVector<FieldInfo, 8> Fields;
    for (uint64_t I = 0; I < NumBytes; ++I)
      Fields.emplace_back(I8, I * 8);
    return TB.getRecordType(Fields, llvm::TypeSize::getFixed(NumBytes * 8),
                            llvm::Align(1), llvm::Align(1),
                            StructPacking::Default, {}, {}, Flags);
  }

  const ArgInfo &classifyReturn(const ABIType *RetTy,
                                std::unique_ptr<FunctionInfo> &FI,
                                std::unique_ptr<TargetInfo> &TI,
                                bool IsInstanceMethod = false) {
    TI = target();
    FI = FunctionInfo::create(llvm::CallingConv::C, RetTy, {});
    FI->setIsInstanceMethod(IsInstanceMethod);
    TI->computeInfo(*FI);
    return FI->getReturnInfo();
  }
};

static void expectDirectInteger(const ArgInfo &Info, unsigned Bits) {
  ASSERT_TRUE(Info.isDirect());
  const ABIType *Coerce = Info.getCoerceToType();
  ASSERT_NE(Coerce, nullptr);
  const auto *IT = llvm::dyn_cast<llvm::abi::IntegerType>(Coerce);
  ASSERT_NE(IT, nullptr);
  EXPECT_EQ(IT->getSizeInBits().getFixedValue(), Bits);
}

TEST_F(WinX86_64TargetInfoTest, ScalarReturns) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;

  // Void -> ignore.
  EXPECT_TRUE(classifyReturn(Void, FI, TI).isIgnore());

  // i1 -> extend (zero extend).
  const ArgInfo &I1Info = classifyReturn(I1, FI, TI);
  EXPECT_TRUE(I1Info.isExtend());
  EXPECT_TRUE(I1Info.isZeroExt());

  // i8 -> direct.
  const ArgInfo &I8Info = classifyReturn(I8, FI, TI);
  EXPECT_TRUE(I8Info.isDirect());
  EXPECT_EQ(I8Info.getCoerceToType(), nullptr);

  // i16 -> direct.
  const ArgInfo &I16Info = classifyReturn(I16, FI, TI);
  EXPECT_TRUE(I16Info.isDirect());
  EXPECT_EQ(I16Info.getCoerceToType(), nullptr);

  // i32 -> direct.
  const ArgInfo &I32Info = classifyReturn(I32, FI, TI);
  EXPECT_TRUE(I32Info.isDirect());
  EXPECT_EQ(I32Info.getCoerceToType(), nullptr);

  // i64 -> direct.
  const ArgInfo &I64Info = classifyReturn(I64, FI, TI);
  EXPECT_TRUE(I64Info.isDirect());
  EXPECT_EQ(I64Info.getCoerceToType(), nullptr);

  // Float -> direct.
  const ArgInfo &F32Info = classifyReturn(F32, FI, TI);
  EXPECT_TRUE(F32Info.isDirect());
  EXPECT_EQ(F32Info.getCoerceToType(), nullptr);

  // Double -> direct.
  const ArgInfo &F64Info = classifyReturn(F64, FI, TI);
  EXPECT_TRUE(F64Info.isDirect());
  EXPECT_EQ(F64Info.getCoerceToType(), nullptr);
}

TEST_F(WinX86_64TargetInfoTest, DirectStructReturns) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;

  // 1-byte struct -> direct, coerced to i8.
  const ABIType *S1 = structOfBytes(1);
  expectDirectInteger(classifyReturn(S1, FI, TI), 8);

  // 2-byte struct -> direct, coerced to i16.
  const ABIType *S2 = structOfBytes(2);
  expectDirectInteger(classifyReturn(S2, FI, TI), 16);

  // 4-byte struct -> direct, coerced to i32.
  const ABIType *S4 = structOfBytes(4);
  expectDirectInteger(classifyReturn(S4, FI, TI), 32);

  // 8-byte struct -> direct, coerced to i64.
  const ABIType *S8 = structOfBytes(8);
  expectDirectInteger(classifyReturn(S8, FI, TI), 64);
}

TEST_F(WinX86_64TargetInfoTest, IndirectStructReturns) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;

  for (uint64_t Bytes : {3u, 5u, 12u, 16u}) {
    const ABIType *S = structOfBytes(Bytes);
    const ArgInfo &Info = classifyReturn(S, FI, TI);
    EXPECT_TRUE(Info.isIndirect());
    EXPECT_FALSE(Info.getIndirectByVal());
    EXPECT_FALSE(Info.isSRetAfterThis());
  }
}

TEST_F(WinX86_64TargetInfoTest, NonTrivialStructReturn) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;

  // 4-byte struct with RecordFlags::None (cannot pass in registers).
  const ABIType *S = structOfBytes(4, RecordFlags::None);
  const ArgInfo &Info = classifyReturn(S, FI, TI);
  EXPECT_TRUE(Info.isIndirect());
  EXPECT_FALSE(Info.getIndirectByVal());
}

TEST_F(WinX86_64TargetInfoTest, InstanceMethodStructReturn) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;

  // Even a 4-byte struct returned from an instance method is indirect with
  // SRetAfterThis = true.
  const ABIType *S4 = structOfBytes(4);
  const ArgInfo &Info = classifyReturn(S4, FI, TI, /*IsInstanceMethod=*/true);
  EXPECT_TRUE(Info.isIndirect());
  EXPECT_FALSE(Info.getIndirectByVal());
  EXPECT_TRUE(Info.isSRetAfterThis());
}

TEST_F(WinX86_64TargetInfoTest, ScalarAndStructArguments) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI = target();

  const ABIType *S4 = structOfBytes(4);
  const ABIType *S16 = structOfBytes(16);

  FI = FunctionInfo::create(llvm::CallingConv::C, Void,
                            {I1, I8, I16, I32, I64, S4, S16});
  TI->computeInfo(*FI);

  // Arg 0: i1 -> extend (zero extend).
  EXPECT_EQ(FI->getArgInfo(0).getType(), I1);
  const ArgInfo &Arg0 = FI->getArgInfo(0).Info;
  EXPECT_TRUE(Arg0.isExtend());
  EXPECT_TRUE(Arg0.isZeroExt());

  // Arg 1: i8 -> direct.
  EXPECT_EQ(FI->getArgInfo(1).getType(), I8);
  const ArgInfo &Arg1 = FI->getArgInfo(1).Info;
  EXPECT_TRUE(Arg1.isDirect());
  EXPECT_EQ(Arg1.getCoerceToType(), nullptr);

  // Arg 2: i16 -> direct.
  EXPECT_EQ(FI->getArgInfo(2).getType(), I16);
  const ArgInfo &Arg2 = FI->getArgInfo(2).Info;
  EXPECT_TRUE(Arg2.isDirect());
  EXPECT_EQ(Arg2.getCoerceToType(), nullptr);

  // Arg 3: i32 -> direct.
  EXPECT_EQ(FI->getArgInfo(3).getType(), I32);
  const ArgInfo &Arg3 = FI->getArgInfo(3).Info;
  EXPECT_TRUE(Arg3.isDirect());
  EXPECT_EQ(Arg3.getCoerceToType(), nullptr);

  // Arg 4: i64 -> direct.
  EXPECT_EQ(FI->getArgInfo(4).getType(), I64);
  const ArgInfo &Arg4 = FI->getArgInfo(4).Info;
  EXPECT_TRUE(Arg4.isDirect());
  EXPECT_EQ(Arg4.getCoerceToType(), nullptr);

  // Arg 5: 4-byte struct -> direct, coerced to i32.
  EXPECT_EQ(FI->getArgInfo(5).getType(), S4);
  const ArgInfo &Arg5 = FI->getArgInfo(5).Info;
  expectDirectInteger(Arg5, 32);

  // Arg 6: 16-byte struct -> indirect, ByVal = false.
  EXPECT_EQ(FI->getArgInfo(6).getType(), S16);
  const ArgInfo &Arg6 = FI->getArgInfo(6).Info;
  EXPECT_TRUE(Arg6.isIndirect());
  EXPECT_FALSE(Arg6.getIndirectByVal());
}

} // namespace
