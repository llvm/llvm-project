//===- AArch64TargetInfoTest.cpp - AArch64 ABI unit tests -----------------===//
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
#include "llvm/IR/CallingConv.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Allocator.h"
#include "gtest/gtest.h"

namespace {

using ABIType = llvm::abi::Type;
using llvm::abi::AArch64ABIKind;
using llvm::abi::ArgInfo;
using llvm::abi::createAArch64TargetInfo;
using llvm::abi::FunctionInfo;
using llvm::abi::TargetInfo;
using llvm::abi::TypeBuilder;

class AArch64TargetInfoTest : public ::testing::Test {
protected:
  llvm::BumpPtrAllocator Alloc;
  TypeBuilder TB;
  const ABIType *Bool;
  const ABIType *I8;
  const ABIType *U8;
  const ABIType *I16;
  const ABIType *U16;
  const ABIType *I32;
  const ABIType *U32;
  const ABIType *I64;
  const ABIType *U64;
  const ABIType *F32;
  const ABIType *F64;
  const ABIType *Ptr;
  const ABIType *Void;
  const ABIType *Matrix;

  AArch64TargetInfoTest()
      : TB(Alloc), Bool(TB.getIntegerType(1, llvm::Align(1), /*Signed=*/false)),
        I8(TB.getIntegerType(8, llvm::Align(1), /*Signed=*/true)),
        U8(TB.getIntegerType(8, llvm::Align(1), /*Signed=*/false)),
        I16(TB.getIntegerType(16, llvm::Align(2), /*Signed=*/true)),
        U16(TB.getIntegerType(16, llvm::Align(2), /*Signed=*/false)),
        I32(TB.getIntegerType(32, llvm::Align(4), /*Signed=*/true)),
        U32(TB.getIntegerType(32, llvm::Align(4), /*Signed=*/false)),
        I64(TB.getIntegerType(64, llvm::Align(8), /*Signed=*/true)),
        U64(TB.getIntegerType(64, llvm::Align(8), /*Signed=*/false)),
        F32(TB.getFloatType(llvm::APFloat::IEEEsingle(), llvm::Align(4))),
        F64(TB.getFloatType(llvm::APFloat::IEEEdouble(), llvm::Align(8))),
        Ptr(TB.getPointerType(64, llvm::Align(8))), Void(TB.getVoidType()),
        Matrix(TB.getArrayType(F32, /*NumElements=*/4, /*SizeInBits=*/128,
                               /*IsMatrixType=*/true)) {}
};

static void expectUncoercedDirect(const ArgInfo &Info) {
  EXPECT_TRUE(Info.isDirect());
  EXPECT_EQ(Info.getCoerceToType(), nullptr);
}

static void expectExtendInteger(const ArgInfo &Info, const ABIType *Ty,
                                bool IsSigned) {
  EXPECT_TRUE(Info.isExtend());
  EXPECT_EQ(Info.isSignExt(), IsSigned);
  EXPECT_EQ(Info.getCoerceToType(), Ty);
}

TEST_F(AArch64TargetInfoTest, ClassifyReturnVoidIsIgnore) {
  std::unique_ptr<TargetInfo> TI =
      createAArch64TargetInfo(TB, AArch64ABIKind::DarwinPCS);
  std::unique_ptr<FunctionInfo> FI =
      FunctionInfo::create(llvm::CallingConv::C, Void, {});

  FI->getReturnInfo() = ArgInfo::getDirect();
  TI->computeInfo(*FI);

  EXPECT_TRUE(FI->getReturnInfo().isIgnore());
  EXPECT_TRUE(FI->arguments().empty());
}

// Non-aggregate scalars, matrix types, and promotable integers take the Direct
// return path under AAPCS.
TEST_F(AArch64TargetInfoTest, ClassifyReturnScalarsDirectAAPCS) {
  std::unique_ptr<TargetInfo> TI =
      createAArch64TargetInfo(TB, AArch64ABIKind::AAPCS);

  for (const ABIType *RetTy :
       {Bool, I8, U8, I16, U16, I32, U32, I64, U64, F32, F64, Ptr, Matrix}) {
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, RetTy, {});
    FI->getReturnInfo() = ArgInfo::getIgnore();
    TI->computeInfo(*FI);
    expectUncoercedDirect(FI->getReturnInfo());
  }
}

// DarwinPCS returns non-promotable scalars directly. Promotable integer
// returns are extended.
TEST_F(AArch64TargetInfoTest, ClassifyReturnScalarsDirectOrPromotableDarwin) {
  std::unique_ptr<TargetInfo> TI =
      createAArch64TargetInfo(TB, AArch64ABIKind::DarwinPCS);

  for (const ABIType *RetTy : {I32, U32, I64, U64, F32, F64, Ptr, Matrix}) {
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, RetTy, {});
    FI->getReturnInfo() = ArgInfo::getIgnore();
    TI->computeInfo(*FI);
    expectUncoercedDirect(FI->getReturnInfo());
  }

  for (const ABIType *RetTy : {Bool, I8, U8, I16, U16}) {
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, RetTy, {});
    FI->getReturnInfo() = ArgInfo::getIgnore();
    TI->computeInfo(*FI);

    bool IsSigned = llvm::cast<llvm::abi::IntegerType>(RetTy)->isSigned();
    expectExtendInteger(FI->getReturnInfo(), RetTy, IsSigned);
  }
}

// Non-aggregate scalars, matrix types, and promotable integers take the Direct
// return path under AAPCSSoft.
TEST_F(AArch64TargetInfoTest, ClassifyReturnScalarsDirectAAPCSSoft) {
  std::unique_ptr<TargetInfo> TI =
      createAArch64TargetInfo(TB, AArch64ABIKind::AAPCSSoft);

  for (const ABIType *RetTy :
       {Bool, I8, U8, I16, U16, I32, U32, I64, U64, F32, F64, Ptr, Matrix}) {
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, RetTy, {});
    FI->getReturnInfo() = ArgInfo::getIgnore();
    TI->computeInfo(*FI);
    expectUncoercedDirect(FI->getReturnInfo());
  }
}

// Non-aggregate scalars, matrix types, and promotable integers take the Direct
// argument path under AAPCS.
TEST_F(AArch64TargetInfoTest, ClassifyArgumentScalarsDirectAAPCS) {
  std::unique_ptr<TargetInfo> TI =
      createAArch64TargetInfo(TB, AArch64ABIKind::AAPCS);

  for (const ABIType *ArgTy :
       {Bool, I8, U8, I16, U16, I32, U32, I64, U64, F32, F64, Ptr, Matrix}) {
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, Void, {ArgTy});
    TI->computeInfo(*FI);
    expectUncoercedDirect(FI->getArgInfo(0).Info);
  }
}

// DarwinPCS passes non-promotable scalars directly. Promotable integer
// arguments are extended.
TEST_F(AArch64TargetInfoTest, ClassifyArgumentScalarsDirectOrPromotableDarwin) {
  std::unique_ptr<TargetInfo> TI =
      createAArch64TargetInfo(TB, AArch64ABIKind::DarwinPCS);

  for (const ABIType *ArgTy : {I32, U32, I64, U64, F32, F64, Ptr, Matrix}) {
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, Void, {ArgTy});
    TI->computeInfo(*FI);
    expectUncoercedDirect(FI->getArgInfo(0).Info);
  }

  for (const ABIType *ArgTy : {Bool, I8, U8, I16, U16}) {
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, Void, {ArgTy});
    TI->computeInfo(*FI);

    bool IsSigned = llvm::cast<llvm::abi::IntegerType>(ArgTy)->isSigned();
    expectExtendInteger(FI->getArgInfo(0).Info, ArgTy, IsSigned);
  }
}

// Non-aggregate scalars, matrix types, and promotable integers take the Direct
// argument path under AAPCSSoft.
TEST_F(AArch64TargetInfoTest, ClassifyArgumentScalarsDirectAAPCSSoft) {
  std::unique_ptr<TargetInfo> TI =
      createAArch64TargetInfo(TB, AArch64ABIKind::AAPCSSoft);

  for (const ABIType *ArgTy :
       {Bool, I8, U8, I16, U16, I32, U32, I64, U64, F32, F64, Ptr, Matrix}) {
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, Void, {ArgTy});
    TI->computeInfo(*FI);
    expectUncoercedDirect(FI->getArgInfo(0).Info);
  }
}

} // namespace
