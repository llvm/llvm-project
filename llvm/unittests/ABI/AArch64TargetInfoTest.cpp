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
#include "llvm/Support/TypeSize.h"
#include "gtest/gtest.h"
#include <cstdint>

namespace {

using ABIType = llvm::abi::Type;
using llvm::abi::AArch64ABIKind;
using llvm::abi::AArch64ABIOptions;
using llvm::abi::ArgInfo;
using llvm::abi::createAArch64TargetInfo;
using llvm::abi::FieldInfo;
using llvm::abi::FunctionInfo;
using llvm::abi::RecordFlags;
using llvm::abi::RequiredArgs;
using llvm::abi::StructPacking;
using llvm::abi::TargetInfo;
using llvm::abi::TypeBuilder;

static void expectUncoercedDirect(const ArgInfo &Info);
static void expectExtendInteger(const ArgInfo &Info, const ABIType *Ty,
                                bool IsSigned);

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
  const ABIType *I128;
  const ABIType *F16;
  const ABIType *F32;
  const ABIType *F64;
  const ABIType *Ptr;
  const ABIType *Void;
  const ABIType *Matrix;
  const ABIType *V2F32;
  const ABIType *V4F32;
  const ABIType *V8F32;
  const ABIType *V16I8;
  const ABIType *V17I8;
  const ABIType *V2I8;
  const ABIType *V3I8;
  const ABIType *V4I8;
  const ABIType *V3F32;
  const ABIType *V5I8;
  const ABIType *V1I128;
  const llvm::abi::VectorType *SVInt32;
  const llvm::abi::VectorType *SVBool;
  const llvm::abi::VectorType *SVCount;
  const llvm::abi::VectorType *SVFloat64;
  const llvm::abi::VectorType *FixedSVInt8;
  const llvm::abi::VectorType *FixedSVInt32;
  const llvm::abi::VectorType *FixedSVInt32VL512;
  const llvm::abi::VectorType *FixedSVUint32;
  const llvm::abi::VectorType *FixedSVFloat64;
  const llvm::abi::VectorType *FixedSVBool;
  const ABIType *SVInt32x2;
  const ABIType *BitInt7;
  const ABIType *UBitInt7;
  const ABIType *BitInt65;
  const ABIType *BitInt128;
  const ABIType *BitInt129;
  const ABIType *BitInt2;
  const ABIType *V4BitInt2;
  const ABIType *V8BitInt2;
  const ABIType *ComplexFloat;

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
        I128(TB.getIntegerType(128, llvm::Align(16), /*Signed=*/false)),
        F16(TB.getFloatType(llvm::APFloat::IEEEhalf(), llvm::Align(2))),
        F32(TB.getFloatType(llvm::APFloat::IEEEsingle(), llvm::Align(4))),
        F64(TB.getFloatType(llvm::APFloat::IEEEdouble(), llvm::Align(8))),
        Ptr(TB.getPointerType(64, llvm::Align(8))), Void(TB.getVoidType()),
        Matrix(TB.getArrayType(F32, /*NumElements=*/4, /*SizeInBits=*/128,
                               /*IsMatrixType=*/true)),
        V2F32(TB.getVectorType(F32, llvm::ElementCount::getFixed(2),
                               llvm::Align(8))),
        V4F32(TB.getVectorType(F32, llvm::ElementCount::getFixed(4),
                               llvm::Align(16))),
        V8F32(TB.getVectorType(F32, llvm::ElementCount::getFixed(8),
                               llvm::Align(16))),
        V16I8(TB.getVectorType(I8, llvm::ElementCount::getFixed(16),
                               llvm::Align(16))),
        V17I8(TB.getVectorType(I8, llvm::ElementCount::getFixed(17),
                               llvm::Align(16))),
        V2I8(TB.getVectorType(I8, llvm::ElementCount::getFixed(2),
                              llvm::Align(2))),
        V3I8(TB.getVectorType(I8, llvm::ElementCount::getFixed(3),
                              llvm::Align(4))),
        V4I8(TB.getVectorType(I8, llvm::ElementCount::getFixed(4),
                              llvm::Align(4))),
        V3F32(TB.getVectorType(F32, llvm::ElementCount::getFixed(3),
                               llvm::Align(16))),
        V5I8(TB.getVectorType(I8, llvm::ElementCount::getFixed(5),
                              llvm::Align(8))),
        V1I128(TB.getVectorType(I128, llvm::ElementCount::getFixed(1),
                                llvm::Align(16))),
        SVInt32(TB.getVectorType(I32, llvm::ElementCount::getScalable(4),
                                 llvm::Align(16),
                                 llvm::abi::VectorKind::SVEData)),
        SVBool(TB.getVectorType(Bool, llvm::ElementCount::getScalable(16),
                                llvm::Align(2),
                                llvm::abi::VectorKind::SVEPredicate)),
        SVCount(TB.getSVECountType(llvm::Align(2))),
        SVFloat64(TB.getVectorType(F64, llvm::ElementCount::getScalable(2),
                                   llvm::Align(16),
                                   llvm::abi::VectorKind::SVEData)),
        FixedSVInt8(TB.getVectorType(I8, llvm::ElementCount::getFixed(32),
                                     llvm::Align(16),
                                     llvm::abi::VectorKind::SVEData)),
        FixedSVInt32(TB.getVectorType(I32, llvm::ElementCount::getFixed(8),
                                      llvm::Align(16),
                                      llvm::abi::VectorKind::SVEData)),
        FixedSVInt32VL512(
            TB.getVectorType(I32, llvm::ElementCount::getFixed(16),
                             llvm::Align(16), llvm::abi::VectorKind::SVEData)),
        FixedSVUint32(TB.getVectorType(U32, llvm::ElementCount::getFixed(8),
                                       llvm::Align(16),
                                       llvm::abi::VectorKind::SVEData)),
        FixedSVFloat64(TB.getVectorType(F64, llvm::ElementCount::getFixed(4),
                                        llvm::Align(16),
                                        llvm::abi::VectorKind::SVEData)),
        FixedSVBool(TB.getVectorType(U8, llvm::ElementCount::getFixed(32),
                                     llvm::Align(2),
                                     llvm::abi::VectorKind::SVEPredicate)),
        SVInt32x2(TB.getTupleType(SVInt32, /*NumVectors=*/2)),
        BitInt7(TB.getIntegerType(7, llvm::Align(1), /*Signed=*/true,
                                  /*IsBitInt=*/true)),
        UBitInt7(TB.getIntegerType(7, llvm::Align(1), /*Signed=*/false,
                                   /*IsBitInt=*/true)),
        BitInt65(TB.getIntegerType(65, llvm::Align(16), /*Signed=*/true,
                                   /*IsBitInt=*/true)),
        BitInt128(TB.getIntegerType(128, llvm::Align(16), /*Signed=*/true,
                                    /*IsBitInt=*/true)),
        BitInt129(TB.getIntegerType(129, llvm::Align(16), /*Signed=*/true,
                                    /*IsBitInt=*/true)),
        BitInt2(TB.getIntegerType(2, llvm::Align(1), /*Signed=*/true,
                                  /*IsBitInt=*/true)),
        V4BitInt2(TB.getVectorType(BitInt2, llvm::ElementCount::getFixed(4),
                                   llvm::Align(4))),
        V8BitInt2(TB.getVectorType(BitInt2, llvm::ElementCount::getFixed(8),
                                   llvm::Align(8))),
        ComplexFloat(TB.getComplexType(F32, llvm::Align(4))) {}

  static RecordFlags passableRecordFlags(bool IsCXX = false) {
    unsigned Flags = RecordFlags::CanPassInRegisters;
    if (IsCXX)
      Flags |= RecordFlags::IsCXXRecord;
    return static_cast<RecordFlags>(Flags);
  }

  const ABIType *makeRecord(llvm::ArrayRef<FieldInfo> Fields, uint64_t SizeBits,
                            llvm::Align Align, llvm::Align UnadjustedAlign,
                            RecordFlags Flags = RecordFlags::CanPassInRegisters,
                            llvm::ArrayRef<FieldInfo> Bases = {},
                            llvm::ArrayRef<FieldInfo> VBases = {}) {
    return TB.getRecordType(Fields, llvm::TypeSize::getFixed(SizeBits), Align,
                            UnadjustedAlign, StructPacking::Default, Bases,
                            VBases, Flags);
  }
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

static void expectDirectCoercedInteger(const ArgInfo &Info, uint64_t BitWidth) {
  EXPECT_TRUE(Info.isDirect());
  const llvm::abi::IntegerType *IT =
      llvm::dyn_cast<llvm::abi::IntegerType>(Info.getCoerceToType());
  ASSERT_NE(IT, nullptr);
  EXPECT_EQ(IT->getSizeInBits().getFixedValue(), BitWidth);
}

static void expectDirectCoercedI32Vector(const ArgInfo &Info,
                                         unsigned NumElts) {
  EXPECT_TRUE(Info.isDirect());
  const llvm::abi::VectorType *VT =
      llvm::dyn_cast<llvm::abi::VectorType>(Info.getCoerceToType());
  ASSERT_NE(VT, nullptr);
  EXPECT_EQ(VT->getNumElements().getKnownMinValue(), NumElts);
  EXPECT_EQ(VT->getElementType()->getSizeInBits().getFixedValue(), 32u);
}

// Checks that \p Info is Direct with a coercion to a scalable SVE data
// vector of \p MinElts copies of \p EltTy.
static void expectDirectCoercedSVEData(const ArgInfo &Info,
                                       const ABIType *EltTy, unsigned MinElts) {
  EXPECT_TRUE(Info.isDirect());
  const auto *VT =
      llvm::dyn_cast<llvm::abi::VectorType>(Info.getCoerceToType());
  ASSERT_NE(VT, nullptr);
  EXPECT_TRUE(VT->isSVEData());
  EXPECT_TRUE(VT->isScalable());
  EXPECT_EQ(VT->getNumElements().getKnownMinValue(), MinElts);
  // The element type is carried over unchanged, so signedness survives.
  EXPECT_EQ(VT->getElementType(), EltTy);
}

// Checks that \p Info is Direct with a coercion to svbool_t.
static void expectDirectCoercedSVEPredicate(const ArgInfo &Info) {
  EXPECT_TRUE(Info.isDirect());
  const auto *VT =
      llvm::dyn_cast<llvm::abi::VectorType>(Info.getCoerceToType());
  ASSERT_NE(VT, nullptr);
  EXPECT_TRUE(VT->isSVEPredicate());
  EXPECT_TRUE(VT->isScalable());
  EXPECT_EQ(VT->getNumElements().getKnownMinValue(), 16u);
  EXPECT_EQ(VT->getElementType()->getSizeInBits().getFixedValue(), 1u);
}

static void expectAlignedIndirect(const ArgInfo &Info, llvm::Align Align,
                                  bool ByVal = false) {
  EXPECT_TRUE(Info.isIndirect());
  EXPECT_EQ(Info.getIndirectAlign(), Align);
  EXPECT_EQ(Info.getIndirectByVal(), ByVal);
}

TEST_F(AArch64TargetInfoTest, ClassifyReturnVoidIsIgnore) {
  std::unique_ptr<TargetInfo> TI =
      createAArch64TargetInfo(TB, AArch64ABIOptions(AArch64ABIKind::DarwinPCS));
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
      createAArch64TargetInfo(TB, AArch64ABIOptions(AArch64ABIKind::AAPCS));

  for (const ABIType *RetTy : {Bool, I8, U8, I16, U16, I32, U32, I64, U64, F16,
                               F32, F64, Ptr, Matrix}) {
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
      createAArch64TargetInfo(TB, AArch64ABIOptions(AArch64ABIKind::DarwinPCS));

  for (const ABIType *RetTy :
       {I32, U32, I64, U64, F16, F32, F64, Ptr, Matrix}) {
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
      createAArch64TargetInfo(TB, AArch64ABIOptions(AArch64ABIKind::AAPCSSoft));

  for (const ABIType *RetTy : {Bool, I8, U8, I16, U16, I32, U32, I64, U64, F16,
                               F32, F64, Ptr, Matrix}) {
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, RetTy, {});
    FI->getReturnInfo() = ArgInfo::getIgnore();
    TI->computeInfo(*FI);
    expectUncoercedDirect(FI->getReturnInfo());
  }
}

// _BitInt types no wider than 128 bits are returned directly under AAPCS.
// Wider _BitInt types are returned indirectly.
TEST_F(AArch64TargetInfoTest, ClassifyReturnBitIntAAPCS) {
  std::unique_ptr<TargetInfo> TI =
      createAArch64TargetInfo(TB, AArch64ABIOptions(AArch64ABIKind::AAPCS));

  for (const ABIType *RetTy : {BitInt7, UBitInt7, BitInt65, BitInt128}) {
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, RetTy, {});
    FI->getReturnInfo() = ArgInfo::getIgnore();
    TI->computeInfo(*FI);
    expectUncoercedDirect(FI->getReturnInfo());
  }

  std::unique_ptr<FunctionInfo> FI =
      FunctionInfo::create(llvm::CallingConv::C, BitInt129, {});
  FI->getReturnInfo() = ArgInfo::getIgnore();
  TI->computeInfo(*FI);
  expectAlignedIndirect(FI->getReturnInfo(), llvm::Align(16), /*ByVal=*/true);
}

// DarwinPCS extends promotable _BitInt returns. Other _BitInt types no wider
// than 128 bits are returned directly. Wider _BitInt types are returned
// indirectly.
TEST_F(AArch64TargetInfoTest, ClassifyReturnBitIntDarwin) {
  std::unique_ptr<TargetInfo> TI =
      createAArch64TargetInfo(TB, AArch64ABIOptions(AArch64ABIKind::DarwinPCS));

  for (const ABIType *RetTy : {BitInt65, BitInt128}) {
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, RetTy, {});
    FI->getReturnInfo() = ArgInfo::getIgnore();
    TI->computeInfo(*FI);
    expectUncoercedDirect(FI->getReturnInfo());
  }

  {
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, BitInt7, {});
    FI->getReturnInfo() = ArgInfo::getIgnore();
    TI->computeInfo(*FI);
    expectExtendInteger(FI->getReturnInfo(), BitInt7, /*IsSigned=*/true);
  }
  {
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, UBitInt7, {});
    FI->getReturnInfo() = ArgInfo::getIgnore();
    TI->computeInfo(*FI);
    expectExtendInteger(FI->getReturnInfo(), UBitInt7, /*IsSigned=*/false);
  }

  std::unique_ptr<FunctionInfo> FI =
      FunctionInfo::create(llvm::CallingConv::C, BitInt129, {});
  FI->getReturnInfo() = ArgInfo::getIgnore();
  TI->computeInfo(*FI);
  expectAlignedIndirect(FI->getReturnInfo(), llvm::Align(16), /*ByVal=*/true);
}

// Non-aggregate scalars, matrix types, and promotable integers take the Direct
// return path under Win64.
TEST_F(AArch64TargetInfoTest, ClassifyReturnScalarsDirectWin64) {
  std::unique_ptr<TargetInfo> TI =
      createAArch64TargetInfo(TB, AArch64ABIOptions(AArch64ABIKind::Win64));

  for (const ABIType *RetTy :
       {Bool, I8, U8, I16, U16, I32, U32, I64, U64, F32, F64, Ptr, Matrix}) {
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, RetTy, {});
    FI->getReturnInfo() = ArgInfo::getIgnore();
    TI->computeInfo(*FI);
    expectUncoercedDirect(FI->getReturnInfo());
  }
}

// Non-SVE vector types no wider than 128 bits are returned directly. Larger
// vector types are returned indirectly.
TEST_F(AArch64TargetInfoTest, ClassifyReturnNonSVEVectors) {
  for (AArch64ABIKind Kind :
       {AArch64ABIKind::AAPCS, AArch64ABIKind::DarwinPCS, AArch64ABIKind::Win64,
        AArch64ABIKind::AAPCSSoft}) {
    std::unique_ptr<TargetInfo> TI =
        createAArch64TargetInfo(TB, AArch64ABIOptions(Kind));

    for (const ABIType *RetTy : {V2F32, V4F32, V16I8}) {
      std::unique_ptr<FunctionInfo> FI =
          FunctionInfo::create(llvm::CallingConv::C, RetTy, {});
      FI->getReturnInfo() = ArgInfo::getIgnore();
      TI->computeInfo(*FI);
      expectUncoercedDirect(FI->getReturnInfo());
    }

    for (const ABIType *RetTy : {V8F32, V17I8}) {
      std::unique_ptr<FunctionInfo> FI =
          FunctionInfo::create(llvm::CallingConv::C, RetTy, {});
      FI->getReturnInfo() = ArgInfo::getIgnore();
      TI->computeInfo(*FI);
      expectAlignedIndirect(FI->getReturnInfo(), llvm::Align(16),
                            /*ByVal=*/true);
    }
  }
}

// Legal 64- and 128-bit non-SVE vectors are passed directly. Illegal vectors
// are coerced, or passed indirectly if larger than 128 bits.
TEST_F(AArch64TargetInfoTest, ClassifyArgumentNonSVEVectors) {
  for (AArch64ABIKind Kind :
       {AArch64ABIKind::AAPCS, AArch64ABIKind::DarwinPCS, AArch64ABIKind::Win64,
        AArch64ABIKind::AAPCSSoft}) {
    std::unique_ptr<TargetInfo> TI =
        createAArch64TargetInfo(TB, AArch64ABIOptions(Kind));

    for (const ABIType *ArgTy : {V2F32, V4F32, V16I8}) {
      std::unique_ptr<FunctionInfo> FI =
          FunctionInfo::create(llvm::CallingConv::C, Void, {ArgTy});
      TI->computeInfo(*FI);
      expectUncoercedDirect(FI->getArgInfo(0).Info);
    }

    for (const ABIType *ArgTy : {V3I8, V4I8}) {
      std::unique_ptr<FunctionInfo> FI =
          FunctionInfo::create(llvm::CallingConv::C, Void, {ArgTy});
      TI->computeInfo(*FI);
      expectDirectCoercedInteger(FI->getArgInfo(0).Info, 32);
    }

    {
      std::unique_ptr<FunctionInfo> FI =
          FunctionInfo::create(llvm::CallingConv::C, Void, {V2I8});
      TI->computeInfo(*FI);
      expectDirectCoercedInteger(FI->getArgInfo(0).Info, 32);
    }

    // A vector whose element count is not a power of 2 is coerced based on
    // its ABI size, which rounds the payload width up to a power of 2. So
    // 5 x i8 is coerced as 64 bits and 3 x float as 128 bits.
    {
      std::unique_ptr<FunctionInfo> FI =
          FunctionInfo::create(llvm::CallingConv::C, Void, {V5I8});
      TI->computeInfo(*FI);
      expectDirectCoercedI32Vector(FI->getArgInfo(0).Info, 2);
    }

    {
      std::unique_ptr<FunctionInfo> FI =
          FunctionInfo::create(llvm::CallingConv::C, Void, {V3F32});
      TI->computeInfo(*FI);
      expectDirectCoercedI32Vector(FI->getArgInfo(0).Info, 4);
    }

    // A sub-byte _BitInt element counts as 8 bits towards the size of the
    // vector, so 4 x _BitInt(2) is an illegal 32-bit vector while
    // 8 x _BitInt(2) is a legal 64-bit one.
    {
      std::unique_ptr<FunctionInfo> FI =
          FunctionInfo::create(llvm::CallingConv::C, Void, {V4BitInt2});
      TI->computeInfo(*FI);
      expectDirectCoercedInteger(FI->getArgInfo(0).Info, 32);
    }

    {
      std::unique_ptr<FunctionInfo> FI =
          FunctionInfo::create(llvm::CallingConv::C, Void, {V8BitInt2});
      TI->computeInfo(*FI);
      expectUncoercedDirect(FI->getArgInfo(0).Info);
    }

    {
      std::unique_ptr<FunctionInfo> FI =
          FunctionInfo::create(llvm::CallingConv::C, Void, {V1I128});
      TI->computeInfo(*FI);
      expectDirectCoercedI32Vector(FI->getArgInfo(0).Info, 4);
    }

    for (const ABIType *ArgTy : {V8F32, V17I8}) {
      std::unique_ptr<FunctionInfo> FI =
          FunctionInfo::create(llvm::CallingConv::C, Void, {ArgTy});
      TI->computeInfo(*FI);
      expectAlignedIndirect(FI->getArgInfo(0).Info, llvm::Align(16),
                            /*ByVal=*/false);
    }
  }
}

// Android and OHOS coerce illegal vectors of at most 16 bits to i16 rather
// than i32.
TEST_F(AArch64TargetInfoTest, ClassifyArgumentIllegalVectorAndroid) {
  AArch64ABIOptions Opts(AArch64ABIKind::AAPCS);
  Opts.IsAndroidOrOHOS = true;
  std::unique_ptr<TargetInfo> TI = createAArch64TargetInfo(TB, Opts);

  std::unique_ptr<FunctionInfo> FI =
      FunctionInfo::create(llvm::CallingConv::C, Void, {V2I8});
  TI->computeInfo(*FI);
  expectDirectCoercedInteger(FI->getArgInfo(0).Info, 16);

  FI = FunctionInfo::create(llvm::CallingConv::C, Void, {V4I8});
  TI->computeInfo(*FI);
  expectDirectCoercedInteger(FI->getArgInfo(0).Info, 32);
}

// arm64_32 MachO treats vectors larger than 32 bits as legal, including
// sizes that other AArch64 ABIs pass indirectly. Non-power-of-2 element
// counts are still illegal.
TEST_F(AArch64TargetInfoTest, ClassifyArgumentVectorILP32MachO) {
  AArch64ABIOptions Opts(AArch64ABIKind::DarwinPCS);
  Opts.IsILP32 = true;
  Opts.IsMachO = true;
  std::unique_ptr<TargetInfo> TI = createAArch64TargetInfo(TB, Opts);

  for (const ABIType *ArgTy : {V2F32, V4F32, V16I8, V8F32, V1I128}) {
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, Void, {ArgTy});
    TI->computeInfo(*FI);
    expectUncoercedDirect(FI->getArgInfo(0).Info);
  }

  {
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, Void, {V4I8});
    TI->computeInfo(*FI);
    expectDirectCoercedInteger(FI->getArgInfo(0).Info, 32);
  }

  {
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, Void, {V17I8});
    TI->computeInfo(*FI);
    expectAlignedIndirect(FI->getArgInfo(0).Info, llvm::Align(16),
                          /*ByVal=*/false);
  }
}

// The sizeless SVE types occupy a register of their own, so they are passed
// and returned directly, without coercion, under every ABI kind.
TEST_F(AArch64TargetInfoTest, ClassifySizelessSVETypesDirect) {
  const ABIType *SVETypes[] = {SVInt32, SVFloat64, SVBool, SVCount};

  for (AArch64ABIKind Kind :
       {AArch64ABIKind::AAPCS, AArch64ABIKind::DarwinPCS, AArch64ABIKind::Win64,
        AArch64ABIKind::AAPCSSoft}) {
    std::unique_ptr<TargetInfo> TI =
        createAArch64TargetInfo(TB, AArch64ABIOptions(Kind));

    for (const ABIType *Ty : SVETypes) {
      std::unique_ptr<FunctionInfo> FI =
          FunctionInfo::create(llvm::CallingConv::C, Ty, {Ty});
      FI->getReturnInfo() = ArgInfo::getIgnore();
      TI->computeInfo(*FI);
      expectUncoercedDirect(FI->getReturnInfo());
      expectUncoercedDirect(FI->getArgInfo(0).Info);
    }
  }
}

// Fixed-length SVE data vectors are coerced to the scalable vector that
// occupies the same register. The scalable element count depends only on the
// element size, so the two vector lengths of the same element type coerce to
// the same scalable type.
TEST_F(AArch64TargetInfoTest, ClassifyFixedLengthSVEDataCoerced) {
  for (AArch64ABIKind Kind :
       {AArch64ABIKind::AAPCS, AArch64ABIKind::DarwinPCS, AArch64ABIKind::Win64,
        AArch64ABIKind::AAPCSSoft}) {
    std::unique_ptr<TargetInfo> TI =
        createAArch64TargetInfo(TB, AArch64ABIOptions(Kind));

    struct {
      const ABIType *Ty;
      const ABIType *EltTy;
      unsigned MinElts;
    } Cases[] = {
        {FixedSVInt8, I8, 16},       {FixedSVInt32, I32, 4},
        {FixedSVInt32VL512, I32, 4}, {FixedSVUint32, U32, 4},
        {FixedSVFloat64, F64, 2},
    };

    for (const auto &Case : Cases) {
      std::unique_ptr<FunctionInfo> FI =
          FunctionInfo::create(llvm::CallingConv::C, Case.Ty, {Case.Ty});
      FI->getReturnInfo() = ArgInfo::getIgnore();
      TI->computeInfo(*FI);
      expectDirectCoercedSVEData(FI->getReturnInfo(), Case.EltTy, Case.MinElts);
      expectDirectCoercedSVEData(FI->getArgInfo(0).Info, Case.EltTy,
                                 Case.MinElts);
    }
  }
}

// Fixed-length SVE predicates are described with 8-bit elements, but they are
// coerced to svbool_t, which has one-bit elements.
TEST_F(AArch64TargetInfoTest, ClassifyFixedLengthSVEPredicateCoerced) {
  for (AArch64ABIKind Kind :
       {AArch64ABIKind::AAPCS, AArch64ABIKind::DarwinPCS, AArch64ABIKind::Win64,
        AArch64ABIKind::AAPCSSoft}) {
    std::unique_ptr<TargetInfo> TI =
        createAArch64TargetInfo(TB, AArch64ABIOptions(Kind));

    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, FixedSVBool, {FixedSVBool});
    FI->getReturnInfo() = ArgInfo::getIgnore();
    TI->computeInfo(*FI);
    expectDirectCoercedSVEPredicate(FI->getReturnInfo());
    expectDirectCoercedSVEPredicate(FI->getArgInfo(0).Info);
  }
}

// Arm64EC variadic functions classify their arguments with the x86-64 rules,
// which are not implemented yet. Every argument is deferred, including the
// named ones. Non-variadic functions are unaffected.
TEST_F(AArch64TargetInfoTest,
       ClassifyArgumentArm64ECVariadicNotYetImplemented) {
  AArch64ABIOptions Opts(AArch64ABIKind::Win64);
  Opts.IsWindowsArm64EC = true;
  std::unique_ptr<TargetInfo> TI = createAArch64TargetInfo(TB, Opts);

  std::unique_ptr<FunctionInfo> FI = FunctionInfo::create(
      llvm::CallingConv::C, Void, {I32, F64}, RequiredArgs(1));
  TI->computeInfo(*FI);
  EXPECT_TRUE(FI->getArgInfo(0).Info.isIgnore());
  EXPECT_TRUE(FI->getArgInfo(1).Info.isIgnore());

  FI = FunctionInfo::create(llvm::CallingConv::C, Void, {I32, F64},
                            RequiredArgs::All);
  TI->computeInfo(*FI);
  expectUncoercedDirect(FI->getArgInfo(0).Info);
  expectUncoercedDirect(FI->getArgInfo(1).Info);
}

// Non-aggregate scalars, matrix types, and promotable integers take the Direct
// argument path under AAPCS.
TEST_F(AArch64TargetInfoTest, ClassifyArgumentScalarsDirectAAPCS) {
  std::unique_ptr<TargetInfo> TI =
      createAArch64TargetInfo(TB, AArch64ABIOptions(AArch64ABIKind::AAPCS));

  for (const ABIType *ArgTy : {Bool, I8, U8, I16, U16, I32, U32, I64, U64, F16,
                               F32, F64, Ptr, Matrix}) {
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
      createAArch64TargetInfo(TB, AArch64ABIOptions(AArch64ABIKind::DarwinPCS));

  for (const ABIType *ArgTy :
       {I32, U32, I64, U64, F16, F32, F64, Ptr, Matrix}) {
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
      createAArch64TargetInfo(TB, AArch64ABIOptions(AArch64ABIKind::AAPCSSoft));

  for (const ABIType *ArgTy : {Bool, I8, U8, I16, U16, I32, U32, I64, U64, F16,
                               F32, F64, Ptr, Matrix}) {
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, Void, {ArgTy});
    TI->computeInfo(*FI);
    expectUncoercedDirect(FI->getArgInfo(0).Info);
  }
}

// _BitInt types no wider than 128 bits are passed directly under AAPCS.
// Wider _BitInt types are passed indirectly without byval.
TEST_F(AArch64TargetInfoTest, ClassifyArgumentBitIntAAPCS) {
  std::unique_ptr<TargetInfo> TI =
      createAArch64TargetInfo(TB, AArch64ABIOptions(AArch64ABIKind::AAPCS));

  for (const ABIType *ArgTy : {BitInt7, UBitInt7, BitInt65, BitInt128}) {
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, Void, {ArgTy});
    TI->computeInfo(*FI);
    expectUncoercedDirect(FI->getArgInfo(0).Info);
  }

  std::unique_ptr<FunctionInfo> FI =
      FunctionInfo::create(llvm::CallingConv::C, Void, {BitInt129});
  TI->computeInfo(*FI);
  expectAlignedIndirect(FI->getArgInfo(0).Info, llvm::Align(16),
                        /*ByVal=*/false);
}

// DarwinPCS extends promotable _BitInt arguments. Other _BitInt types no
// wider than 128 bits are passed directly. Wider _BitInt types are passed
// indirectly without byval.
TEST_F(AArch64TargetInfoTest, ClassifyArgumentBitIntDarwin) {
  std::unique_ptr<TargetInfo> TI =
      createAArch64TargetInfo(TB, AArch64ABIOptions(AArch64ABIKind::DarwinPCS));

  for (const ABIType *ArgTy : {BitInt65, BitInt128}) {
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, Void, {ArgTy});
    TI->computeInfo(*FI);
    expectUncoercedDirect(FI->getArgInfo(0).Info);
  }

  {
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, Void, {BitInt7});
    TI->computeInfo(*FI);
    expectExtendInteger(FI->getArgInfo(0).Info, BitInt7, /*IsSigned=*/true);
  }
  {
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, Void, {UBitInt7});
    TI->computeInfo(*FI);
    expectExtendInteger(FI->getArgInfo(0).Info, UBitInt7, /*IsSigned=*/false);
  }

  std::unique_ptr<FunctionInfo> FI =
      FunctionInfo::create(llvm::CallingConv::C, Void, {BitInt129});
  TI->computeInfo(*FI);
  expectAlignedIndirect(FI->getArgInfo(0).Info, llvm::Align(16),
                        /*ByVal=*/false);
}

// Non-aggregate scalars, matrix types, and promotable integers take the Direct
// argument path under Win64.
TEST_F(AArch64TargetInfoTest, ClassifyArgumentScalarsDirectWin64) {
  std::unique_ptr<TargetInfo> TI =
      createAArch64TargetInfo(TB, AArch64ABIOptions(AArch64ABIKind::Win64));

  for (const ABIType *ArgTy :
       {Bool, I8, U8, I16, U16, I32, U32, I64, U64, F32, F64, Ptr, Matrix}) {
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, Void, {ArgTy});
    TI->computeInfo(*FI);
    expectUncoercedDirect(FI->getArgInfo(0).Info);
  }
}

static void expectNaturalAlignIndirect(const ArgInfo &Info,
                                       llvm::Align ExpectedAlign, bool ByVal) {
  EXPECT_TRUE(Info.isIndirect());
  EXPECT_EQ(Info.getIndirectAlign(), ExpectedAlign);
  EXPECT_EQ(Info.getIndirectByVal(), ByVal);
}

static void expectHFADirectArg(const ArgInfo &Info, const ABIType *Base,
                               uint64_t Members, llvm::MaybeAlign DirectAlign) {
  EXPECT_TRUE(Info.isDirect());
  const llvm::abi::ArrayType *AT =
      llvm::dyn_cast<llvm::abi::ArrayType>(Info.getCoerceToType());
  ASSERT_NE(AT, nullptr);
  EXPECT_EQ(AT->getElementType(), Base);
  EXPECT_EQ(AT->getNumElements(), Members);
  EXPECT_EQ(Info.getDirectOffset(), 0u);
  EXPECT_EQ(Info.getDirectAlign(), DirectAlign);
}

// Records that cannot be passed in registers (e.g. non-trivial C++ types) are
// classified as Indirect with ByVal=false under all AArch64 ABI kinds.
TEST_F(AArch64TargetInfoTest, ClassifyArgumentRecordCannotPassInRegisters) {
  // A record without CanPassInRegisters is treated like a C++ type with a
  // non-trivial copy constructor or destructor.
  const ABIType *CannotPass = TB.getRecordType(
      {llvm::abi::FieldInfo(I32)}, llvm::TypeSize::getFixed(32), llvm::Align(4),
      /*UnadjustedAlign=*/llvm::Align(4), llvm::abi::StructPacking::Default,
      /*BaseClasses=*/{},
      /*VirtualBaseClasses=*/{}, llvm::abi::RecordFlags::IsCXXRecord);

  for (AArch64ABIKind Kind :
       {AArch64ABIKind::AAPCS, AArch64ABIKind::DarwinPCS, AArch64ABIKind::Win64,
        AArch64ABIKind::AAPCSSoft}) {
    std::unique_ptr<TargetInfo> TI =
        createAArch64TargetInfo(TB, AArch64ABIOptions(Kind));
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, Void, {CannotPass});
    TI->computeInfo(*FI);
    expectNaturalAlignIndirect(FI->getArgInfo(0).Info, llvm::Align(4),
                               /*ByVal=*/false);
  }
}

// Transparent unions are classified as their first field type.
TEST_F(AArch64TargetInfoTest, ClassifyArgumentTransparentUnion) {
  using llvm::abi::FieldInfo;
  using llvm::abi::RecordFlags;
  using llvm::abi::StructPacking;

  // First field is i32; second field is ignored for classification.
  const ABIType *TUInt = TB.getUnionType(
      {FieldInfo(I32), FieldInfo(F32)}, llvm::TypeSize::getFixed(32),
      llvm::Align(4), /*UnadjustedAlign=*/llvm::Align(4),
      StructPacking::Default, RecordFlags::IsTransparent);

  for (AArch64ABIKind Kind :
       {AArch64ABIKind::AAPCS, AArch64ABIKind::DarwinPCS, AArch64ABIKind::Win64,
        AArch64ABIKind::AAPCSSoft}) {
    std::unique_ptr<TargetInfo> TI =
        createAArch64TargetInfo(TB, AArch64ABIOptions(Kind));
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, Void, {TUInt});
    TI->computeInfo(*FI);
    expectUncoercedDirect(FI->getArgInfo(0).Info);
  }

  // First field is a promotable integer: DarwinPCS extends; others are Direct.
  const ABIType *TUChar = TB.getUnionType(
      {FieldInfo(I8), FieldInfo(U8)}, llvm::TypeSize::getFixed(8),
      llvm::Align(1), /*UnadjustedAlign=*/llvm::Align(1),
      StructPacking::Default, RecordFlags::IsTransparent);

  {
    std::unique_ptr<TargetInfo> TI = createAArch64TargetInfo(
        TB, AArch64ABIOptions(AArch64ABIKind::DarwinPCS));
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, Void, {TUChar});
    TI->computeInfo(*FI);
    expectExtendInteger(FI->getArgInfo(0).Info, I8, /*IsSigned=*/true);
  }

  for (AArch64ABIKind Kind : {AArch64ABIKind::AAPCS, AArch64ABIKind::Win64,
                              AArch64ABIKind::AAPCSSoft}) {
    std::unique_ptr<TargetInfo> TI =
        createAArch64TargetInfo(TB, AArch64ABIOptions(Kind));
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, Void, {TUChar});
    TI->computeInfo(*FI);
    expectUncoercedDirect(FI->getArgInfo(0).Info);
  }
}

// Homogeneous floating-point aggregates of at most four members are returned
// directly under AAPCS, DarwinPCS, and Win64.
TEST_F(AArch64TargetInfoTest, ClassifyReturnHFADirect) {
  RecordFlags CXXFlags = passableRecordFlags(/*IsCXX=*/true);

  const ABIType *HFA2f =
      makeRecord({FieldInfo(F32, 0), FieldInfo(F32, 32)}, 64, llvm::Align(4),
                 /*UnadjustedAlign=*/llvm::Align(4));
  const ABIType *HFA4d = makeRecord({FieldInfo(F64, 0), FieldInfo(F64, 64),
                                     FieldInfo(F64, 128), FieldInfo(F64, 192)},
                                    256, llvm::Align(8),
                                    /*UnadjustedAlign=*/llvm::Align(8));
  const ABIType *HFA3arr =
      makeRecord({FieldInfo(TB.getArrayType(F32, 3, /*SizeInBits=*/96), 0)}, 96,
                 llvm::Align(4), /*UnadjustedAlign=*/llvm::Align(4));
  const ABIType *HFA2h =
      makeRecord({FieldInfo(F16, 0), FieldInfo(F16, 16)}, 32, llvm::Align(2),
                 /*UnadjustedAlign=*/llvm::Align(2));
  const ABIType *HFANested =
      makeRecord({FieldInfo(HFA2f, 0), FieldInfo(F32, 64)}, 96, llvm::Align(4),
                 /*UnadjustedAlign=*/llvm::Align(4));
  const ABIType *HFAZeroBF =
      makeRecord({FieldInfo(I32, 0, /*IsBitField=*/true, /*BitFieldWidth=*/0),
                  FieldInfo(F32, 0), FieldInfo(F32, 32)},
                 64, llvm::Align(4), /*UnadjustedAlign=*/llvm::Align(4));
  const ABIType *HFAUnion = TB.getUnionType(
      {FieldInfo(F32, 0),
       FieldInfo(TB.getArrayType(F32, 3, /*SizeInBits=*/96), 0)},
      llvm::TypeSize::getFixed(96), llvm::Align(4),
      /*UnadjustedAlign=*/llvm::Align(4), StructPacking::Default,
      RecordFlags::CanPassInRegisters);

  // Short-vector aggregates (HVAs) follow the same rules.
  const ABIType *HVA2x64 =
      makeRecord({FieldInfo(V2F32, 0), FieldInfo(V2F32, 64)}, 128,
                 llvm::Align(8), /*UnadjustedAlign=*/llvm::Align(8));
  const ABIType *HVA2x128 =
      makeRecord({FieldInfo(V4F32, 0), FieldInfo(V4F32, 128)}, 256,
                 llvm::Align(16), /*UnadjustedAlign=*/llvm::Align(16));

  // 3 x float has 96 bits of payload; Clang's type size is 128, so it is an
  // HVA base like a 128-bit short vector.
  const ABIType *V3F32 =
      TB.getVectorType(F32, llvm::ElementCount::getFixed(3), llvm::Align(16));
  const ABIType *HVA3x32 =
      makeRecord({FieldInfo(V3F32, 0)}, 128, llvm::Align(16),
                 /*UnadjustedAlign=*/llvm::Align(16));
  const ABIType *HVA2xV3F32 = makeRecord(
      {FieldInfo(V3F32, 0), FieldInfo(V3F32, 128)}, 256, llvm::Align(16),
      /*UnadjustedAlign=*/llvm::Align(16));

  // A 2x2 float matrix is four homogeneous float members.
  const ABIType *HFAMatrix =
      makeRecord({FieldInfo(Matrix, 0)}, 128, llvm::Align(4),
                 /*UnadjustedAlign=*/llvm::Align(4));
  const ABIType *M2x1 = TB.getArrayType(F32, /*NumElements=*/2,
                                        /*SizeInBits=*/64,
                                        /*IsMatrixType=*/true);
  const ABIType *HFAMatrix2 =
      makeRecord({FieldInfo(M2x1, 0)}, 64, llvm::Align(4),
                 /*UnadjustedAlign=*/llvm::Align(4));

  // C++ records: empty bases are skipped and non-empty bases contribute
  // members.
  const ABIType *EmptyRecord = makeRecord(
      {}, 0, llvm::Align(1), /*UnadjustedAlign=*/llvm::Align(1), CXXFlags);
  const ABIType *HFAEmptyBase =
      makeRecord({FieldInfo(F32, 0), FieldInfo(F32, 32)}, 64, llvm::Align(4),
                 /*UnadjustedAlign=*/llvm::Align(4), CXXFlags,
                 {FieldInfo(EmptyRecord, 0)});
  const ABIType *FloatBase =
      makeRecord({FieldInfo(F32, 0)}, 32, llvm::Align(4),
                 /*UnadjustedAlign=*/llvm::Align(4), CXXFlags);
  const ABIType *HFADerived = makeRecord(
      {FieldInfo(F32, 32)}, 64, llvm::Align(4),
      /*UnadjustedAlign=*/llvm::Align(4), CXXFlags, {FieldInfo(FloatBase, 0)});

  for (AArch64ABIKind Kind : {AArch64ABIKind::AAPCS, AArch64ABIKind::DarwinPCS,
                              AArch64ABIKind::Win64}) {
    std::unique_ptr<TargetInfo> TI =
        createAArch64TargetInfo(TB, AArch64ABIOptions(Kind));
    for (const ABIType *RetTy :
         {ComplexFloat, HFA2f, HFA4d, HFA3arr, HFA2h, HFANested, HFAZeroBF,
          HFAUnion, HVA2x64, HVA2x128, HVA3x32, HVA2xV3F32, HFAMatrix,
          HFAMatrix2, HFAEmptyBase, HFADerived}) {
      std::unique_ptr<FunctionInfo> FI =
          FunctionInfo::create(llvm::CallingConv::C, RetTy, {});
      FI->getReturnInfo() = ArgInfo::getIgnore();
      TI->computeInfo(*FI);
      expectUncoercedDirect(FI->getReturnInfo());
    }
  }
}

// Homogeneous floating-point and short-vector aggregates are passed as a
// coerced array of the base type. AAPCS overrides stack alignment; DarwinPCS
// and Win64 do not.
TEST_F(AArch64TargetInfoTest, ClassifyArgumentHFADirect) {
  RecordFlags CXXFlags = passableRecordFlags(/*IsCXX=*/true);

  const ABIType *HFA2f =
      makeRecord({FieldInfo(F32, 0), FieldInfo(F32, 32)}, 64, llvm::Align(4),
                 /*UnadjustedAlign=*/llvm::Align(4));
  const ABIType *HFA4d = makeRecord({FieldInfo(F64, 0), FieldInfo(F64, 64),
                                     FieldInfo(F64, 128), FieldInfo(F64, 192)},
                                    256, llvm::Align(8),
                                    /*UnadjustedAlign=*/llvm::Align(8));
  const ABIType *HFA3arr =
      makeRecord({FieldInfo(TB.getArrayType(F32, 3, /*SizeInBits=*/96), 0)}, 96,
                 llvm::Align(4), /*UnadjustedAlign=*/llvm::Align(4));
  const ABIType *HFA2h =
      makeRecord({FieldInfo(F16, 0), FieldInfo(F16, 16)}, 32, llvm::Align(2),
                 /*UnadjustedAlign=*/llvm::Align(2));
  const ABIType *HFANested =
      makeRecord({FieldInfo(HFA2f, 0), FieldInfo(F32, 64)}, 96, llvm::Align(4),
                 /*UnadjustedAlign=*/llvm::Align(4));
  const ABIType *HFAZeroBF =
      makeRecord({FieldInfo(I32, 0, /*IsBitField=*/true, /*BitFieldWidth=*/0),
                  FieldInfo(F32, 0), FieldInfo(F32, 32)},
                 64, llvm::Align(4), /*UnadjustedAlign=*/llvm::Align(4));
  const ABIType *HFAUnion = TB.getUnionType(
      {FieldInfo(F32, 0),
       FieldInfo(TB.getArrayType(F32, 3, /*SizeInBits=*/96), 0)},
      llvm::TypeSize::getFixed(96), llvm::Align(4),
      /*UnadjustedAlign=*/llvm::Align(4), StructPacking::Default,
      RecordFlags::CanPassInRegisters);
  const ABIType *HVA2x64 =
      makeRecord({FieldInfo(V2F32, 0), FieldInfo(V2F32, 64)}, 128,
                 llvm::Align(8), /*UnadjustedAlign=*/llvm::Align(8));
  const ABIType *HVA2x128 =
      makeRecord({FieldInfo(V4F32, 0), FieldInfo(V4F32, 128)}, 256,
                 llvm::Align(16), /*UnadjustedAlign=*/llvm::Align(16));

  const ABIType *EmptyRecord = makeRecord(
      {}, 0, llvm::Align(1), /*UnadjustedAlign=*/llvm::Align(1), CXXFlags);
  const ABIType *HFAEmptyBase =
      makeRecord({FieldInfo(F32, 0), FieldInfo(F32, 32)}, 64, llvm::Align(4),
                 /*UnadjustedAlign=*/llvm::Align(4), CXXFlags,
                 {FieldInfo(EmptyRecord, 0)});
  const ABIType *FloatBase =
      makeRecord({FieldInfo(F32, 0)}, 32, llvm::Align(4),
                 /*UnadjustedAlign=*/llvm::Align(4), CXXFlags);
  const ABIType *HFADerived = makeRecord(
      {FieldInfo(F32, 32)}, 64, llvm::Align(4),
      /*UnadjustedAlign=*/llvm::Align(4), CXXFlags, {FieldInfo(FloatBase, 0)});
  FieldInfo VirtualFloatBase(FloatBase, 0);

  struct HFACase {
    const ABIType *Ty;
    const ABIType *Base;
    uint64_t Members;
    llvm::Align AAPCSAlign;
  };
  const HFACase Cases[] = {
      {ComplexFloat, F32, 2, llvm::Align(8)},
      {HFA2f, F32, 2, llvm::Align(8)},
      {HFA4d, F64, 4, llvm::Align(8)},
      {HFA3arr, F32, 3, llvm::Align(8)},
      {HFA2h, F16, 2, llvm::Align(8)},
      {HFANested, F32, 3, llvm::Align(8)},
      {HFAZeroBF, F32, 2, llvm::Align(8)},
      {HFAUnion, F32, 3, llvm::Align(8)},
      {HVA2x64, V2F32, 2, llvm::Align(8)},
      {HVA2x128, V4F32, 2, llvm::Align(16)},
      {HFAEmptyBase, F32, 2, llvm::Align(8)},
      {HFADerived, F32, 2, llvm::Align(8)},
  };

  {
    std::unique_ptr<TargetInfo> TI =
        createAArch64TargetInfo(TB, AArch64ABIOptions(AArch64ABIKind::AAPCS));
    for (const HFACase &C : Cases) {
      std::unique_ptr<FunctionInfo> FI =
          FunctionInfo::create(llvm::CallingConv::C, Void, {C.Ty});
      TI->computeInfo(*FI);
      expectHFADirectArg(FI->getArgInfo(0).Info, C.Base, C.Members,
                         C.AAPCSAlign);
    }
  }

  for (AArch64ABIKind Kind :
       {AArch64ABIKind::DarwinPCS, AArch64ABIKind::Win64}) {
    std::unique_ptr<TargetInfo> TI =
        createAArch64TargetInfo(TB, AArch64ABIOptions(Kind));
    for (const HFACase &C : Cases) {
      std::unique_ptr<FunctionInfo> FI =
          FunctionInfo::create(llvm::CallingConv::C, Void, {C.Ty});
      TI->computeInfo(*FI);
      expectHFADirectArg(FI->getArgInfo(0).Info, C.Base, C.Members,
                         /*DirectAlign=*/std::nullopt);
    }
  }
}

// Records that cannot pass in registers are returned indirectly before HFA
// classification.
TEST_F(AArch64TargetInfoTest, ClassifyReturnCXXCannotPassInRegistersIndirect) {
  std::unique_ptr<TargetInfo> TI =
      createAArch64TargetInfo(TB, AArch64ABIOptions(AArch64ABIKind::AAPCS));

  const ABIType *NonPassableHFA =
      makeRecord({FieldInfo(F32, 0), FieldInfo(F32, 32)}, 64, llvm::Align(4),
                 /*UnadjustedAlign=*/llvm::Align(4), RecordFlags::IsCXXRecord);

  // struct FloatBase { float f; };
  // struct VirtualDerived : virtual FloatBase {};
  // Virtual inheritance makes the copy constructor non-trivial, so the derived
  // record cannot pass in registers even though its virtual base would
  // otherwise supply a homogeneous float member. The vbase pointer at offset 0
  // places the FloatBase subobject at offset 8, giving sizeof == 16.
  const ABIType *FloatBase = makeRecord({FieldInfo(F32, 0)}, 32, llvm::Align(4),
                                        /*UnadjustedAlign=*/llvm::Align(4),
                                        passableRecordFlags(/*IsCXX=*/true));
  const ABIType *VirtualDerived =
      makeRecord({}, 128, llvm::Align(8), /*UnadjustedAlign=*/llvm::Align(8),
                 RecordFlags::IsCXXRecord,
                 /*Bases=*/{}, /*VBases=*/{FieldInfo(FloatBase, 64)});

  const struct {
    const ABIType *RetTy;
    llvm::Align ExpectedAlign;
  } Cases[] = {{NonPassableHFA, llvm::Align(4)},
               {VirtualDerived, llvm::Align(8)}};

  for (const auto &Case : Cases) {
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, Case.RetTy, {});
    FI->getReturnInfo() = ArgInfo::getIgnore();
    TI->computeInfo(*FI);
    expectNaturalAlignIndirect(FI->getReturnInfo(), Case.ExpectedAlign,
                               /*ByVal=*/false);
  }
}

// AAPCS HFA stack alignment uses unadjusted alignment. A record-level aligned
// attribute raises getAlignment() but must not change the 8/16 stack cap.
TEST_F(AArch64TargetInfoTest, ClassifyArgumentOveralignedHFAAlign) {
  // Two doubles already occupy 16 bytes, so aligned(16) does not add padding
  // and the type remains an HFA. Unadjusted alignment stays 8.
  const ABIType *RecordAlignedHFA =
      makeRecord({FieldInfo(F64, 0), FieldInfo(F64, 64)}, 128, llvm::Align(16),
                 /*UnadjustedAlign=*/llvm::Align(8));

  // Four doubles occupy 32 bytes, so aligned(32) also remains an HFA.
  // Unadjusted alignment is still 8, so AAPCS must not cap up to 16.
  const ABIType *RecordAligned32HFA =
      makeRecord({FieldInfo(F64, 0), FieldInfo(F64, 64), FieldInfo(F64, 128),
                  FieldInfo(F64, 192)},
                 256, llvm::Align(32), /*UnadjustedAlign=*/llvm::Align(8));

  // Field-driven alignment of 16 is visible in unadjusted alignment, so AAPCS
  // uses the 16-byte cap.
  const ABIType *FieldAlignedHFA =
      makeRecord({FieldInfo(F64, 0), FieldInfo(F64, 64)}, 128, llvm::Align(16),
                 /*UnadjustedAlign=*/llvm::Align(16));

  // Field-driven alignment of 32 is capped at 16.
  const ABIType *FieldAligned32HFA =
      makeRecord({FieldInfo(F64, 0), FieldInfo(F64, 64), FieldInfo(F64, 128),
                  FieldInfo(F64, 192)},
                 256, llvm::Align(32), /*UnadjustedAlign=*/llvm::Align(32));

  {
    std::unique_ptr<TargetInfo> TI =
        createAArch64TargetInfo(TB, AArch64ABIOptions(AArch64ABIKind::AAPCS));

    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, Void, {RecordAlignedHFA});
    TI->computeInfo(*FI);
    expectHFADirectArg(FI->getArgInfo(0).Info, F64, 2, llvm::Align(8));

    FI = FunctionInfo::create(llvm::CallingConv::C, Void, {RecordAligned32HFA});
    TI->computeInfo(*FI);
    expectHFADirectArg(FI->getArgInfo(0).Info, F64, 4, llvm::Align(8));

    FI = FunctionInfo::create(llvm::CallingConv::C, Void, {FieldAlignedHFA});
    TI->computeInfo(*FI);
    expectHFADirectArg(FI->getArgInfo(0).Info, F64, 2, llvm::Align(16));

    FI = FunctionInfo::create(llvm::CallingConv::C, Void, {FieldAligned32HFA});
    TI->computeInfo(*FI);
    expectHFADirectArg(FI->getArgInfo(0).Info, F64, 4, llvm::Align(16));
  }

  // DarwinPCS and Win64 coerce HFAs to an array but do not set DirectAlign,
  // even when the record is overaligned.
  for (AArch64ABIKind Kind :
       {AArch64ABIKind::DarwinPCS, AArch64ABIKind::Win64}) {
    std::unique_ptr<TargetInfo> TI =
        createAArch64TargetInfo(TB, AArch64ABIOptions(Kind));
    for (const ABIType *ArgTy : {RecordAlignedHFA, RecordAligned32HFA,
                                 FieldAlignedHFA, FieldAligned32HFA}) {
      std::unique_ptr<FunctionInfo> FI =
          FunctionInfo::create(llvm::CallingConv::C, Void, {ArgTy});
      TI->computeInfo(*FI);
      EXPECT_TRUE(FI->getArgInfo(0).Info.isDirect());
      EXPECT_EQ(FI->getArgInfo(0).Info.getDirectAlign(), std::nullopt);
    }
  }
}

// Empty records and zero-size types are ignored as returns under all AArch64
// ABI kinds. Empty C arguments are ignored; Darwin also ignores empty C++
// arguments. C++ AAPCS/Win64 only ignore zero-size types.
TEST_F(AArch64TargetInfoTest, ClassifyEmptyAndZeroSizeIgnore) {
  const ABIType *EmptyC = makeRecord({}, 0, llvm::Align(1),
                                     /*UnadjustedAlign=*/llvm::Align(1));
  const ABIType *EmptyUnion =
      TB.getUnionType({}, llvm::TypeSize::getFixed(0), llvm::Align(1),
                      /*UnadjustedAlign=*/llvm::Align(1),
                      StructPacking::Default, RecordFlags::CanPassInRegisters);
  const ABIType *EmptyCXX =
      makeRecord({}, 8, llvm::Align(1), /*UnadjustedAlign=*/llvm::Align(1),
                 passableRecordFlags(/*IsCXX=*/true));
  const ABIType *ZeroArr = TB.getArrayType(I32, /*NumElements=*/0,
                                           /*SizeInBits=*/0);
  const ABIType *ZeroSizeCXX = makeRecord(
      {FieldInfo(ZeroArr, 0)}, 0, llvm::Align(1),
      /*UnadjustedAlign=*/llvm::Align(1), passableRecordFlags(/*IsCXX=*/true));
  const ABIType *NestedZeroSize = makeRecord(
      {FieldInfo(ZeroSizeCXX, 0)}, 0, llvm::Align(1),
      /*UnadjustedAlign=*/llvm::Align(1), passableRecordFlags(/*IsCXX=*/true));

  auto ClassifyArg = [&](AArch64ABIKind Kind, bool IsCXX, const ABIType *Ty) {
    AArch64ABIOptions Opts(Kind);
    Opts.IsCXX = IsCXX;
    std::unique_ptr<TargetInfo> TI = createAArch64TargetInfo(TB, Opts);
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, Void, {Ty});
    TI->computeInfo(*FI);
    return FI->getArgInfo(0).Info;
  };

  auto ClassifyReturn = [&](AArch64ABIKind Kind, bool IsCXX,
                            const ABIType *Ty) {
    AArch64ABIOptions Opts(Kind);
    Opts.IsCXX = IsCXX;
    std::unique_ptr<TargetInfo> TI = createAArch64TargetInfo(TB, Opts);
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, Ty, {});
    TI->computeInfo(*FI);
    return FI->getReturnInfo();
  };

  for (AArch64ABIKind Kind :
       {AArch64ABIKind::AAPCS, AArch64ABIKind::DarwinPCS, AArch64ABIKind::Win64,
        AArch64ABIKind::AAPCSSoft}) {
    EXPECT_TRUE(ClassifyReturn(Kind, /*IsCXX=*/false, EmptyC).isIgnore());
    EXPECT_TRUE(ClassifyReturn(Kind, /*IsCXX=*/false, EmptyUnion).isIgnore());
    EXPECT_TRUE(ClassifyReturn(Kind, /*IsCXX=*/true, EmptyCXX).isIgnore());
    EXPECT_TRUE(ClassifyReturn(Kind, /*IsCXX=*/true, ZeroSizeCXX).isIgnore());

    EXPECT_TRUE(ClassifyArg(Kind, /*IsCXX=*/false, EmptyC).isIgnore());
    EXPECT_TRUE(ClassifyArg(Kind, /*IsCXX=*/false, EmptyUnion).isIgnore());
    EXPECT_TRUE(ClassifyArg(Kind, /*IsCXX=*/true, ZeroSizeCXX).isIgnore());
    EXPECT_TRUE(ClassifyArg(Kind, /*IsCXX=*/true, NestedZeroSize).isIgnore());
  }

  // Darwin ignores empty C++ records even when they occupy a byte.
  EXPECT_TRUE(ClassifyArg(AArch64ABIKind::DarwinPCS, /*IsCXX=*/true, EmptyCXX)
                  .isIgnore());

  // An ignored empty argument does not affect classification of later args.
  {
    AArch64ABIOptions Opts(AArch64ABIKind::AAPCS);
    std::unique_ptr<TargetInfo> TI = createAArch64TargetInfo(TB, Opts);
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, Void, {EmptyC, I32});
    TI->computeInfo(*FI);
    EXPECT_TRUE(FI->getArgInfo(0).Info.isIgnore());
    expectUncoercedDirect(FI->getArgInfo(1).Info);
  }
}

} // namespace
