//===- SystemZTargetInfoTest.cpp - SystemZ ABI unit tests
//------------------===//
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
#include "llvm/Support/Allocator.h"
#include "gtest/gtest.h"

namespace {

using namespace llvm;
using ABIType = llvm::abi::Type;
using namespace llvm::abi;

class SystemZTargetInfoTest : public ::testing::Test {
protected:
  BumpPtrAllocator Alloc;
  TypeBuilder TB{Alloc};
  const ABIType *I8 = TB.getIntegerType(8, Align(1), true);
  const ABIType *I32 = TB.getIntegerType(32, Align(4), true);
  const ABIType *I64 = TB.getIntegerType(64, Align(8), true);
  const ABIType *F16 = TB.getFloatType(APFloat::IEEEhalf(), Align(2));
  const ABIType *F32 = TB.getFloatType(APFloat::IEEEsingle(), Align(4));
  const ABIType *F64 = TB.getFloatType(APFloat::IEEEdouble(), Align(8));

  const ABIType *record(ArrayRef<FieldInfo> Fields, uint64_t Size,
                        Align Alignment,
                        RecordFlags Flags = RecordFlags::CanPassInRegisters,
                        ArrayRef<FieldInfo> Bases = {}) {
    return TB.getRecordType(Fields, TypeSize::getFixed(Size), Alignment,
                            Alignment, StructPacking::Default, Bases, {},
                            Flags);
  }

  std::unique_ptr<FunctionInfo> classify(const ABIType *Ty,
                                         SystemZABIOptions Opts = {}) {
    auto FI = FunctionInfo::create(CallingConv::C, Ty, {Ty});
    createSystemZTargetInfo(TB, Opts)->computeInfo(*FI);
    return FI;
  }

  void expectIndirect(const ArgInfo &Info, Align Alignment, bool ByVal) {
    ASSERT_TRUE(Info.isIndirect());
    EXPECT_EQ(Info.getIndirectAlign(), Alignment);
    EXPECT_EQ(Info.getIndirectByVal(), ByVal);
    EXPECT_EQ(Info.getIndirectAddrSpace(), 0u);
  }

  void expectFloatCoercion(const ArgInfo &Info, const fltSemantics *Semantics) {
    ASSERT_TRUE(Info.isDirect());
    const auto *FT =
        dyn_cast_or_null<llvm::abi::FloatType>(Info.getCoerceToType());
    ASSERT_NE(FT, nullptr);
    EXPECT_EQ(FT->getSemantics(), Semantics);
  }

  void expectIntegerCoercion(const ArgInfo &Info, unsigned Width, bool NoExt) {
    if (NoExt) {
      ASSERT_TRUE(Info.isExtend());
      EXPECT_TRUE(Info.isNoExt());
    } else {
      ASSERT_TRUE(Info.isDirect());
    }
    const auto *IT =
        dyn_cast_or_null<llvm::abi::IntegerType>(Info.getCoerceToType());
    ASSERT_NE(IT, nullptr);
    EXPECT_EQ(IT->getSizeInBits().getFixedValue(), Width);
  }
};

// Cases mirror the signatures in Clang's CodeGen/SystemZ/systemz-abi*.c tests.
TEST_F(SystemZTargetInfoTest, VoidReturn) {
  auto FI = FunctionInfo::create(CallingConv::C, TB.getVoidType(), {});
  createSystemZTargetInfo(TB, {})->computeInfo(*FI);
  EXPECT_TRUE(FI->getReturnInfo().isIgnore());
}

TEST_F(SystemZTargetInfoTest, IntegerExtensionAndBitIntBoundary) {
  for (unsigned Width : {1u, 8u, 16u, 32u, 33u, 63u, 64u}) {
    for (bool Signed : {false, true}) {
      for (bool BitInt : {false, true}) {
        if (!BitInt && Width > 32 && Width < 64)
          continue;
        const ABIType *Ty = TB.getIntegerType(Width, Align(8), Signed, BitInt);
        auto FI = classify(Ty);
        for (const ArgInfo *Info :
             {&FI->getReturnInfo(), &FI->getArgInfo(0).Info}) {
          if (Width == 64) {
            EXPECT_TRUE(Info->isDirect());
            continue;
          }
          ASSERT_TRUE(Info->isExtend());
          EXPECT_EQ(Info->isSignExt(), Signed);
          EXPECT_EQ(Info->isZeroExt(), !Signed);
          EXPECT_EQ(Info->getCoerceToType(), Ty);
        }
      }
    }
  }
}

TEST_F(SystemZTargetInfoTest, ScalarsAndWideValues) {
  for (const ABIType *Ty :
       {I64, F16, F32, F64,
        static_cast<const ABIType *>(TB.getPointerType(64, Align(8)))}) {
    auto FI = classify(Ty);
    EXPECT_TRUE(FI->getReturnInfo().isDirect());
    EXPECT_TRUE(FI->getArgInfo(0).Info.isDirect());
  }
  for (const ABIType *Ty :
       {static_cast<const ABIType *>(TB.getIntegerType(128, Align(8), true)),
        static_cast<const ABIType *>(
            TB.getIntegerType(65, Align(8), true, true)),
        static_cast<const ABIType *>(
            TB.getFloatType(APFloat::IEEEquad(), Align(8))),
        static_cast<const ABIType *>(TB.getComplexType(F32, Align(4)))}) {
    auto FI = classify(Ty);
    expectIndirect(FI->getReturnInfo(), Ty->getAlignment(), true);
    expectIndirect(FI->getArgInfo(0).Info, Ty->getAlignment(), false);
  }
}

TEST_F(SystemZTargetInfoTest, MemberPointers) {
  const ABIType *DataMember = TB.getMemberPointerType(false, 64, Align(8));
  auto FI = classify(DataMember);
  EXPECT_TRUE(FI->getReturnInfo().isDirect());
  EXPECT_TRUE(FI->getArgInfo(0).Info.isDirect());

  FI = classify(TB.getAtomicType(DataMember, 64, Align(8)));
  EXPECT_TRUE(FI->getReturnInfo().isDirect());
  EXPECT_TRUE(FI->getArgInfo(0).Info.isDirect());

  FI = classify(TB.getMemberPointerType(true, 128, Align(8)));
  expectIndirect(FI->getReturnInfo(), Align(8), true);
  expectIndirect(FI->getArgInfo(0).Info, Align(8), false);
}

TEST_F(SystemZTargetInfoTest, RecordSizeAndNoExt) {
  for (unsigned Bytes : {0u, 1u, 2u, 3u, 4u, 8u, 9u}) {
    SCOPED_TRACE(Bytes);
    auto *Array = TB.getArrayType(I8, Bytes, Bytes * 8);
    const ABIType *Ty = record({FieldInfo(Array)}, Bytes * 8, Align(1));
    auto FI = classify(Ty);
    expectIndirect(FI->getReturnInfo(), Align(1), true);
    if (Bytes == 1 || Bytes == 2 || Bytes == 4 || Bytes == 8)
      expectIntegerCoercion(FI->getArgInfo(0).Info, Bytes * 8, Bytes <= 4);
    else
      expectIndirect(FI->getArgInfo(0).Info, Align(1), false);
  }
}

TEST_F(SystemZTargetInfoTest, FloatRecordsAndSoftFloat) {
  for (const ABIType *Elt : {F16, F32, F64}) {
    unsigned Size = Elt->getABISizeInBits();
    const ABIType *Ty = record({FieldInfo(Elt)}, Size, Elt->getAlignment());
    auto FI = classify(Ty);
    ASSERT_TRUE(FI->getArgInfo(0).Info.isDirect());
    const auto *FT = dyn_cast<llvm::abi::FloatType>(
        FI->getArgInfo(0).Info.getCoerceToType());
    ASSERT_NE(FT, nullptr);
    EXPECT_EQ(FT->getSemantics(),
              cast<llvm::abi::FloatType>(Elt)->getSemantics());
    expectIndirect(FI->getReturnInfo(), Elt->getAlignment(), true);

    SystemZABIOptions Soft;
    Soft.IsSoftFloatABI = true;
    auto SoftFI = classify(Ty, Soft);
    expectIntegerCoercion(SoftFI->getArgInfo(0).Info, Size, Size <= 32);
  }
}

TEST_F(SystemZTargetInfoTest, NestedAndPaddedFloatRecords) {
  for (const ABIType *Elt : {F16, F32}) {
    const ABIType *Inner =
        record({FieldInfo(Elt)}, Elt->getABISizeInBits(), Elt->getAlignment());
    for (unsigned Size : {32u, 64u}) {
      const ABIType *Ty = record({FieldInfo(Inner)}, Size, Align(Size / 8));
      auto FI = classify(Ty);
      ASSERT_TRUE(FI->getArgInfo(0).Info.isDirect());
      const auto *FT = dyn_cast<llvm::abi::FloatType>(
          FI->getArgInfo(0).Info.getCoerceToType());
      ASSERT_NE(FT, nullptr);
      EXPECT_EQ(FT->getSemantics(),
                Size == 32 ? &APFloat::IEEEsingle() : &APFloat::IEEEdouble());
    }
  }
}

TEST_F(SystemZTargetInfoTest, ArraysUnionsAndEmptyFieldsDoNotReduceToFloat) {
  const ABIType *Array = TB.getArrayType(F32, 1, 32);
  const ABIType *Empty = record({}, 0, Align(1));
  for (const ABIType *Ty :
       {record({FieldInfo(Array)}, 32, Align(4)),
        record({FieldInfo(F32)}, 32, Align(4),
               RecordFlags::CanPassInRegisters | RecordFlags::IsUnion),
        record({FieldInfo(Empty), FieldInfo(F32)}, 32, Align(4)),
        record({FieldInfo(I32, 0, true, 1, true), FieldInfo(F32)}, 64,
               Align(4))}) {
    auto FI = classify(Ty);
    unsigned Size = Ty->getABISizeInBits();
    expectIntegerCoercion(FI->getArgInfo(0).Info, Size, Size <= 32);
  }
}

TEST_F(SystemZTargetInfoTest, FlexibleArraysAndNontrivialRecords) {
  const ABIType *Flexible = record({FieldInfo(I64)}, 64, Align(8),
                                   RecordFlags::CanPassInRegisters |
                                       RecordFlags::HasFlexibleArrayMember);
  auto FI = classify(Flexible);
  expectIndirect(FI->getArgInfo(0).Info, Align(8), false);

  const ABIType *Nontrivial =
      record({FieldInfo(F32)}, 32, Align(4), RecordFlags::IsCXXRecord);
  FI = classify(Nontrivial);
  expectIndirect(FI->getReturnInfo(), Align(4), false);
  expectIndirect(FI->getArgInfo(0).Info, Align(4), false);
}

TEST_F(SystemZTargetInfoTest, TransparentUnionUsesFirstField) {
  const ABIType *Ty =
      record({FieldInfo(I32), FieldInfo(F32)}, 32, Align(4),
             RecordFlags::CanPassInRegisters | RecordFlags::IsUnion |
                 RecordFlags::IsTransparent);
  auto FI = classify(Ty);
  ASSERT_TRUE(FI->getArgInfo(0).Info.isExtend());
  EXPECT_TRUE(FI->getArgInfo(0).Info.isSignExt());
  EXPECT_EQ(FI->getArgInfo(0).Info.getCoerceToType(), I32);
  expectIndirect(FI->getReturnInfo(), Align(4), true);
}

TEST_F(SystemZTargetInfoTest, VectorFacilityAndRecordPadding) {
  for (unsigned Bytes : {1u, 8u, 16u, 32u}) {
    const ABIType *Vec =
        TB.getVectorType(I8, ElementCount::getFixed(Bytes), Align(8));
    for (bool HasVector : {false, true}) {
      SystemZABIOptions Opts;
      Opts.HasVector = HasVector;
      auto FI = classify(Vec, Opts);
      if (HasVector && Bytes <= 16) {
        EXPECT_TRUE(FI->getReturnInfo().isDirect());
        EXPECT_TRUE(FI->getArgInfo(0).Info.isDirect());
        EXPECT_EQ(FI->getArgInfo(0).Info.getCoerceToType(), Vec);
      } else {
        expectIndirect(FI->getReturnInfo(), Align(8), true);
        expectIndirect(FI->getArgInfo(0).Info, Align(8), false);
      }
    }
  }

  const ABIType *Vec =
      TB.getVectorType(I8, ElementCount::getFixed(8), Align(8));
  SystemZABIOptions Opts;
  Opts.HasVector = true;
  auto FI = classify(record({FieldInfo(Vec)}, 64, Align(8)), Opts);
  ASSERT_TRUE(FI->getArgInfo(0).Info.isDirect());
  EXPECT_EQ(FI->getArgInfo(0).Info.getCoerceToType(), Vec);
  expectIndirect(FI->getReturnInfo(), Align(8), true);
  FI = classify(record({FieldInfo(Vec)}, 128, Align(8)), Opts);
  expectIndirect(FI->getArgInfo(0).Info, Align(8), false);
}

TEST_F(SystemZTargetInfoTest, VariadicClassification) {
  const ABIType *Vec =
      TB.getVectorType(I8, ElementCount::getFixed(16), Align(8));
  auto FI =
      FunctionInfo::create(CallingConv::C, I32, {I32, Vec}, RequiredArgs(1));
  SystemZABIOptions Opts;
  Opts.HasVector = true;
  createSystemZTargetInfo(TB, Opts)->computeInfo(*FI);
  EXPECT_TRUE(FI->getArgInfo(0).Info.isExtend());
  EXPECT_TRUE(FI->getArgInfo(1).Info.isDirect());
}

TEST_F(SystemZTargetInfoTest, XPLINKClassification) {
  SystemZABIOptions Opts;
  Opts.Kind = SystemZABIKind::XPLINK;

  // Scalar integers are extended.
  auto FI = classify(I32, Opts);
  EXPECT_TRUE(FI->getReturnInfo().isExtend());
  EXPECT_TRUE(FI->getArgInfo(0).Info.isExtend());

  // Floats are direct.
  FI = classify(F64, Opts);
  EXPECT_TRUE(FI->getReturnInfo().isDirect());
  EXPECT_TRUE(FI->getArgInfo(0).Info.isDirect());

  // Aggregate <= 24 bytes (3 GPRs) is coerced to array of i64 on return.
  const ABIType *SmallRec =
      record({FieldInfo(I32), FieldInfo(I32)}, 64, Align(8));
  FI = classify(SmallRec, Opts);
  EXPECT_TRUE(FI->getReturnInfo().isDirect());
  ASSERT_TRUE(FI->getReturnInfo().getCoerceToType());
  EXPECT_TRUE(FI->getReturnInfo().getCoerceToType()->isArray());
  EXPECT_TRUE(FI->getArgInfo(0).Info.isDirect());
}

TEST_F(SystemZTargetInfoTest, SingleElementSkipsEmptyBaseAndMember) {
  const ABIType *Empty = record({}, 8, Align(1));

  // An empty base does not stop the struct from being a single float.
  const ABIType *WithBase =
      record({FieldInfo(F32)}, 32, Align(4),
             RecordFlags::CanPassInRegisters | RecordFlags::IsCXXRecord,
             {FieldInfo(Empty)});
  auto FI = classify(WithBase);
  ASSERT_TRUE(FI->getArgInfo(0).Info.isDirect());
  expectFloatCoercion(FI->getArgInfo(0).Info, &APFloat::IEEEsingle());

  // An empty [[no_unique_address]] member is ignored.
  const ABIType *WithEmptyMember =
      record({FieldInfo(Empty, 0, false, 0, false, true), FieldInfo(F32)}, 32,
             Align(4));
  FI = classify(WithEmptyMember);
  ASSERT_TRUE(FI->getArgInfo(0).Info.isDirect());
  expectFloatCoercion(FI->getArgInfo(0).Info, &APFloat::IEEEsingle());

  // A non-empty base plus a field is not a single element.
  const ABIType *TwoElements =
      record({FieldInfo(F32)}, 64, Align(4),
             RecordFlags::CanPassInRegisters | RecordFlags::IsCXXRecord,
             {FieldInfo(record({FieldInfo(F32)}, 32, Align(4)))});
  FI = classify(TwoElements);
  expectIntegerCoercion(FI->getArgInfo(0).Info, 64, false);
}

TEST_F(SystemZTargetInfoTest, XPLINKComplexLikeStructs) {
  SystemZABIOptions Opts;
  Opts.Kind = SystemZABIKind::XPLINK;
  const ABIType *Empty = record({}, 8, Align(1));
  auto IsCoercedToRecord = [&](const ABIType *Ty) {
    auto FI = classify(Ty, Opts);
    const ArgInfo &Info = FI->getArgInfo(0).Info;
    return Info.isDirect() &&
           isa_and_nonnull<llvm::abi::RecordType>(Info.getCoerceToType());
  };

  // Two floats are complex-like.
  EXPECT_TRUE(IsCoercedToRecord(
      record({FieldInfo(F32), FieldInfo(F32)}, 64, Align(4))));

  // An empty base does not stop it from being complex-like.
  EXPECT_TRUE(IsCoercedToRecord(
      record({FieldInfo(F32), FieldInfo(F32)}, 64, Align(4),
             RecordFlags::CanPassInRegisters | RecordFlags::IsCXXRecord,
             {FieldInfo(Empty)})));

  // A non-empty base does.
  EXPECT_FALSE(IsCoercedToRecord(
      record({FieldInfo(F32), FieldInfo(F32)}, 96, Align(4),
             RecordFlags::CanPassInRegisters | RecordFlags::IsCXXRecord,
             {FieldInfo(record({FieldInfo(F32)}, 32, Align(4)))})));

  // Half precision floats are not complex-like.
  EXPECT_FALSE(IsCoercedToRecord(
      record({FieldInfo(F16), FieldInfo(F16)}, 32, Align(2))));
}

TEST_F(SystemZTargetInfoTest, XPLINKExtendsLongButNotLongLong) {
  SystemZABIOptions Opts;
  Opts.Kind = SystemZABIKind::XPLINK;
  const ABIType *Long = TB.getIntegerType(64, Align(8), /*Signed=*/true,
                                          /*IsBitInt=*/false, /*IsLong=*/true);

  // long is extended, long long is passed directly.
  auto FI = classify(Long, Opts);
  EXPECT_TRUE(FI->getArgInfo(0).Info.isExtend());
  EXPECT_TRUE(FI->getReturnInfo().isExtend());
  FI = classify(I64, Opts);
  EXPECT_TRUE(FI->getArgInfo(0).Info.isDirect());
  EXPECT_TRUE(FI->getReturnInfo().isDirect());

  // The ELF ABI extends neither.
  FI = classify(Long);
  EXPECT_TRUE(FI->getArgInfo(0).Info.isDirect());
}

TEST_F(SystemZTargetInfoTest, XPLINKAggregateReturnIsInReg) {
  SystemZABIOptions Opts;
  Opts.Kind = SystemZABIKind::XPLINK;
  const ABIType *Rec = record({FieldInfo(I32), FieldInfo(I32)}, 64, Align(8));
  auto FI = classify(Rec, Opts);
  ASSERT_TRUE(FI->getReturnInfo().isDirect());
  EXPECT_TRUE(FI->getReturnInfo().getInReg());
  // Arguments are not marked inreg.
  EXPECT_FALSE(FI->getArgInfo(0).Info.getInReg());
}

TEST_F(SystemZTargetInfoTest, XPLINKSingleElementUnions) {
  SystemZABIOptions Opts;
  Opts.Kind = SystemZABIKind::XPLINK;
  const ABIType *F32Again = TB.getFloatType(APFloat::IEEEsingle(), Align(4));

  // union { float a, b; } behaves like a float, so a struct of two of them is
  // complex-like and is passed as a struct of two floats.
  const ABIType *SameUnion =
      record({FieldInfo(F32), FieldInfo(F32Again)}, 32, Align(4),
             RecordFlags::CanPassInRegisters | RecordFlags::IsUnion);
  const ABIType *Pair =
      record({FieldInfo(SameUnion), FieldInfo(SameUnion)}, 64, Align(4));
  auto FI = classify(Pair, Opts);
  const ArgInfo &Info = FI->getArgInfo(0).Info;
  ASSERT_TRUE(Info.isDirect());
  EXPECT_TRUE(isa<llvm::abi::RecordType>(Info.getCoerceToType()));

  // union { float a; double b; } has two different types, so it stays a union.
  const ABIType *MixedUnion =
      record({FieldInfo(F32), FieldInfo(F64)}, 64, Align(8),
             RecordFlags::CanPassInRegisters | RecordFlags::IsUnion);
  const ABIType *MixedPair =
      record({FieldInfo(MixedUnion), FieldInfo(MixedUnion)}, 128, Align(8));
  FI = classify(MixedPair, Opts);
  EXPECT_FALSE(isa_and_nonnull<llvm::abi::RecordType>(
      FI->getArgInfo(0).Info.getCoerceToType()));
}

} // namespace
