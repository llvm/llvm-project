//===- TargetInfoTest.cpp - shared ABI TargetInfo unit tests --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for target-independent helpers on TargetInfo and the shared default
// classification in DefaultTargetInfo. These are exercised through a minimal
// concrete target so any new shared default lands here rather than in a
// per-target test file.
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/TargetInfo.h"
#include "llvm/ABI/DefaultTargetInfo.h"
#include "llvm/ABI/FunctionInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Allocator.h"
#include "gtest/gtest.h"

namespace {

// RecordFlags' bitmask operators are declared in namespace llvm, so combining
// two of them needs that namespace visible.
using namespace llvm;

using ABIType = llvm::abi::Type;
using llvm::abi::ArgInfo;
using llvm::abi::DefaultTargetInfo;
using llvm::abi::FieldInfo;
using llvm::abi::FunctionInfo;
using llvm::abi::RecordFlags;
using llvm::abi::StructPacking;
using llvm::abi::TargetInfo;
using llvm::abi::TypeBuilder;

// A minimal concrete target that inherits the shared default classifiers so
// they can be exercised directly, independent of any real target. Only
// getABICompatInfo() is left to supply.
class TestTargetInfo : public DefaultTargetInfo {
public:
  explicit TestTargetInfo(TypeBuilder &Builder) : DefaultTargetInfo(Builder) {}
  const llvm::abi::ABICompatInfo &getABICompatInfo() const override {
    return Compat;
  }
  using TargetInfo::getNaturalAlignIndirect;
  using TargetInfo::isSingleElementStruct;

private:
  llvm::abi::ABICompatInfo Compat;
};

// A target whose stack/alloca lives in a non-zero address space, so indirect
// arguments must be allocated there rather than in AS 0.
class AllocaAS5TargetInfo : public TargetInfo {
public:
  explicit AllocaAS5TargetInfo(TypeBuilder &Builder) : TargetInfo(Builder) {}
  void computeInfo(FunctionInfo &) const override {}
  const llvm::abi::ABICompatInfo &getABICompatInfo() const override {
    return Compat;
  }
  unsigned getAllocaAddrSpace() const override { return 5; }
  using TargetInfo::getNaturalAlignIndirect;

private:
  llvm::abi::ABICompatInfo Compat;
};

// Exposes convertTypeForMem, which is protected on TargetInfo.
class MemTargetInfo : public TestTargetInfo {
public:
  using TargetInfo::convertTypeForMem;
  using TestTargetInfo::TestTargetInfo;
};

// A target lacking a 128-bit integer type, so the _BitInt register threshold
// is the width of `long long`.
class NoInt128TargetInfo : public TestTargetInfo {
public:
  using TestTargetInfo::TestTargetInfo;
  bool hasInt128Type() const override { return false; }
};

// A default-classifying target whose alloca space is non-zero, so indirect
// classifications must carry that address space.
class AllocaAS5DefaultTargetInfo : public TestTargetInfo {
public:
  using TestTargetInfo::TestTargetInfo;
  unsigned getAllocaAddrSpace() const override { return 5; }
};

class TargetInfoTest : public ::testing::Test {
protected:
  llvm::BumpPtrAllocator Alloc;
  TypeBuilder TB;
  const ABIType *I16;
  const ABIType *I32;
  const ABIType *F32;
  const ABIType *Void;
  /// An empty class: a record with no fields, one byte wide.
  const ABIType *Empty;
  /// A _BitInt wider than 128 bits, which cannot be passed in registers.
  const ABIType *WideBitInt;
  /// A _BitInt between `long long` (64) and 128 bits wide.
  const ABIType *MidBitInt;

  TargetInfoTest()
      : TB(Alloc), I16(TB.getIntegerType(16, llvm::Align(2), /*Signed=*/true)),
        I32(TB.getIntegerType(32, llvm::Align(4), /*Signed=*/true)),
        F32(TB.getFloatType(llvm::APFloat::IEEEsingle(), llvm::Align(4))),
        Void(TB.getVoidType()),
        Empty(TB.getRecordType({}, llvm::TypeSize::getFixed(8), llvm::Align(1),
                               /*UnadjustedAlign=*/llvm::Align(1),
                               StructPacking::Default, {}, {},
                               RecordFlags::CanPassInRegisters)),
        WideBitInt(TB.getIntegerType(129, llvm::Align(8), /*Signed=*/true,
                                     /*IsBitInt=*/true)),
        MidBitInt(TB.getIntegerType(100, llvm::Align(8), /*Signed=*/true,
                                    /*IsBitInt=*/true)) {}

  /// A record with a single int field, passable in registers.
  const ABIType *recordInReg() {
    return TB.getRecordType({FieldInfo(I32, 0)}, llvm::TypeSize::getFixed(32),
                            llvm::Align(4), /*UnadjustedAlign=*/llvm::Align(4),
                            StructPacking::Default, {}, {},
                            RecordFlags::CanPassInRegisters);
  }

  /// The same record marked as unable to pass in registers, e.g. a non-trivial
  /// C++ type. This takes the RAA_Indirect path.
  const ABIType *recordInMemory() {
    return TB.getRecordType({FieldInfo(I32, 0)}, llvm::TypeSize::getFixed(32),
                            llvm::Align(4), /*UnadjustedAlign=*/llvm::Align(4),
                            StructPacking::Default, {}, {}, RecordFlags::None);
  }

  ArgInfo classifyArg(const ABIType *Ty) {
    TestTargetInfo TI(TB);
    return TI.classifyArgumentType(Ty);
  }

  ArgInfo classifyRet(const ABIType *Ty) {
    TestTargetInfo TI(TB);
    return TI.classifyReturnType(Ty);
  }

  ArgInfo classifyArgNoInt128(const ABIType *Ty) {
    NoInt128TargetInfo TI(TB);
    return TI.classifyArgumentType(Ty);
  }

  const ABIType *recordOf(llvm::ArrayRef<FieldInfo> Fields, uint64_t SizeInBits,
                          llvm::Align Alignment) {
    return TB.getRecordType(Fields, llvm::TypeSize::getFixed(SizeInBits),
                            Alignment, Alignment, StructPacking::Default, {},
                            {}, RecordFlags::CanPassInRegisters);
  }

  /// The single-element reduction of \p Ty, or null if it is not a
  /// single-element struct. Exercises the shared TargetInfo helper directly.
  const ABIType *singleElement(const ABIType *Ty) {
    TestTargetInfo TI(TB);
    return TI.isSingleElementStruct(Ty);
  }
};

// --- Argument classification -------------------------------------------------

// A word-sized integer is passed directly.
TEST_F(TargetInfoTest, DefaultArgIntIsDirect) {
  ArgInfo Info = classifyArg(I32);
  EXPECT_TRUE(Info.isDirect());
}

// A sub-word integer is promoted.
TEST_F(TargetInfoTest, DefaultArgSmallIntIsExtended) {
  EXPECT_TRUE(classifyArg(I16).isExtend());
}

// A record that fits in registers is passed indirectly by value.
TEST_F(TargetInfoTest, DefaultArgRecordInRegIsIndirectByVal) {
  ArgInfo Info = classifyArg(recordInReg());
  ASSERT_TRUE(Info.isIndirect());
  EXPECT_TRUE(Info.getIndirectByVal());
}

// A record that cannot pass in registers goes indirect without ByVal.
TEST_F(TargetInfoTest, DefaultArgRecordInMemoryIsIndirectNoByVal) {
  ArgInfo Info = classifyArg(recordInMemory());
  ASSERT_TRUE(Info.isIndirect());
  EXPECT_FALSE(Info.getIndirectByVal());
}

// The default classifier routes indirect args through the target's alloca
// space.
TEST_F(TargetInfoTest, DefaultArgIndirectUsesAllocaAddrSpace) {
  AllocaAS5DefaultTargetInfo TI(TB);
  ArgInfo Info = TI.classifyArgumentType(recordInMemory());
  ASSERT_TRUE(Info.isIndirect());
  EXPECT_EQ(Info.getIndirectAddrSpace(), 5u);
}

// A _BitInt wider than 128 bits is passed indirectly.
TEST_F(TargetInfoTest, DefaultArgWideBitIntIsIndirect) {
  EXPECT_TRUE(classifyArg(WideBitInt).isIndirect());
}

// With a 128-bit integer type, a _BitInt no wider than 128 stays direct.
TEST_F(TargetInfoTest, DefaultArgMidBitIntIsDirectWithInt128) {
  EXPECT_TRUE(classifyArg(MidBitInt).isDirect());
}

// Without a 128-bit integer type the threshold drops to long long (64), so the
// same _BitInt is passed indirectly.
TEST_F(TargetInfoTest, DefaultArgMidBitIntIsIndirectWithoutInt128) {
  EXPECT_TRUE(classifyArgNoInt128(MidBitInt).isIndirect());
}

// A transparent union is classified as its first field, so a union of one int
// is passed directly rather than as an aggregate.
TEST_F(TargetInfoTest, DefaultArgTransparentUnionUsesFirstField) {
  const ABIType *U = TB.getUnionType(
      {FieldInfo(I32, 0)}, llvm::TypeSize::getFixed(32), llvm::Align(4),
      /*UnadjustedAlign=*/llvm::Align(4), StructPacking::Default,
      RecordFlags::IsTransparent | RecordFlags::CanPassInRegisters);
  EXPECT_TRUE(classifyArg(U).isDirect());
}

// --- Return classification ---------------------------------------------------

// Void returns are ignored.
TEST_F(TargetInfoTest, DefaultReturnVoidIsIgnored) {
  EXPECT_TRUE(classifyRet(Void).isIgnore());
}

// A word-sized integer is returned directly.
TEST_F(TargetInfoTest, DefaultReturnIntIsDirect) {
  EXPECT_TRUE(classifyRet(I32).isDirect());
}

// A sub-word integer is promoted on return.
TEST_F(TargetInfoTest, DefaultReturnSmallIntIsExtended) {
  EXPECT_TRUE(classifyRet(I16).isExtend());
}

// An aggregate is returned indirectly, with ByVal=true.
TEST_F(TargetInfoTest, DefaultReturnRecordIsIndirect) {
  ArgInfo Info = classifyRet(recordInReg());
  ASSERT_TRUE(Info.isIndirect());
  EXPECT_TRUE(Info.getIndirectByVal());
}

// A _BitInt wider than 128 bits is returned indirectly.
TEST_F(TargetInfoTest, DefaultReturnWideBitIntIsIndirect) {
  ArgInfo Info = classifyRet(WideBitInt);
  ASSERT_TRUE(Info.isIndirect());
  EXPECT_TRUE(Info.getIndirectByVal());
}

// --- Single-element struct reduction -----------------------------------------

// The shared single-element-struct reduction, used by getByteVectorType and by
// other targets, looks through single-element wrappers to the scalar element.

// A struct of one float reduces to its single float element.
TEST_F(TargetInfoTest, SingleElementStructSingleScalarFieldReduces) {
  const ABIType *S = recordOf({FieldInfo(F32, 0)}, 32, llvm::Align(4));
  EXPECT_EQ(singleElement(S), F32);
}

// A single-element array is transparent, so a struct of one float[1] reduces
// to the element type.
TEST_F(TargetInfoTest, SingleElementStructSingleElementArrayReduces) {
  const ABIType *Arr =
      TB.getArrayType(F32, /*NumElements=*/1, /*SizeInBits=*/32);
  const ABIType *S = recordOf({FieldInfo(Arr, 0)}, 32, llvm::Align(4));
  EXPECT_EQ(singleElement(S), F32);
}

// A multi-element array is an aggregate that does not itself reduce, so a
// struct of one short[2] is NOT a single-element struct.
TEST_F(TargetInfoTest, SingleElementStructMultiElementArrayDoesNotReduce) {
  const ABIType *Arr =
      TB.getArrayType(I16, /*NumElements=*/2, /*SizeInBits=*/32);
  const ABIType *S = recordOf({FieldInfo(Arr, 0)}, 32, llvm::Align(2));
  EXPECT_EQ(singleElement(S), nullptr);
}

// Two data members: not a single-element struct.
TEST_F(TargetInfoTest, SingleElementStructTwoFieldsDoNotReduce) {
  const ABIType *S =
      recordOf({FieldInfo(I32, 0), FieldInfo(I32, 32)}, 64, llvm::Align(4));
  EXPECT_EQ(singleElement(S), nullptr);
}

// A single member that leaves tail padding does not cover the record, so it
// does not reduce.
TEST_F(TargetInfoTest, SingleElementStructTailPaddingDoesNotReduce) {
  const ABIType *S = recordOf({FieldInfo(I32, 0)}, 64, llvm::Align(8));
  EXPECT_EQ(singleElement(S), nullptr);
}

// An empty member supplies no data and is skipped, so a struct of an empty
// member plus an int reduces to the int.
TEST_F(TargetInfoTest, SingleElementStructEmptyMemberIsSkipped) {
  const ABIType *S =
      recordOf({FieldInfo(Empty, 0), FieldInfo(I32, 0)}, 32, llvm::Align(4));
  EXPECT_EQ(singleElement(S), I32);
}

// A nested single-element struct reduces to the inner scalar.
TEST_F(TargetInfoTest, SingleElementStructNestedSingleElementReduces) {
  const ABIType *Inner = recordOf({FieldInfo(F32, 0)}, 32, llvm::Align(4));
  const ABIType *Outer = recordOf({FieldInfo(Inner, 0)}, 32, llvm::Align(4));
  EXPECT_EQ(singleElement(Outer), F32);
}

// A non-record type is never a single-element struct.
TEST_F(TargetInfoTest, SingleElementStructNonRecordReturnsNull) {
  EXPECT_EQ(singleElement(I32), nullptr);
}

// Indirect args land in the target's alloca space.
TEST_F(TargetInfoTest, NaturalAlignIndirectUsesAllocaAddrSpace) {
  AllocaAS5TargetInfo TI(TB);
  ArgInfo AI = TI.getNaturalAlignIndirect(I32, TI.getAllocaAddrSpace());
  EXPECT_TRUE(AI.isIndirect());
  EXPECT_EQ(AI.getIndirectAddrSpace(), 5u);
  EXPECT_TRUE(AI.getIndirectByVal());
}

// The default alloca space is 0, matching classic's DefaultABIInfo.
TEST_F(TargetInfoTest, NaturalAlignIndirectDefaultsToZeroAddrSpace) {
  TestTargetInfo TI(TB);
  ArgInfo AI = TI.getNaturalAlignIndirect(I32, TI.getAllocaAddrSpace());
  EXPECT_TRUE(AI.isIndirect());
  EXPECT_EQ(AI.getIndirectAddrSpace(), 0u);
}

// --- convertTypeForMem -------------------------------------------------------

static const llvm::abi::RecordType *asRecord(const ABIType *Ty) {
  return llvm::dyn_cast<llvm::abi::RecordType>(Ty);
}

static const llvm::abi::ArrayType *asI8Array(const ABIType *Ty) {
  const auto *AT = llvm::dyn_cast<llvm::abi::ArrayType>(Ty);
  if (!AT || AT->getElementType()->getSizeInBits().getFixedValue() != 8)
    return nullptr;
  return AT;
}

// A naturally aligned struct keeps the gap implicit and is not packed.
TEST_F(TargetInfoTest, ConvertTypeForMemNaturalStructIsUnpacked) {
  MemTargetInfo TI(TB);
  const ABIType *I8 = TB.getIntegerType(8, llvm::Align(1), /*Signed=*/true);
  const ABIType *S =
      recordOf({FieldInfo(I8, 0), FieldInfo(I32, 32)}, 64, llvm::Align(4));
  const llvm::abi::RecordType *Mem = asRecord(TI.convertTypeForMem(S));
  ASSERT_NE(Mem, nullptr);
  EXPECT_EQ(Mem->getPacking(), StructPacking::Default);
  ASSERT_EQ(Mem->getNumFields(), 2u);
  EXPECT_EQ(Mem->getFields()[0].FieldType, I8);
  EXPECT_EQ(Mem->getFields()[1].FieldType, I32);
}

// The record has tail padding, but rounding the members up by the most aligned
// member reaches the record size, so a non-packed struct produces that padding
// without an array for it.
TEST_F(TargetInfoTest, ConvertTypeForMemTailCoveredByMemberAlignIsImplicit) {
  MemTargetInfo TI(TB);
  const ABIType *I8 = TB.getIntegerType(8, llvm::Align(1), /*Signed=*/true);
  // int at 0, char at byte 4, size 8 from the int's alignment.
  const ABIType *S =
      recordOf({FieldInfo(I32, 0), FieldInfo(I8, 32)}, 64, llvm::Align(4));
  const llvm::abi::RecordType *Mem = asRecord(TI.convertTypeForMem(S));
  ASSERT_NE(Mem, nullptr);
  EXPECT_EQ(Mem->getPacking(), StructPacking::Default);
  ASSERT_EQ(Mem->getNumFields(), 2u);
  EXPECT_EQ(Mem->getFields()[0].FieldType, I32);
  EXPECT_EQ(Mem->getFields()[1].FieldType, I8);
}

// alignas on the record does not raise the converted member alignment, so the
// tail is an explicit i8 array. The record stays unpacked.
TEST_F(TargetInfoTest, ConvertTypeForMemOveralignedTailIsExplicit) {
  MemTargetInfo TI(TB);
  const ABIType *S = recordOf({FieldInfo(I32, 0)}, 128, llvm::Align(16));
  const llvm::abi::RecordType *Mem = asRecord(TI.convertTypeForMem(S));
  ASSERT_NE(Mem, nullptr);
  EXPECT_EQ(Mem->getPacking(), StructPacking::Default);
  ASSERT_EQ(Mem->getNumFields(), 2u);
  EXPECT_EQ(Mem->getFields()[0].FieldType, I32);
  const llvm::abi::ArrayType *Tail = asI8Array(Mem->getFields()[1].FieldType);
  ASSERT_NE(Tail, nullptr);
  EXPECT_EQ(Tail->getNumElements(), 12u);
}

// An over-aligned member widens the gap before it and the tail. Both are
// explicit, and the record stays unpacked.
TEST_F(TargetInfoTest, ConvertTypeForMemOveralignedMemberPadsGapAndTail) {
  MemTargetInfo TI(TB);
  const ABIType *I8 = TB.getIntegerType(8, llvm::Align(1), /*Signed=*/true);
  // char at 0, int at byte 16, size 32 from the member's alignment.
  const ABIType *S =
      recordOf({FieldInfo(I8, 0), FieldInfo(I32, 128)}, 256, llvm::Align(16));
  const llvm::abi::RecordType *Mem = asRecord(TI.convertTypeForMem(S));
  ASSERT_NE(Mem, nullptr);
  EXPECT_EQ(Mem->getPacking(), StructPacking::Default);
  ASSERT_EQ(Mem->getNumFields(), 4u);
  EXPECT_EQ(Mem->getFields()[0].FieldType, I8);
  const llvm::abi::ArrayType *Gap = asI8Array(Mem->getFields()[1].FieldType);
  ASSERT_NE(Gap, nullptr);
  EXPECT_EQ(Gap->getNumElements(), 15u);
  EXPECT_EQ(Mem->getFields()[2].FieldType, I32);
  const llvm::abi::ArrayType *Tail = asI8Array(Mem->getFields()[3].FieldType);
  ASSERT_NE(Tail, nullptr);
  EXPECT_EQ(Tail->getNumElements(), 12u);
}

// A misaligned field makes the record packed. A later field that sits at its
// natural alignment still gets an explicit gap, and so does the tail.
TEST_F(TargetInfoTest, ConvertTypeForMemPackedRecordKeepsNaturalGap) {
  MemTargetInfo TI(TB);
  const ABIType *I8 = TB.getIntegerType(8, llvm::Align(1), /*Signed=*/true);
  // char, misaligned int at byte 1, naturally aligned int at byte 8, char at
  // byte 12. Size 16.
  const ABIType *S = recordOf({FieldInfo(I8, 0), FieldInfo(I32, 8),
                               FieldInfo(I32, 64), FieldInfo(I8, 96)},
                              128, llvm::Align(4));
  const llvm::abi::RecordType *Mem = asRecord(TI.convertTypeForMem(S));
  ASSERT_NE(Mem, nullptr);
  EXPECT_EQ(Mem->getPacking(), StructPacking::Packed);
  ASSERT_EQ(Mem->getNumFields(), 6u);
  EXPECT_EQ(Mem->getFields()[0].FieldType, I8);
  EXPECT_EQ(Mem->getFields()[1].FieldType, I32);
  const llvm::abi::ArrayType *Gap = asI8Array(Mem->getFields()[2].FieldType);
  ASSERT_NE(Gap, nullptr);
  EXPECT_EQ(Gap->getNumElements(), 3u);
  EXPECT_EQ(Mem->getFields()[3].FieldType, I32);
  EXPECT_EQ(Mem->getFields()[4].FieldType, I8);
  const llvm::abi::ArrayType *Tail = asI8Array(Mem->getFields()[5].FieldType);
  ASSERT_NE(Tail, nullptr);
  EXPECT_EQ(Tail->getNumElements(), 3u);
}

// A packed member has converted alignment 1, so placing it at byte 1 does not
// pack the outer record, and the naturally aligned field after it has no
// explicit padding array.
TEST_F(TargetInfoTest, ConvertTypeForMemPackedMemberHasAlignOne) {
  MemTargetInfo TI(TB);
  const ABIType *I8 = TB.getIntegerType(8, llvm::Align(1), /*Signed=*/true);
  const ABIType *Inner =
      recordOf({FieldInfo(I8, 0), FieldInfo(I32, 8)}, 40, llvm::Align(1));
  // char, packed inner at byte 1, int at byte 8. Size 12.
  const ABIType *Outer =
      recordOf({FieldInfo(I8, 0), FieldInfo(Inner, 8), FieldInfo(I32, 64)}, 96,
               llvm::Align(4));
  const llvm::abi::RecordType *Mem = asRecord(TI.convertTypeForMem(Outer));
  ASSERT_NE(Mem, nullptr);
  EXPECT_EQ(Mem->getPacking(), StructPacking::Default);
  ASSERT_EQ(Mem->getNumFields(), 3u);
  EXPECT_EQ(Mem->getFields()[0].FieldType, I8);
  const llvm::abi::RecordType *InnerMem =
      asRecord(Mem->getFields()[1].FieldType);
  ASSERT_NE(InnerMem, nullptr);
  EXPECT_EQ(InnerMem->getPacking(), StructPacking::Packed);
  EXPECT_EQ(Mem->getFields()[2].FieldType, I32);
}

} // namespace
