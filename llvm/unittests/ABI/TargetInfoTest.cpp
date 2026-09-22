//===- TargetInfoTest.cpp - shared ABI TargetInfo unit tests --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for target-independent helpers on the TargetInfo base class. These are
// exercised through a minimal concrete target so any new shared default lands
// here rather than in a per-target test file.
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/TargetInfo.h"
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
using llvm::abi::FieldInfo;
using llvm::abi::FunctionInfo;
using llvm::abi::RecordFlags;
using llvm::abi::StructPacking;
using llvm::abi::TargetInfo;
using llvm::abi::TypeBuilder;

// A minimal concrete target that re-exposes the shared, protected helpers so
// they can be exercised directly, independent of any real target.
class TestTargetInfo : public TargetInfo {
public:
  explicit TestTargetInfo(TypeBuilder &Builder) : TargetInfo(Builder) {}
  void computeInfo(FunctionInfo &) const override {}
  const llvm::abi::ABICompatInfo &getABICompatInfo() const override {
    return Compat;
  }
  using TargetInfo::isSingleElementStruct;

private:
  llvm::abi::ABICompatInfo Compat;
};

class TargetInfoTest : public ::testing::Test {
protected:
  llvm::BumpPtrAllocator Alloc;
  TypeBuilder TB;
  const ABIType *I16;
  const ABIType *I32;
  const ABIType *F32;
  /// An empty class: a record with no fields, one byte wide.
  const ABIType *Empty;

  TargetInfoTest()
      : TB(Alloc), I16(TB.getIntegerType(16, llvm::Align(2), /*Signed=*/true)),
        I32(TB.getIntegerType(32, llvm::Align(4), /*Signed=*/true)),
        F32(TB.getFloatType(llvm::APFloat::IEEEsingle(), llvm::Align(4))),
        Empty(TB.getRecordType({}, llvm::TypeSize::getFixed(8), llvm::Align(1),
                               /*UnadjustedAlign=*/llvm::Align(1),
                               StructPacking::Default, {}, {},
                               RecordFlags::CanPassInRegisters)) {}

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

} // namespace
