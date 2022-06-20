//=== - llvm/unittest/Support/Alignment.cpp - Alignment utility tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Alignment.h"
#include "gtest/gtest.h"

#include <vector>

#ifdef _MSC_VER
// Disable warnings about potential divide by 0.
#pragma warning(push)
#pragma warning(disable : 4723)
#endif

using namespace llvm;

namespace {

TEST(AlignmentTest, AlignOfConstant) {
  EXPECT_EQ(Align::Of<uint8_t>(), Align(alignof(uint8_t)));
  EXPECT_EQ(Align::Of<uint16_t>(), Align(alignof(uint16_t)));
  EXPECT_EQ(Align::Of<uint32_t>(), Align(alignof(uint32_t)));
  EXPECT_EQ(Align::Of<uint64_t>(), Align(alignof(uint64_t)));
}

TEST(AlignmentTest, AlignConstant) {
  EXPECT_EQ(Align::Constant<1>(), Align(1));
  EXPECT_EQ(Align::Constant<2>(), Align(2));
  EXPECT_EQ(Align::Constant<4>(), Align(4));
  EXPECT_EQ(Align::Constant<8>(), Align(8));
  EXPECT_EQ(Align::Constant<16>(), Align(16));
  EXPECT_EQ(Align::Constant<32>(), Align(32));
  EXPECT_EQ(Align::Constant<64>(), Align(64));
}

TEST(AlignmentTest, AlignConstexprConstant) {
  constexpr Align kConstantAlign = Align::Of<uint64_t>();
  EXPECT_EQ(Align(alignof(uint64_t)), kConstantAlign);
}

std::vector<uint64_t> getValidAlignments() {
  std::vector<uint64_t> Out;
  for (size_t Shift = 0; Shift < 64; ++Shift)
    Out.push_back(1ULL << Shift);
  return Out;
}

TEST(AlignmentTest, AlignDefaultCTor) {
  EXPECT_EQ(Align().value(), 1ULL);
}

TEST(AlignmentTest, MaybeAlignDefaultCTor) {
  EXPECT_FALSE(MaybeAlign().hasValue());
}

TEST(AlignmentTest, ValidCTors) {
  for (uint64_t Value : getValidAlignments()) {
    EXPECT_EQ(Align(Value).value(), Value);
    EXPECT_EQ((*MaybeAlign(Value)).value(), Value);
  }
}

TEST(AlignmentTest, CheckMaybeAlignHasValue) {
  EXPECT_TRUE(MaybeAlign(1));
  EXPECT_TRUE(MaybeAlign(1).hasValue());
  EXPECT_FALSE(MaybeAlign(0));
  EXPECT_FALSE(MaybeAlign(0).hasValue());
  EXPECT_FALSE(MaybeAlign());
  EXPECT_FALSE(MaybeAlign().hasValue());
}

TEST(AlignmentTest, Division) {
  for (uint64_t Value : getValidAlignments()) {
    if (Value > 1) {
      EXPECT_EQ(Align(Value) / 2, Value / 2);
    }
  }
}

TEST(AlignmentTest, AlignTo) {
  struct {
    uint64_t alignment;
    uint64_t offset;
    uint64_t rounded;
    const void *forgedAddr() const {
      //  A value of any integral or enumeration type can be converted to a
      //  pointer type.
      return reinterpret_cast<const void *>(offset);
    }
  } kTests[] = {
      // MaybeAlign
      {0, 0, 0},
      {0, 1, 1},
      {0, 5, 5},
      // MaybeAlign / Align
      {1, 0, 0},
      {1, 1, 1},
      {1, 5, 5},
      {2, 0, 0},
      {2, 1, 2},
      {2, 2, 2},
      {2, 7, 8},
      {2, 16, 16},
      {4, 0, 0},
      {4, 1, 4},
      {4, 4, 4},
      {4, 6, 8},
  };
  for (const auto &T : kTests) {
    MaybeAlign A(T.alignment);
    // Test MaybeAlign
    EXPECT_EQ(alignTo(T.offset, A), T.rounded);
    // Test Align
    if (A) {
      EXPECT_EQ(alignTo(T.offset, A.getValue()), T.rounded);
      EXPECT_EQ(alignAddr(T.forgedAddr(), A.getValue()), T.rounded);
    }
  }
}

TEST(AlignmentTest, AlignToWithSkew) {
  EXPECT_EQ(alignTo(5, Align(8), 0), alignTo(5, Align(8)));
  EXPECT_EQ(alignTo(5, Align(8), 7), 7U);
  EXPECT_EQ(alignTo(17, Align(8), 1), 17U);
  EXPECT_EQ(alignTo(~0LL, Align(8), 3), 3U);
}

TEST(AlignmentTest, Log2) {
  for (uint64_t Value : getValidAlignments()) {
    EXPECT_EQ(Log2(Align(Value)), Log2_64(Value));
  }
}

TEST(AlignmentTest, MinAlign) {
  struct {
    uint64_t A;
    uint64_t B;
    uint64_t MinAlign;
  } kTests[] = {
      // MaybeAlign
      {0, 0, 0},
      {0, 8, 8},
      {2, 0, 2},
      // MaybeAlign / Align
      {1, 2, 1},
      {8, 4, 4},
  };
  for (const auto &T : kTests) {
    EXPECT_EQ(commonAlignment(MaybeAlign(T.A), MaybeAlign(T.B)), T.MinAlign);
    EXPECT_EQ(MinAlign(T.A, T.B), T.MinAlign);
    if (T.A) {
      EXPECT_EQ(commonAlignment(Align(T.A), MaybeAlign(T.B)), T.MinAlign);
    }
    if (T.B) {
      EXPECT_EQ(commonAlignment(MaybeAlign(T.A), Align(T.B)), T.MinAlign);
    }
    if (T.A && T.B) {
      EXPECT_EQ(commonAlignment(Align(T.A), Align(T.B)), T.MinAlign);
    }
  }
}

TEST(AlignmentTest, Encode_Decode) {
  for (uint64_t Value : getValidAlignments()) {
    {
      Align Actual(Value);
      Align Expected = decodeMaybeAlign(encode(Actual)).getValue();
      EXPECT_EQ(Expected, Actual);
    }
    {
      MaybeAlign Actual(Value);
      MaybeAlign Expected = decodeMaybeAlign(encode(Actual));
      EXPECT_EQ(Expected, Actual);
    }
  }
  MaybeAlign Actual(0);
  MaybeAlign Expected = decodeMaybeAlign(encode(Actual));
  EXPECT_EQ(Expected, Actual);
}

TEST(AlignmentTest, isAligned_isAddrAligned) {
  struct {
    uint64_t alignment;
    uint64_t offset;
    bool isAligned;
    const void *forgedAddr() const {
      //  A value of any integral or enumeration type can be converted to a
      //  pointer type.
      return reinterpret_cast<const void *>(offset);
    }
  } kTests[] = {
      {1, 0, true},  {1, 1, true},  {1, 5, true},  {2, 0, true},
      {2, 1, false}, {2, 2, true},  {2, 7, false}, {2, 16, true},
      {4, 0, true},  {4, 1, false}, {4, 4, true},  {4, 6, false},
  };
  for (const auto &T : kTests) {
    MaybeAlign A(T.alignment);
    // Test Align
    if (A) {
      EXPECT_EQ(isAligned(A.getValue(), T.offset), T.isAligned);
      EXPECT_EQ(isAddrAligned(A.getValue(), T.forgedAddr()), T.isAligned);
    }
  }
}

TEST(AlignmentTest, offsetToAlignment) {
  struct {
    uint64_t alignment;
    uint64_t offset;
    uint64_t alignedOffset;
    const void *forgedAddr() const {
      //  A value of any integral or enumeration type can be converted to a
      //  pointer type.
      return reinterpret_cast<const void *>(offset);
    }
  } kTests[] = {
      {1, 0, 0}, {1, 1, 0},  {1, 5, 0}, {2, 0, 0}, {2, 1, 1}, {2, 2, 0},
      {2, 7, 1}, {2, 16, 0}, {4, 0, 0}, {4, 1, 3}, {4, 4, 0}, {4, 6, 2},
  };
  for (const auto &T : kTests) {
    const Align A(T.alignment);
    EXPECT_EQ(offsetToAlignment(T.offset, A), T.alignedOffset);
    EXPECT_EQ(offsetToAlignedAddr(T.forgedAddr(), A), T.alignedOffset);
  }
}

TEST(AlignmentTest, AlignComparisons) {
  std::vector<uint64_t> ValidAlignments = getValidAlignments();
  std::sort(ValidAlignments.begin(), ValidAlignments.end());
  for (size_t I = 1; I < ValidAlignments.size(); ++I) {
    assert(I >= 1);
    const Align A(ValidAlignments[I - 1]);
    const Align B(ValidAlignments[I]);
    EXPECT_EQ(A, A);
    EXPECT_NE(A, B);
    EXPECT_LT(A, B);
    EXPECT_GT(B, A);
    EXPECT_LE(A, B);
    EXPECT_GE(B, A);
    EXPECT_LE(A, A);
    EXPECT_GE(A, A);

    EXPECT_EQ(A, A.value());
    EXPECT_NE(A, B.value());
    EXPECT_LT(A, B.value());
    EXPECT_GT(B, A.value());
    EXPECT_LE(A, B.value());
    EXPECT_GE(B, A.value());
    EXPECT_LE(A, A.value());
    EXPECT_GE(A, A.value());

    EXPECT_EQ(std::max(A, B), B);
    EXPECT_EQ(std::min(A, B), A);

    const MaybeAlign MA(ValidAlignments[I - 1]);
    const MaybeAlign MB(ValidAlignments[I]);
    EXPECT_EQ(MA, MA);
    EXPECT_NE(MA, MB);

    EXPECT_EQ(MA, MA ? (*MA).value() : 0);
    EXPECT_NE(MA, MB ? (*MB).value() : 0);

    EXPECT_EQ(std::max(A, B), B);
    EXPECT_EQ(std::min(A, B), A);
  }
}

TEST(AlignmentTest, AssumeAligned) {
  EXPECT_EQ(assumeAligned(0), Align(1));
  EXPECT_EQ(assumeAligned(0), Align());
  EXPECT_EQ(assumeAligned(1), Align(1));
  EXPECT_EQ(assumeAligned(1), Align());
}

// Death tests reply on assert which is disabled in release mode.
#ifndef NDEBUG

// We use a subset of valid alignments for DEATH_TESTs as they are particularly
// slow.
std::vector<uint64_t> getValidAlignmentsForDeathTest() {
  return {1, 1ULL << 31, 1ULL << 63};
}

std::vector<uint64_t> getNonPowerOfTwo() { return {3, 10, 15}; }

TEST(AlignmentDeathTest, CantConvertUnsetMaybe) {
  EXPECT_DEATH((MaybeAlign(0).getValue()), ".*");
}

TEST(AlignmentDeathTest, Division) {
  EXPECT_DEATH(Align(1) / 2, "Can't halve byte alignment");

  EXPECT_DEATH(Align(8) / 0, "Divisor must be positive and a power of 2");
  EXPECT_DEATH(Align(8) / 3, "Divisor must be positive and a power of 2");
}

TEST(AlignmentDeathTest, InvalidCTors) {
  EXPECT_DEATH((Align(0)), "Value must not be 0");
  for (uint64_t Value : getNonPowerOfTwo()) {
    EXPECT_DEATH((Align(Value)), "Alignment is not a power of 2");
    EXPECT_DEATH((MaybeAlign(Value)),
                 "Alignment is neither 0 nor a power of 2");
  }
}

TEST(AlignmentDeathTest, ComparisonsWithZero) {
  for (uint64_t Value : getValidAlignmentsForDeathTest()) {
    EXPECT_DEATH((void)(Align(Value) == 0), ".* should be defined");
    EXPECT_DEATH((void)(Align(Value) != 0), ".* should be defined");
    EXPECT_DEATH((void)(Align(Value) >= 0), ".* should be defined");
    EXPECT_DEATH((void)(Align(Value) <= 0), ".* should be defined");
    EXPECT_DEATH((void)(Align(Value) > 0), ".* should be defined");
    EXPECT_DEATH((void)(Align(Value) < 0), ".* should be defined");
  }
}

TEST(AlignmentDeathTest, CompareMaybeAlignToZero) {
  for (uint64_t Value : getValidAlignmentsForDeathTest()) {
    // MaybeAlign is allowed to be == or != 0
    (void)(MaybeAlign(Value) == 0);
    (void)(MaybeAlign(Value) != 0);
  }
}

TEST(AlignmentDeathTest, AlignAddr) {
  const void *const unaligned_high_ptr =
      reinterpret_cast<const void *>(std::numeric_limits<uintptr_t>::max() - 1);
  EXPECT_DEATH(alignAddr(unaligned_high_ptr, Align(16)), "Overflow");
}

#endif // NDEBUG

} // end anonymous namespace

#ifdef _MSC_VER
#pragma warning(pop)
#endif
