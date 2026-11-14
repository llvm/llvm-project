//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements unit tests for the RISCVZicfilpHash header.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/RISCVZicfilpHash.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MD5.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::RISCVCFI;
namespace {
MD5::MD5Result makeResult(uint64_t Low, uint64_t High) {
  MD5::MD5Result R;
  support::endian::write<uint64_t, endianness::little>(R.data(), Low);
  support::endian::write<uint64_t, endianness::little>(R.data() + 8, High);
  return R;
}

// creates an MD5Result where the first x bits are 0, followed by 'value' to
// test the fallback logic.
llvm::MD5::MD5Result makeResultWithFirstNonzeroAt(unsigned SkipBits,
                                                  uint32_t Value) {
  uint64_t Low = 0, High = 0;
  if (SkipBits < 64) {
    Low = static_cast<uint64_t>(Value) << SkipBits;
    if (SkipBits + 20 > 64)
      High = static_cast<uint64_t>(Value) >> (64 - SkipBits);
  } else {
    High = static_cast<uint64_t>(Value) << (SkipBits - 64);
  }
  return makeResult(Low, High);
}

} // namespace

TEST(RISCVZicfilpHash, SkipsLeadingZeroSegment) {
  // Lower 20 bits zero, next 20 bits = 0x12345.
  uint64_t Low = 0x12345ull << 20;
  auto Result = makeResult(Low, 0);
  EXPECT_EQ(0x12345u, reduceZicfilpFuncSigLabel(Result));
}

TEST(RISCVZicfilpHash, AllZeroFallsBackToConstant) {
  EXPECT_EQ(ZicfilpFuncSigFallbackLabel,
            reduceZicfilpFuncSigLabel(makeResult(0, 0)));
}

TEST(RISCVZicfilpHash, FallbackConstantMatchesRiscVHash) {
  EXPECT_EQ(ZicfilpFuncSigFallbackLabel,
            getZicfilpFuncSigLabel(llvm::StringRef("RISC-V")));
}

TEST(RISCVZicfilpHash, FirstNonzeroAt0) {
  EXPECT_EQ(0x12345u, reduceZicfilpFuncSigLabel(
                          makeResultWithFirstNonzeroAt(0, 0x12345)));
}
TEST(RISCVZicfilpHash, FirstNonzeroAt20) {
  EXPECT_EQ(0x6789Au, reduceZicfilpFuncSigLabel(
                          makeResultWithFirstNonzeroAt(20, 0x6789A)));
}
TEST(RISCVZicfilpHash, FirstNonzeroAt40) {
  EXPECT_EQ(0xABCDEu, reduceZicfilpFuncSigLabel(
                          makeResultWithFirstNonzeroAt(40, 0xABCDE)));
}
TEST(RISCVZicfilpHash, FirstNonzeroAt60) {
  EXPECT_EQ(0x3ABCDu, reduceZicfilpFuncSigLabel(
                          makeResultWithFirstNonzeroAt(60, 0x3ABCD)));
} // straddles Low/High
TEST(RISCVZicfilpHash, FirstNonzeroAt80) {
  EXPECT_EQ(0x54321u, reduceZicfilpFuncSigLabel(
                          makeResultWithFirstNonzeroAt(80, 0x54321)));
}
TEST(RISCVZicfilpHash, FirstNonzeroAt100) {
  EXPECT_EQ(0xFEDCBu, reduceZicfilpFuncSigLabel(
                          makeResultWithFirstNonzeroAt(100, 0xFEDCB)));
}
TEST(RISCVZicfilpHash, FirstNonzeroAt120) {
  // Only 8 bits remain (120..127); the rest of the 20-bit field is 0-padded
  EXPECT_EQ(0xABu,
            reduceZicfilpFuncSigLabel(makeResultWithFirstNonzeroAt(120, 0xAB)));
}

TEST(RISCVZicfilpHash, RiscVHashLower20BitsMatchesFallbackConstant) {
  llvm::MD5 Hash;
  Hash.update(llvm::StringRef("RISC-V"));
  llvm::MD5::MD5Result Result;
  Hash.final(Result);
  EXPECT_EQ(ZicfilpFuncSigFallbackLabel,
            static_cast<uint32_t>(Result.low() & 0xFFFFF));
}
