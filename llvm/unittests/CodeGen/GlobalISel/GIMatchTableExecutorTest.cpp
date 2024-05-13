//===- GIMatchTableExecutorTest.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/GIMatchTableExecutor.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(GlobalISelLEB128Test, fastDecodeULEB128) {
#define EXPECT_DECODE_ULEB128_EQ(EXPECTED, VALUE)                              \
  do {                                                                         \
    uint64_t ActualSize = 0;                                                   \
    uint64_t Actual = GIMatchTableExecutor::fastDecodeULEB128(                 \
        reinterpret_cast<const uint8_t *>(VALUE), ActualSize);                 \
    EXPECT_EQ(sizeof(VALUE) - 1, ActualSize);                                  \
    EXPECT_EQ(EXPECTED, Actual);                                               \
  } while (0)

  EXPECT_DECODE_ULEB128_EQ(0u, "\x00");
  EXPECT_DECODE_ULEB128_EQ(1u, "\x01");
  EXPECT_DECODE_ULEB128_EQ(63u, "\x3f");
  EXPECT_DECODE_ULEB128_EQ(64u, "\x40");
  EXPECT_DECODE_ULEB128_EQ(0x7fu, "\x7f");
  EXPECT_DECODE_ULEB128_EQ(0x80u, "\x80\x01");
  EXPECT_DECODE_ULEB128_EQ(0x81u, "\x81\x01");
  EXPECT_DECODE_ULEB128_EQ(0x90u, "\x90\x01");
  EXPECT_DECODE_ULEB128_EQ(0xffu, "\xff\x01");
  EXPECT_DECODE_ULEB128_EQ(0x100u, "\x80\x02");
  EXPECT_DECODE_ULEB128_EQ(0x101u, "\x81\x02");
  EXPECT_DECODE_ULEB128_EQ(4294975616ULL, "\x80\xc1\x80\x80\x10");

  // Decode ULEB128 with extra padding bytes
  EXPECT_DECODE_ULEB128_EQ(0u, "\x80\x00");
  EXPECT_DECODE_ULEB128_EQ(0u, "\x80\x80\x00");
  EXPECT_DECODE_ULEB128_EQ(0x7fu, "\xff\x00");
  EXPECT_DECODE_ULEB128_EQ(0x7fu, "\xff\x80\x00");
  EXPECT_DECODE_ULEB128_EQ(0x80u, "\x80\x81\x00");
  EXPECT_DECODE_ULEB128_EQ(0x80u, "\x80\x81\x80\x00");
  EXPECT_DECODE_ULEB128_EQ(0x80u, "\x80\x81\x80\x80\x80\x80\x80\x80\x80\x00");
  EXPECT_DECODE_ULEB128_EQ(0x80000000'00000000ul,
                           "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x01");

#undef EXPECT_DECODE_ULEB128_EQ
}
