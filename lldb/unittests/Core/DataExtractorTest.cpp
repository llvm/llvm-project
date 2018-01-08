//===-- DataExtractorTest.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Utility/DataExtractor.h"

using namespace lldb_private;

TEST(DataExtractorTest, GetBitfield) {
  uint8_t buffer[] = {0x01, 0x23, 0x45, 0x67};
  DataExtractor LE(buffer, sizeof(buffer), lldb::eByteOrderLittle,
                   sizeof(void *));
  DataExtractor BE(buffer, sizeof(buffer), lldb::eByteOrderBig, sizeof(void *));

  lldb::offset_t offset;

  offset = 0;
  ASSERT_EQ(buffer[1], LE.GetMaxU64Bitfield(&offset, sizeof(buffer), 8, 8));
  offset = 0;
  ASSERT_EQ(buffer[1], BE.GetMaxU64Bitfield(&offset, sizeof(buffer), 8, 8));

  offset = 0;
  ASSERT_EQ(int8_t(buffer[1]),
            LE.GetMaxS64Bitfield(&offset, sizeof(buffer), 8, 8));
  offset = 0;
  ASSERT_EQ(int8_t(buffer[1]),
            BE.GetMaxS64Bitfield(&offset, sizeof(buffer), 8, 8));
}

TEST(DataExtractorTest, PeekData) {
  uint8_t buffer[] = {0x01, 0x02, 0x03, 0x04};
  DataExtractor E(buffer, sizeof buffer, lldb::eByteOrderLittle, 4);

  EXPECT_EQ(buffer + 0, E.PeekData(0, 0));
  EXPECT_EQ(buffer + 0, E.PeekData(0, 4));
  EXPECT_EQ(nullptr, E.PeekData(0, 5));

  EXPECT_EQ(buffer + 2, E.PeekData(2, 0));
  EXPECT_EQ(buffer + 2, E.PeekData(2, 2));
  EXPECT_EQ(nullptr, E.PeekData(2, 3));

  EXPECT_EQ(buffer + 4, E.PeekData(4, 0));
  EXPECT_EQ(nullptr, E.PeekData(4, 1));
}

TEST(DataExtractorTest, GetMaxU64) {
  uint8_t buffer[] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
  DataExtractor LE(buffer, sizeof(buffer), lldb::eByteOrderLittle,
                   sizeof(void *));
  DataExtractor BE(buffer, sizeof(buffer), lldb::eByteOrderBig, sizeof(void *));

  lldb::offset_t offset;

  // Check with the minimum allowed byte size.
  offset = 0;
  EXPECT_EQ(0x01U, LE.GetMaxU64(&offset, 1));
  EXPECT_EQ(1U, offset);
  offset = 0;
  EXPECT_EQ(0x01U, BE.GetMaxU64(&offset, 1));
  EXPECT_EQ(1U, offset);

  // Check with a non-zero offset.
  offset = 1;
  EXPECT_EQ(0x0302U, LE.GetMaxU64(&offset, 2));
  EXPECT_EQ(3U, offset);
  offset = 1;
  EXPECT_EQ(0x0203U, BE.GetMaxU64(&offset, 2));
  EXPECT_EQ(3U, offset);

  // Check with the byte size not being a multiple of 2.
  offset = 0;
  EXPECT_EQ(0x07060504030201U, LE.GetMaxU64(&offset, 7));
  EXPECT_EQ(7U, offset);
  offset = 0;
  EXPECT_EQ(0x01020304050607U, BE.GetMaxU64(&offset, 7));
  EXPECT_EQ(7U, offset);

  // Check with the maximum allowed byte size.
  offset = 0;
  EXPECT_EQ(0x0807060504030201U, LE.GetMaxU64(&offset, 8));
  EXPECT_EQ(8U, offset);
  offset = 0;
  EXPECT_EQ(0x0102030405060708U, BE.GetMaxU64(&offset, 8));
  EXPECT_EQ(8U, offset);
}

TEST(DataExtractorTest, GetMaxS64) {
  uint8_t buffer[] = {0x01, 0x02, 0x83, 0x04, 0x05, 0x06, 0x07, 0x08};
  DataExtractor LE(buffer, sizeof(buffer), lldb::eByteOrderLittle,
                   sizeof(void *));
  DataExtractor BE(buffer, sizeof(buffer), lldb::eByteOrderBig, sizeof(void *));

  lldb::offset_t offset;

  // Check with the minimum allowed byte size.
  offset = 0;
  EXPECT_EQ(0x01, LE.GetMaxS64(&offset, 1));
  EXPECT_EQ(1U, offset);
  offset = 0;
  EXPECT_EQ(0x01, BE.GetMaxS64(&offset, 1));
  EXPECT_EQ(1U, offset);

  // Check that sign extension works correctly.
  offset = 0;
  int64_t value = LE.GetMaxS64(&offset, 3);
  EXPECT_EQ(0xffffffffff830201U, *reinterpret_cast<uint64_t *>(&value));
  EXPECT_EQ(3U, offset);
  offset = 2;
  value = BE.GetMaxS64(&offset, 3);
  EXPECT_EQ(0xffffffffff830405U, *reinterpret_cast<uint64_t *>(&value));
  EXPECT_EQ(5U, offset);

  // Check with the maximum allowed byte size.
  offset = 0;
  EXPECT_EQ(0x0807060504830201, LE.GetMaxS64(&offset, 8));
  EXPECT_EQ(8U, offset);
  offset = 0;
  EXPECT_EQ(0x0102830405060708, BE.GetMaxS64(&offset, 8));
  EXPECT_EQ(8U, offset);
}

TEST(DataExtractorTest, GetMaxU64_unchecked) {
  uint8_t buffer[] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
  DataExtractor LE(buffer, sizeof(buffer), lldb::eByteOrderLittle,
                   sizeof(void *));
  DataExtractor BE(buffer, sizeof(buffer), lldb::eByteOrderBig, sizeof(void *));

  lldb::offset_t offset;

  // Check with the minimum allowed byte size.
  offset = 0;
  EXPECT_EQ(0x01U, LE.GetMaxU64_unchecked(&offset, 1));
  EXPECT_EQ(1U, offset);
  offset = 0;
  EXPECT_EQ(0x01U, BE.GetMaxU64_unchecked(&offset, 1));
  EXPECT_EQ(1U, offset);

  // Check with a non-zero offset.
  offset = 1;
  EXPECT_EQ(0x0302U, LE.GetMaxU64_unchecked(&offset, 2));
  EXPECT_EQ(3U, offset);
  offset = 1;
  EXPECT_EQ(0x0203U, BE.GetMaxU64_unchecked(&offset, 2));
  EXPECT_EQ(3U, offset);

  // Check with the byte size not being a multiple of 2.
  offset = 0;
  EXPECT_EQ(0x07060504030201U, LE.GetMaxU64_unchecked(&offset, 7));
  EXPECT_EQ(7U, offset);
  offset = 0;
  EXPECT_EQ(0x01020304050607U, BE.GetMaxU64_unchecked(&offset, 7));
  EXPECT_EQ(7U, offset);

  // Check with the maximum allowed byte size.
  offset = 0;
  EXPECT_EQ(0x0807060504030201U, LE.GetMaxU64_unchecked(&offset, 8));
  EXPECT_EQ(8U, offset);
  offset = 0;
  EXPECT_EQ(0x0102030405060708U, BE.GetMaxU64_unchecked(&offset, 8));
  EXPECT_EQ(8U, offset);
}
