//===-- DataExtractorTest.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#if defined(_MSC_VER) && (_HAS_EXCEPTIONS == 0)
// Workaround for MSVC standard library bug, which fails to include <thread>
// when
// exceptions are disabled.
#include <eh.h>
#endif

#include "gtest/gtest.h"

#include "lldb/Core/DataExtractor.h"

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
