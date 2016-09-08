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
  char buffer[] = {0x01, 0x23, 0x45, 0x67};
  DataExtractor LE(buffer, sizeof(buffer), lldb::eByteOrderLittle,
                   sizeof(void *));
  DataExtractor BE(buffer, sizeof(buffer), lldb::eByteOrderBig, sizeof(void *));

  lldb::offset_t offset;

  offset = 0;
  ASSERT_EQ(buffer[1], LE.GetMaxU64Bitfield(&offset, sizeof(buffer), 8, 8));
  offset = 0;
  ASSERT_EQ(buffer[1], BE.GetMaxU64Bitfield(&offset, sizeof(buffer), 8, 8));

  offset = 0;
  ASSERT_EQ(buffer[1], LE.GetMaxS64Bitfield(&offset, sizeof(buffer), 8, 8));
  offset = 0;
  ASSERT_EQ(buffer[1], BE.GetMaxS64Bitfield(&offset, sizeof(buffer), 8, 8));
}
