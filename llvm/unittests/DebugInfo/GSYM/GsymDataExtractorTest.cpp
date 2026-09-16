//===- GsymDataExtractorTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/GSYM/GsymDataExtractor.h"
#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace gsym;

TEST(GsymDataExtractorTest, DefaultStringOffsetSize) {
  GsymDataExtractor DE(StringRef("\0", 1), false);
  EXPECT_EQ(8u, DE.getStringOffsetSize());
}

TEST(GsymDataExtractorTest, ExplicitStringOffsetSize) {
  for (uint8_t Size = 1; Size <= 8; ++Size) {
    GsymDataExtractor DE(StringRef("\0", 1), false, Size);
    EXPECT_EQ(Size, DE.getStringOffsetSize());
  }
}

TEST(GsymDataExtractorTest, SubrangeConstructor) {
  const char Data[] = "\x01\x02\x03\x04\x05\x06\x07\x08";
  GsymDataExtractor Parent(StringRef(Data, 8), true, 5);

  GsymDataExtractor Sub(Parent, 2, 4);
  EXPECT_EQ(5u, Sub.getStringOffsetSize());
  EXPECT_TRUE(Sub.isLittleEndian());
  EXPECT_EQ(4u, Sub.getData().size());

  uint64_t Offset = 0;
  EXPECT_EQ(0x03u, Sub.getU8(&Offset));
}

TEST(GsymDataExtractorTest, GetStringOffset) {
  // Data: 0x01 0x02 0x03 0x04 0x05 0x06 0x07 0x08
  const char Data[] = "\x01\x02\x03\x04\x05\x06\x07\x08";

  // Expected little-endian values for sizes 1-8 reading from offset 0.
  const uint64_t ExpectedLE[] = {
      0x01u,               // size 1
      0x0201u,             // size 2
      0x030201u,           // size 3
      0x04030201u,         // size 4
      0x0504030201u,       // size 5
      0x060504030201u,     // size 6
      0x07060504030201u,   // size 7
      0x0807060504030201u, // size 8
  };
  // Expected big-endian values for sizes 1-8 reading from offset 0.
  const uint64_t ExpectedBE[] = {
      0x01u,               // size 1
      0x0102u,             // size 2
      0x010203u,           // size 3
      0x01020304u,         // size 4
      0x0102030405u,       // size 5
      0x010203040506u,     // size 6
      0x01020304050607u,   // size 7
      0x0102030405060708u, // size 8
  };

  for (uint8_t Size = 1; Size <= 8; ++Size) {
    GsymDataExtractor LE(StringRef(Data, 8), true, Size);
    GsymDataExtractor BE(StringRef(Data, 8), false, Size);
    uint64_t Offset = 0;
    EXPECT_EQ(ExpectedLE[Size - 1], LE.getStringOffset(&Offset))
        << "LE size=" << (int)Size;
    EXPECT_EQ(Size, Offset);
    Offset = 0;
    EXPECT_EQ(ExpectedBE[Size - 1], BE.getStringOffset(&Offset))
        << "BE size=" << (int)Size;
    EXPECT_EQ(Size, Offset);
  }
}

TEST(GsymDataExtractorTest, GetStringOffsetCursor) {
  const char Data[] = "\x01\x02\x03\x04\x05\x06\x07\x08";

  // Test Cursor-based reads for all sizes 1-8.
  for (uint8_t Size = 1; Size <= 8; ++Size) {
    GsymDataExtractor DE(StringRef(Data, 8), true, Size);
    DataExtractor::Cursor C(0);
    // Read the first element.
    uint64_t Val = DE.getStringOffset(C);
    EXPECT_EQ(Size, C.tell()) << "size=" << (int)Size;
    EXPECT_NE(0u, Val) << "size=" << (int)Size;
    EXPECT_THAT_ERROR(C.takeError(), Succeeded());
  }

  // Verify reading past end fails.
  GsymDataExtractor DE(StringRef(Data, 8), true, 4);
  DataExtractor::Cursor C(0);
  DE.getStringOffset(C); // offset 0-3
  DE.getStringOffset(C); // offset 4-7
  EXPECT_EQ(8u, C.tell());
  EXPECT_THAT_ERROR(C.takeError(), Succeeded());
  EXPECT_EQ(0u, DE.getStringOffset(C)); // past end
  EXPECT_THAT_ERROR(C.takeError(), Failed());
}
