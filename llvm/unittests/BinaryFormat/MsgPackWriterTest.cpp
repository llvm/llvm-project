//===- MsgPackWriterTest.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/MsgPackWriter.h"
#include "llvm/BinaryFormat/MsgPack.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::msgpack;

struct MsgPackWriter : testing::Test {
  std::string Buffer;
  llvm::raw_string_ostream OStream;
  Writer MPWriter;

  MsgPackWriter() : OStream(Buffer), MPWriter(OStream) {}
};

TEST_F(MsgPackWriter, TestWriteNil) {
  MPWriter.writeNil();
  EXPECT_EQ(OStream.str(), "\xc0");
}

TEST_F(MsgPackWriter, TestWriteBool) {
  MPWriter.write(true);
  MPWriter.write(false);
  EXPECT_EQ(OStream.str(), "\xc3\xc2");
}

TEST_F(MsgPackWriter, TestWriteFixPositiveInt) {
  // FixPositiveInt form bitpattern starts with 0, so max FixPositiveInt
  // is 01111111 = 127
  for (uint64_t u = 0; u <= 127; ++u) {
    Buffer.clear();
    MPWriter.write(u);
    std::string Output = OStream.str();
    EXPECT_EQ(Output.size(), 1u);
    EXPECT_EQ(Output.data()[0], static_cast<uint8_t>(u));
  }
}

TEST_F(MsgPackWriter, TestWriteUInt8Min) {
  // See TestWriteFixPositiveInt for why 128 is the min non-fix Int8
  uint64_t u = 128;
  MPWriter.write(u);
  EXPECT_EQ(OStream.str(), "\xcc\x80");
}

TEST_F(MsgPackWriter, TestWriteUInt8) {
  uint64_t u = 221;
  MPWriter.write(u);
  EXPECT_EQ(OStream.str(), "\xcc\xdd");
}

TEST_F(MsgPackWriter, TestWriteUInt8Max) {
  uint64_t u = UINT8_MAX;
  MPWriter.write(u);
  EXPECT_EQ(OStream.str(), "\xcc\xff");
}

TEST_F(MsgPackWriter, TestWriteUInt16Min) {
  uint64_t u = static_cast<uint64_t>(UINT8_MAX) + 1;
  MPWriter.write(u);
  EXPECT_EQ(OStream.str(), std::string("\xcd\x01\x00", 3));
}

TEST_F(MsgPackWriter, TestWriteUInt16) {
  uint64_t u = 43981;
  MPWriter.write(u);
  EXPECT_EQ(OStream.str(), "\xcd\xab\xcd");
}

TEST_F(MsgPackWriter, TestWriteUInt16Max) {
  uint64_t u = UINT16_MAX;
  MPWriter.write(u);
  EXPECT_EQ(OStream.str(), "\xcd\xff\xff");
}

TEST_F(MsgPackWriter, TestWriteUInt32Min) {
  uint64_t u = static_cast<uint64_t>(UINT16_MAX) + 1;
  MPWriter.write(u);
  EXPECT_EQ(OStream.str(), std::string("\xce\x00\x01\x00\x00", 5));
}

TEST_F(MsgPackWriter, TestWriteUInt32) {
  uint64_t u = 2882400186;
  MPWriter.write(u);
  EXPECT_EQ(OStream.str(), "\xce\xab\xcd\xef\xba");
}

TEST_F(MsgPackWriter, TestWriteUInt32Max) {
  uint64_t u = UINT32_MAX;
  MPWriter.write(u);
  EXPECT_EQ(OStream.str(), "\xce\xff\xff\xff\xff");
}

TEST_F(MsgPackWriter, TestWriteUInt64Min) {
  uint64_t u = static_cast<uint64_t>(UINT32_MAX) + 1;
  MPWriter.write(u);
  EXPECT_EQ(OStream.str(),
            std::string("\xcf\x00\x00\x00\x01\x00\x00\x00\x00", 9));
}

TEST_F(MsgPackWriter, TestWriteUInt64) {
  uint64_t u = 0x010203040506074a;
  MPWriter.write(u);
  EXPECT_EQ(OStream.str(), "\xcf\x01\x02\x03\x04\x05\x06\x07\x4a");
}

TEST_F(MsgPackWriter, TestWriteUInt64Max) {
  uint64_t u = UINT64_MAX;
  MPWriter.write(u);
  EXPECT_EQ(OStream.str(), "\xcf\xff\xff\xff\xff\xff\xff\xff\xff");
}

TEST_F(MsgPackWriter, TestWriteFixNegativeInt) {
  // Positive values will be written in a UInt form, so max FixNegativeInt is -1
  //
  // FixNegativeInt form bitpattern starts with 111, so min FixNegativeInt
  // is 11100000 = -32
  for (int64_t i = -1; i >= -32; --i) {
    Buffer.clear();
    MPWriter.write(i);
    std::string Output = OStream.str();
    EXPECT_EQ(Output.size(), 1u);
    EXPECT_EQ(static_cast<int8_t>(Output.data()[0]), static_cast<int8_t>(i));
  }
}

TEST_F(MsgPackWriter, TestWriteInt8Max) {
  // See TestWriteFixNegativeInt for why -33 is the max non-fix Int8
  int64_t i = -33;
  MPWriter.write(i);
  EXPECT_EQ(OStream.str(), "\xd0\xdf");
}

TEST_F(MsgPackWriter, TestWriteInt8) {
  int64_t i = -40;
  MPWriter.write(i);
  EXPECT_EQ(OStream.str(), "\xd0\xd8");
}

TEST_F(MsgPackWriter, TestWriteInt8Min) {
  int64_t i = INT8_MIN;
  MPWriter.write(i);
  EXPECT_EQ(OStream.str(), "\xd0\x80");
}

TEST_F(MsgPackWriter, TestWriteInt16Max) {
  int64_t i = static_cast<int64_t>(INT8_MIN) - 1;
  MPWriter.write(i);
  EXPECT_EQ(OStream.str(), "\xd1\xff\x7f");
}

TEST_F(MsgPackWriter, TestWriteInt16) {
  int64_t i = -4369;
  MPWriter.write(i);
  EXPECT_EQ(OStream.str(), "\xd1\xee\xef");
}

TEST_F(MsgPackWriter, TestWriteInt16Min) {
  int64_t i = INT16_MIN;
  MPWriter.write(i);
  EXPECT_EQ(OStream.str(), std::string("\xd1\x80\x00", 3));
}

TEST_F(MsgPackWriter, TestWriteInt32Max) {
  int64_t i = static_cast<int64_t>(INT16_MIN) - 1;
  MPWriter.write(i);
  EXPECT_EQ(OStream.str(), "\xd2\xff\xff\x7f\xff");
}

TEST_F(MsgPackWriter, TestWriteInt32) {
  int64_t i = -286331153;
  MPWriter.write(i);
  EXPECT_EQ(OStream.str(), "\xd2\xee\xee\xee\xef");
}

TEST_F(MsgPackWriter, TestWriteInt32Min) {
  int64_t i = INT32_MIN;
  MPWriter.write(i);
  EXPECT_EQ(OStream.str(), std::string("\xd2\x80\x00\x00\x00", 5));
}

TEST_F(MsgPackWriter, TestWriteInt64Max) {
  int64_t i = static_cast<int64_t>(INT32_MIN) - 1;
  MPWriter.write(i);
  EXPECT_EQ(OStream.str(), "\xd3\xff\xff\xff\xff\x7f\xff\xff\xff");
}

TEST_F(MsgPackWriter, TestWriteInt64) {
  int64_t i = -1229782938247303441;
  MPWriter.write(i);
  EXPECT_EQ(OStream.str(), "\xd3\xee\xee\xee\xee\xee\xee\xee\xef");
}

TEST_F(MsgPackWriter, TestWriteInt64Min) {
  int64_t i = INT64_MIN;
  MPWriter.write(i);
  EXPECT_EQ(OStream.str(),
            std::string("\xd3\x80\x00\x00\x00\x00\x00\x00\x00", 9));
}

TEST_F(MsgPackWriter, TestWriteFloat32) {
  float f = -3.6973142664068907e+28;
  MPWriter.write(f);
  EXPECT_EQ(OStream.str(), "\xca\xee\xee\xee\xef");
}

TEST_F(MsgPackWriter, TestWriteFloat64) {
  double d = -2.2899894549927042e+226;
  MPWriter.write(d);
  EXPECT_EQ(OStream.str(), "\xcb\xee\xee\xee\xee\xee\xee\xee\xef");
}

TEST_F(MsgPackWriter, TestWriteFixStrMin) {
  std::string s;
  MPWriter.write(s);
  EXPECT_EQ(OStream.str(), "\xa0");
}

TEST_F(MsgPackWriter, TestWriteFixStr) {
  std::string s = "foo";
  MPWriter.write(s);
  EXPECT_EQ(OStream.str(), "\xa3"
                           "foo");
}

TEST_F(MsgPackWriter, TestWriteFixStrMax) {
  // FixStr format's size is a 5 bit unsigned integer, so max is 11111 = 31
  std::string s(31, 'a');
  MPWriter.write(s);
  EXPECT_EQ(OStream.str(), std::string("\xbf") + s);
}

TEST_F(MsgPackWriter, TestWriteStr8Min) {
  // See TestWriteFixStrMax for why 32 is the min non-fix Str8
  std::string s(32, 'a');
  MPWriter.write(s);
  EXPECT_EQ(OStream.str(), std::string("\xd9\x20") + s);
}

TEST_F(MsgPackWriter, TestWriteStr8) {
  std::string s(33, 'a');
  MPWriter.write(s);
  EXPECT_EQ(OStream.str(), std::string("\xd9\x21") + s);
}

TEST_F(MsgPackWriter, TestWriteStr8Max) {
  std::string s(UINT8_MAX, 'a');
  MPWriter.write(s);
  EXPECT_EQ(OStream.str(), std::string("\xd9\xff") + s);
}

TEST_F(MsgPackWriter, TestWriteStr16Min) {
  std::string s(static_cast<uint64_t>(UINT8_MAX) + 1, 'a');
  MPWriter.write(s);
  EXPECT_EQ(OStream.str(), std::string("\xda\x01\x00", 3) + s);
}

TEST_F(MsgPackWriter, TestWriteStr16) {
  std::string s(511, 'a');
  MPWriter.write(s);
  EXPECT_EQ(OStream.str(), std::string("\xda\x01\xff") + s);
}

TEST_F(MsgPackWriter, TestWriteStr16Max) {
  std::string s(UINT16_MAX, 'a');
  MPWriter.write(s);
  EXPECT_EQ(OStream.str(), std::string("\xda\xff\xff") + s);
}

TEST_F(MsgPackWriter, TestWriteStr32Min) {
  std::string s(static_cast<uint64_t>(UINT16_MAX) + 1, 'a');
  MPWriter.write(s);
  EXPECT_EQ(OStream.str(), std::string("\xdb\x00\x01\x00\x00", 5) + s);
}

TEST_F(MsgPackWriter, TestWriteStr32) {
  std::string s(131071, 'a');
  MPWriter.write(s);
  EXPECT_EQ(OStream.str(), std::string("\xdb\x00\x01\xff\xff", 5) + s);
}

TEST_F(MsgPackWriter, TestWriteBin8Min) {
  std::string s;
  MPWriter.write(MemoryBufferRef(s, ""));
  EXPECT_EQ(OStream.str(), std::string("\xc4\x00", 2) + s);
}

TEST_F(MsgPackWriter, TestWriteBin8) {
  std::string s(5, 'a');
  MPWriter.write(MemoryBufferRef(s, ""));
  EXPECT_EQ(OStream.str(), std::string("\xc4\x05") + s);
}

TEST_F(MsgPackWriter, TestWriteBin8Max) {
  std::string s(UINT8_MAX, 'a');
  MPWriter.write(MemoryBufferRef(s, ""));
  EXPECT_EQ(OStream.str(), std::string("\xc4\xff") + s);
}

TEST_F(MsgPackWriter, TestWriteBin16Min) {
  std::string s(static_cast<uint64_t>(UINT8_MAX) + 1, 'a');
  MPWriter.write(MemoryBufferRef(s, ""));
  EXPECT_EQ(OStream.str(), std::string("\xc5\x01\x00", 3) + s);
}

TEST_F(MsgPackWriter, TestWriteBin16) {
  std::string s(511, 'a');
  MPWriter.write(MemoryBufferRef(s, ""));
  EXPECT_EQ(OStream.str(), "\xc5\x01\xff" + s);
}

TEST_F(MsgPackWriter, TestWriteBin16Max) {
  std::string s(UINT16_MAX, 'a');
  MPWriter.write(MemoryBufferRef(s, ""));
  EXPECT_EQ(OStream.str(), std::string("\xc5\xff\xff") + s);
}

TEST_F(MsgPackWriter, TestWriteBin32Min) {
  std::string s(static_cast<uint64_t>(UINT16_MAX) + 1, 'a');
  MPWriter.write(MemoryBufferRef(s, ""));
  EXPECT_EQ(OStream.str(), std::string("\xc6\x00\x01\x00\x00", 5) + s);
}

TEST_F(MsgPackWriter, TestWriteBin32) {
  std::string s(131071, 'a');
  MPWriter.write(MemoryBufferRef(s, ""));
  EXPECT_EQ(OStream.str(), std::string("\xc6\x00\x01\xff\xff", 5) + s);
}

TEST_F(MsgPackWriter, TestWriteFixArrayMin) {
  MPWriter.writeArraySize(0);
  EXPECT_EQ(OStream.str(), "\x90");
}

TEST_F(MsgPackWriter, TestWriteFixArray) {
  MPWriter.writeArraySize(4);
  EXPECT_EQ(OStream.str(), "\x94");
}

TEST_F(MsgPackWriter, TestWriteFixArrayMax) {
  // FixArray format's size is a 4 bit unsigned integer, so max is 1111 = 15
  MPWriter.writeArraySize(15);
  EXPECT_EQ(OStream.str(), "\x9f");
}

TEST_F(MsgPackWriter, TestWriteArray16Min) {
  // See TestWriteFixArrayMax for why 16 is the min non-fix Array16
  MPWriter.writeArraySize(16);
  EXPECT_EQ(OStream.str(), std::string("\xdc\x00\x10", 3));
}

TEST_F(MsgPackWriter, TestWriteArray16) {
  MPWriter.writeArraySize(273);
  EXPECT_EQ(OStream.str(), "\xdc\x01\x11");
}

TEST_F(MsgPackWriter, TestWriteArray16Max) {
  MPWriter.writeArraySize(UINT16_MAX);
  EXPECT_EQ(OStream.str(), "\xdc\xff\xff");
}

TEST_F(MsgPackWriter, TestWriteArray32Min) {
  MPWriter.writeArraySize(static_cast<uint64_t>(UINT16_MAX) + 1);
  EXPECT_EQ(OStream.str(), std::string("\xdd\x00\x01\x00\x00", 5));
}

TEST_F(MsgPackWriter, TestWriteArray32) {
  MPWriter.writeArraySize(131071);
  EXPECT_EQ(OStream.str(), std::string("\xdd\x00\x01\xff\xff", 5));
}

TEST_F(MsgPackWriter, TestWriteArray32Max) {
  MPWriter.writeArraySize(UINT32_MAX);
  EXPECT_EQ(OStream.str(), "\xdd\xff\xff\xff\xff");
}

TEST_F(MsgPackWriter, TestWriteFixMapMin) {
  MPWriter.writeMapSize(0);
  EXPECT_EQ(OStream.str(), "\x80");
}

TEST_F(MsgPackWriter, TestWriteFixMap) {
  MPWriter.writeMapSize(4);
  EXPECT_EQ(OStream.str(), "\x84");
}

TEST_F(MsgPackWriter, TestWriteFixMapMax) {
  // FixMap format's size is a 4 bit unsigned integer, so max is 1111 = 15
  MPWriter.writeMapSize(15);
  EXPECT_EQ(OStream.str(), "\x8f");
}

TEST_F(MsgPackWriter, TestWriteMap16Min) {
  // See TestWriteFixMapMax for why 16 is the min non-fix Map16
  MPWriter.writeMapSize(16);
  EXPECT_EQ(OStream.str(), std::string("\xde\x00\x10", 3));
}

TEST_F(MsgPackWriter, TestWriteMap16) {
  MPWriter.writeMapSize(273);
  EXPECT_EQ(OStream.str(), "\xde\x01\x11");
}

TEST_F(MsgPackWriter, TestWriteMap16Max) {
  MPWriter.writeMapSize(UINT16_MAX);
  EXPECT_EQ(OStream.str(), "\xde\xff\xff");
}

TEST_F(MsgPackWriter, TestWriteMap32Min) {
  MPWriter.writeMapSize(static_cast<uint64_t>(UINT16_MAX) + 1);
  EXPECT_EQ(OStream.str(), std::string("\xdf\x00\x01\x00\x00", 5));
}

TEST_F(MsgPackWriter, TestWriteMap32) {
  MPWriter.writeMapSize(131071);
  EXPECT_EQ(OStream.str(), std::string("\xdf\x00\x01\xff\xff", 5));
}

TEST_F(MsgPackWriter, TestWriteMap32Max) {
  MPWriter.writeMapSize(UINT32_MAX);
  EXPECT_EQ(OStream.str(), std::string("\xdf\xff\xff\xff\xff", 5));
}

// FixExt formats are only available for these specific lengths: 1, 2, 4, 8, 16

TEST_F(MsgPackWriter, TestWriteFixExt1) {
  std::string s(1, 'a');
  MPWriter.writeExt(0x01, MemoryBufferRef(s, ""));
  EXPECT_EQ(OStream.str(), std::string("\xd4\x01") + s);
}

TEST_F(MsgPackWriter, TestWriteFixExt2) {
  std::string s(2, 'a');
  MPWriter.writeExt(0x01, MemoryBufferRef(s, ""));
  EXPECT_EQ(OStream.str(), std::string("\xd5\x01") + s);
}

TEST_F(MsgPackWriter, TestWriteFixExt4) {
  std::string s(4, 'a');
  MPWriter.writeExt(0x01, MemoryBufferRef(s, ""));
  EXPECT_EQ(OStream.str(), std::string("\xd6\x01") + s);
}

TEST_F(MsgPackWriter, TestWriteFixExt8) {
  std::string s(8, 'a');
  MPWriter.writeExt(0x01, MemoryBufferRef(s, ""));
  EXPECT_EQ(OStream.str(), std::string("\xd7\x01") + s);
}

TEST_F(MsgPackWriter, TestWriteFixExt16) {
  std::string s(16, 'a');
  MPWriter.writeExt(0x01, MemoryBufferRef(s, ""));
  EXPECT_EQ(OStream.str(), std::string("\xd8\x01") + s);
}

TEST_F(MsgPackWriter, TestWriteExt8Min) {
  std::string s;
  MPWriter.writeExt(0x01, MemoryBufferRef(s, ""));
  EXPECT_EQ(OStream.str(), std::string("\xc7\x00\x01", 3) + s);
}

TEST_F(MsgPackWriter, TestWriteExt8) {
  std::string s(0x2a, 'a');
  MPWriter.writeExt(0x01, MemoryBufferRef(s, ""));
  EXPECT_EQ(OStream.str(), std::string("\xc7\x2a\x01") + s);
}

TEST_F(MsgPackWriter, TestWriteExt8Max) {
  std::string s(UINT8_MAX, 'a');
  MPWriter.writeExt(0x01, MemoryBufferRef(s, ""));
  EXPECT_EQ(OStream.str(), std::string("\xc7\xff\x01") + s);
}

TEST_F(MsgPackWriter, TestWriteExt16Min) {
  std::string s(static_cast<uint16_t>(UINT8_MAX) + 1, 'a');
  MPWriter.writeExt(0x02, MemoryBufferRef(s, ""));
  EXPECT_EQ(OStream.str(), std::string("\xc8\x01\x00\x02", 4) + s);
}

TEST_F(MsgPackWriter, TestWriteExt16) {
  std::string s(273, 'a');
  MPWriter.writeExt(0x01, MemoryBufferRef(s, ""));
  EXPECT_EQ(OStream.str(), std::string("\xc8\x01\x11\x01") + s);
}

TEST_F(MsgPackWriter, TestWriteExt16Max) {
  std::string s(UINT16_MAX, 'a');
  MPWriter.writeExt(0x01, MemoryBufferRef(s, ""));
  EXPECT_EQ(OStream.str(), std::string("\xc8\xff\xff\x01") + s);
}

TEST_F(MsgPackWriter, TestWriteExt32Min) {
  std::string s(static_cast<uint32_t>(UINT16_MAX) + 1, 'a');
  MPWriter.writeExt(0x02, MemoryBufferRef(s, ""));
  EXPECT_EQ(OStream.str(), std::string("\xc9\x00\x01\x00\x00\x02", 6) + s);
}

TEST_F(MsgPackWriter, TestWriteCompatibleNoStr8) {
  Writer CompatWriter(OStream, true);
  std::string s(32, 'a');
  CompatWriter.write(s);
  EXPECT_EQ(OStream.str(), std::string("\xda\x00\x20", 3) + s);
}

TEST_F(MsgPackWriter, TestWriteCompatibleNoBin) {
  Writer CompatWriter(OStream, true);
  std::string s;

#ifdef GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
  EXPECT_DEATH(CompatWriter.write(MemoryBufferRef(s, "")), "compatible mode");
#endif
#endif
}
