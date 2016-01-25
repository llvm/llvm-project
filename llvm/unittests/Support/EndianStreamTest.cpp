//===- unittests/Support/EndianStreamTest.cpp - EndianStream.h tests ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/DataTypes.h"
#include "gtest/gtest.h"
using namespace llvm;
using namespace support;

namespace {

TEST(EndianStream, WriteInt32LE) {
  SmallString<16> data;

  {
    raw_svector_ostream OS(data);
    endian::Writer<little> LE(OS);
    LE.write(static_cast<int32_t>(-1362446643));
  }

  EXPECT_EQ(static_cast<uint8_t>(data[0]), 0xCD);
  EXPECT_EQ(static_cast<uint8_t>(data[1]), 0xB6);
  EXPECT_EQ(static_cast<uint8_t>(data[2]), 0xCA);
  EXPECT_EQ(static_cast<uint8_t>(data[3]), 0xAE);
}

TEST(EndianStream, WriteInt32BE) {
  SmallVector<char, 16> data;

  {
    raw_svector_ostream OS(data);
    endian::Writer<big> BE(OS);
    BE.write(static_cast<int32_t>(-1362446643));
  }

  EXPECT_EQ(static_cast<uint8_t>(data[0]), 0xAE);
  EXPECT_EQ(static_cast<uint8_t>(data[1]), 0xCA);
  EXPECT_EQ(static_cast<uint8_t>(data[2]), 0xB6);
  EXPECT_EQ(static_cast<uint8_t>(data[3]), 0xCD);
}


TEST(EndianStream, WriteFloatLE) {
  SmallString<16> data;

  {
    raw_svector_ostream OS(data);
    endian::Writer<little> LE(OS);
    LE.write(12345.0f);
  }

  EXPECT_EQ(static_cast<uint8_t>(data[0]), 0x00);
  EXPECT_EQ(static_cast<uint8_t>(data[1]), 0xE4);
  EXPECT_EQ(static_cast<uint8_t>(data[2]), 0x40);
  EXPECT_EQ(static_cast<uint8_t>(data[3]), 0x46);
}

TEST(EndianStream, WriteFloatBE) {
  SmallVector<char, 16> data;

  {
    raw_svector_ostream OS(data);
    endian::Writer<big> BE(OS);
    BE.write(12345.0f);
  }

  EXPECT_EQ(static_cast<uint8_t>(data[0]), 0x46);
  EXPECT_EQ(static_cast<uint8_t>(data[1]), 0x40);
  EXPECT_EQ(static_cast<uint8_t>(data[2]), 0xE4);
  EXPECT_EQ(static_cast<uint8_t>(data[3]), 0x00);
}

TEST(EndianStream, WriteInt64LE) {
  SmallString<16> data;

  {
    raw_svector_ostream OS(data);
    endian::Writer<little> LE(OS);
    LE.write(static_cast<int64_t>(-136244664332342323));
  }

  EXPECT_EQ(static_cast<uint8_t>(data[0]), 0xCD);
  EXPECT_EQ(static_cast<uint8_t>(data[1]), 0xAB);
  EXPECT_EQ(static_cast<uint8_t>(data[2]), 0xED);
  EXPECT_EQ(static_cast<uint8_t>(data[3]), 0x1B);
  EXPECT_EQ(static_cast<uint8_t>(data[4]), 0x33);
  EXPECT_EQ(static_cast<uint8_t>(data[5]), 0xF6);
  EXPECT_EQ(static_cast<uint8_t>(data[6]), 0x1B);
  EXPECT_EQ(static_cast<uint8_t>(data[7]), 0xFE);
}

TEST(EndianStream, WriteInt64BE) {
  SmallVector<char, 16> data;

  {
    raw_svector_ostream OS(data);
    endian::Writer<big> BE(OS);
    BE.write(static_cast<int64_t>(-136244664332342323));
  }

  EXPECT_EQ(static_cast<uint8_t>(data[0]), 0xFE);
  EXPECT_EQ(static_cast<uint8_t>(data[1]), 0x1B);
  EXPECT_EQ(static_cast<uint8_t>(data[2]), 0xF6);
  EXPECT_EQ(static_cast<uint8_t>(data[3]), 0x33);
  EXPECT_EQ(static_cast<uint8_t>(data[4]), 0x1B);
  EXPECT_EQ(static_cast<uint8_t>(data[5]), 0xED);
  EXPECT_EQ(static_cast<uint8_t>(data[6]), 0xAB);
  EXPECT_EQ(static_cast<uint8_t>(data[7]), 0xCD);
}

TEST(EndianStream, WriteDoubleLE) {
  SmallString<16> data;

  {
    raw_svector_ostream OS(data);
    endian::Writer<little> LE(OS);
    LE.write(-2349214918.58107);
  }

  EXPECT_EQ(static_cast<uint8_t>(data[0]), 0x20);
  EXPECT_EQ(static_cast<uint8_t>(data[1]), 0x98);
  EXPECT_EQ(static_cast<uint8_t>(data[2]), 0xD2);
  EXPECT_EQ(static_cast<uint8_t>(data[3]), 0x98);
  EXPECT_EQ(static_cast<uint8_t>(data[4]), 0xC5);
  EXPECT_EQ(static_cast<uint8_t>(data[5]), 0x80);
  EXPECT_EQ(static_cast<uint8_t>(data[6]), 0xE1);
  EXPECT_EQ(static_cast<uint8_t>(data[7]), 0xC1);
}

TEST(EndianStream, WriteDoubleBE) {
  SmallVector<char, 16> data;

  {
    raw_svector_ostream OS(data);
    endian::Writer<big> BE(OS);
    BE.write(-2349214918.58107);
  }

  EXPECT_EQ(static_cast<uint8_t>(data[0]), 0xC1);
  EXPECT_EQ(static_cast<uint8_t>(data[1]), 0xE1);
  EXPECT_EQ(static_cast<uint8_t>(data[2]), 0x80);
  EXPECT_EQ(static_cast<uint8_t>(data[3]), 0xC5);
  EXPECT_EQ(static_cast<uint8_t>(data[4]), 0x98);
  EXPECT_EQ(static_cast<uint8_t>(data[5]), 0xD2);
  EXPECT_EQ(static_cast<uint8_t>(data[6]), 0x98);
  EXPECT_EQ(static_cast<uint8_t>(data[7]), 0x20);
}

TEST(EndianStream, WriteArrayLE) {
  SmallString<16> Data;

  {
    raw_svector_ostream OS(Data);
    endian::Writer<little> LE(OS);
    LE.write<uint16_t>({0x1234, 0x5678});
  }

  EXPECT_EQ(static_cast<uint8_t>(Data[0]), 0x34);
  EXPECT_EQ(static_cast<uint8_t>(Data[1]), 0x12);
  EXPECT_EQ(static_cast<uint8_t>(Data[2]), 0x78);
  EXPECT_EQ(static_cast<uint8_t>(Data[3]), 0x56);
}

TEST(EndianStream, WriteVectorLE) {
  SmallString<16> Data;

  {
    raw_svector_ostream OS(Data);
    endian::Writer<little> LE(OS);
    std::vector<uint16_t> Vec{0x1234, 0x5678};
    LE.write<uint16_t>(Vec);
  }

  EXPECT_EQ(static_cast<uint8_t>(Data[0]), 0x34);
  EXPECT_EQ(static_cast<uint8_t>(Data[1]), 0x12);
  EXPECT_EQ(static_cast<uint8_t>(Data[2]), 0x78);
  EXPECT_EQ(static_cast<uint8_t>(Data[3]), 0x56);
}

TEST(EndianStream, WriteFloatArrayLE) {
  SmallString<16> Data;

  {
    raw_svector_ostream OS(Data);
    endian::Writer<little> LE(OS);
    LE.write<float>({12345.0f, 12346.0f});
  }

  EXPECT_EQ(static_cast<uint8_t>(Data[0]), 0x00);
  EXPECT_EQ(static_cast<uint8_t>(Data[1]), 0xE4);
  EXPECT_EQ(static_cast<uint8_t>(Data[2]), 0x40);
  EXPECT_EQ(static_cast<uint8_t>(Data[3]), 0x46);

  EXPECT_EQ(static_cast<uint8_t>(Data[4]), 0x00);
  EXPECT_EQ(static_cast<uint8_t>(Data[5]), 0xE8);
  EXPECT_EQ(static_cast<uint8_t>(Data[6]), 0x40);
  EXPECT_EQ(static_cast<uint8_t>(Data[7]), 0x46);
}


} // end anon namespace
