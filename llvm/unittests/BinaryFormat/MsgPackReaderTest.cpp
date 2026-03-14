//===- MsgPackReaderTest.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/MsgPackReader.h"
#include "llvm/BinaryFormat/MsgPack.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::msgpack;

struct MsgPackReader : testing::Test {
  std::string Buffer;
  Object Obj;
};

TEST_F(MsgPackReader, TestReadMultiple) {
  Buffer = "\xc0\xc2";
  Reader MPReader(Buffer);
  {
    auto ContinueOrErr = MPReader.read(Obj);
    EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
    EXPECT_TRUE(*ContinueOrErr);
    EXPECT_EQ(Obj.Kind, Type::Nil);
  }
  {
    auto ContinueOrErr = MPReader.read(Obj);
    EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
    EXPECT_TRUE(*ContinueOrErr);
    EXPECT_EQ(Obj.Kind, Type::Boolean);
    EXPECT_EQ(Obj.Bool, false);
  }
  {
    auto ContinueOrErr = MPReader.read(Obj);
    EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
    EXPECT_FALSE(*ContinueOrErr);
  }
}

TEST_F(MsgPackReader, TestReadNil) {
  Buffer = "\xc0";
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Nil);
}

TEST_F(MsgPackReader, TestReadBoolFalse) {
  Buffer = "\xc2";
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Boolean);
  EXPECT_EQ(Obj.Bool, false);
}

TEST_F(MsgPackReader, TestReadBoolTrue) {
  Buffer = "\xc3";
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Boolean);
  EXPECT_EQ(Obj.Bool, true);
}

TEST_F(MsgPackReader, TestReadFixNegativeInt) {
  // Positive values will be written in a UInt form, so max FixNegativeInt is -1
  //
  // FixNegativeInt form bitpattern starts with 111, so min FixNegativeInt
  // is 11100000 = -32
  for (int8_t i = -1; i >= -32; --i) {
    Buffer.assign(1, static_cast<char>(i));
    Reader MPReader(Buffer);
    auto ContinueOrErr = MPReader.read(Obj);
    EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
    EXPECT_TRUE(*ContinueOrErr);
    EXPECT_EQ(Obj.Kind, Type::Int);
    EXPECT_EQ(Obj.Int, i);
  }
}

TEST_F(MsgPackReader, TestReadInt8Max) {
  Buffer = "\xd0\x7f";
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Int);
  EXPECT_EQ(Obj.Int, INT8_MAX);
}

TEST_F(MsgPackReader, TestReadInt8Zero) {
  Buffer.assign("\xd0\x00", 2);
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Int);
  EXPECT_EQ(Obj.Int, 0);
}

TEST_F(MsgPackReader, TestReadInt8Min) {
  Buffer = "\xd0\x80";
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Int);
  EXPECT_EQ(Obj.Int, INT8_MIN);
}

TEST_F(MsgPackReader, TestReadInt16Max) {
  Buffer = "\xd1\x7f\xff";
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Int);
  EXPECT_EQ(Obj.Int, INT16_MAX);
}

TEST_F(MsgPackReader, TestReadInt16Zero) {
  Buffer.assign("\xd1\x00\x00", 3);
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Int);
  EXPECT_EQ(Obj.Int, 0);
}

TEST_F(MsgPackReader, TestReadInt16Min) {
  Buffer.assign("\xd1\x80\x00", 3);
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Int);
  EXPECT_EQ(Obj.Int, INT16_MIN);
}

TEST_F(MsgPackReader, TestReadInt32Max) {
  Buffer = "\xd2\x7f\xff\xff\xff";
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Int);
  EXPECT_EQ(Obj.Int, INT32_MAX);
}

TEST_F(MsgPackReader, TestReadInt32Zero) {
  Buffer.assign("\xd2\x00\x00\x00\x00", 5);
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Int);
  EXPECT_EQ(Obj.Int, 0);
}

TEST_F(MsgPackReader, TestReadInt32Min) {
  Buffer.assign("\xd2\x80\x00\x00\x00", 5);
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Int);
  EXPECT_EQ(Obj.Int, INT32_MIN);
}

TEST_F(MsgPackReader, TestReadInt64Max) {
  Buffer = "\xd3\x7f\xff\xff\xff\xff\xff\xff\xff";
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Int);
  EXPECT_EQ(Obj.Int, INT64_MAX);
}

TEST_F(MsgPackReader, TestReadInt64Zero) {
  Buffer.assign("\xd3\x00\x00\x00\x00\x00\x00\x00\x00", 9);
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Int);
  EXPECT_EQ(Obj.Int, 0);
}

TEST_F(MsgPackReader, TestReadInt64Min) {
  Buffer.assign("\xd3\x80\x00\x00\x00\x00\x00\x00\x00", 9);
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Int);
  EXPECT_EQ(Obj.Int, INT64_MIN);
}

TEST_F(MsgPackReader, TestReadFixPositiveInt) {
  // FixPositiveInt form bitpattern starts with 0, so max FixPositiveInt
  // is 01111111 = 127
  for (uint64_t u = 0; u <= 127; ++u) {
    Buffer.assign(1, static_cast<char>(u));
    Reader MPReader(Buffer);
    auto ContinueOrErr = MPReader.read(Obj);
    EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
    EXPECT_TRUE(*ContinueOrErr);
    EXPECT_EQ(Obj.Kind, Type::UInt);
    EXPECT_EQ(Obj.UInt, u);
  }
}

TEST_F(MsgPackReader, TestReadUInt8Zero) {
  Buffer.assign("\xcc\x00", 2);
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::UInt);
  EXPECT_EQ(Obj.UInt, 0u);
}

TEST_F(MsgPackReader, TestReadUInt8One) {
  Buffer = "\xcc\x01";
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::UInt);
  EXPECT_EQ(Obj.UInt, 1u);
}

TEST_F(MsgPackReader, TestReadUInt8Max) {
  Buffer = "\xcc\xff";
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::UInt);
  EXPECT_EQ(Obj.UInt, static_cast<uint8_t>(UINT8_MAX));
}

TEST_F(MsgPackReader, TestReadUInt16Zero) {
  Buffer.assign("\xcd\x00\x00", 3);
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::UInt);
  EXPECT_EQ(Obj.UInt, 0u);
}

TEST_F(MsgPackReader, TestReadUInt16One) {
  Buffer.assign("\xcd\x00\x01", 3);
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::UInt);
  EXPECT_EQ(Obj.UInt, 1u);
}

TEST_F(MsgPackReader, TestReadUInt16Max) {
  Buffer = "\xcd\xff\xff";
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::UInt);
  EXPECT_EQ(Obj.UInt, static_cast<uint16_t>(UINT16_MAX));
}

TEST_F(MsgPackReader, TestReadUInt32Zero) {
  Buffer.assign("\xce\x00\x00\x00\x00", 5);
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::UInt);
  EXPECT_EQ(Obj.UInt, 0u);
}

TEST_F(MsgPackReader, TestReadUInt32One) {
  Buffer.assign("\xce\x00\x00\x00\x01", 5);
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::UInt);
  EXPECT_EQ(Obj.UInt, 1u);
}

TEST_F(MsgPackReader, TestReadUInt32Max) {
  Buffer = "\xce\xff\xff\xff\xff";
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::UInt);
  EXPECT_EQ(Obj.UInt, static_cast<uint32_t>(UINT32_MAX));
}

TEST_F(MsgPackReader, TestReadUInt64Zero) {
  Buffer.assign("\xcf\x00\x00\x00\x00\x00\x00\x00\x00", 9);
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::UInt);
  EXPECT_EQ(Obj.UInt, 0u);
}

TEST_F(MsgPackReader, TestReadUInt64One) {
  Buffer.assign("\xcf\x00\x00\x00\x00\x00\x00\x00\x01", 9);
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::UInt);
  EXPECT_EQ(Obj.UInt, 1u);
}

TEST_F(MsgPackReader, TestReadUInt64Max) {
  Buffer = "\xcf\xff\xff\xff\xff\xff\xff\xff\xff";
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::UInt);
  EXPECT_EQ(Obj.UInt, static_cast<uint64_t>(UINT64_MAX));
}

TEST_F(MsgPackReader, TestReadFloat32) {
  Buffer = "\xca\xee\xee\xee\xef";
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Float);
  EXPECT_EQ(Obj.Float, -3.6973142664068907e+28f);
}

TEST_F(MsgPackReader, TestReadFloat64) {
  Buffer = "\xcb\xee\xee\xee\xee\xee\xee\xee\xef";
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Float);
  EXPECT_EQ(Obj.Float, -2.2899894549927042e+226);
}

TEST_F(MsgPackReader, TestReadFixStrZero) {
  Buffer = "\xa0";
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::String);
  EXPECT_EQ(Obj.Raw, StringRef());
}

TEST_F(MsgPackReader, TestReadFixStrOne) {
  std::string Result(1, 'a');
  Buffer = std::string("\xa1") + Result;
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::String);
  EXPECT_EQ(Obj.Raw, Result);
}

TEST_F(MsgPackReader, TestReadFixStrMax) {
  // FixStr format's size is a 5 bit unsigned integer, so max is 11111 = 31
  std::string Result(31, 'a');
  Buffer = std::string("\xbf") + Result;
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::String);
  EXPECT_EQ(Obj.Raw, Result);
}

TEST_F(MsgPackReader, TestReadStr8Zero) {
  Buffer.assign("\xd9\x00", 2);
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::String);
  EXPECT_EQ(Obj.Raw, StringRef());
}

TEST_F(MsgPackReader, TestReadStr8One) {
  std::string Result(1, 'a');
  Buffer = std::string("\xd9\x01") + Result;
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::String);
  EXPECT_EQ(Obj.Raw, Result);
}

TEST_F(MsgPackReader, TestReadStr8Max) {
  std::string Result(UINT8_MAX, 'a');
  Buffer = std::string("\xd9\xff") + Result;
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::String);
  EXPECT_EQ(Obj.Raw, Result);
}

TEST_F(MsgPackReader, TestReadStr16Zero) {
  Buffer.assign("\xda\x00\x00", 3);
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::String);
  EXPECT_EQ(Obj.Raw, StringRef());
}

TEST_F(MsgPackReader, TestReadStr16One) {
  std::string Result(1, 'a');
  Buffer = std::string("\xda\x00\x01", 3) + Result;
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::String);
  EXPECT_EQ(Obj.Raw, Result);
}

TEST_F(MsgPackReader, TestReadStr16Max) {
  std::string Result(UINT16_MAX, 'a');
  Buffer = std::string("\xda\xff\xff") + Result;
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::String);
  EXPECT_EQ(Obj.Raw, Result);
}

TEST_F(MsgPackReader, TestReadStr32Zero) {
  Buffer.assign("\xdb\x00\x00\x00\x00", 5);
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::String);
  EXPECT_EQ(Obj.Raw, StringRef());
}

TEST_F(MsgPackReader, TestReadStr32One) {
  std::string Result(1, 'a');
  Buffer = std::string("\xdb\x00\x00\x00\x01", 5) + Result;
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::String);
  EXPECT_EQ(Obj.Raw, Result);
}

TEST_F(MsgPackReader, TestReadStr32Max) {
  std::string Result(static_cast<uint32_t>(UINT16_MAX) + 1, 'a');
  Buffer = std::string("\xdb\x00\x01\x00\x00", 5) + Result;
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::String);
  EXPECT_EQ(Obj.Raw, Result);
}

TEST_F(MsgPackReader, TestReadBin8Zero) {
  Buffer.assign("\xc4\x00", 2);
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Binary);
  EXPECT_EQ(Obj.Raw, StringRef());
}

TEST_F(MsgPackReader, TestReadBin8One) {
  std::string Result(1, 'a');
  Buffer = std::string("\xc4\x01") + Result;
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Binary);
  EXPECT_EQ(Obj.Raw, Result);
}

TEST_F(MsgPackReader, TestReadBin8Max) {
  std::string Result(UINT8_MAX, 'a');
  Buffer = std::string("\xc4\xff") + Result;
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Binary);
  EXPECT_EQ(Obj.Raw, Result);
}

TEST_F(MsgPackReader, TestReadBin16Zero) {
  Buffer.assign("\xc5\x00\x00", 3);
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Binary);
  EXPECT_EQ(Obj.Raw, StringRef());
}

TEST_F(MsgPackReader, TestReadBin16One) {
  std::string Result(1, 'a');
  Buffer = std::string("\xc5\x00\x01", 3) + Result;
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Binary);
  EXPECT_EQ(Obj.Raw, Result);
}

TEST_F(MsgPackReader, TestReadBin16Max) {
  std::string Result(UINT16_MAX, 'a');
  Buffer = std::string("\xc5\xff\xff") + Result;
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Binary);
  EXPECT_EQ(Obj.Raw, Result);
}

TEST_F(MsgPackReader, TestReadBin32Zero) {
  Buffer.assign("\xc6\x00\x00\x00\x00", 5);
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Binary);
  EXPECT_EQ(Obj.Raw, StringRef());
}

TEST_F(MsgPackReader, TestReadBin32One) {
  std::string Result(1, 'a');
  Buffer = std::string("\xc6\x00\x00\x00\x01", 5) + Result;
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Binary);
  EXPECT_EQ(Obj.Raw, Result);
}

TEST_F(MsgPackReader, TestReadBin32Max) {
  std::string Result(static_cast<uint32_t>(UINT16_MAX) + 1, 'a');
  Buffer = std::string("\xc6\x00\x01\x00\x00", 5) + Result;
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Binary);
  EXPECT_EQ(Obj.Raw, Result);
}

TEST_F(MsgPackReader, TestReadFixArrayZero) {
  Buffer = "\x90";
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Array);
  EXPECT_EQ(Obj.Length, 0u);
}

TEST_F(MsgPackReader, TestReadFixArrayOne) {
  Buffer = "\x91";
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Array);
  EXPECT_EQ(Obj.Length, 1u);
}

TEST_F(MsgPackReader, TestReadFixArrayMax) {
  Buffer = "\x9f";
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Array);
  // FixArray format's size is a 4 bit unsigned integer, so max is 1111 = 15
  EXPECT_EQ(Obj.Length, 15u);
}

TEST_F(MsgPackReader, TestReadArray16Zero) {
  Buffer.assign("\xdc\x00\x00", 3);
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Array);
  EXPECT_EQ(Obj.Length, 0u);
}

TEST_F(MsgPackReader, TestReadArray16One) {
  Buffer.assign("\xdc\x00\x01", 3);
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Array);
  EXPECT_EQ(Obj.Length, 1u);
}

TEST_F(MsgPackReader, TestReadArray16Max) {
  Buffer = "\xdc\xff\xff";
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Array);
  EXPECT_EQ(Obj.Length, static_cast<uint16_t>(UINT16_MAX));
}

TEST_F(MsgPackReader, TestReadArray32Zero) {
  Buffer.assign("\xdd\x00\x00\x00\x00", 5);
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Array);
  EXPECT_EQ(Obj.Length, 0u);
}

TEST_F(MsgPackReader, TestReadArray32One) {
  Buffer.assign("\xdd\x00\x00\x00\x01", 5);
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Array);
  EXPECT_EQ(Obj.Length, 1u);
}

TEST_F(MsgPackReader, TestReadArray32Max) {
  Buffer = "\xdd\xff\xff\xff\xff";
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Array);
  EXPECT_EQ(Obj.Length, static_cast<uint32_t>(UINT32_MAX));
}

TEST_F(MsgPackReader, TestReadFixMapZero) {
  Buffer = "\x80";
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Map);
  EXPECT_EQ(Obj.Length, 0u);
}

TEST_F(MsgPackReader, TestReadFixMapOne) {
  Buffer = "\x81";
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Map);
  EXPECT_EQ(Obj.Length, 1u);
}

TEST_F(MsgPackReader, TestReadFixMapMax) {
  Buffer = "\x8f";
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Map);
  // FixMap format's size is a 4 bit unsigned integer, so max is 1111 = 15
  EXPECT_EQ(Obj.Length, 15u);
}

TEST_F(MsgPackReader, TestReadMap16Zero) {
  Buffer.assign("\xde\x00\x00", 3);
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Map);
  EXPECT_EQ(Obj.Length, 0u);
}

TEST_F(MsgPackReader, TestReadMap16One) {
  Buffer.assign("\xde\x00\x01", 3);
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Map);
  EXPECT_EQ(Obj.Length, 1u);
}

TEST_F(MsgPackReader, TestReadMap16Max) {
  Buffer = "\xde\xff\xff";
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Map);
  EXPECT_EQ(Obj.Length, static_cast<uint16_t>(UINT16_MAX));
}

TEST_F(MsgPackReader, TestReadMap32Zero) {
  Buffer.assign("\xdf\x00\x00\x00\x00", 5);
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Map);
  EXPECT_EQ(Obj.Length, 0u);
}

TEST_F(MsgPackReader, TestReadMap32One) {
  Buffer.assign("\xdf\x00\x00\x00\x01", 5);
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Map);
  EXPECT_EQ(Obj.Length, 1u);
}

TEST_F(MsgPackReader, TestReadMap32Max) {
  Buffer = "\xdf\xff\xff\xff\xff";
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Map);
  EXPECT_EQ(Obj.Length, static_cast<uint32_t>(UINT32_MAX));
}

// FixExt formats are only available for these specific lengths: 1, 2, 4, 8, 16

TEST_F(MsgPackReader, TestReadFixExt1) {
  std::string Result(1, 'a');
  Buffer = std::string("\xd4\x01") + Result;
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Extension);
  EXPECT_EQ(Obj.Extension.Type, 0x01);
  EXPECT_EQ(Obj.Extension.Bytes, Result);
}

TEST_F(MsgPackReader, TestReadFixExt2) {
  std::string Result(2, 'a');
  Buffer = std::string("\xd5\x01") + Result;
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Extension);
  EXPECT_EQ(Obj.Extension.Type, 0x01);
  EXPECT_EQ(Obj.Extension.Bytes, Result);
}

TEST_F(MsgPackReader, TestReadFixExt4) {
  std::string Result(4, 'a');
  Buffer = std::string("\xd6\x01") + Result;
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Extension);
  EXPECT_EQ(Obj.Extension.Type, 0x01);
  EXPECT_EQ(Obj.Extension.Bytes, Result);
}

TEST_F(MsgPackReader, TestReadFixExt8) {
  std::string Result(8, 'a');
  Buffer = std::string("\xd7\x01") + Result;
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Extension);
  EXPECT_EQ(Obj.Extension.Type, 0x01);
  EXPECT_EQ(Obj.Extension.Bytes, Result);
}

TEST_F(MsgPackReader, TestReadFixExt16) {
  std::string Result(16, 'a');
  Buffer = std::string("\xd8\x01") + Result;
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Extension);
  EXPECT_EQ(Obj.Extension.Type, 0x01);
  EXPECT_EQ(Obj.Extension.Bytes, Result);
}

TEST_F(MsgPackReader, TestReadExt8Min) {
  // There are fix variants for sizes 1 and 2
  Buffer.assign("\xc7\x00\x01", 3);
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Extension);
  EXPECT_EQ(Obj.Extension.Type, 0x01);
  EXPECT_EQ(Obj.Extension.Bytes, StringRef());
}

TEST_F(MsgPackReader, TestReadExt8Max) {
  std::string Result(UINT8_MAX, 'a');
  Buffer = std::string("\xc7\xff\x01", 3) + Result;
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Extension);
  EXPECT_EQ(Obj.Extension.Type, 0x01);
  EXPECT_EQ(Obj.Extension.Bytes, Result);
}

TEST_F(MsgPackReader, TestReadExt16Min) {
  std::string Result(static_cast<uint16_t>(UINT8_MAX) + 1, 'a');
  Buffer = std::string("\xc8\x01\x00\x01", 4) + Result;
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Extension);
  EXPECT_EQ(Obj.Extension.Type, 0x01);
  EXPECT_EQ(Obj.Extension.Bytes, Result);
}

TEST_F(MsgPackReader, TestReadExt16Max) {
  std::string Result(UINT16_MAX, 'a');
  Buffer = std::string("\xc8\xff\xff\x01") + Result;
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Extension);
  EXPECT_EQ(Obj.Extension.Type, 0x01);
  EXPECT_EQ(Obj.Extension.Bytes, Result);
}

TEST_F(MsgPackReader, TestReadExt32Min) {
  std::string Result(static_cast<uint32_t>(UINT16_MAX) + 1, 'a');
  Buffer = std::string("\xc9\x00\x01\x00\x00\x01", 6) + Result;
  Reader MPReader(Buffer);
  auto ContinueOrErr = MPReader.read(Obj);
  EXPECT_TRUE(static_cast<bool>(ContinueOrErr));
  EXPECT_TRUE(*ContinueOrErr);
  EXPECT_EQ(Obj.Kind, Type::Extension);
  EXPECT_EQ(Obj.Extension.Type, 0x01);
  EXPECT_EQ(Obj.Extension.Bytes, Result);
}
