//===- WholeProgramDevirt.cpp - Unit tests for whole-program devirt -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/WholeProgramDevirt.h"
#include "llvm/ADT/ArrayRef.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace wholeprogramdevirt;

TEST(WholeProgramDevirt, findLowestOffset) {
  VTableBits VT1;
  VT1.ObjectSize = 8;
  VT1.Before.BytesUsed = {1 << 0};
  VT1.After.BytesUsed = {1 << 1};

  VTableBits VT2;
  VT2.ObjectSize = 8;
  VT2.Before.BytesUsed = {1 << 1};
  VT2.After.BytesUsed = {1 << 0};

  TypeMemberInfo TM1{&VT1, 0};
  TypeMemberInfo TM2{&VT2, 0};
  VirtualCallTarget Targets[] = {
    {&TM1, /*IsBigEndian=*/false},
    {&TM2, /*IsBigEndian=*/false},
  };

  EXPECT_EQ(2ull, findLowestOffset(Targets, /*IsAfter=*/false, 1));
  EXPECT_EQ(66ull, findLowestOffset(Targets, /*IsAfter=*/true, 1));

  EXPECT_EQ(8ull, findLowestOffset(Targets, /*IsAfter=*/false, 8));
  EXPECT_EQ(72ull, findLowestOffset(Targets, /*IsAfter=*/true, 8));

  TM1.Offset = 4;
  EXPECT_EQ(33ull, findLowestOffset(Targets, /*IsAfter=*/false, 1));
  EXPECT_EQ(65ull, findLowestOffset(Targets, /*IsAfter=*/true, 1));

  EXPECT_EQ(40ull, findLowestOffset(Targets, /*IsAfter=*/false, 8));
  EXPECT_EQ(72ull, findLowestOffset(Targets, /*IsAfter=*/true, 8));

  TM1.Offset = 8;
  TM2.Offset = 8;
  EXPECT_EQ(66ull, findLowestOffset(Targets, /*IsAfter=*/false, 1));
  EXPECT_EQ(2ull, findLowestOffset(Targets, /*IsAfter=*/true, 1));

  EXPECT_EQ(72ull, findLowestOffset(Targets, /*IsAfter=*/false, 8));
  EXPECT_EQ(8ull, findLowestOffset(Targets, /*IsAfter=*/true, 8));

  VT1.After.BytesUsed = {0xff, 0, 0, 0, 0xff};
  VT2.After.BytesUsed = {0xff, 1, 0, 0, 0};
  EXPECT_EQ(16ull, findLowestOffset(Targets, /*IsAfter=*/true, 16));
  EXPECT_EQ(40ull, findLowestOffset(Targets, /*IsAfter=*/true, 32));
}

TEST(WholeProgramDevirt, setReturnValues) {
  VTableBits VT1;
  VT1.ObjectSize = 8;

  VTableBits VT2;
  VT2.ObjectSize = 8;

  TypeMemberInfo TM1{&VT1, 0};
  TypeMemberInfo TM2{&VT2, 0};
  VirtualCallTarget Targets[] = {
    {&TM1, /*IsBigEndian=*/false},
    {&TM2, /*IsBigEndian=*/false},
  };

  TM1.Offset = 4;
  TM2.Offset = 4;

  int64_t OffsetByte;
  uint64_t OffsetBit;

  Targets[0].RetVal = 1;
  Targets[1].RetVal = 0;
  setBeforeReturnValues(Targets, 32, 1, OffsetByte, OffsetBit);
  EXPECT_EQ(-5ll, OffsetByte);
  EXPECT_EQ(0ull, OffsetBit);
  EXPECT_EQ(std::vector<uint8_t>{1}, VT1.Before.Bytes);
  EXPECT_EQ(std::vector<uint8_t>{1}, VT1.Before.BytesUsed);
  EXPECT_EQ(std::vector<uint8_t>{0}, VT2.Before.Bytes);
  EXPECT_EQ(std::vector<uint8_t>{1}, VT2.Before.BytesUsed);

  Targets[0].RetVal = 0;
  Targets[1].RetVal = 1;
  setBeforeReturnValues(Targets, 39, 1, OffsetByte, OffsetBit);
  EXPECT_EQ(-5ll, OffsetByte);
  EXPECT_EQ(7ull, OffsetBit);
  EXPECT_EQ(std::vector<uint8_t>{1}, VT1.Before.Bytes);
  EXPECT_EQ(std::vector<uint8_t>{0x81}, VT1.Before.BytesUsed);
  EXPECT_EQ(std::vector<uint8_t>{0x80}, VT2.Before.Bytes);
  EXPECT_EQ(std::vector<uint8_t>{0x81}, VT2.Before.BytesUsed);

  Targets[0].RetVal = 12;
  Targets[1].RetVal = 34;
  setBeforeReturnValues(Targets, 40, 8, OffsetByte, OffsetBit);
  EXPECT_EQ(-6ll, OffsetByte);
  EXPECT_EQ(0ull, OffsetBit);
  EXPECT_EQ((std::vector<uint8_t>{1, 12}), VT1.Before.Bytes);
  EXPECT_EQ((std::vector<uint8_t>{0x81, 0xff}), VT1.Before.BytesUsed);
  EXPECT_EQ((std::vector<uint8_t>{0x80, 34}), VT2.Before.Bytes);
  EXPECT_EQ((std::vector<uint8_t>{0x81, 0xff}), VT2.Before.BytesUsed);

  Targets[0].RetVal = 56;
  Targets[1].RetVal = 78;
  setBeforeReturnValues(Targets, 48, 16, OffsetByte, OffsetBit);
  EXPECT_EQ(-8ll, OffsetByte);
  EXPECT_EQ(0ull, OffsetBit);
  EXPECT_EQ((std::vector<uint8_t>{1, 12, 0, 56}), VT1.Before.Bytes);
  EXPECT_EQ((std::vector<uint8_t>{0x81, 0xff, 0xff, 0xff}),
            VT1.Before.BytesUsed);
  EXPECT_EQ((std::vector<uint8_t>{0x80, 34, 0, 78}), VT2.Before.Bytes);
  EXPECT_EQ((std::vector<uint8_t>{0x81, 0xff, 0xff, 0xff}),
            VT2.Before.BytesUsed);

  Targets[0].RetVal = 1;
  Targets[1].RetVal = 0;
  setAfterReturnValues(Targets, 32, 1, OffsetByte, OffsetBit);
  EXPECT_EQ(4ll, OffsetByte);
  EXPECT_EQ(0ull, OffsetBit);
  EXPECT_EQ(std::vector<uint8_t>{1}, VT1.After.Bytes);
  EXPECT_EQ(std::vector<uint8_t>{1}, VT1.After.BytesUsed);
  EXPECT_EQ(std::vector<uint8_t>{0}, VT2.After.Bytes);
  EXPECT_EQ(std::vector<uint8_t>{1}, VT2.After.BytesUsed);

  Targets[0].RetVal = 0;
  Targets[1].RetVal = 1;
  setAfterReturnValues(Targets, 39, 1, OffsetByte, OffsetBit);
  EXPECT_EQ(4ll, OffsetByte);
  EXPECT_EQ(7ull, OffsetBit);
  EXPECT_EQ(std::vector<uint8_t>{1}, VT1.After.Bytes);
  EXPECT_EQ(std::vector<uint8_t>{0x81}, VT1.After.BytesUsed);
  EXPECT_EQ(std::vector<uint8_t>{0x80}, VT2.After.Bytes);
  EXPECT_EQ(std::vector<uint8_t>{0x81}, VT2.After.BytesUsed);

  Targets[0].RetVal = 12;
  Targets[1].RetVal = 34;
  setAfterReturnValues(Targets, 40, 8, OffsetByte, OffsetBit);
  EXPECT_EQ(5ll, OffsetByte);
  EXPECT_EQ(0ull, OffsetBit);
  EXPECT_EQ((std::vector<uint8_t>{1, 12}), VT1.After.Bytes);
  EXPECT_EQ((std::vector<uint8_t>{0x81, 0xff}), VT1.After.BytesUsed);
  EXPECT_EQ((std::vector<uint8_t>{0x80, 34}), VT2.After.Bytes);
  EXPECT_EQ((std::vector<uint8_t>{0x81, 0xff}), VT2.After.BytesUsed);

  Targets[0].RetVal = 56;
  Targets[1].RetVal = 78;
  setAfterReturnValues(Targets, 48, 16, OffsetByte, OffsetBit);
  EXPECT_EQ(6ll, OffsetByte);
  EXPECT_EQ(0ull, OffsetBit);
  EXPECT_EQ((std::vector<uint8_t>{1, 12, 56, 0}), VT1.After.Bytes);
  EXPECT_EQ((std::vector<uint8_t>{0x81, 0xff, 0xff, 0xff}),
            VT1.After.BytesUsed);
  EXPECT_EQ((std::vector<uint8_t>{0x80, 34, 78, 0}), VT2.After.Bytes);
  EXPECT_EQ((std::vector<uint8_t>{0x81, 0xff, 0xff, 0xff}),
            VT2.After.BytesUsed);
}
