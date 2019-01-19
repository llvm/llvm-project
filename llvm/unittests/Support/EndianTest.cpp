//===- unittests/Support/EndianTest.cpp - Endian.h tests ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Endian.h"
#include "llvm/Support/DataTypes.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <ctime>
using namespace llvm;
using namespace support;

#undef max

namespace {

TEST(Endian, Read) {
  // These are 5 bytes so we can be sure at least one of the reads is unaligned.
  unsigned char bigval[] = {0x00, 0x01, 0x02, 0x03, 0x04};
  unsigned char littleval[] = {0x00, 0x04, 0x03, 0x02, 0x01};
  int32_t BigAsHost = 0x00010203;
  EXPECT_EQ(BigAsHost, (endian::read<int32_t, big, unaligned>(bigval)));
  int32_t LittleAsHost = 0x02030400;
  EXPECT_EQ(LittleAsHost,(endian::read<int32_t, little, unaligned>(littleval)));

  EXPECT_EQ((endian::read<int32_t, big, unaligned>(bigval + 1)),
            (endian::read<int32_t, little, unaligned>(littleval + 1)));
}

TEST(Endian, ReadBitAligned) {
  // Simple test to make sure we properly pull out the 0x0 word.
  unsigned char littleval[] = {0x3f, 0x00, 0x00, 0x00, 0xc0, 0xff, 0xff, 0xff};
  unsigned char bigval[] = {0x00, 0x00, 0x00, 0x3f, 0xff, 0xff, 0xff, 0xc0};
  EXPECT_EQ(
      (endian::readAtBitAlignment<int, little, unaligned>(&littleval[0], 6)),
      0x0);
  EXPECT_EQ((endian::readAtBitAlignment<int, big, unaligned>(&bigval[0], 6)),
            0x0);
  // Test to make sure that signed right shift of 0xf0000000 is masked
  // properly.
  unsigned char littleval2[] = {0x00, 0x00, 0x00, 0xf0, 0x00, 0x00, 0x00, 0x00};
  unsigned char bigval2[] = {0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  EXPECT_EQ(
      (endian::readAtBitAlignment<int, little, unaligned>(&littleval2[0], 4)),
      0x0f000000);
  EXPECT_EQ((endian::readAtBitAlignment<int, big, unaligned>(&bigval2[0], 4)),
            0x0f000000);
  // Test to make sure left shift of start bit doesn't overflow.
  EXPECT_EQ(
      (endian::readAtBitAlignment<int, little, unaligned>(&littleval2[0], 1)),
      0x78000000);
  EXPECT_EQ((endian::readAtBitAlignment<int, big, unaligned>(&bigval2[0], 1)),
            0x78000000);
  // Test to make sure 64-bit int doesn't overflow.
  unsigned char littleval3[] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0,
                                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  unsigned char bigval3[] = {0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                             0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  EXPECT_EQ((endian::readAtBitAlignment<int64_t, little, unaligned>(
                &littleval3[0], 4)),
            0x0f00000000000000);
  EXPECT_EQ(
      (endian::readAtBitAlignment<int64_t, big, unaligned>(&bigval3[0], 4)),
      0x0f00000000000000);
}

TEST(Endian, WriteBitAligned) {
  // This test ensures that signed right shift of 0xffffaa is masked
  // properly.
  unsigned char bigval[8] = {0x00};
  endian::writeAtBitAlignment<int32_t, big, unaligned>(bigval, (int)0xffffaaaa,
                                                       4);
  EXPECT_EQ(bigval[0], 0xff);
  EXPECT_EQ(bigval[1], 0xfa);
  EXPECT_EQ(bigval[2], 0xaa);
  EXPECT_EQ(bigval[3], 0xa0);
  EXPECT_EQ(bigval[4], 0x00);
  EXPECT_EQ(bigval[5], 0x00);
  EXPECT_EQ(bigval[6], 0x00);
  EXPECT_EQ(bigval[7], 0x0f);

  unsigned char littleval[8] = {0x00};
  endian::writeAtBitAlignment<int32_t, little, unaligned>(littleval,
                                                          (int)0xffffaaaa, 4);
  EXPECT_EQ(littleval[0], 0xa0);
  EXPECT_EQ(littleval[1], 0xaa);
  EXPECT_EQ(littleval[2], 0xfa);
  EXPECT_EQ(littleval[3], 0xff);
  EXPECT_EQ(littleval[4], 0x0f);
  EXPECT_EQ(littleval[5], 0x00);
  EXPECT_EQ(littleval[6], 0x00);
  EXPECT_EQ(littleval[7], 0x00);

  // This test makes sure 1<<31 doesn't overflow.
  // Test to make sure left shift of start bit doesn't overflow.
  unsigned char bigval2[8] = {0x00};
  endian::writeAtBitAlignment<int32_t, big, unaligned>(bigval2, (int)0xffffffff,
                                                       1);
  EXPECT_EQ(bigval2[0], 0xff);
  EXPECT_EQ(bigval2[1], 0xff);
  EXPECT_EQ(bigval2[2], 0xff);
  EXPECT_EQ(bigval2[3], 0xfe);
  EXPECT_EQ(bigval2[4], 0x00);
  EXPECT_EQ(bigval2[5], 0x00);
  EXPECT_EQ(bigval2[6], 0x00);
  EXPECT_EQ(bigval2[7], 0x01);

  unsigned char littleval2[8] = {0x00};
  endian::writeAtBitAlignment<int32_t, little, unaligned>(littleval2,
                                                          (int)0xffffffff, 1);
  EXPECT_EQ(littleval2[0], 0xfe);
  EXPECT_EQ(littleval2[1], 0xff);
  EXPECT_EQ(littleval2[2], 0xff);
  EXPECT_EQ(littleval2[3], 0xff);
  EXPECT_EQ(littleval2[4], 0x01);
  EXPECT_EQ(littleval2[5], 0x00);
  EXPECT_EQ(littleval2[6], 0x00);
  EXPECT_EQ(littleval2[7], 0x00);

  // Test to make sure 64-bit int doesn't overflow.
  unsigned char bigval64[16] = {0x00};
  endian::writeAtBitAlignment<int64_t, big, unaligned>(
      bigval64, (int64_t)0xffffffffffffffff, 1);
  EXPECT_EQ(bigval64[0], 0xff);
  EXPECT_EQ(bigval64[1], 0xff);
  EXPECT_EQ(bigval64[2], 0xff);
  EXPECT_EQ(bigval64[3], 0xff);
  EXPECT_EQ(bigval64[4], 0xff);
  EXPECT_EQ(bigval64[5], 0xff);
  EXPECT_EQ(bigval64[6], 0xff);
  EXPECT_EQ(bigval64[7], 0xfe);
  EXPECT_EQ(bigval64[8], 0x00);
  EXPECT_EQ(bigval64[9], 0x00);
  EXPECT_EQ(bigval64[10], 0x00);
  EXPECT_EQ(bigval64[11], 0x00);
  EXPECT_EQ(bigval64[12], 0x00);
  EXPECT_EQ(bigval64[13], 0x00);
  EXPECT_EQ(bigval64[14], 0x00);
  EXPECT_EQ(bigval64[15], 0x01);

  unsigned char littleval64[16] = {0x00};
  endian::writeAtBitAlignment<int64_t, little, unaligned>(
      littleval64, (int64_t)0xffffffffffffffff, 1);
  EXPECT_EQ(littleval64[0], 0xfe);
  EXPECT_EQ(littleval64[1], 0xff);
  EXPECT_EQ(littleval64[2], 0xff);
  EXPECT_EQ(littleval64[3], 0xff);
  EXPECT_EQ(littleval64[4], 0xff);
  EXPECT_EQ(littleval64[5], 0xff);
  EXPECT_EQ(littleval64[6], 0xff);
  EXPECT_EQ(littleval64[7], 0xff);
  EXPECT_EQ(littleval64[8], 0x01);
  EXPECT_EQ(littleval64[9], 0x00);
  EXPECT_EQ(littleval64[10], 0x00);
  EXPECT_EQ(littleval64[11], 0x00);
  EXPECT_EQ(littleval64[12], 0x00);
  EXPECT_EQ(littleval64[13], 0x00);
  EXPECT_EQ(littleval64[14], 0x00);
  EXPECT_EQ(littleval64[15], 0x00);
}

TEST(Endian, Write) {
  unsigned char data[5];
  endian::write<int32_t, big, unaligned>(data, -1362446643);
  EXPECT_EQ(data[0], 0xAE);
  EXPECT_EQ(data[1], 0xCA);
  EXPECT_EQ(data[2], 0xB6);
  EXPECT_EQ(data[3], 0xCD);
  endian::write<int32_t, big, unaligned>(data + 1, -1362446643);
  EXPECT_EQ(data[1], 0xAE);
  EXPECT_EQ(data[2], 0xCA);
  EXPECT_EQ(data[3], 0xB6);
  EXPECT_EQ(data[4], 0xCD);

  endian::write<int32_t, little, unaligned>(data, -1362446643);
  EXPECT_EQ(data[0], 0xCD);
  EXPECT_EQ(data[1], 0xB6);
  EXPECT_EQ(data[2], 0xCA);
  EXPECT_EQ(data[3], 0xAE);
  endian::write<int32_t, little, unaligned>(data + 1, -1362446643);
  EXPECT_EQ(data[1], 0xCD);
  EXPECT_EQ(data[2], 0xB6);
  EXPECT_EQ(data[3], 0xCA);
  EXPECT_EQ(data[4], 0xAE);
}

TEST(Endian, PackedEndianSpecificIntegral) {
  // These are 5 bytes so we can be sure at least one of the reads is unaligned.
  unsigned char big[] = {0x00, 0x01, 0x02, 0x03, 0x04};
  unsigned char little[] = {0x00, 0x04, 0x03, 0x02, 0x01};
  big32_t    *big_val    =
    reinterpret_cast<big32_t *>(big + 1);
  little32_t *little_val =
    reinterpret_cast<little32_t *>(little + 1);

  EXPECT_EQ(*big_val, *little_val);
}

} // end anon namespace
