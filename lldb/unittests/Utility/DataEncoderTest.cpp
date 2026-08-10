//===-- DataEncoderTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Utility/DataEncoder.h"
#include "llvm/ADT/ArrayRef.h"
#include <vector>
using namespace lldb_private;
using namespace llvm;

TEST(DataEncoderTest, PutU8) {
  const std::vector<uint8_t> init = {1, 2, 3, 4, 5, 6, 7, 8};
  const uint32_t addr_size = 4;

  uint32_t offset = 0;
  DataEncoder encoder(init.data(), init.size(), lldb::eByteOrderLittle,
                      addr_size);
  offset = encoder.PutU8(offset, 11);
  ASSERT_EQ(offset, 1U);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({11, 2, 3, 4, 5, 6, 7, 8}));
  offset = encoder.PutU8(offset, 12);
  ASSERT_EQ(offset, 2U);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({11, 12, 3, 4, 5, 6, 7, 8}));
  offset = encoder.PutU8(offset, 13);
  ASSERT_EQ(offset, 3U);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({11, 12, 13, 4, 5, 6, 7, 8}));
  offset = encoder.PutU8(offset, 14);
  ASSERT_EQ(offset, 4U);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({11, 12, 13, 14, 5, 6, 7, 8}));
  // Check that putting a number to an invalid offset doesn't work and returns
  // an error offset and doesn't modify the buffer.
  ASSERT_EQ(encoder.PutU8(init.size(), 15), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({11, 12, 13, 14, 5, 6, 7, 8}));
}

TEST(DataEncoderTest, AppendUnsignedLittle) {
  const uint32_t addr_size = 4;
  DataEncoder encoder(lldb::eByteOrderLittle, addr_size);
  encoder.AppendU8(0x11);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({0x11}));
  encoder.AppendU16(0x2233);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({0x11, 0x33, 0x22}));
  encoder.AppendU32(0x44556677);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x11, 0x33, 0x22, 0x77, 0x66, 0x55, 0x44}));
  encoder.AppendU64(0x8899AABBCCDDEEFF);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x11, 0x33, 0x22, 0x77, 0x66, 0x55, 0x44,
                               0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA, 0x99, 0x88}));
  encoder.AppendU64(0x8899AABBCCDDEEFF);
}

TEST(DataEncoderTest, AppendUnsignedBig) {
  const uint32_t addr_size = 4;
  DataEncoder encoder(lldb::eByteOrderBig, addr_size);
  encoder.AppendU8(0x11);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({0x11}));
  encoder.AppendU16(0x2233);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({0x11, 0x22, 0x33}));
  encoder.AppendU32(0x44556677);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77}));
  encoder.AppendU64(0x8899AABBCCDDEEFF);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                               0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF}));
}

TEST(DataEncoderTest, AppendAddress4Little) {
  const uint32_t addr_size = 4;
  DataEncoder encoder(lldb::eByteOrderLittle, addr_size);
  encoder.AppendAddress(0x11223344);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({0x44, 0x33, 0x22, 0x11}));
  encoder.AppendAddress(0x55);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x44, 0x33, 0x22, 0x11, 0x55, 0x00, 0x00, 0x00}));
}

TEST(DataEncoderTest, AppendAddress4Big) {
  const uint32_t addr_size = 4;
  DataEncoder encoder(lldb::eByteOrderBig, addr_size);
  encoder.AppendAddress(0x11223344);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({0x11, 0x22, 0x33, 0x44}));
  encoder.AppendAddress(0x55);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x11, 0x22, 0x33, 0x44, 0x00, 0x00, 0x00, 0x55}));
}

TEST(DataEncoderTest, AppendAddress8Little) {
  const uint32_t addr_size = 8;
  DataEncoder encoder(lldb::eByteOrderLittle, addr_size);
  encoder.AppendAddress(0x11223344);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x44, 0x33, 0x22, 0x11, 0x00, 0x00, 0x00, 0x00}));
  encoder.AppendAddress(0x5566778899AABBCC);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x44, 0x33, 0x22, 0x11, 0x00, 0x00, 0x00, 0x00,
                               0xCC, 0xBB, 0xAA, 0x99, 0x88, 0x77, 0x66, 0x55}));
}

TEST(DataEncoderTest, AppendAddress8Big) {
  const uint32_t addr_size = 8;
  DataEncoder encoder(lldb::eByteOrderBig, addr_size);
  encoder.AppendAddress(0x11223344);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x00, 0x00, 0x00, 0x00, 0x11, 0x22, 0x33, 0x44}));
  encoder.AppendAddress(0x5566778899AABBCC);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x00, 0x00, 0x00, 0x00, 0x11, 0x22, 0x33, 0x44,
                               0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC}));
}

TEST(DataEncoderTest, AppendData) {
  const uint32_t addr_size = 4;
  DataEncoder encoder(lldb::eByteOrderBig, addr_size);
  // Make sure default constructed StringRef appends nothing
  encoder.AppendData(StringRef());
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({}));
  // Make sure empty StringRef appends nothing
  encoder.AppendData(StringRef(""));
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({}));
  // Append some bytes that contains a NULL character
  encoder.AppendData(StringRef("\x11\x00\x22", 3));
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({0x11, 0x00, 0x22}));
}

TEST(DataEncoderTest, AppendCString) {
  const uint32_t addr_size = 4;
  DataEncoder encoder(lldb::eByteOrderBig, addr_size);
  // Make sure default constructed StringRef appends nothing
  encoder.AppendCString(StringRef());
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({}));
  // Make sure empty StringRef appends a NULL character since the StringRef
  // doesn't contain a NULL in the referenced string.
  encoder.AppendCString(StringRef(""));
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({0x00}));
  // Make sure empty StringRef appends only one NULL character if StringRef
  // does contain a NULL in the referenced string.
  encoder.AppendCString(StringRef("\0", 1));
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({0x00, 0x00}));
  // Append a string where the StringRef doesn't contain a NULL termination
  // and verify the NULL terminate gets added
  encoder.AppendCString(StringRef("hello"));
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x00, 0x00, 'h', 'e', 'l', 'l', 'o', 0x00}));
  // Append a string where the StringRef does contain a NULL termination and
  // verify only one NULL is added
  encoder.AppendCString(StringRef("world", 6));
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x00, 0x00, 'h', 'e', 'l', 'l', 'o', 0x00,
                               'w', 'o', 'r', 'l', 'd', '\0'}));
}

TEST(DataEncoderTest, PutU16Little) {
  const std::vector<uint8_t> init = {1, 2, 3, 4, 5, 6, 7, 8};
  const uint32_t addr_size = 4;
  uint32_t offset = 0;
  DataEncoder encoder(init.data(), init.size(), lldb::eByteOrderLittle,
                      addr_size);
  offset = encoder.PutU16(offset, 11);
  ASSERT_EQ(offset, 2U);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({11, 0, 3, 4, 5, 6, 7, 8}));
  offset = encoder.PutU16(offset, 12);
  ASSERT_EQ(offset, 4U);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({11, 0, 12, 0, 5, 6, 7, 8}));
  offset = encoder.PutU16(offset, 13);
  ASSERT_EQ(offset, 6U);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({11, 0, 12, 0, 13, 0, 7, 8}));
  offset = encoder.PutU16(offset, 14);
  ASSERT_EQ(offset, 8U);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({11, 0, 12, 0, 13, 0, 14, 0}));
  // Check that putting a number to an invalid offset doesn't work and returns
  // an error offset and doesn't modify the buffer.
  ASSERT_EQ(encoder.PutU16(init.size(), 15), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({11, 0, 12, 0, 13, 0, 14, 0}));
}

TEST(DataEncoderTest, PutU16Big) {
  const std::vector<uint8_t> init = {1, 2, 3, 4, 5, 6, 7, 8};
  const uint32_t addr_size = 4;
  uint32_t offset = 0;
  DataEncoder encoder(init.data(), init.size(), lldb::eByteOrderBig,
                      addr_size);
  offset = encoder.PutU16(offset, 11);
  ASSERT_EQ(offset, 2U);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({0, 11, 3, 4, 5, 6, 7, 8}));
  offset = encoder.PutU16(offset, 12);
  ASSERT_EQ(offset, 4U);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({0, 11, 0, 12, 5, 6, 7, 8}));
  offset = encoder.PutU16(offset, 13);
  ASSERT_EQ(offset, 6U);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({0, 11, 0, 12, 0, 13, 7, 8}));
  offset = encoder.PutU16(offset, 14);
  ASSERT_EQ(offset, 8U);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({0, 11, 0, 12, 0, 13, 0, 14}));
  // Check that putting a number to an invalid offset doesn't work and returns
  // an error offset and doesn't modify the buffer.
  ASSERT_EQ(encoder.PutU16(init.size(), 15), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({0, 11, 0, 12, 0, 13, 0, 14}));
}

TEST(DataEncoderTest, PutU32Little) {
  const std::vector<uint8_t> init = {1, 2, 3, 4, 5, 6, 7, 8};
  const uint32_t addr_size = 4;

  uint32_t offset = 0;
  DataEncoder encoder(init.data(), init.size(), lldb::eByteOrderLittle,
                      addr_size);
  offset = encoder.PutU32(offset, 11);
  ASSERT_EQ(offset, 4U);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({11, 0, 0, 0, 5, 6, 7, 8}));
  offset = encoder.PutU32(offset, 12);
  ASSERT_EQ(offset, 8u);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({11, 0, 0, 0, 12, 0, 0, 0}));
  // Check that putting a number to an invalid offset doesn't work and returns
  // an error offset and doesn't modify the buffer.
  ASSERT_EQ(encoder.PutU32(init.size(), 15), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({11, 0, 0, 0, 12, 0, 0, 0}));
}

TEST(DataEncoderTest, PutU32Big) {
  const std::vector<uint8_t> init = {1, 2, 3, 4, 5, 6, 7, 8};
  const uint32_t addr_size = 4;

  uint32_t offset = 0;
  DataEncoder encoder(init.data(), init.size(), lldb::eByteOrderBig,
                      addr_size);
  offset = encoder.PutU32(offset, 11);
  ASSERT_EQ(offset, 4U);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({0, 0, 0, 11, 5, 6, 7, 8}));
  offset = encoder.PutU32(offset, 12);
  ASSERT_EQ(offset, 8U);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({0, 0, 0, 11, 0, 0, 0, 12}));
  // Check that putting a number to an invalid offset doesn't work and returns
  // an error offset and doesn't modify the buffer.
  ASSERT_EQ(encoder.PutU32(init.size(), 15), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({0, 0, 0, 11, 0, 0, 0, 12}));
}

TEST(DataEncoderTest, PutU64Little) {
  const std::vector<uint8_t> init = {1, 2, 3, 4, 5, 6, 7, 8};
  const uint32_t addr_size = 4;
  uint32_t offset = 0;
  DataEncoder encoder(init.data(), init.size(), lldb::eByteOrderLittle,
                      addr_size);
  offset = encoder.PutU64(offset, 11);
  ASSERT_EQ(offset, 8U);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({11, 0, 0, 0, 0, 0, 0, 0}));
  // Check that putting a number to an invalid offset doesn't work and returns
  // an error offset and doesn't modify the buffer.
  ASSERT_EQ(encoder.PutU64(init.size(), 15), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({11, 0, 0, 0, 0, 0, 0, 0}));
}

TEST(DataEncoderTest, PutU64Big) {
  const std::vector<uint8_t> init = {1, 2, 3, 4, 5, 6, 7, 8};
  const uint32_t addr_size = 4;
  uint32_t offset = 0;
  DataEncoder encoder(init.data(), init.size(), lldb::eByteOrderBig,
                      addr_size);
  offset = encoder.PutU64(offset, 11);
  ASSERT_EQ(offset, 8U);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({0, 0, 0, 0, 0, 0, 0, 11}));
  // Check that putting a number to an invalid offset doesn't work and returns
  // an error offset and doesn't modify the buffer.
  ASSERT_EQ(encoder.PutU64(init.size(), 15), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({0, 0, 0, 0, 0, 0, 0, 11}));
}

TEST(DataEncoderTest, PutUnsignedLittle) {
  const std::vector<uint8_t> init = {1, 2, 3, 4, 5, 6, 7, 8};
  const uint32_t addr_size = 4;
  uint32_t offset = 0;
  DataEncoder encoder(init.data(), init.size(), lldb::eByteOrderLittle,
                      addr_size);
  // Put only the least significant byte from the uint64_t into the encoder
  offset = encoder.PutUnsigned(0, 1, 0x1122334455667788ULL);
  ASSERT_EQ(offset, 1U);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({0x88, 2, 3, 4, 5, 6, 7, 8}));

  // Put only the least significant 2 byte2 from the uint64_t into the encoder
  offset = encoder.PutUnsigned(0, 2, 0x1122334455667788ULL);
  ASSERT_EQ(offset, 2U);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x88, 0x77, 3, 4, 5, 6, 7, 8}));

  // Put only the least significant 4 bytes from the uint64_t into the encoder
  offset = encoder.PutUnsigned(0, 4, 0x1122334455667788ULL);
  ASSERT_EQ(offset, 4U);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x88, 0x77, 0x66, 0x55, 5, 6, 7, 8}));

  // Put the full uint64_t value into the encoder
  offset = encoder.PutUnsigned(0, 8, 0x1122334455667788ULL);
  ASSERT_EQ(offset, 8U);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11}));
}

TEST(DataEncoderTest, PutUnsignedBig) {
  const std::vector<uint8_t> init = {1, 2, 3, 4, 5, 6, 7, 8};
  const uint32_t addr_size = 4;
  uint32_t offset = 0;
  DataEncoder encoder(init.data(), init.size(), lldb::eByteOrderBig,
                      addr_size);
  // Put only the least significant byte from the uint64_t into the encoder
  offset = encoder.PutUnsigned(0, 1, 0x1122334455667788ULL);
  ASSERT_EQ(offset, 1U);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x88, 2, 3, 4, 5, 6, 7, 8}));

  // Put only the least significant 2 byte2 from the uint64_t into the encoder
  offset = encoder.PutUnsigned(0, 2, 0x1122334455667788ULL);
  ASSERT_EQ(offset, 2U);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x77, 0x88, 3, 4, 5, 6, 7, 8}));

  // Put only the least significant 4 bytes from the uint64_t into the encoder
  offset = encoder.PutUnsigned(0, 4, 0x1122334455667788ULL);
  ASSERT_EQ(offset, 4U);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x55, 0x66, 0x77, 0x88, 5, 6, 7, 8}));

  // Put the full uint64_t value into the encoder
  offset = encoder.PutUnsigned(0, 8, 0x1122334455667788ULL);
  ASSERT_EQ(offset, 8U);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88}));
}

TEST(DataEncoderTest, PutData) {
  const std::vector<uint8_t> init = {1, 2, 3, 4, 5, 6, 7, 8};
  const uint32_t addr_size = 4;
  char one_byte[] = {11};
  char two_bytes[] = {12, 13};
  char to_many_bytes[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  DataEncoder encoder(init.data(), init.size(), lldb::eByteOrderLittle,
                      addr_size);
  uint32_t offset = 0;
  // Test putting zero bytes from a invalid array (NULL)
  offset = encoder.PutData(offset, nullptr, 0);
  ASSERT_EQ(offset, 0U);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>(init));
  // Test putting zero bytes from a valid array
  offset = encoder.PutData(offset, one_byte, 0);
  ASSERT_EQ(offset, 0U);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>(init));
  // Test putting one byte from a valid array
  offset = encoder.PutData(offset, one_byte, sizeof(one_byte));
  ASSERT_EQ(offset, 1U);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({11, 2, 3, 4, 5, 6, 7, 8}));
  offset = encoder.PutData(offset, two_bytes, sizeof(two_bytes));
  ASSERT_EQ(offset, 3U);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({11, 12, 13, 4, 5, 6, 7, 8}));
  offset = encoder.PutData(0, to_many_bytes, sizeof(to_many_bytes));
  ASSERT_EQ(offset, UINT32_MAX);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({11, 12, 13, 4, 5, 6, 7, 8}));
}

TEST(DataEncoderTest, PutCString) {
  const std::vector<uint8_t> init = {1, 2, 3, 4, 5, 6, 7, 8};
  const uint32_t addr_size = 4;
  DataEncoder encoder(init.data(), init.size(), lldb::eByteOrderLittle,
                      addr_size);
  // Test putting invalid string pointer
  ASSERT_EQ(encoder.PutCString(0, nullptr), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>(init));
  // Test putting an empty string
  uint32_t offset = 0;
  offset = encoder.PutCString(offset, "");
  ASSERT_EQ(offset, 1U);
  ASSERT_EQ(encoder.GetData(), ArrayRef<uint8_t>({'\0', 2, 3, 4, 5, 6, 7, 8}));
  // Test putting valid C string
  offset = encoder.PutCString(offset, "hello");
  ASSERT_EQ(offset, 7U);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({'\0', 'h', 'e', 'l', 'l', 'o', '\0', 8}));
  // Test putting valid C string but where it won't fit in existing data and
  // make sure data stay unchanged.
  offset = encoder.PutCString(offset, "world");
  ASSERT_EQ(offset, UINT32_MAX);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({'\0', 'h', 'e', 'l', 'l', 'o', '\0', 8}));
}

TEST(DataEncoderTest, PutAddressLittle4) {
  const std::vector<uint8_t> init = {1, 2, 3, 4, 5, 6, 7, 8};
  const uint32_t addr_size = 4;
  DataEncoder encoder(init.data(), init.size(), lldb::eByteOrderLittle,
                      addr_size);
  uint32_t offset = 0;
  offset = encoder.PutAddress(offset, 0x11223344);
  ASSERT_EQ(offset, addr_size);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x44, 0x33, 0x22, 0x11, 5, 6, 7, 8}));
  offset = encoder.PutAddress(offset, 0x55667788);
  ASSERT_EQ(offset, addr_size*2);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55}));
  // Make sure we can put an address when it won't fit in the existing buffer
  // and that the buffer doesn't get modified.
  ASSERT_EQ(encoder.PutAddress(addr_size+1, 0x10203040), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55}));
  ASSERT_EQ(encoder.PutAddress(addr_size+2, 0x10203040), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55}));
  ASSERT_EQ(encoder.PutAddress(addr_size+3, 0x10203040), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55}));
  ASSERT_EQ(encoder.PutAddress(addr_size+4, 0x10203040), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55}));
}

TEST(DataEncoderTest, PutAddressBig4) {
  const std::vector<uint8_t> init = {1, 2, 3, 4, 5, 6, 7, 8};
  const uint32_t addr_size = 4;
  DataEncoder encoder(init.data(), init.size(), lldb::eByteOrderBig,
                      addr_size);
  uint32_t offset = 0;
  offset = encoder.PutAddress(offset, 0x11223344);
  ASSERT_EQ(offset, addr_size);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x11, 0x22, 0x33, 0x44, 5, 6, 7, 8}));
  offset = encoder.PutAddress(offset, 0x55667788);
  ASSERT_EQ(offset, addr_size*2);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88}));
  // Make sure we can put an address when it won't fit in the existing buffer
  // and that the buffer doesn't get modified.
  ASSERT_EQ(encoder.PutAddress(addr_size+1, 0x10203040), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88}));
  ASSERT_EQ(encoder.PutAddress(addr_size+2, 0x10203040), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88}));
  ASSERT_EQ(encoder.PutAddress(addr_size+3, 0x10203040), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88}));
  ASSERT_EQ(encoder.PutAddress(addr_size+4, 0x10203040), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88}));
}

TEST(DataEncoderTest, PutAddressLittle8) {
  const std::vector<uint8_t> init = {1, 2, 3, 4, 5, 6, 7, 8};
  const uint32_t addr_size = 8;
  DataEncoder encoder(init.data(), init.size(), lldb::eByteOrderLittle,
                      addr_size);
  uint32_t offset = 0;
  offset = encoder.PutAddress(offset, 0x11223344);
  ASSERT_EQ(offset, addr_size);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x44, 0x33, 0x22, 0x11, 0x00, 0x00, 0x00, 0x00}));
  // Make sure we can put an address when it won't fit in the existing buffer
  // and that the buffer doesn't get modified.
  ASSERT_EQ(encoder.PutAddress(1, 0x10203040), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x44, 0x33, 0x22, 0x11, 0x00, 0x00, 0x00, 0x00}));
  ASSERT_EQ(encoder.PutAddress(2, 0x10203040), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x44, 0x33, 0x22, 0x11, 0x00, 0x00, 0x00, 0x00}));
  ASSERT_EQ(encoder.PutAddress(3, 0x10203040), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x44, 0x33, 0x22, 0x11, 0x00, 0x00, 0x00, 0x00}));
  ASSERT_EQ(encoder.PutAddress(4, 0x10203040), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x44, 0x33, 0x22, 0x11, 0x00, 0x00, 0x00, 0x00}));
  ASSERT_EQ(encoder.PutAddress(5, 0x10203040), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x44, 0x33, 0x22, 0x11, 0x00, 0x00, 0x00, 0x00}));
  ASSERT_EQ(encoder.PutAddress(6, 0x10203040), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x44, 0x33, 0x22, 0x11, 0x00, 0x00, 0x00, 0x00}));
  ASSERT_EQ(encoder.PutAddress(7, 0x10203040), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x44, 0x33, 0x22, 0x11, 0x00, 0x00, 0x00, 0x00}));
  ASSERT_EQ(encoder.PutAddress(8, 0x10203040), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x44, 0x33, 0x22, 0x11, 0x00, 0x00, 0x00, 0x00}));
}

TEST(DataEncoderTest, PutAddressBig8) {
  const std::vector<uint8_t> init = {1, 2, 3, 4, 5, 6, 7, 8};
  const uint32_t addr_size = 8;
  DataEncoder encoder(init.data(), init.size(), lldb::eByteOrderBig,
                      addr_size);
  uint32_t offset = 0;
  offset = encoder.PutAddress(offset, 0x1122334455667788);
  ASSERT_EQ(offset, addr_size);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88}));
  // Make sure we can put an address when it won't fit in the existing buffer
  // and that the buffer doesn't get modified.
  ASSERT_EQ(encoder.PutAddress(1, 0x10203040), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88}));
  ASSERT_EQ(encoder.PutAddress(2, 0x10203040), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88}));
  ASSERT_EQ(encoder.PutAddress(3, 0x10203040), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88}));
  ASSERT_EQ(encoder.PutAddress(4, 0x10203040), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88}));
  ASSERT_EQ(encoder.PutAddress(5, 0x10203040), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88}));
  ASSERT_EQ(encoder.PutAddress(6, 0x10203040), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88}));
  ASSERT_EQ(encoder.PutAddress(7, 0x10203040), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88}));
  ASSERT_EQ(encoder.PutAddress(8, 0x10203040), UINT32_MAX);
  ASSERT_EQ(encoder.GetData(),
            ArrayRef<uint8_t>({0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88}));
}
