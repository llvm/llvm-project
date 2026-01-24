//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/VirtualDataExtractor.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb;

using Table = VirtualDataExtractor::LookupTable;
using Entry = Table::Entry;

TEST(VirtualDataExtractorTest, BasicConstruction) {
  uint8_t buffer[] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 4, Table{Entry(0x1000, 8, 0)});

  EXPECT_EQ(extractor->GetByteSize(), 8U);
}

TEST(VirtualDataExtractorTest, GetDataAtVirtualOffset) {
  uint8_t buffer[] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 4, Table{Entry(0x1000, 8, 0)});

  offset_t virtual_offset = 0x1000;
  const void *data = extractor->GetData(&virtual_offset, 4);

  ASSERT_NE(data, nullptr);
  EXPECT_EQ(virtual_offset, 0x1004U);
  EXPECT_EQ(memcmp(data, buffer, 4), 0);
}

TEST(VirtualDataExtractorTest, GetDataAtVirtualOffsetInvalid) {
  uint8_t buffer[] = {0x01, 0x02, 0x03, 0x04};

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 4, Table{Entry(0x1000, 4, 0)});

  // Try to read from an invalid virtual address.
  offset_t virtual_offset = 0x2000;
  const void *data = extractor->GetData(&virtual_offset, 4);

  EXPECT_EQ(data, nullptr);
}

TEST(VirtualDataExtractorTest, GetU8AtVirtualOffset) {
  uint8_t buffer[] = {0x12, 0x34, 0x56, 0x78};

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 4, Table{Entry(0x1000, 4, 0)});

  offset_t virtual_offset = 0x1000;
  EXPECT_EQ(extractor->GetU8(&virtual_offset), 0x12U);
  EXPECT_EQ(virtual_offset, 0x1001U);

  EXPECT_EQ(extractor->GetU8(&virtual_offset), 0x34U);
  EXPECT_EQ(virtual_offset, 0x1002U);
}

TEST(VirtualDataExtractorTest, GetU16AtVirtualOffset) {
  uint8_t buffer[] = {0x12, 0x34, 0x56, 0x78};

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 4, Table{Entry(0x1000, 4, 0)});

  offset_t virtual_offset = 0x1000;
  EXPECT_EQ(extractor->GetU16(&virtual_offset), 0x3412U);
  EXPECT_EQ(virtual_offset, 0x1002U);

  EXPECT_EQ(extractor->GetU16(&virtual_offset), 0x7856U);
  EXPECT_EQ(virtual_offset, 0x1004U);
}

TEST(VirtualDataExtractorTest, GetU32AtVirtualOffset) {
  uint8_t buffer[] = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0};

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 4, Table{Entry(0x1000, 8, 0)});

  offset_t virtual_offset = 0x1000;
  EXPECT_EQ(extractor->GetU32(&virtual_offset), 0x78563412U);
  EXPECT_EQ(virtual_offset, 0x1004U);

  EXPECT_EQ(extractor->GetU32(&virtual_offset), 0xF0DEBC9AU);
  EXPECT_EQ(virtual_offset, 0x1008U);
}

TEST(VirtualDataExtractorTest, GetU64AtVirtualOffset) {
  uint8_t buffer[] = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0};

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 8, Table{Entry(0x1000, 8, 0)});

  offset_t virtual_offset = 0x1000;
  EXPECT_EQ(extractor->GetU64(&virtual_offset), 0xF0DEBC9A78563412ULL);
  EXPECT_EQ(virtual_offset, 0x1008U);
}

TEST(VirtualDataExtractorTest, GetAddressAtVirtualOffset) {
  uint8_t buffer[] = {0x12, 0x34, 0x56, 0x78};

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 4, Table{Entry(0x1000, 4, 0)});

  offset_t virtual_offset = 0x1000;
  EXPECT_EQ(extractor->GetAddress(&virtual_offset), 0x78563412U);
  EXPECT_EQ(virtual_offset, 0x1004U);
}

TEST(VirtualDataExtractorTest, BigEndian) {
  uint8_t buffer[] = {0x12, 0x34, 0x56, 0x78};

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderBig, 4, Table{Entry(0x1000, 4, 0)});

  offset_t virtual_offset = 0x1000;
  EXPECT_EQ(extractor->GetU16(&virtual_offset), 0x1234U);
  EXPECT_EQ(virtual_offset, 0x1002U);

  EXPECT_EQ(extractor->GetU16(&virtual_offset), 0x5678U);
  EXPECT_EQ(virtual_offset, 0x1004U);
}

TEST(VirtualDataExtractorTest, MultipleEntries) {
  // Create a buffer with distinct patterns for each section.
  uint8_t buffer[] = {
      0x01, 0x02, 0x03, 0x04, // Physical offset 0-3.
      0x11, 0x12, 0x13, 0x14, // Physical offset 4-7.
      0x21, 0x22, 0x23, 0x24  // Physical offset 8-11.
  };

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 4,
      Table{Entry(0x1000, 4, 0),   // Virt 0x1000-0x1004
            Entry(0x2000, 4, 4),   // Virt 0x2000-0x2004
            Entry(0x3000, 4, 8)}); // Virt 0x3000-0x3004

  // Test reading from first virtual range.
  offset_t virtual_offset = 0x1000;
  EXPECT_EQ(extractor->GetU8(&virtual_offset), 0x01U);

  // Test reading from second virtual range.
  virtual_offset = 0x2000;
  EXPECT_EQ(extractor->GetU8(&virtual_offset), 0x11U);

  // Test reading from third virtual range.
  virtual_offset = 0x3000;
  EXPECT_EQ(extractor->GetU8(&virtual_offset), 0x21U);
}

TEST(VirtualDataExtractorTest, NonContiguousVirtualAddresses) {
  uint8_t buffer[] = {0xAA, 0xBB, 0xCC, 0xDD};

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 4,
      Table{Entry(0x1000, 2, 0),   // Virt 0x1000-0x1002
            Entry(0x5000, 2, 2)}); // Virt 0x5000-0x5002

  // Test reading from first virtual range.
  offset_t virtual_offset = 0x1000;
  EXPECT_EQ(extractor->GetU16(&virtual_offset), 0xBBAAU);

  // Test reading from second virtual range (non-contiguous).
  virtual_offset = 0x5000;
  EXPECT_EQ(extractor->GetU16(&virtual_offset), 0xDDCCU);

  // Test that gap between ranges is invalid.
  virtual_offset = 0x3000;
  EXPECT_EQ(extractor->GetU8(&virtual_offset), 0U);
}

TEST(VirtualDataExtractorTest, SharedDataBuffer) {
  // Test with shared_ptr to DataBuffer.
  uint8_t buffer[] = {0x01, 0x02, 0x03, 0x04};
  auto data_sp = std::make_shared<DataBufferHeap>(buffer, sizeof(buffer));

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      data_sp, eByteOrderLittle, 4, Table{Entry(0x1000, 4, 0)});

  offset_t virtual_offset = 0x1000;
  EXPECT_EQ(extractor->GetU32(&virtual_offset), 0x04030201U);
}

TEST(VirtualDataExtractorTest, NullPointerHandling) {
  uint8_t buffer[] = {0x01, 0x02, 0x03, 0x04};

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 4, Table{Entry(0x1000, 4, 0)});

  // Test that passing nullptr returns default values.
  EXPECT_EQ(extractor->GetU8(nullptr), 0U);
  EXPECT_EQ(extractor->GetU16(nullptr), 0U);
  EXPECT_EQ(extractor->GetU32(nullptr), 0U);
  EXPECT_EQ(extractor->GetU64(nullptr), 0U);
  EXPECT_EQ(extractor->GetAddress(nullptr), 0U);
  EXPECT_EQ(extractor->GetData(nullptr, 4), nullptr);
}

TEST(VirtualDataExtractorTest, OffsetMapping) {
  // Test that virtual to physical offset mapping works correctly.
  uint8_t buffer[] = {0x00, 0x00, 0x00, 0x00, 0xAA, 0xBB, 0xCC, 0xDD};

  // Map virtual address 0x1000 to physical offset 4 (skipping first 4 bytes).
  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 4, Table{Entry(0x1000, 4, 4)});

  offset_t virtual_offset = 0x1000;
  // Should read from physical offset 4, not 0.
  EXPECT_EQ(extractor->GetU32(&virtual_offset), 0xDDCCBBAAU);
}

TEST(VirtualDataExtractorTest, GetU8Unchecked) {
  uint8_t buffer[] = {0x12, 0x34, 0x56, 0x78};

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 4, Table{Entry(0x1000, 4, 0)});

  offset_t virtual_offset = 0x1000;
  EXPECT_EQ(extractor->GetU8_unchecked(&virtual_offset), 0x12U);
  EXPECT_EQ(virtual_offset, 0x1001U);

  EXPECT_EQ(extractor->GetU8_unchecked(&virtual_offset), 0x34U);
  EXPECT_EQ(virtual_offset, 0x1002U);
}

TEST(VirtualDataExtractorTest, GetU16Unchecked) {
  uint8_t buffer[] = {0x12, 0x34, 0x56, 0x78};

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 4, Table{Entry(0x1000, 4, 0)});

  offset_t virtual_offset = 0x1000;
  EXPECT_EQ(extractor->GetU16_unchecked(&virtual_offset), 0x3412U);
  EXPECT_EQ(virtual_offset, 0x1002U);

  EXPECT_EQ(extractor->GetU16_unchecked(&virtual_offset), 0x7856U);
  EXPECT_EQ(virtual_offset, 0x1004U);
}

TEST(VirtualDataExtractorTest, GetU32Unchecked) {
  uint8_t buffer[] = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0};

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 4, Table{Entry(0x1000, 8, 0)});

  offset_t virtual_offset = 0x1000;
  EXPECT_EQ(extractor->GetU32_unchecked(&virtual_offset), 0x78563412U);
  EXPECT_EQ(virtual_offset, 0x1004U);

  EXPECT_EQ(extractor->GetU32_unchecked(&virtual_offset), 0xF0DEBC9AU);
  EXPECT_EQ(virtual_offset, 0x1008U);
}

TEST(VirtualDataExtractorTest, GetU64Unchecked) {
  uint8_t buffer[] = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0};

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 8, Table{Entry(0x1000, 8, 0)});

  offset_t virtual_offset = 0x1000;
  EXPECT_EQ(extractor->GetU64_unchecked(&virtual_offset),
            0xF0DEBC9A78563412ULL);
  EXPECT_EQ(virtual_offset, 0x1008U);
}

TEST(VirtualDataExtractorTest, GetMaxU64Unchecked) {
  uint8_t buffer[] = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0};

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 4, Table{Entry(0x1000, 8, 0)});

  // Test various byte sizes.
  offset_t virtual_offset = 0x1000;
  EXPECT_EQ(extractor->GetMaxU64_unchecked(&virtual_offset, 1), 0x12U);
  EXPECT_EQ(virtual_offset, 0x1001U);

  virtual_offset = 0x1000;
  EXPECT_EQ(extractor->GetMaxU64_unchecked(&virtual_offset, 2), 0x3412U);
  EXPECT_EQ(virtual_offset, 0x1002U);

  virtual_offset = 0x1000;
  EXPECT_EQ(extractor->GetMaxU64_unchecked(&virtual_offset, 4), 0x78563412U);
  EXPECT_EQ(virtual_offset, 0x1004U);

  virtual_offset = 0x1000;
  EXPECT_EQ(extractor->GetMaxU64_unchecked(&virtual_offset, 8),
            0xF0DEBC9A78563412ULL);
  EXPECT_EQ(virtual_offset, 0x1008U);
}

TEST(VirtualDataExtractorTest, GetAddressUnchecked) {
  uint8_t buffer[] = {0x12, 0x34, 0x56, 0x78};

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 4, Table{Entry(0x1000, 4, 0)});

  offset_t virtual_offset = 0x1000;
  EXPECT_EQ(extractor->GetAddress_unchecked(&virtual_offset), 0x78563412U);
  EXPECT_EQ(virtual_offset, 0x1004U);
}

TEST(VirtualDataExtractorTest, UncheckedWithBigEndian) {
  uint8_t buffer[] = {0x12, 0x34, 0x56, 0x78};

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderBig, 4, Table{Entry(0x1000, 4, 0)});

  offset_t virtual_offset = 0x1000;
  EXPECT_EQ(extractor->GetU16_unchecked(&virtual_offset), 0x1234U);
  EXPECT_EQ(virtual_offset, 0x1002U);

  EXPECT_EQ(extractor->GetU16_unchecked(&virtual_offset), 0x5678U);
  EXPECT_EQ(virtual_offset, 0x1004U);
}

TEST(VirtualDataExtractorTest, GetCStr) {
  // Create buffer with null-terminated strings.
  uint8_t buffer[] = {'H', 'e', 'l', 'l',  'o', '\0', 'W', 'o',
                      'r', 'l', 'd', '\0', 'F', 'o',  'o', '\0'};

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 4,
      Table{Entry(0x1000, 6, 0), Entry(0x2000, 12, 6)});

  // Test reading first string.
  offset_t virtual_offset = 0x1000;
  const char *str1 = extractor->GetCStr(&virtual_offset);
  ASSERT_NE(str1, nullptr);
  EXPECT_STREQ(str1, "Hello");
  EXPECT_EQ(virtual_offset, 0x1006U); // After "Hello\0"

  // Test reading second string.
  virtual_offset = 0x2000;
  const char *str2 = extractor->GetCStr(&virtual_offset);
  ASSERT_NE(str2, nullptr);
  EXPECT_STREQ(str2, "World");
  EXPECT_EQ(virtual_offset, 0x2006U); // After "World\0"
}

TEST(VirtualDataExtractorTest, GetFloat) {
  // Create buffer with float value (IEEE 754 single precision).
  // 3.14159f in little endian: 0xDB 0x0F 0x49 0x40
  uint8_t buffer[] = {0xDB, 0x0F, 0x49, 0x40};

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 4, Table{Entry(0x1000, 4, 0)});

  offset_t virtual_offset = 0x1000;
  float value = extractor->GetFloat(&virtual_offset);
  EXPECT_NEAR(value, 3.14159f, 0.00001f);
  EXPECT_EQ(virtual_offset, 0x1004U);
}

TEST(VirtualDataExtractorTest, GetDouble) {
  // Create buffer with double value (IEEE 754 double precision).
  // 3.14159265358979 in little endian
  uint8_t buffer[] = {0x18, 0x2D, 0x44, 0x54, 0xFB, 0x21, 0x09, 0x40};

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 8, Table{Entry(0x1000, 8, 0)});

  offset_t virtual_offset = 0x1000;
  double value = extractor->GetDouble(&virtual_offset);
  EXPECT_NEAR(value, 3.14159265358979, 0.00000000000001);
  EXPECT_EQ(virtual_offset, 0x1008U);
}

TEST(VirtualDataExtractorTest, GetULEB128) {
  // ULEB128 encoding: 0x624 (1572 decimal) = 0xA4 0x0C
  uint8_t buffer[] = {0xA4, 0x0C, 0xFF, 0x00, 0x7F, 0x80, 0x01};

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 4, Table{Entry(0x1000, 7, 0)});

  // Test reading first ULEB128 value (1572).
  offset_t virtual_offset = 0x1000;
  EXPECT_EQ(extractor->GetULEB128(&virtual_offset), 1572U);
  EXPECT_EQ(virtual_offset, 0x1002U);

  // Test reading second ULEB128 value (127).
  virtual_offset = 0x1004;
  EXPECT_EQ(extractor->GetULEB128(&virtual_offset), 127U);
  EXPECT_EQ(virtual_offset, 0x1005U);

  // Test reading third ULEB128 value (128).
  EXPECT_EQ(extractor->GetULEB128(&virtual_offset), 128U);
  EXPECT_EQ(virtual_offset, 0x1007U);
}

TEST(VirtualDataExtractorTest, GetSLEB128) {
  // SLEB128 encoding: -123 = 0x85 0x7F, 123 = 0xFB 0x00
  uint8_t buffer[] = {0x85, 0x7F, 0xFB, 0x00};

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 4, Table{Entry(0x1000, 4, 0)});

  // Test reading negative SLEB128 value (-123).
  offset_t virtual_offset = 0x1000;
  EXPECT_EQ(extractor->GetSLEB128(&virtual_offset), -123);
  EXPECT_EQ(virtual_offset, 0x1002U);

  // Test reading positive SLEB128 value (123).
  EXPECT_EQ(extractor->GetSLEB128(&virtual_offset), 123);
  EXPECT_EQ(virtual_offset, 0x1004U);
}

TEST(VirtualDataExtractorTest, GetU8Array) {
  uint8_t buffer[] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 4, Table{Entry(0x1000, 8, 0)});

  // Test reading array of 4 bytes.
  offset_t virtual_offset = 0x1000;
  uint8_t dst[4] = {0};
  void *result = extractor->GetU8(&virtual_offset, dst, 4);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(dst[0], 0x01U);
  EXPECT_EQ(dst[1], 0x02U);
  EXPECT_EQ(dst[2], 0x03U);
  EXPECT_EQ(dst[3], 0x04U);
  EXPECT_EQ(virtual_offset, 0x1004U);
}

TEST(VirtualDataExtractorTest, GetU16Array) {
  uint8_t buffer[] = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0};

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 4, Table{Entry(0x1000, 8, 0)});

  // Test reading array of 3 uint16_t values.
  offset_t virtual_offset = 0x1000;
  uint16_t dst[3] = {0};
  void *result = extractor->GetU16(&virtual_offset, dst, 3);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(dst[0], 0x3412U);
  EXPECT_EQ(dst[1], 0x7856U);
  EXPECT_EQ(dst[2], 0xBC9AU);
  EXPECT_EQ(virtual_offset, 0x1006U);
}

TEST(VirtualDataExtractorTest, GetU32Array) {
  uint8_t buffer[] = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0};

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 4, Table{Entry(0x1000, 8, 0)});

  // Test reading array of 2 uint32_t values.
  offset_t virtual_offset = 0x1000;
  uint32_t dst[2] = {0};
  void *result = extractor->GetU32(&virtual_offset, dst, 2);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(dst[0], 0x78563412U);
  EXPECT_EQ(dst[1], 0xF0DEBC9AU);
  EXPECT_EQ(virtual_offset, 0x1008U);
}

TEST(VirtualDataExtractorTest, GetU64Array) {
  uint8_t buffer[] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                      0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18};

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 8, Table{Entry(0x1000, 16, 0)});

  // Test reading array of 2 uint64_t values.
  offset_t virtual_offset = 0x1000;
  uint64_t dst[2] = {0};
  void *result = extractor->GetU64(&virtual_offset, dst, 2);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(dst[0], 0x0807060504030201ULL);
  EXPECT_EQ(dst[1], 0x1817161514131211ULL);
  EXPECT_EQ(virtual_offset, 0x1010U);
}

TEST(VirtualDataExtractorTest, GetMaxU64WithVariableSizes) {
  uint8_t buffer[] = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0};

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 4, Table{Entry(0x1000, 8, 0)});

  // Test reading 3-byte value.
  offset_t virtual_offset = 0x1000;
  EXPECT_EQ(extractor->GetMaxU64(&virtual_offset, 3), 0x563412U);
  EXPECT_EQ(virtual_offset, 0x1003U);

  // Test reading 5-byte value.
  virtual_offset = 0x1000;
  EXPECT_EQ(extractor->GetMaxU64(&virtual_offset, 5), 0x9A78563412ULL);
  EXPECT_EQ(virtual_offset, 0x1005U);
}

TEST(VirtualDataExtractorTest, GetMaxS64) {
  // Test with negative number (sign extension).
  uint8_t buffer[] = {0xFF, 0xFF, 0xFF, 0xFF};

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 4, Table{Entry(0x1000, 4, 0)});

  // Test reading 1-byte signed value (-1).
  offset_t virtual_offset = 0x1000;
  EXPECT_EQ(extractor->GetMaxS64(&virtual_offset, 1), -1);
  EXPECT_EQ(virtual_offset, 0x1001U);

  // Test reading 2-byte signed value (-1).
  virtual_offset = 0x1000;
  EXPECT_EQ(extractor->GetMaxS64(&virtual_offset, 2), -1);
  EXPECT_EQ(virtual_offset, 0x1002U);
}

TEST(VirtualDataExtractorTest, CannotReadAcrossEntryBoundaries) {
  // Create buffer with two separate regions.
  uint8_t buffer[] = {0x01, 0x02, 0x03, 0x04, 0x11, 0x12, 0x13, 0x14};

  // First entry: virtual 0x1000-0x1004 maps to physical 0-4.
  // Second entry: virtual 0x2000-0x2004 maps to physical 4-8.
  // Note: there's a gap in virtual addresses (0x1004-0x2000).
  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 4,
      Table{Entry(0x1000, 4, 0), Entry(0x2000, 4, 4)});

  // Verify we can read within the first entry.
  offset_t virtual_offset = 0x1000;
  EXPECT_EQ(extractor->GetU32(&virtual_offset), 0x04030201U);
  EXPECT_EQ(virtual_offset, 0x1004U);

  // Verify we can read within the second entry.
  virtual_offset = 0x2000;
  EXPECT_EQ(extractor->GetU32(&virtual_offset), 0x14131211U);
  EXPECT_EQ(virtual_offset, 0x2004U);

  // Verify we CANNOT read in the gap between entries.
  // This address is not in any lookup table entry.
  virtual_offset = 0x1500;
  EXPECT_EQ(extractor->GetU8(&virtual_offset), 0U);
  EXPECT_EQ(virtual_offset, 0x1500U);

  // Verify we CANNOT read data pointer from the gap.
  virtual_offset = 0x1800;
  const void *data = extractor->GetData(&virtual_offset, 1);
  EXPECT_EQ(data, nullptr);
  EXPECT_EQ(virtual_offset, 0x1800U); // Offset unchanged.

  // Verify we can read individual bytes within each entry.
  virtual_offset = 0x1003;
  EXPECT_EQ(extractor->GetU8(&virtual_offset), 0x04U);
  EXPECT_EQ(virtual_offset, 0x1004U);

  // Verify we CANNOT read past the end of an entry.
  virtual_offset = 0x1004;
  EXPECT_EQ(extractor->GetU8(&virtual_offset), 0U);
  EXPECT_EQ(virtual_offset, 0x1004U);
}

TEST(VirtualDataExtractorTest, ReadExactlyAtEntryEnd) {
  uint8_t buffer[] = {0x01, 0x02, 0x03, 0x04};

  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer, sizeof(buffer), eByteOrderLittle, 4, Table{Entry(0x1000, 4, 0)});

  // Reading exactly to the end of an entry should work.
  offset_t virtual_offset = 0x1000;
  EXPECT_EQ(extractor->GetU32(&virtual_offset), 0x04030201U);
  EXPECT_EQ(virtual_offset, 0x1004U);

  // But reading one byte past the end should fail.
  virtual_offset = 0x1004;
  EXPECT_EQ(extractor->GetU8(&virtual_offset), 0U);
  EXPECT_EQ(virtual_offset, 0x1004U);

  // Reading from just before the end should work for smaller sizes.
  virtual_offset = 0x1003;
  EXPECT_EQ(extractor->GetU8(&virtual_offset), 0x04U);
  EXPECT_EQ(virtual_offset, 0x1004U);
}

TEST(VirtualDataExtractorTest, SubsetExtractorGetU32) {
  uint32_t buffer[16];
  // 0x11111111 0x22222222 ... 0xffffffff
  for (int i = 0; i < 16; i++)
    buffer[i] =
        i << 28 | i << 24 | i << 20 | i << 16 | i << 12 | i << 8 | i << 4 | i;
  DataBufferSP buffer_sp =
      std::make_shared<DataBufferUnowned>((uint8_t *)&buffer, sizeof(buffer));
  lldb::DataExtractorSP extractor = std::make_shared<VirtualDataExtractor>(
      buffer_sp, eByteOrderLittle, 8,
      Table{Entry(0x0, 4 * sizeof(uint32_t), 12 * sizeof(uint32_t)),
            Entry(0x10, 4 * sizeof(uint32_t), 0 * sizeof(uint32_t)),
            Entry(0x20, 4 * sizeof(uint32_t), 8 * sizeof(uint32_t)),
            Entry(0x30, 4 * sizeof(uint32_t), 4 * sizeof(uint32_t))});

  offset_t virtual_offset = 0;
  // Entry(0x0, 4*sizeof(uint32_t), 12*sizeof(uint32_t))
  EXPECT_EQ(extractor->GetU32(&virtual_offset), 0xccccccccU);
  EXPECT_EQ(extractor->GetU32(&virtual_offset), 0xddddddddU);
  EXPECT_EQ(extractor->GetU32(&virtual_offset), 0xeeeeeeeeU);
  EXPECT_EQ(extractor->GetU32(&virtual_offset), 0xffffffffU);
  // Entry(0x10, 4*sizeof(uint32_t), 0*sizeof(uint32_t))
  EXPECT_EQ(extractor->GetU32(&virtual_offset), 0x00000000U);
  EXPECT_EQ(extractor->GetU32(&virtual_offset), 0x11111111U);
  EXPECT_EQ(extractor->GetU32(&virtual_offset), 0x22222222U);
  EXPECT_EQ(extractor->GetU32(&virtual_offset), 0x33333333U);
  // Entry(0x20, 4*sizeof(uint32_t), 8*sizeof(uint32_t))
  EXPECT_EQ(extractor->GetU32(&virtual_offset), 0x88888888U);
  EXPECT_EQ(extractor->GetU32(&virtual_offset), 0x99999999U);
  EXPECT_EQ(extractor->GetU32(&virtual_offset), 0xAAAAAAAAU);
  EXPECT_EQ(extractor->GetU32(&virtual_offset), 0xBBBBBBBBU);
  // Entry(0x30, 4*sizeof(uint32_t), 4*sizeof(uint32_t))
  EXPECT_EQ(extractor->GetU32(&virtual_offset), 0x44444444U);
  EXPECT_EQ(extractor->GetU32(&virtual_offset), 0x55555555U);
  EXPECT_EQ(extractor->GetU32(&virtual_offset), 0x66666666U);
  EXPECT_EQ(extractor->GetU32(&virtual_offset), 0x77777777U);

  // sub_extractor starts at buffer[4] for 4 uint32_t's, aligned
  // to the start of a LookupTable entry.
  lldb::DataExtractorSP aligned_sub_extractor = extractor->GetSubsetExtractorSP(
      4 * sizeof(uint32_t), 4 * sizeof(uint32_t));

  virtual_offset = 0;
  // Entry(0x10, 4*sizeof(uint32_t), 0*sizeof(uint32_t))
  // {subset virtual offset: 0x0}
  EXPECT_EQ(aligned_sub_extractor->GetU32(&virtual_offset), 0x00000000U);
  EXPECT_EQ(aligned_sub_extractor->GetU32(&virtual_offset), 0x11111111U);
  EXPECT_EQ(aligned_sub_extractor->GetU32(&virtual_offset), 0x22222222U);
  EXPECT_EQ(aligned_sub_extractor->GetU32(&virtual_offset), 0x33333333U);

  // sub_extractor starts at buffer[10] for 2 uint32_t's,
  // only PART of a LookupTable entry.
  lldb::DataExtractorSP misaligned_sub_extractor =
      extractor->GetSubsetExtractorSP(10 * sizeof(uint32_t),
                                      2 * sizeof(uint32_t));
  virtual_offset = 0;
  EXPECT_EQ(misaligned_sub_extractor->GetU32(&virtual_offset), 0xAAAAAAAAU);
  EXPECT_EQ(misaligned_sub_extractor->GetU32(&virtual_offset), 0xBBBBBBBBU);

  lldb::DataExtractorSP contiguous_subset = extractor->GetSubsetExtractorSP(0);
  EXPECT_EQ(contiguous_subset->GetByteSize(), 4 * sizeof(uint32_t));

  lldb::DataExtractorSP misaligned_contiguous_subset =
      extractor->GetSubsetExtractorSP(2 * sizeof(uint32_t));
  EXPECT_EQ(misaligned_contiguous_subset->GetByteSize(), 2 * sizeof(uint32_t));

  // Ask for a subset in the second LookupTable entry.
  lldb::DataExtractorSP middle_contiguous_subset =
      extractor->GetSubsetExtractorSP(4 * sizeof(uint32_t));
  EXPECT_EQ(middle_contiguous_subset->GetByteSize(), 4 * sizeof(uint32_t));
}
