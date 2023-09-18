//===-- DataDumpExtractorTest.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/DumpDataExtractor.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/StreamString.h"
#include "gtest/gtest.h"
#include <complex>
#include <limits>

using namespace lldb;
using namespace lldb_private;

// This is needed for the tests because they rely on the Target global
// properties.
class DumpDataExtractorTest : public ::testing::Test {
public:
  void SetUp() override {
    FileSystem::Initialize();
    HostInfo::Initialize();
  }
  void TearDown() override {
    HostInfo::Terminate();
    FileSystem::Terminate();
  }
};

static void TestDumpWithAddress(uint64_t base_addr, size_t item_count,
                                llvm::StringRef expected) {
  std::vector<uint8_t> data{0x11, 0x22};
  StreamString result;
  DataBufferHeap dumpbuffer(&data[0], data.size());
  DataExtractor extractor(dumpbuffer.GetBytes(), dumpbuffer.GetByteSize(),
                          endian::InlHostByteOrder(), /*addr_size=*/4);

  DumpDataExtractor(extractor, &result, 0, lldb::Format::eFormatHex,
                    /*item_byte_size=*/1, item_count,
                    /*num_per_line=*/1, base_addr, 0, 0);
  ASSERT_EQ(expected, result.GetString());
}

TEST(DumpDataExtractorTest, BaseAddress) {
  TestDumpWithAddress(0x12341234, 1, "0x12341234: 0x11");
  TestDumpWithAddress(LLDB_INVALID_ADDRESS, 1, "0x11");
  TestDumpWithAddress(0x12341234, 2, "0x12341234: 0x11\n0x12341235: 0x22");
  TestDumpWithAddress(LLDB_INVALID_ADDRESS, 2, "0x11\n0x22");
}

static void TestDumpWithOffset(offset_t start_offset,
                               llvm::StringRef expected) {
  std::vector<uint8_t> data{0x11, 0x22, 0x33};
  StreamString result;
  DataBufferHeap dumpbuffer(&data[0], data.size());
  DataExtractor extractor(dumpbuffer.GetBytes(), dumpbuffer.GetByteSize(),
                          endian::InlHostByteOrder(), /*addr_size=*/4);

  DumpDataExtractor(extractor, &result, start_offset, lldb::Format::eFormatHex,
                    /*item_byte_size=*/1, /*item_count=*/data.size(),
                    /*num_per_line=*/data.size(), /*base_addr=*/0, 0, 0);
  ASSERT_EQ(expected, result.GetString());
}

TEST(DumpDataExtractorTest, StartOffset) {
  TestDumpWithOffset(0, "0x00000000: 0x11 0x22 0x33");
  // The offset applies to the DataExtractor, not the address used when
  // formatting.
  TestDumpWithOffset(1, "0x00000000: 0x22 0x33");
  // If the offset is outside the DataExtractor's range we do nothing.
  TestDumpWithOffset(3, "");
}

TEST(DumpDataExtractorTest, NullStream) {
  // We don't do any work if there is no output stream.
  uint8_t c = 0x11;
  StreamString result;
  DataBufferHeap dumpbuffer(&c, 0);
  DataExtractor extractor(dumpbuffer.GetBytes(), dumpbuffer.GetByteSize(),
                          endian::InlHostByteOrder(), /*addr_size=*/4);

  DumpDataExtractor(extractor, nullptr, 0, lldb::Format::eFormatHex,
                    /*item_byte_size=*/1, /*item_count=*/1,
                    /*num_per_line=*/1, /*base_addr=*/0, 0, 0);
  ASSERT_EQ("", result.GetString());
}

static void TestDumpImpl(const void *data, size_t data_size,
                         size_t item_byte_size, size_t item_count,
                         size_t num_per_line, uint64_t base_addr,
                         lldb::Format format, llvm::StringRef expected) {
  StreamString result;
  DataBufferHeap dumpbuffer(data, data_size);
  DataExtractor extractor(dumpbuffer.GetBytes(), dumpbuffer.GetByteSize(),
                          endian::InlHostByteOrder(),
                          /*addr_size=*/4);
  DumpDataExtractor(extractor, &result, 0, format, item_byte_size, item_count,
                    num_per_line, base_addr, 0, 0);
  ASSERT_EQ(expected, result.GetString());
}

template <typename T>
static void TestDump(T data, lldb::Format format, llvm::StringRef expected) {
  TestDumpImpl(&data, sizeof(T), sizeof(T), 1, 1, LLDB_INVALID_ADDRESS, format,
               expected);
}

static void TestDump(llvm::StringRef str, lldb::Format format,
                     llvm::StringRef expected) {
  TestDumpImpl(str.bytes_begin(),
               // +1 to include the NULL char as the last byte
               str.size() + 1, str.size() + 1, 1, 1, LLDB_INVALID_ADDRESS,
               format, expected);
}

template <typename T>
static void TestDump(const std::vector<T> data, lldb::Format format,
                     llvm::StringRef expected) {
  size_t sz_bytes = data.size() * sizeof(T);
  TestDumpImpl(&data[0], sz_bytes, sz_bytes, data.size(), 1,
               LLDB_INVALID_ADDRESS, format, expected);
}

TEST(DumpDataExtractorTest, Formats) {
  TestDump<uint8_t>(1, lldb::eFormatDefault, "0x01");
  TestDump<uint8_t>(1, lldb::eFormatBoolean, "true");
  TestDump<uint8_t>(0xAA, lldb::eFormatBinary, "0b10101010");
  TestDump<uint8_t>(1, lldb::eFormatBytes, "01");
  TestDump<uint8_t>(1, lldb::eFormatBytesWithASCII, "01  .");
  TestDump('?', lldb::eFormatChar, "'?'");
  TestDump('\x1A', lldb::eFormatCharPrintable, ".");
  TestDump('#', lldb::eFormatCharPrintable, "#");
  TestDump(std::complex<float>(1.2, 3.4), lldb::eFormatComplex, "1.2 + 3.4i");
  TestDump(std::complex<double>(4.5, 6.7), lldb::eFormatComplex, "4.5 + 6.7i");

  // long double is not tested here because for some platforms we treat it as 10
  // bytes when the compiler allocates 16 bytes of space for it. (see
  // DataExtractor::GetLongDouble) Meaning that when we extract the second one,
  // it gets the wrong value (it's 6 bytes off). You could manually construct a
  // set of bytes to match the 10 byte format but then if the test runs on a
  // machine where we don't use 10 it'll break.

  // Test printable characters.
  TestDump(llvm::StringRef("aardvark"), lldb::Format::eFormatCString,
           "\"aardvark\"");
  // Test unprintable characters.
  TestDump(llvm::StringRef("\xcf\xfa\xed\xfe\f"), lldb::Format::eFormatCString,
           "\"\\xcf\\xfa\\xed\\xfe\\f\"");
  // Test a mix of printable and unprintable characters.
  TestDump(llvm::StringRef("\xcf\xfa\ffoo"), lldb::Format::eFormatCString,
           "\"\\xcf\\xfa\\ffoo\"");

  TestDump<uint16_t>(99, lldb::Format::eFormatDecimal, "99");
  // Just prints as a signed integer.
  TestDump(-1, lldb::Format::eFormatEnum, "-1");
  TestDump(0xcafef00d, lldb::Format::eFormatHex, "0xcafef00d");
  TestDump(0xcafef00d, lldb::Format::eFormatHexUppercase, "0xCAFEF00D");
  TestDump(0.456, lldb::Format::eFormatFloat, "0.45600000000000002");
  TestDump(9, lldb::Format::eFormatOctal, "011");
  // Chars packed into an integer.
  TestDump<uint32_t>(0x4C4C4442, lldb::Format::eFormatOSType, "'LLDB'");
  // Unicode8 doesn't have a specific formatter.
  TestDump<uint8_t>(0x34, lldb::Format::eFormatUnicode8, "0x34");
  TestDump<uint16_t>(0x1122, lldb::Format::eFormatUnicode16, "U+1122");
  TestDump<uint32_t>(0x12345678, lldb::Format::eFormatUnicode32,
                     "U+0x12345678");
  TestDump<unsigned int>(654321, lldb::Format::eFormatUnsigned, "654321");
  // This pointer is printed based on the size of uint64_t, so the test is the
  // same for 32/64 bit host.
  TestDump<uint64_t>(0x4444555566667777, lldb::Format::eFormatPointer,
                     "0x4444555566667777");

  TestDump(std::vector<char>{'A', '\x01', 'C'},
           lldb::Format::eFormatVectorOfChar, "{A\\x01C}");
  TestDump(std::vector<int8_t>{0, -1, std::numeric_limits<int8_t>::max()},
           lldb::Format::eFormatVectorOfSInt8, "{0 -1 127}");
  TestDump(std::vector<uint8_t>{12, 0xFF, 34},
           lldb::Format::eFormatVectorOfUInt8, "{0x0c 0xff 0x22}");
  TestDump(std::vector<int16_t>{-1, 1234, std::numeric_limits<int16_t>::max()},
           lldb::Format::eFormatVectorOfSInt16, "{-1 1234 32767}");
  TestDump(std::vector<uint16_t>{0xffff, 0xabcd, 0x1234},
           lldb::Format::eFormatVectorOfUInt16, "{0xffff 0xabcd 0x1234}");
  TestDump(std::vector<int32_t>{0, -1, std::numeric_limits<int32_t>::max()},
           lldb::Format::eFormatVectorOfSInt32, "{0 -1 2147483647}");
  TestDump(std::vector<uint32_t>{0, 0xffffffff, 0x1234abcd},
           lldb::Format::eFormatVectorOfUInt32,
           "{0x00000000 0xffffffff 0x1234abcd}");
  TestDump(std::vector<int64_t>{0, -1, std::numeric_limits<int64_t>::max()},
           lldb::Format::eFormatVectorOfSInt64, "{0 -1 9223372036854775807}");
  TestDump(std::vector<uint64_t>{0, 0xaaaabbbbccccdddd},
           lldb::Format::eFormatVectorOfUInt64,
           "{0x0000000000000000 0xaaaabbbbccccdddd}");

  // See half2float for format details.
  // Test zeroes.
  TestDump(std::vector<uint16_t>{0x0000, 0x8000},
           lldb::Format::eFormatVectorOfFloat16, "{0 -0}");
  // Some subnormal numbers.
  TestDump(std::vector<uint16_t>{0x0001, 0x8001},
           lldb::Format::eFormatVectorOfFloat16, "{5.9605E-8 -5.9605E-8}");
  // A full mantisse and empty expontent.
  TestDump(std::vector<uint16_t>{0x83ff, 0x03ff},
           lldb::Format::eFormatVectorOfFloat16, "{-6.0976E-5 6.0976E-5}");
  // Some normal numbers.
  TestDump(std::vector<uint16_t>{0b0100001001001000},
           lldb::Format::eFormatVectorOfFloat16, "{3.1406}");
  // Largest and smallest normal number.
  TestDump(std::vector<uint16_t>{0x0400, 0x7bff},
           lldb::Format::eFormatVectorOfFloat16, "{6.1035E-5 65504}");
  TestDump(std::vector<uint16_t>{0xabcd, 0x1234},
           lldb::Format::eFormatVectorOfFloat16, "{-0.060944 7.5722E-4}");

  // quiet/signaling NaNs.
  TestDump(std::vector<uint16_t>{0xffff, 0xffc0, 0x7fff, 0x7fc0},
           lldb::Format::eFormatVectorOfFloat16, "{NaN NaN NaN NaN}");
  // +/-Inf.
  TestDump(std::vector<uint16_t>{0xfc00, 0x7c00},
           lldb::Format::eFormatVectorOfFloat16, "{-Inf +Inf}");

  TestDump(std::vector<float>{std::numeric_limits<float>::min(),
                              std::numeric_limits<float>::max()},
           lldb::Format::eFormatVectorOfFloat32,
           "{1.17549435E-38 3.40282347E+38}");
  TestDump(std::vector<float>{std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::signaling_NaN(),
                              -std::numeric_limits<float>::quiet_NaN(),
                              -std::numeric_limits<float>::signaling_NaN()},
           lldb::Format::eFormatVectorOfFloat32, "{NaN NaN NaN NaN}");
  TestDump(std::vector<double>{std::numeric_limits<double>::min(),
                               std::numeric_limits<double>::max()},
           lldb::Format::eFormatVectorOfFloat64,
           "{2.2250738585072014E-308 1.7976931348623157E+308}");
  TestDump(
      std::vector<double>{
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::signaling_NaN(),
          -std::numeric_limits<double>::quiet_NaN(),
          -std::numeric_limits<double>::signaling_NaN(),
      },
      lldb::Format::eFormatVectorOfFloat64, "{NaN NaN NaN NaN}");

  // Not sure we can rely on having uint128_t everywhere so emulate with
  // uint64_t.
  TestDump(
      std::vector<uint64_t>{0x1, 0x1111222233334444, 0xaaaabbbbccccdddd, 0x0},
      lldb::Format::eFormatVectorOfUInt128,
      "{0x11112222333344440000000000000001 "
      "0x0000000000000000aaaabbbbccccdddd}");

  TestDump(std::vector<int>{2, 4}, lldb::Format::eFormatComplexInteger,
           "2 + 4i");

  // Without an execution context this just prints the pointer on its own.
  TestDump<uint32_t>(0x11223344, lldb::Format::eFormatAddressInfo,
                     "0x11223344");

  // Input not written in hex form because that requires C++17.
  TestDump<float>(10, lldb::Format::eFormatHexFloat, "0x1.4p3");
  TestDump<double>(10, lldb::Format::eFormatHexFloat, "0x1.4p3");
  // long double not supported, see ItemByteSizeErrors.

  // Can't disassemble without an execution context.
  TestDump<uint32_t>(0xcafef00d, lldb::Format::eFormatInstruction,
                     "invalid target");

  // Has no special handling, intended for use elsewhere.
  TestDump<int>(99, lldb::Format::eFormatVoid, "0x00000063");
}

TEST(DumpDataExtractorTest, FormatCharArray) {
  // Unlike the other formats, charArray isn't 1 array of N chars.
  // It must be passed as N chars of 1 byte each.
  // (eFormatVectorOfChar does this swap for you)
  std::vector<char> data{'A', '\x01', '#'};
  StreamString result;
  DataBufferHeap dumpbuffer(&data[0], data.size());
  DataExtractor extractor(dumpbuffer.GetBytes(), dumpbuffer.GetByteSize(),
                          endian::InlHostByteOrder(), /*addr_size=*/4);

  DumpDataExtractor(extractor, &result, 0, lldb::Format::eFormatCharArray,
                    /*item_byte_size=*/1,
                    /*item_count=*/data.size(),
                    /*num_per_line=*/data.size(), 0, 0, 0);
  ASSERT_EQ("0x00000000: A\\x01#", result.GetString());

  result.Clear();
  DumpDataExtractor(extractor, &result, 0, lldb::Format::eFormatCharArray, 1,
                    data.size(), 1, 0, 0, 0);
  // ASSERT macro thinks the split strings are multiple arguments so make a var.
  const char *expected = "0x00000000: A\n"
                         "0x00000001: \\x01\n"
                         "0x00000002: #";
  ASSERT_EQ(expected, result.GetString());
}

template <typename T>
void TestDumpMultiLine(std::vector<T> data, lldb::Format format,
                       size_t num_per_line, llvm::StringRef expected) {
  size_t sz_bytes = data.size() * sizeof(T);
  TestDumpImpl(&data[0], sz_bytes, data.size(), sz_bytes, num_per_line,
               0x80000000, format, expected);
}

template <typename T>
void TestDumpMultiLine(const T *data, size_t num_items, lldb::Format format,
                       size_t num_per_line, llvm::StringRef expected) {
  TestDumpImpl(data, sizeof(T) * num_items, sizeof(T), num_items, num_per_line,
               0x80000000, format, expected);
}

TEST(DumpDataExtractorTest, MultiLine) {
  // A vector counts as 1 item regardless of size.
  TestDumpMultiLine(std::vector<uint8_t>{0x11},
                    lldb::Format::eFormatVectorOfUInt8, 1,
                    "0x80000000: {0x11}");
  TestDumpMultiLine(std::vector<uint8_t>{0x11, 0x22},
                    lldb::Format::eFormatVectorOfUInt8, 1,
                    "0x80000000: {0x11 0x22}");

  // If you have multiple vectors then that's multiple items.
  // Here we say that these 2 bytes are actually 2 1 byte vectors.
  const std::vector<uint8_t> vector_data{0x11, 0x22};
  TestDumpMultiLine(vector_data.data(), 2, lldb::Format::eFormatVectorOfUInt8,
                    1, "0x80000000: {0x11}\n0x80000001: {0x22}");

  // Single value formats can span multiple lines.
  const std::vector<uint8_t> bytes{0x11, 0x22, 0x33};
  const char *expected_bytes_3_line = "0x80000000: 0x11\n"
                                      "0x80000001: 0x22\n"
                                      "0x80000002: 0x33";
  TestDumpMultiLine(bytes.data(), bytes.size(), lldb::Format::eFormatHex, 1,
                    expected_bytes_3_line);

  // Lines may not have the full number of items.
  TestDumpMultiLine(bytes.data(), bytes.size(), lldb::Format::eFormatHex, 4,
                    "0x80000000: 0x11 0x22 0x33");
  const char *expected_bytes_2_line = "0x80000000: 0x11 0x22\n"
                                      "0x80000002: 0x33";
  TestDumpMultiLine(bytes.data(), bytes.size(), lldb::Format::eFormatHex, 2,
                    expected_bytes_2_line);

  // The line address accounts for item sizes other than 1 byte.
  const std::vector<uint16_t> shorts{0x1111, 0x2222, 0x3333};
  const char *expected_shorts_2_line = "0x80000000: 0x1111 0x2222\n"
                                       "0x80000004: 0x3333";
  TestDumpMultiLine(shorts.data(), shorts.size(), lldb::Format::eFormatHex, 2,
                    expected_shorts_2_line);

  // The ascii column is positioned using the maximum line length.
  const std::vector<char> chars{'L', 'L', 'D', 'B'};
  const char *expected_chars_2_lines = "0x80000000: 4c 4c 44  LLD\n"
                                       "0x80000003: 42        B";
  TestDumpMultiLine(chars.data(), chars.size(),
                    lldb::Format::eFormatBytesWithASCII, 3,
                    expected_chars_2_lines);
}

void TestDumpWithItemByteSize(size_t item_byte_size, lldb::Format format,
                              llvm::StringRef expected) {
  // We won't be reading this data so anything will do.
  uint8_t dummy = 0;
  TestDumpImpl(&dummy, 1, item_byte_size, 1, 1, LLDB_INVALID_ADDRESS, format,
               expected);
}

TEST(DumpDataExtractorTest, ItemByteSizeErrors) {
  TestDumpWithItemByteSize(
      16, lldb::Format::eFormatBoolean,
      "error: unsupported byte size (16) for boolean format");
  TestDumpWithItemByteSize(21, lldb::Format::eFormatChar,
                           "error: unsupported byte size (21) for char format");
  TestDumpWithItemByteSize(
      18, lldb::Format::eFormatComplexInteger,
      "error: unsupported byte size (18) for complex integer format");

  // The code uses sizeof(long double) for these checks. This changes by host
  // but we know it won't be >16.
  TestDumpWithItemByteSize(
      34, lldb::Format::eFormatComplex,
      "error: unsupported byte size (34) for complex float format");
  TestDumpWithItemByteSize(
      18, lldb::Format::eFormatFloat,
      "error: unsupported byte size (18) for float format");

  // We want sizes to exactly match one of float/double.
  TestDumpWithItemByteSize(
      14, lldb::Format::eFormatComplex,
      "error: unsupported byte size (14) for complex float format");
  TestDumpWithItemByteSize(3, lldb::Format::eFormatFloat,
                           "error: unsupported byte size (3) for float format");

  // We only allow float and double size.
  TestDumpWithItemByteSize(
      1, lldb::Format::eFormatHexFloat,
      "error: unsupported byte size (1) for hex float format");
  TestDumpWithItemByteSize(
      17, lldb::Format::eFormatHexFloat,
      "error: unsupported byte size (17) for hex float format");
}
