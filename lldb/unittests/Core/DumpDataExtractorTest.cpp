//===-- DataDumpExtractorTest.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/DumpDataExtractor.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/StreamString.h"
#include "gtest/gtest.h"
#include <complex>

using namespace lldb;
using namespace lldb_private;

static void test_format_impl(const void *data, size_t data_size,
                             size_t item_count, lldb::Format format,
                             llvm::StringRef expected) {
  StreamString result;
  DataBufferHeap dumpbuffer(data, data_size);
  DataExtractor extractor(dumpbuffer.GetBytes(), dumpbuffer.GetByteSize(),
                          endian::InlHostByteOrder(),
                          /*addr_size=*/4);
  DumpDataExtractor(extractor, &result, 0, format, data_size, item_count, 1, 0,
                    0, 0);
  ASSERT_EQ(expected, result.GetString());
}

template <typename T>
static void test_format(T data, lldb::Format format, llvm::StringRef expected) {
  test_format_impl(&data, sizeof(T), 1, format, expected);
}

static void test_format(llvm::StringRef str, lldb::Format format,
                        llvm::StringRef expected) {
  test_format_impl(str.bytes_begin(),
                   // +1 to include the NULL char as the last byte
                   str.size() + 1, 1, format, expected);
}

template <typename T>
static void test_format(const std::vector<T> data, lldb::Format format,
                        llvm::StringRef expected) {
  test_format_impl(&data[0], data.size() * sizeof(T), data.size(), format,
                   expected);
}

TEST(DumpDataExtractorTest, Formats) {
  test_format<uint8_t>(1, lldb::eFormatDefault, "0x00000000: 0x01");
  test_format<uint8_t>(1, lldb::eFormatBoolean, "0x00000000: true");
  test_format<uint8_t>(0xAA, lldb::eFormatBinary, "0x00000000: 0b10101010");
  test_format<uint8_t>(1, lldb::eFormatBytes, "0x00000000: 01");
  test_format<uint8_t>(1, lldb::eFormatBytesWithASCII, "0x00000000: 01  .");
  test_format('?', lldb::eFormatChar, "0x00000000: '?'");
  test_format('\x1A', lldb::eFormatCharPrintable, "0x00000000: .");
  test_format('#', lldb::eFormatCharPrintable, "0x00000000: #");
  test_format(std::complex<float>(1.2, 3.4), lldb::eFormatComplex,
              "0x00000000: 1.2 + 3.4i");
  test_format(std::complex<double>(4.5, 6.7), lldb::eFormatComplex,
              "0x00000000: 4.5 + 6.7i");

  // long double is not tested here because for some platforms we treat it as 10
  // bytes when the compiler allocates 16 bytes of space for it. (see
  // DataExtractor::GetLongDouble) Meaning that when we extract the second one,
  // it gets the wrong value (it's 6 bytes off). You could manually construct a
  // set of bytes to match the 10 byte format but then if the test runs on a
  // machine where we don't use 10 it'll break.

  test_format(llvm::StringRef("aardvark"), lldb::Format::eFormatCString,
              "0x00000000: \"aardvark\"");
  test_format<uint16_t>(99, lldb::Format::eFormatDecimal, "0x00000000: 99");
  // Just prints as a signed integer.
  test_format(-1, lldb::Format::eFormatEnum, "0x00000000: -1");
  test_format(0xcafef00d, lldb::Format::eFormatHex, "0x00000000: 0xcafef00d");
  test_format(0xcafef00d, lldb::Format::eFormatHexUppercase,
              "0x00000000: 0xCAFEF00D");
  test_format(0.456, lldb::Format::eFormatFloat, "0x00000000: 0.456");
  test_format(9, lldb::Format::eFormatOctal, "0x00000000: 011");
  // Chars packed into an integer.
  test_format<uint32_t>(0x4C4C4442, lldb::Format::eFormatOSType,
                        "0x00000000: 'LLDB'");
  // Unicode8 doesn't have a specific formatter.
  test_format<uint8_t>(0x34, lldb::Format::eFormatUnicode8, "0x00000000: 0x34");
  test_format<uint16_t>(0x1122, lldb::Format::eFormatUnicode16,
                        "0x00000000: U+1122");
  test_format<uint32_t>(0x12345678, lldb::Format::eFormatUnicode32,
                        "0x00000000: U+0x12345678");
  test_format<unsigned int>(654321, lldb::Format::eFormatUnsigned,
                            "0x00000000: 654321");
  // This pointer is printed based on the size of uint64_t, so the test is the
  // same for 32/64 bit host.
  test_format<uint64_t>(0x4444555566667777, lldb::Format::eFormatPointer,
                        "0x00000000: 0x4444555566667777");

  test_format(std::vector<char>{'A', '\x01', 'C'},
              lldb::Format::eFormatVectorOfChar, "0x00000000: {A\\x01C}");
  test_format(std::vector<int8_t>{0, -1, std::numeric_limits<int8_t>::max()},
              lldb::Format::eFormatVectorOfSInt8, "0x00000000: {0 -1 127}");
  test_format(std::vector<uint8_t>{12, 0xFF, 34},
              lldb::Format::eFormatVectorOfUInt8,
              "0x00000000: {0x0c 0xff 0x22}");
  test_format(
      std::vector<int16_t>{-1, 1234, std::numeric_limits<int16_t>::max()},
      lldb::Format::eFormatVectorOfSInt16, "0x00000000: {-1 1234 32767}");
  test_format(std::vector<uint16_t>{0xffff, 0xabcd, 0x1234},
              lldb::Format::eFormatVectorOfUInt16,
              "0x00000000: {0xffff 0xabcd 0x1234}");
  test_format(std::vector<int32_t>{0, -1, std::numeric_limits<int32_t>::max()},
              lldb::Format::eFormatVectorOfSInt32,
              "0x00000000: {0 -1 2147483647}");
  test_format(std::vector<uint32_t>{0, 0xffffffff, 0x1234abcd},
              lldb::Format::eFormatVectorOfUInt32,
              "0x00000000: {0x00000000 0xffffffff 0x1234abcd}");
  test_format(std::vector<int64_t>{0, -1, std::numeric_limits<int64_t>::max()},
              lldb::Format::eFormatVectorOfSInt64,
              "0x00000000: {0 -1 9223372036854775807}");
  test_format(std::vector<uint64_t>{0, 0xaaaabbbbccccdddd},
              lldb::Format::eFormatVectorOfUInt64,
              "0x00000000: {0x0000000000000000 0xaaaabbbbccccdddd}");

  // See half2float for format details.
  test_format(std::vector<uint16_t>{0xabcd, 0x1234},
              lldb::Format::eFormatVectorOfFloat16,
              "0x00000000: {-0.0609436 0.000757217}");
  test_format(std::vector<float>{std::numeric_limits<float>::min(),
                                 std::numeric_limits<float>::max()},
              lldb::Format::eFormatVectorOfFloat32,
              "0x00000000: {1.17549e-38 3.40282e+38}");
  test_format(std::vector<double>{std::numeric_limits<double>::min(),
                                  std::numeric_limits<double>::max()},
              lldb::Format::eFormatVectorOfFloat64,
              "0x00000000: {2.2250738585072e-308 1.79769313486232e+308}");

  // Not sure we can rely on having uint128_t everywhere so emulate with
  // uint64_t.
  test_format(
      std::vector<uint64_t>{0x1, 0x1111222233334444, 0xaaaabbbbccccdddd, 0x0},
      lldb::Format::eFormatVectorOfUInt128,
      "0x00000000: {0x11112222333344440000000000000001 "
      "0x0000000000000000aaaabbbbccccdddd}");

  test_format(std::vector<int>{2, 4}, lldb::Format::eFormatComplexInteger,
              "0x00000000: 2 + 4i");

  // Without an execution context this just prints the pointer on its own.
  test_format<uint32_t>(0x11223344, lldb::Format::eFormatAddressInfo,
                        "0x00000000: 0x11223344");

  // Input not written in hex form because that requires C++17.
  test_format<float>(10, lldb::Format::eFormatHexFloat, "0x00000000: 0x1.4p3");

  // Can't disassemble without an execution context.
  test_format<uint32_t>(0xcafef00d, lldb::Format::eFormatInstruction,
                        "invalid target");

  // Has no special handling, intended for use elsewhere.
  test_format<int>(99, lldb::Format::eFormatVoid, "0x00000000: 0x00000063");
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
