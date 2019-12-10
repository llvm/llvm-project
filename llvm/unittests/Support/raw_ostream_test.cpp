//===- llvm/unittest/Support/raw_ostream_test.cpp - raw_ostream tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

template<typename T> std::string printToString(const T &Value) {
  std::string res;
  return (llvm::raw_string_ostream(res) << Value).str();
}

/// printToString - Print the given value to a stream which only has \arg
/// BytesLeftInBuffer bytes left in the buffer. This is useful for testing edge
/// cases in the buffer handling logic.
template<typename T> std::string printToString(const T &Value,
                                               unsigned BytesLeftInBuffer) {
  // FIXME: This is relying on internal knowledge of how raw_ostream works to
  // get the buffer position right.
  SmallString<256> SVec;
  assert(BytesLeftInBuffer < 256 && "Invalid buffer count!");
  llvm::raw_svector_ostream OS(SVec);
  unsigned StartIndex = 256 - BytesLeftInBuffer;
  for (unsigned i = 0; i != StartIndex; ++i)
    OS << '?';
  OS << Value;
  return OS.str().substr(StartIndex);
}

template<typename T> std::string printToStringUnbuffered(const T &Value) {
  std::string res;
  llvm::raw_string_ostream OS(res);
  OS.SetUnbuffered();
  OS << Value;
  return res;
}

struct X {};

raw_ostream &operator<<(raw_ostream &OS, const X &) { return OS << 'X'; }

TEST(raw_ostreamTest, Types_Buffered) {
  // Char
  EXPECT_EQ("c", printToString('c'));

  // String
  EXPECT_EQ("hello", printToString("hello"));
  EXPECT_EQ("hello", printToString(std::string("hello")));

  // Int
  EXPECT_EQ("0", printToString(0));
  EXPECT_EQ("2425", printToString(2425));
  EXPECT_EQ("-2425", printToString(-2425));

  // Long long
  EXPECT_EQ("0", printToString(0LL));
  EXPECT_EQ("257257257235709", printToString(257257257235709LL));
  EXPECT_EQ("-257257257235709", printToString(-257257257235709LL));

  // Double
  EXPECT_EQ("1.100000e+00", printToString(1.1));

  // void*
  EXPECT_EQ("0x0", printToString((void*) nullptr));
  EXPECT_EQ("0xbeef", printToString((void*) 0xbeefLL));
  EXPECT_EQ("0xdeadbeef", printToString((void*) 0xdeadbeefLL));

  // Min and max.
  EXPECT_EQ("18446744073709551615", printToString(UINT64_MAX));
  EXPECT_EQ("-9223372036854775808", printToString(INT64_MIN));

  // X, checking free operator<<().
  EXPECT_EQ("X", printToString(X{}));
}

TEST(raw_ostreamTest, Types_Unbuffered) {  
  // Char
  EXPECT_EQ("c", printToStringUnbuffered('c'));

  // String
  EXPECT_EQ("hello", printToStringUnbuffered("hello"));
  EXPECT_EQ("hello", printToStringUnbuffered(std::string("hello")));

  // Int
  EXPECT_EQ("0", printToStringUnbuffered(0));
  EXPECT_EQ("2425", printToStringUnbuffered(2425));
  EXPECT_EQ("-2425", printToStringUnbuffered(-2425));

  // Long long
  EXPECT_EQ("0", printToStringUnbuffered(0LL));
  EXPECT_EQ("257257257235709", printToStringUnbuffered(257257257235709LL));
  EXPECT_EQ("-257257257235709", printToStringUnbuffered(-257257257235709LL));

  // Double
  EXPECT_EQ("1.100000e+00", printToStringUnbuffered(1.1));

  // void*
  EXPECT_EQ("0x0", printToStringUnbuffered((void*) nullptr));
  EXPECT_EQ("0xbeef", printToStringUnbuffered((void*) 0xbeefLL));
  EXPECT_EQ("0xdeadbeef", printToStringUnbuffered((void*) 0xdeadbeefLL));

  // Min and max.
  EXPECT_EQ("18446744073709551615", printToStringUnbuffered(UINT64_MAX));
  EXPECT_EQ("-9223372036854775808", printToStringUnbuffered(INT64_MIN));

  // X, checking free operator<<().
  EXPECT_EQ("X", printToString(X{}));
}

TEST(raw_ostreamTest, BufferEdge) {  
  EXPECT_EQ("1.20", printToString(format("%.2f", 1.2), 1));
  EXPECT_EQ("1.20", printToString(format("%.2f", 1.2), 2));
  EXPECT_EQ("1.20", printToString(format("%.2f", 1.2), 3));
  EXPECT_EQ("1.20", printToString(format("%.2f", 1.2), 4));
  EXPECT_EQ("1.20", printToString(format("%.2f", 1.2), 10));
}

TEST(raw_ostreamTest, TinyBuffer) {
  std::string Str;
  raw_string_ostream OS(Str);
  OS.SetBufferSize(1);
  OS << "hello";
  OS << 1;
  OS << 'w' << 'o' << 'r' << 'l' << 'd';
  EXPECT_EQ("hello1world", OS.str());
}

TEST(raw_ostreamTest, WriteEscaped) {
  std::string Str;

  Str = "";
  raw_string_ostream(Str).write_escaped("hi");
  EXPECT_EQ("hi", Str);

  Str = "";
  raw_string_ostream(Str).write_escaped("\\\t\n\"");
  EXPECT_EQ("\\\\\\t\\n\\\"", Str);

  Str = "";
  raw_string_ostream(Str).write_escaped("\1\10\200");
  EXPECT_EQ("\\001\\010\\200", Str);
}

TEST(raw_ostreamTest, Justify) {  
  EXPECT_EQ("xyz   ", printToString(left_justify("xyz", 6), 6));
  EXPECT_EQ("abc",    printToString(left_justify("abc", 3), 3));
  EXPECT_EQ("big",    printToString(left_justify("big", 1), 3));
  EXPECT_EQ("   xyz", printToString(right_justify("xyz", 6), 6));
  EXPECT_EQ("abc",    printToString(right_justify("abc", 3), 3));
  EXPECT_EQ("big",    printToString(right_justify("big", 1), 3));
  EXPECT_EQ("   on    ",    printToString(center_justify("on", 9), 9));
  EXPECT_EQ("   off    ",    printToString(center_justify("off", 10), 10));
  EXPECT_EQ("single ",    printToString(center_justify("single", 7), 7));
  EXPECT_EQ("none",    printToString(center_justify("none", 1), 4));
  EXPECT_EQ("none",    printToString(center_justify("none", 1), 1));
}

TEST(raw_ostreamTest, FormatHex) {  
  EXPECT_EQ("0x1234",     printToString(format_hex(0x1234, 6), 6));
  EXPECT_EQ("0x001234",   printToString(format_hex(0x1234, 8), 8));
  EXPECT_EQ("0x00001234", printToString(format_hex(0x1234, 10), 10));
  EXPECT_EQ("0x1234",     printToString(format_hex(0x1234, 4), 6));
  EXPECT_EQ("0xff",       printToString(format_hex(255, 4), 4));
  EXPECT_EQ("0xFF",       printToString(format_hex(255, 4, true), 4));
  EXPECT_EQ("0x1",        printToString(format_hex(1, 3), 3));
  EXPECT_EQ("0x12",       printToString(format_hex(0x12, 3), 4));
  EXPECT_EQ("0x123",      printToString(format_hex(0x123, 3), 5));
  EXPECT_EQ("FF",         printToString(format_hex_no_prefix(0xFF, 2, true), 4));
  EXPECT_EQ("ABCD",       printToString(format_hex_no_prefix(0xABCD, 2, true), 4));
  EXPECT_EQ("0xffffffffffffffff",     
                          printToString(format_hex(UINT64_MAX, 18), 18));
  EXPECT_EQ("0x8000000000000000",     
                          printToString(format_hex((INT64_MIN), 18), 18));
}

TEST(raw_ostreamTest, FormatDecimal) {  
  EXPECT_EQ("   0",        printToString(format_decimal(0, 4), 4));
  EXPECT_EQ("  -1",        printToString(format_decimal(-1, 4), 4));
  EXPECT_EQ("    -1",      printToString(format_decimal(-1, 6), 6));
  EXPECT_EQ("1234567890",  printToString(format_decimal(1234567890, 10), 10));
  EXPECT_EQ("  9223372036854775807", 
                          printToString(format_decimal(INT64_MAX, 21), 21));
  EXPECT_EQ(" -9223372036854775808", 
                          printToString(format_decimal(INT64_MIN, 21), 21));
}

static std::string formatted_bytes_str(ArrayRef<uint8_t> Bytes,
                                       llvm::Optional<uint64_t> Offset = None,
                                       uint32_t NumPerLine = 16,
                                       uint8_t ByteGroupSize = 4) {
  std::string S;
  raw_string_ostream Str(S);
  Str << format_bytes(Bytes, Offset, NumPerLine, ByteGroupSize);
  Str.flush();
  return S;
}

static std::string format_bytes_with_ascii_str(ArrayRef<uint8_t> Bytes,
                                               Optional<uint64_t> Offset = None,
                                               uint32_t NumPerLine = 16,
                                               uint8_t ByteGroupSize = 4) {
  std::string S;
  raw_string_ostream Str(S);
  Str << format_bytes_with_ascii(Bytes, Offset, NumPerLine, ByteGroupSize);
  Str.flush();
  return S;
}

TEST(raw_ostreamTest, FormattedHexBytes) {
  std::vector<uint8_t> Buf = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
                              'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                              's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0',
                              '1', '2', '3', '4', '5', '6', '7', '8', '9'};
  ArrayRef<uint8_t> B(Buf);

  // Test invalid input.
  EXPECT_EQ("", formatted_bytes_str(ArrayRef<uint8_t>()));
  EXPECT_EQ("", format_bytes_with_ascii_str(ArrayRef<uint8_t>()));
  //----------------------------------------------------------------------
  // Test hex byte output with the default 4 byte groups
  //----------------------------------------------------------------------
  EXPECT_EQ("61", formatted_bytes_str(B.take_front()));
  EXPECT_EQ("61626364 65", formatted_bytes_str(B.take_front(5)));
  // Test that 16 bytes get written to a line correctly.
  EXPECT_EQ("61626364 65666768 696a6b6c 6d6e6f70",
            formatted_bytes_str(B.take_front(16)));
  // Test raw bytes with default 16 bytes per line wrapping.
  EXPECT_EQ("61626364 65666768 696a6b6c 6d6e6f70\n71",
            formatted_bytes_str(B.take_front(17)));
  // Test raw bytes with 1 bytes per line wrapping.
  EXPECT_EQ("61\n62\n63\n64\n65\n66",
            formatted_bytes_str(B.take_front(6), None, 1));
  // Test raw bytes with 7 bytes per line wrapping.
  EXPECT_EQ("61626364 656667\n68696a6b 6c6d6e\n6f7071",
            formatted_bytes_str(B.take_front(17), None, 7));
  // Test raw bytes with 8 bytes per line wrapping.
  EXPECT_EQ("61626364 65666768\n696a6b6c 6d6e6f70\n71",
            formatted_bytes_str(B.take_front(17), None, 8));
  //----------------------------------------------------------------------
  // Test hex byte output with the 1 byte groups
  //----------------------------------------------------------------------
  EXPECT_EQ("61 62 63 64 65",
            formatted_bytes_str(B.take_front(5), None, 16, 1));
  // Test that 16 bytes get written to a line correctly.
  EXPECT_EQ("61 62 63 64 65 66 67 68 69 6a 6b 6c 6d 6e 6f 70",
            formatted_bytes_str(B.take_front(16), None, 16, 1));
  // Test raw bytes with default 16 bytes per line wrapping.
  EXPECT_EQ("61 62 63 64 65 66 67 68 69 6a 6b 6c 6d 6e 6f 70\n71",
            formatted_bytes_str(B.take_front(17), None, 16, 1));
  // Test raw bytes with 7 bytes per line wrapping.
  EXPECT_EQ("61 62 63 64 65 66 67\n68 69 6a 6b 6c 6d 6e\n6f 70 71",
            formatted_bytes_str(B.take_front(17), None, 7, 1));
  // Test raw bytes with 8 bytes per line wrapping.
  EXPECT_EQ("61 62 63 64 65 66 67 68\n69 6a 6b 6c 6d 6e 6f 70\n71",
            formatted_bytes_str(B.take_front(17), None, 8, 1));

  //----------------------------------------------------------------------
  // Test hex byte output with the 2 byte groups
  //----------------------------------------------------------------------
  EXPECT_EQ("6162 6364 65", formatted_bytes_str(B.take_front(5), None, 16, 2));
  // Test that 16 bytes get written to a line correctly.
  EXPECT_EQ("6162 6364 6566 6768 696a 6b6c 6d6e 6f70",
            formatted_bytes_str(B.take_front(16), None, 16, 2));
  // Test raw bytes with default 16 bytes per line wrapping.
  EXPECT_EQ("6162 6364 6566 6768 696a 6b6c 6d6e 6f70\n71",
            formatted_bytes_str(B.take_front(17), None, 16, 2));
  // Test raw bytes with 7 bytes per line wrapping.
  EXPECT_EQ("6162 6364 6566 67\n6869 6a6b 6c6d 6e\n6f70 71",
            formatted_bytes_str(B.take_front(17), None, 7, 2));
  // Test raw bytes with 8 bytes per line wrapping.
  EXPECT_EQ("6162 6364 6566 6768\n696a 6b6c 6d6e 6f70\n71",
            formatted_bytes_str(B.take_front(17), None, 8, 2));

  //----------------------------------------------------------------------
  // Test hex bytes with offset with the default 4 byte groups.
  //----------------------------------------------------------------------
  EXPECT_EQ("0000: 61", formatted_bytes_str(B.take_front(), 0x0));
  EXPECT_EQ("1000: 61", formatted_bytes_str(B.take_front(), 0x1000));
  EXPECT_EQ("1000: 61\n1001: 62",
            formatted_bytes_str(B.take_front(2), 0x1000, 1));
  //----------------------------------------------------------------------
  // Test hex bytes with ASCII with the default 4 byte groups.
  //----------------------------------------------------------------------
  EXPECT_EQ("61626364 65666768 696a6b6c 6d6e6f70  |abcdefghijklmnop|",
            format_bytes_with_ascii_str(B.take_front(16)));
  EXPECT_EQ("61626364 65666768  |abcdefgh|\n"
            "696a6b6c 6d6e6f70  |ijklmnop|",
            format_bytes_with_ascii_str(B.take_front(16), None, 8));
  EXPECT_EQ("61626364 65666768  |abcdefgh|\n696a6b6c           |ijkl|",
            format_bytes_with_ascii_str(B.take_front(12), None, 8));
  std::vector<uint8_t> Unprintable = {'a', '\x1e', 'b', '\x1f'};
  // Make sure the ASCII is still lined up correctly when fewer bytes than 16
  // bytes per line are available. The ASCII should still be aligned as if 16
  // bytes of hex might be displayed.
  EXPECT_EQ("611e621f                             |a.b.|",
            format_bytes_with_ascii_str(Unprintable));
  //----------------------------------------------------------------------
  // Test hex bytes with ASCII with offsets with the default 4 byte groups.
  //----------------------------------------------------------------------
  EXPECT_EQ("0000: 61626364 65666768 "
            "696a6b6c 6d6e6f70  |abcdefghijklmnop|",
            format_bytes_with_ascii_str(B.take_front(16), 0));
  EXPECT_EQ("0000: 61626364 65666768  |abcdefgh|\n"
            "0008: 696a6b6c 6d6e6f70  |ijklmnop|",
            format_bytes_with_ascii_str(B.take_front(16), 0, 8));
  EXPECT_EQ("0000: 61626364 656667  |abcdefg|\n"
            "0007: 68696a6b 6c      |hijkl|",
            format_bytes_with_ascii_str(B.take_front(12), 0, 7));

  //----------------------------------------------------------------------
  // Test hex bytes with ASCII with offsets with the default 2 byte groups.
  //----------------------------------------------------------------------
  EXPECT_EQ("0000: 6162 6364 6566 6768 "
            "696a 6b6c 6d6e 6f70  |abcdefghijklmnop|",
            format_bytes_with_ascii_str(B.take_front(16), 0, 16, 2));
  EXPECT_EQ("0000: 6162 6364 6566 6768  |abcdefgh|\n"
            "0008: 696a 6b6c 6d6e 6f70  |ijklmnop|",
            format_bytes_with_ascii_str(B.take_front(16), 0, 8, 2));
  EXPECT_EQ("0000: 6162 6364 6566 67  |abcdefg|\n"
            "0007: 6869 6a6b 6c       |hijkl|",
            format_bytes_with_ascii_str(B.take_front(12), 0, 7, 2));

  //----------------------------------------------------------------------
  // Test hex bytes with ASCII with offsets with the default 1 byte groups.
  //----------------------------------------------------------------------
  EXPECT_EQ("0000: 61 62 63 64 65 66 67 68 "
            "69 6a 6b 6c 6d 6e 6f 70  |abcdefghijklmnop|",
            format_bytes_with_ascii_str(B.take_front(16), 0, 16, 1));
  EXPECT_EQ("0000: 61 62 63 64 65 66 67 68  |abcdefgh|\n"
            "0008: 69 6a 6b 6c 6d 6e 6f 70  |ijklmnop|",
            format_bytes_with_ascii_str(B.take_front(16), 0, 8, 1));
  EXPECT_EQ("0000: 61 62 63 64 65 66 67  |abcdefg|\n"
            "0007: 68 69 6a 6b 6c        |hijkl|",
            format_bytes_with_ascii_str(B.take_front(12), 0, 7, 1));
}

TEST(raw_fd_ostreamTest, multiple_raw_fd_ostream_to_stdout) {
  std::error_code EC;

  { raw_fd_ostream("-", EC, sys::fs::OpenFlags::OF_None); }
  { raw_fd_ostream("-", EC, sys::fs::OpenFlags::OF_None); }
}
}
