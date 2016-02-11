//===- llvm/unittest/Support/raw_ostream_test.cpp - raw_ostream tests -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

template<typename T> std::string printToString(const T &Value) {
  std::string res;
  llvm::raw_string_ostream(res) << Value;
  return res;    
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


}
