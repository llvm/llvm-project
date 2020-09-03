//===-- StreamTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/StreamString.h"
#include "gtest/gtest.h"

using namespace lldb_private;

namespace {
struct StreamTest : ::testing::Test {
  // Note: Stream is an abstract class, so we use StreamString to test it. To
  // make it easier to change this later, only methods in this class explicitly
  // refer to the StringStream class.
  StreamString s;
  // We return here a std::string because that way gtest can print better
  // assertion messages.
  std::string TakeValue() {
    std::string result = s.GetString().str();
    s.Clear();
    return result;
  }
};
}

namespace {
// A StreamTest where we expect the Stream output to be binary.
struct BinaryStreamTest : StreamTest {
  void SetUp() override {
    s.GetFlags().Set(Stream::eBinary);
  }
};
}

TEST_F(StreamTest, AddressPrefix) {
  DumpAddress(s.AsRawOstream(), 0x1, 1, "foo");
  EXPECT_EQ("foo0x01", TakeValue());
}

TEST_F(StreamTest, AddressEmptyPrefix) {
  DumpAddress(s.AsRawOstream(), 0x1, 1, nullptr);
  EXPECT_EQ("0x01", TakeValue());
  DumpAddress(s.AsRawOstream(), 0x1, 1, "");
  EXPECT_EQ("0x01", TakeValue());
}

TEST_F(StreamTest, AddressSuffix) {
  DumpAddress(s.AsRawOstream(), 0x1, 1, nullptr, "foo");
  EXPECT_EQ("0x01foo", TakeValue());
}

TEST_F(StreamTest, AddressNoSuffix) {
  DumpAddress(s.AsRawOstream(), 0x1, 1, nullptr, nullptr);
  EXPECT_EQ("0x01", TakeValue());
  DumpAddress(s.AsRawOstream(), 0x1, 1, nullptr, "");
  EXPECT_EQ("0x01", TakeValue());
}

TEST_F(StreamTest, AddressPrefixAndSuffix) {
  DumpAddress(s.AsRawOstream(), 0x1, 1, "foo", "bar");
  EXPECT_EQ("foo0x01bar", TakeValue());
}

TEST_F(StreamTest, AddressSize) {
  DumpAddress(s.AsRawOstream(), 0x0, 0);
  EXPECT_EQ("0x0", TakeValue());
  DumpAddress(s.AsRawOstream(), 0x1, 0);
  EXPECT_EQ("0x1", TakeValue());

  DumpAddress(s.AsRawOstream(), 0x1, 1);
  EXPECT_EQ("0x01", TakeValue());
  DumpAddress(s.AsRawOstream(), 0xf1, 1);
  EXPECT_EQ("0xf1", TakeValue());
  DumpAddress(s.AsRawOstream(), 0xff, 1);
  EXPECT_EQ("0xff", TakeValue());
  DumpAddress(s.AsRawOstream(), 0x100, 1);
  EXPECT_EQ("0x100", TakeValue());

  DumpAddress(s.AsRawOstream(), 0xf00, 4);
  EXPECT_EQ("0x00000f00", TakeValue());
  DumpAddress(s.AsRawOstream(), 0x100, 8);
  EXPECT_EQ("0x0000000000000100", TakeValue());
}

TEST_F(StreamTest, AddressRange) {
  DumpAddressRange(s.AsRawOstream(), 0x100, 0x101, 2);
  EXPECT_EQ("[0x0100-0x0101)", TakeValue());
}

TEST_F(StreamTest, AddressRangeEmptyRange) {
  DumpAddressRange(s.AsRawOstream(), 0x100, 0x100, 2);
  EXPECT_EQ("[0x0100-0x0100)", TakeValue());
  DumpAddressRange(s.AsRawOstream(), 0x0, 0x0, 2);
  EXPECT_EQ("[0x0000-0x0000)", TakeValue());
}

TEST_F(StreamTest, AddressRangeInvalidRange) {
  DumpAddressRange(s.AsRawOstream(), 0x100, 0x0FF, 2);
  EXPECT_EQ("[0x0100-0x00ff)", TakeValue());
  DumpAddressRange(s.AsRawOstream(), 0x100, 0x0, 2);
  EXPECT_EQ("[0x0100-0x0000)", TakeValue());
}

TEST_F(StreamTest, AddressRangeSize) {
  DumpAddressRange(s.AsRawOstream(), 0x100, 0x101, 0);
  EXPECT_EQ("[0x100-0x101)", TakeValue());
  DumpAddressRange(s.AsRawOstream(), 0x100, 0x101, 2);
  EXPECT_EQ("[0x0100-0x0101)", TakeValue());
  DumpAddressRange(s.AsRawOstream(), 0x100, 0x101, 4);
  EXPECT_EQ("[0x00000100-0x00000101)", TakeValue());

  DumpAddressRange(s.AsRawOstream(), 0x100, 0x101, 4);
  EXPECT_EQ("[0x00000100-0x00000101)", TakeValue());
  DumpAddressRange(s.AsRawOstream(), 0x1, 0x101, 4);
  EXPECT_EQ("[0x00000001-0x00000101)", TakeValue());
  DumpAddressRange(s.AsRawOstream(), 0x101, 0x1, 4);
  EXPECT_EQ("[0x00000101-0x00000001)", TakeValue());

  DumpAddressRange(s.AsRawOstream(), 0x1, 0x101, 1);
  EXPECT_EQ("[0x01-0x101)", TakeValue());
}

TEST_F(StreamTest, ChangingByteOrder) {
  s.SetByteOrder(lldb::eByteOrderPDP);
  EXPECT_EQ(lldb::eByteOrderPDP, s.GetByteOrder());
}

TEST_F(StreamTest, SetIndentLevel) {
  s.Indent("a");
  EXPECT_EQ("a", TakeValue());

  s.SetIndentLevel(3);
  s.Indent("a");
  EXPECT_EQ("   a", TakeValue());

  s.SetIndentLevel(2);
  s.Indent("a");
  EXPECT_EQ("  a", TakeValue());

  s.SetIndentLevel(0);
  s.Indent("a");
  EXPECT_EQ("a", TakeValue());
}

TEST_F(StreamTest, Indent) {
  s.SetIndentLevel(2);
  const char *nullptr_cstring = nullptr;
  s.Indent(nullptr_cstring);
  EXPECT_EQ("  ", TakeValue());

  s.Indent("");
  EXPECT_EQ("  ", TakeValue());

  s.Indent(" ");
  EXPECT_EQ("   ", TakeValue());

  s.Indent(" aa");
  EXPECT_EQ("   aa", TakeValue());
}

TEST_F(StreamTest, PutChar) {
  s.PutChar('a');
  EXPECT_EQ(1U, s.GetWrittenBytes());
  EXPECT_EQ("a", TakeValue());

  s.PutChar('1');
  EXPECT_EQ(1U, s.GetWrittenBytes());
  EXPECT_EQ("1", TakeValue());
}

TEST_F(StreamTest, PutCharWhitespace) {
  s.PutChar(' ');
  EXPECT_EQ(1U, s.GetWrittenBytes());
  EXPECT_EQ(" ", TakeValue());

  s.PutChar('\n');
  EXPECT_EQ(1U, s.GetWrittenBytes());
  EXPECT_EQ("\n", TakeValue());

  s.PutChar('\r');
  EXPECT_EQ(1U, s.GetWrittenBytes());
  EXPECT_EQ("\r", TakeValue());

  s.PutChar('\t');
  EXPECT_EQ(1U, s.GetWrittenBytes());
  EXPECT_EQ("\t", TakeValue());
}

TEST_F(StreamTest, PutCString) {
  s.PutCString("");
  EXPECT_EQ(0U, s.GetWrittenBytes());
  EXPECT_EQ("", TakeValue());

  s.PutCString("foobar");
  EXPECT_EQ(6U, s.GetWrittenBytes());
  EXPECT_EQ("foobar", TakeValue());

  s.PutCString(" ");
  EXPECT_EQ(1U, s.GetWrittenBytes());
  EXPECT_EQ(" ", TakeValue());
}

TEST_F(StreamTest, PutCStringWithStringRef) {
  s.PutCString(llvm::StringRef(""));
  EXPECT_EQ(0U, s.GetWrittenBytes());
  EXPECT_EQ("", TakeValue());

  s.PutCString(llvm::StringRef("foobar"));
  EXPECT_EQ(6U, s.GetWrittenBytes());
  EXPECT_EQ("foobar", TakeValue());

  s.PutCString(llvm::StringRef(" "));
  EXPECT_EQ(1U, s.GetWrittenBytes());
  EXPECT_EQ(" ", TakeValue());
}

TEST_F(StreamTest, QuotedCString) {
  s.QuotedCString("foo");
  EXPECT_EQ(5U, s.GetWrittenBytes());
  EXPECT_EQ(R"("foo")", TakeValue());

  s.QuotedCString("ba r");
  EXPECT_EQ(6U, s.GetWrittenBytes());
  EXPECT_EQ(R"("ba r")", TakeValue());

  s.QuotedCString(" ");
  EXPECT_EQ(3U, s.GetWrittenBytes());
  EXPECT_EQ(R"(" ")", TakeValue());
}

TEST_F(StreamTest, PutCharNull) {
  s.PutChar('\0');
  EXPECT_EQ(1U, s.GetWrittenBytes());
  EXPECT_EQ(std::string("\0", 1), TakeValue());

  s.PutChar('a');
  EXPECT_EQ(1U, s.GetWrittenBytes());
  EXPECT_EQ(std::string("a", 1), TakeValue());
}

TEST_F(StreamTest, PutStringAsRawHex8) {
  s.PutStringAsRawHex8("");
  EXPECT_EQ(0U, s.GetWrittenBytes());
  EXPECT_EQ("", TakeValue());

  s.PutStringAsRawHex8("foobar");
  EXPECT_EQ(12U, s.GetWrittenBytes());
  EXPECT_EQ("666f6f626172", TakeValue());

  s.PutStringAsRawHex8(" ");
  EXPECT_EQ(2U, s.GetWrittenBytes());
  EXPECT_EQ("20", TakeValue());
}

TEST_F(StreamTest, PutHex8) {
  s.PutHex8((uint8_t)55);
  EXPECT_EQ(2U, s.GetWrittenBytes());
  EXPECT_EQ("37", TakeValue());

  s.PutHex8(std::numeric_limits<uint8_t>::max());
  EXPECT_EQ(2U, s.GetWrittenBytes());
  EXPECT_EQ("ff", TakeValue());

  s.PutHex8((uint8_t)0);
  EXPECT_EQ(2U, s.GetWrittenBytes());
  EXPECT_EQ("00", TakeValue());
}

TEST_F(StreamTest, PutNHex8) {
  s.PutNHex8(0, (uint8_t)55);
  EXPECT_EQ(0U, s.GetWrittenBytes());
  EXPECT_EQ("", TakeValue());

  s.PutNHex8(1, (uint8_t)55);
  EXPECT_EQ(2U, s.GetWrittenBytes());
  EXPECT_EQ("37", TakeValue());

  s.PutNHex8(2, (uint8_t)55);
  EXPECT_EQ(4U, s.GetWrittenBytes());
  EXPECT_EQ("3737", TakeValue());

  s.PutNHex8(1, (uint8_t)56);
  EXPECT_EQ(2U, s.GetWrittenBytes());
  EXPECT_EQ("38", TakeValue());
}

TEST_F(StreamTest, PutHex16ByteOrderLittle) {
  s.PutHex16(0x1234U, lldb::eByteOrderLittle);
  EXPECT_EQ(4U, s.GetWrittenBytes());
  EXPECT_EQ("3412", TakeValue());

  s.PutHex16(std::numeric_limits<uint16_t>::max(), lldb::eByteOrderLittle);
  EXPECT_EQ(4U, s.GetWrittenBytes());
  EXPECT_EQ("ffff", TakeValue());

  s.PutHex16(0U, lldb::eByteOrderLittle);
  EXPECT_EQ(4U, s.GetWrittenBytes());
  EXPECT_EQ("0000", TakeValue());
}

TEST_F(StreamTest, PutHex16ByteOrderBig) {
  s.PutHex16(0x1234U, lldb::eByteOrderBig);
  EXPECT_EQ(4U, s.GetWrittenBytes());
  EXPECT_EQ("1234", TakeValue());

  s.PutHex16(std::numeric_limits<uint16_t>::max(), lldb::eByteOrderBig);
  EXPECT_EQ(4U, s.GetWrittenBytes());
  EXPECT_EQ("ffff", TakeValue());

  s.PutHex16(0U, lldb::eByteOrderBig);
  EXPECT_EQ(4U, s.GetWrittenBytes());
  EXPECT_EQ("0000", TakeValue());
}

TEST_F(StreamTest, PutHex32ByteOrderLittle) {
  s.PutHex32(0x12345678U, lldb::eByteOrderLittle);
  EXPECT_EQ(8U, s.GetWrittenBytes());
  EXPECT_EQ("78563412", TakeValue());

  s.PutHex32(std::numeric_limits<uint32_t>::max(), lldb::eByteOrderLittle);
  EXPECT_EQ(8U, s.GetWrittenBytes());
  EXPECT_EQ("ffffffff", TakeValue());

  s.PutHex32(0U, lldb::eByteOrderLittle);
  EXPECT_EQ(8U, s.GetWrittenBytes());
  EXPECT_EQ("00000000", TakeValue());
}

TEST_F(StreamTest, PutHex32ByteOrderBig) {
  s.PutHex32(0x12345678U, lldb::eByteOrderBig);
  EXPECT_EQ(8U, s.GetWrittenBytes());
  EXPECT_EQ("12345678", TakeValue());

  s.PutHex32(std::numeric_limits<uint32_t>::max(), lldb::eByteOrderBig);
  EXPECT_EQ(8U, s.GetWrittenBytes());
  EXPECT_EQ("ffffffff", TakeValue());

  s.PutHex32(0U, lldb::eByteOrderBig);
  EXPECT_EQ(8U, s.GetWrittenBytes());
  EXPECT_EQ("00000000", TakeValue());
}

TEST_F(StreamTest, PutHex64ByteOrderLittle) {
  s.PutHex64(0x1234567890ABCDEFU, lldb::eByteOrderLittle);
  EXPECT_EQ(16U, s.GetWrittenBytes());
  EXPECT_EQ("efcdab9078563412", TakeValue());

  s.PutHex64(std::numeric_limits<uint64_t>::max(), lldb::eByteOrderLittle);
  EXPECT_EQ(16U, s.GetWrittenBytes());
  EXPECT_EQ("ffffffffffffffff", TakeValue());

  s.PutHex64(0U, lldb::eByteOrderLittle);
  EXPECT_EQ(16U, s.GetWrittenBytes());
  EXPECT_EQ("0000000000000000", TakeValue());
}

TEST_F(StreamTest, PutHex64ByteOrderBig) {
  s.PutHex64(0x1234567890ABCDEFU, lldb::eByteOrderBig);
  EXPECT_EQ(16U, s.GetWrittenBytes());
  EXPECT_EQ("1234567890abcdef", TakeValue());

  s.PutHex64(std::numeric_limits<uint64_t>::max(), lldb::eByteOrderBig);
  EXPECT_EQ(16U, s.GetWrittenBytes());
  EXPECT_EQ("ffffffffffffffff", TakeValue());

  s.PutHex64(0U, lldb::eByteOrderBig);
  EXPECT_EQ(16U, s.GetWrittenBytes());
  EXPECT_EQ("0000000000000000", TakeValue());
}

TEST_F(StreamTest, PutMaxHex64ByteOrderBig) {
  std::size_t bytes;
  bytes = s.PutMaxHex64(0x12U, 1, lldb::eByteOrderBig);
  EXPECT_EQ(2U, bytes);
  bytes = s.PutMaxHex64(0x1234U, 2, lldb::eByteOrderBig);
  EXPECT_EQ(4U, bytes);
  bytes = s.PutMaxHex64(0x12345678U, 4, lldb::eByteOrderBig);
  EXPECT_EQ(8U, bytes);
  bytes = s.PutMaxHex64(0x1234567890ABCDEFU, 8, lldb::eByteOrderBig);
  EXPECT_EQ(16U, bytes);
  EXPECT_EQ(30U, s.GetWrittenBytes());
  EXPECT_EQ("121234123456781234567890abcdef", TakeValue());
}

TEST_F(StreamTest, PutMaxHex64ByteOrderLittle) {
  std::size_t bytes;
  bytes = s.PutMaxHex64(0x12U, 1, lldb::eByteOrderLittle);
  EXPECT_EQ(2U, bytes);
  bytes = s.PutMaxHex64(0x1234U, 2, lldb::eByteOrderLittle);
  EXPECT_EQ(4U, bytes);
  bytes = s.PutMaxHex64(0x12345678U, 4, lldb::eByteOrderLittle);
  EXPECT_EQ(8U, bytes);
  bytes = s.PutMaxHex64(0x1234567890ABCDEFU, 8, lldb::eByteOrderLittle);
  EXPECT_EQ(16U, bytes);
  EXPECT_EQ(30U, s.GetWrittenBytes());
  EXPECT_EQ("12341278563412efcdab9078563412", TakeValue());
}

// Shift operator tests.

TEST_F(StreamTest, ShiftOperatorChars) {
  s << 'a' << 'b';
  EXPECT_EQ(2U, s.GetWrittenBytes());
  EXPECT_EQ("ab", TakeValue());
}

TEST_F(StreamTest, ShiftOperatorStrings) {
  s << "cstring\n";
  EXPECT_EQ(8U, s.GetWrittenBytes());
  s << llvm::StringRef("llvm::StringRef\n");
  EXPECT_EQ(24U, s.GetWrittenBytes());
  EXPECT_EQ("cstring\nllvm::StringRef\n", TakeValue());
}

TEST_F(StreamTest, ShiftOperatorPtr) {
  // This test is a bit tricky because pretty much everything related to
  // pointer printing seems to lead to UB or IB. So let's make the most basic
  // test that just checks that we print *something*. This way we at least know
  // that pointer printing doesn't do really bad things (e.g. crashing, reading
  // OOB/uninitialized memory which the sanitizers would spot).

  // Shift our own pointer to the output.
  int i = 3;
  int *ptr = &i;
  s << ptr;

  EXPECT_NE(0U, s.GetWrittenBytes());
  EXPECT_TRUE(!TakeValue().empty());
}

TEST_F(StreamTest, PutPtr) {
  // See the ShiftOperatorPtr test for the rationale.
  int i = 3;
  int *ptr = &i;
  s.PutPointer(ptr);

  EXPECT_NE(0U, s.GetWrittenBytes());
  EXPECT_TRUE(!TakeValue().empty());
}

// Alias to make it more clear that 'invalid' means for the Stream interface
// that it should use the host byte order.
const static auto hostByteOrder = lldb::eByteOrderInvalid;

// PutRawBytes/PutBytesAsRawHex tests.

TEST_F(StreamTest, PutBytesAsRawHex8ToBigEndian) {
  uint32_t value = 0x12345678;
  s.PutBytesAsRawHex8(static_cast<void*>(&value), sizeof(value),
                      hostByteOrder, lldb::eByteOrderBig);
  EXPECT_EQ(8U, s.GetWrittenBytes());
  EXPECT_EQ("78563412", TakeValue());
}

TEST_F(StreamTest, PutRawBytesToBigEndian) {
  uint32_t value = 0x12345678;
  s.PutRawBytes(static_cast<void*>(&value), sizeof(value),
                      hostByteOrder, lldb::eByteOrderBig);
  EXPECT_EQ(4U, s.GetWrittenBytes());
  EXPECT_EQ("\x78\x56\x34\x12", TakeValue());
}

TEST_F(StreamTest, PutBytesAsRawHex8ToLittleEndian) {
  uint32_t value = 0x12345678;
  s.PutBytesAsRawHex8(static_cast<void*>(&value), sizeof(value),
                      hostByteOrder, lldb::eByteOrderLittle);
  EXPECT_EQ(8U, s.GetWrittenBytes());
  EXPECT_EQ("12345678", TakeValue());
}

TEST_F(StreamTest, PutRawBytesToLittleEndian) {
  uint32_t value = 0x12345678;
  s.PutRawBytes(static_cast<void*>(&value), sizeof(value),
                      hostByteOrder, lldb::eByteOrderLittle);
  EXPECT_EQ(4U, s.GetWrittenBytes());
  EXPECT_EQ("\x12\x34\x56\x78", TakeValue());
}

TEST_F(StreamTest, PutBytesAsRawHex8ToMixedEndian) {
  uint32_t value = 0x12345678;
  s.PutBytesAsRawHex8(static_cast<void*>(&value), sizeof(value),
                      hostByteOrder, lldb::eByteOrderPDP);

  // FIXME: PDP byte order is not actually implemented but Stream just silently
  // prints the value in some random byte order...
#if 0
  EXPECT_EQ("34127856", TakeValue());
#endif
}

TEST_F(StreamTest, PutRawBytesToMixedEndian) {
  uint32_t value = 0x12345678;
  s.PutRawBytes(static_cast<void*>(&value), sizeof(value),
                      lldb::eByteOrderInvalid, lldb::eByteOrderPDP);

  // FIXME: PDP byte order is not actually implemented but Stream just silently
  // prints the value in some random byte order...
#if 0
  EXPECT_EQ("\x34\x12\x78\x56", TakeValue());
#endif
}

// ULEB128 support for binary streams.

TEST_F(BinaryStreamTest, PutULEB128OneByte) {
  auto bytes = s.PutULEB128(0x74ULL);
  EXPECT_EQ(1U, s.GetWrittenBytes());
  EXPECT_EQ("\x74", TakeValue());
  EXPECT_EQ(1U, bytes);
}

TEST_F(BinaryStreamTest, PutULEB128TwoBytes) {
  auto bytes = s.PutULEB128(0x1985ULL);
  EXPECT_EQ(2U, s.GetWrittenBytes());
  EXPECT_EQ("\x85\x33", TakeValue());
  EXPECT_EQ(2U, bytes);
}

TEST_F(BinaryStreamTest, PutULEB128ThreeBytes) {
  auto bytes = s.PutULEB128(0x5023ULL);
  EXPECT_EQ(3U, s.GetWrittenBytes());
  EXPECT_EQ("\xA3\xA0\x1", TakeValue());
  EXPECT_EQ(3U, bytes);
}

TEST_F(BinaryStreamTest, PutULEB128FourBytes) {
  auto bytes = s.PutULEB128(0xA48032ULL);
  EXPECT_EQ(4U, s.GetWrittenBytes());
  EXPECT_EQ("\xB2\x80\x92\x5", TakeValue());
  EXPECT_EQ(4U, bytes);
}

TEST_F(BinaryStreamTest, PutULEB128FiveBytes) {
  auto bytes = s.PutULEB128(0x12345678ULL);
  EXPECT_EQ(5U, s.GetWrittenBytes());
  EXPECT_EQ("\xF8\xAC\xD1\x91\x1", TakeValue());
  EXPECT_EQ(5U, bytes);
}

TEST_F(BinaryStreamTest, PutULEB128SixBytes) {
  auto bytes = s.PutULEB128(0xABFE3FAFDFULL);
  EXPECT_EQ(6U, s.GetWrittenBytes());
  EXPECT_EQ("\xDF\xDF\xFE\xF1\xBF\x15", TakeValue());
  EXPECT_EQ(6U, bytes);
}

TEST_F(BinaryStreamTest, PutULEB128SevenBytes) {
  auto bytes = s.PutULEB128(0xDABFE3FAFDFULL);
  EXPECT_EQ(7U, s.GetWrittenBytes());
  EXPECT_EQ("\xDF\xDF\xFE\xF1\xBF\xB5\x3", TakeValue());
  EXPECT_EQ(7U, bytes);
}

TEST_F(BinaryStreamTest, PutULEB128EightBytes) {
  auto bytes = s.PutULEB128(0x7CDABFE3FAFDFULL);
  EXPECT_EQ(8U, s.GetWrittenBytes());
  EXPECT_EQ("\xDF\xDF\xFE\xF1\xBF\xB5\xF3\x3", TakeValue());
  EXPECT_EQ(8U, bytes);
}

TEST_F(BinaryStreamTest, PutULEB128NineBytes) {
  auto bytes = s.PutULEB128(0x327CDABFE3FAFDFULL);
  EXPECT_EQ(9U, s.GetWrittenBytes());
  EXPECT_EQ("\xDF\xDF\xFE\xF1\xBF\xB5\xF3\x93\x3", TakeValue());
  EXPECT_EQ(9U, bytes);
}

TEST_F(BinaryStreamTest, PutULEB128MaxValue) {
  auto bytes = s.PutULEB128(std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(10U, s.GetWrittenBytes());
  EXPECT_EQ("\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\x1", TakeValue());
  EXPECT_EQ(10U, bytes);
}

TEST_F(BinaryStreamTest, PutULEB128Zero) {
  auto bytes = s.PutULEB128(0x0U);
  EXPECT_EQ(1U, s.GetWrittenBytes());
  EXPECT_EQ(std::string("\0", 1), TakeValue());
  EXPECT_EQ(1U, bytes);
}

TEST_F(BinaryStreamTest, PutULEB128One) {
  auto bytes = s.PutULEB128(0x1U);
  EXPECT_EQ(1U, s.GetWrittenBytes());
  EXPECT_EQ("\x1", TakeValue());
  EXPECT_EQ(1U, bytes);
}

// SLEB128 support for binary streams.

TEST_F(BinaryStreamTest, PutSLEB128OneByte) {
  auto bytes = s.PutSLEB128(0x74LL);
  EXPECT_EQ(2U, s.GetWrittenBytes());
  EXPECT_EQ(std::string("\xF4\0", 2), TakeValue());
  EXPECT_EQ(2U, bytes);
}

TEST_F(BinaryStreamTest, PutSLEB128TwoBytes) {
  auto bytes = s.PutSLEB128(0x1985LL);
  EXPECT_EQ(2U, s.GetWrittenBytes());
  EXPECT_EQ("\x85\x33", TakeValue());
  EXPECT_EQ(2U, bytes);
}

TEST_F(BinaryStreamTest, PutSLEB128ThreeBytes) {
  auto bytes = s.PutSLEB128(0x5023LL);
  EXPECT_EQ(3U, s.GetWrittenBytes());
  EXPECT_EQ("\xA3\xA0\x1", TakeValue());
  EXPECT_EQ(3U, bytes);
}

TEST_F(BinaryStreamTest, PutSLEB128FourBytes) {
  auto bytes = s.PutSLEB128(0xA48032LL);
  EXPECT_EQ(4U, s.GetWrittenBytes());
  EXPECT_EQ("\xB2\x80\x92\x5", TakeValue());
  EXPECT_EQ(4U, bytes);
}

TEST_F(BinaryStreamTest, PutSLEB128FiveBytes) {
  auto bytes = s.PutSLEB128(0x12345678LL);
  EXPECT_EQ(5U, s.GetWrittenBytes());
  EXPECT_EQ("\xF8\xAC\xD1\x91\x1", TakeValue());
  EXPECT_EQ(5U, bytes);
}

TEST_F(BinaryStreamTest, PutSLEB128SixBytes) {
  auto bytes = s.PutSLEB128(0xABFE3FAFDFLL);
  EXPECT_EQ(6U, s.GetWrittenBytes());
  EXPECT_EQ("\xDF\xDF\xFE\xF1\xBF\x15", TakeValue());
  EXPECT_EQ(6U, bytes);
}

TEST_F(BinaryStreamTest, PutSLEB128SevenBytes) {
  auto bytes = s.PutSLEB128(0xDABFE3FAFDFLL);
  EXPECT_EQ(7U, s.GetWrittenBytes());
  EXPECT_EQ("\xDF\xDF\xFE\xF1\xBF\xB5\x3", TakeValue());
  EXPECT_EQ(7U, bytes);
}

TEST_F(BinaryStreamTest, PutSLEB128EightBytes) {
  auto bytes = s.PutSLEB128(0x7CDABFE3FAFDFLL);
  EXPECT_EQ(8U, s.GetWrittenBytes());
  EXPECT_EQ("\xDF\xDF\xFE\xF1\xBF\xB5\xF3\x3", TakeValue());
  EXPECT_EQ(8U, bytes);
}

TEST_F(BinaryStreamTest, PutSLEB128NineBytes) {
  auto bytes = s.PutSLEB128(0x327CDABFE3FAFDFLL);
  EXPECT_EQ(9U, s.GetWrittenBytes());
  EXPECT_EQ("\xDF\xDF\xFE\xF1\xBF\xB5\xF3\x93\x3", TakeValue());
  EXPECT_EQ(9U, bytes);
}

TEST_F(BinaryStreamTest, PutSLEB128MaxValue) {
  auto bytes = s.PutSLEB128(std::numeric_limits<int64_t>::max());
  EXPECT_EQ(10U, s.GetWrittenBytes());
  EXPECT_EQ(std::string("\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\0", 10), TakeValue());
  EXPECT_EQ(10U, bytes);
}

TEST_F(BinaryStreamTest, PutSLEB128Zero) {
  auto bytes = s.PutSLEB128(0x0);
  EXPECT_EQ(1U, s.GetWrittenBytes());
  EXPECT_EQ(std::string("\0", 1), TakeValue());
  EXPECT_EQ(1U, bytes);
}

TEST_F(BinaryStreamTest, PutSLEB128One) {
  auto bytes = s.PutSLEB128(0x1);
  EXPECT_EQ(1U, s.GetWrittenBytes());
  EXPECT_EQ(std::string("\x1", 1), TakeValue());
  EXPECT_EQ(1U, bytes);
}

// SLEB128/ULEB128 support for non-binary streams.

// The logic for this is very simple, so it should be enough to test some basic
// use cases.

TEST_F(StreamTest, PutULEB128) {
  auto bytes = s.PutULEB128(0x74ULL);
  EXPECT_EQ(4U, s.GetWrittenBytes());
  EXPECT_EQ("0x74", TakeValue());
  EXPECT_EQ(4U, bytes);
}

TEST_F(StreamTest, PutSLEB128) {
  auto bytes = s.PutSLEB128(0x1985LL);
  EXPECT_EQ(6U, s.GetWrittenBytes());
  EXPECT_EQ("0x6533", TakeValue());
  EXPECT_EQ(6U, bytes);
}
