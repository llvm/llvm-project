//===- unittests/Support/SwapByteOrderTest.cpp - swap byte order test -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/Support/SwapByteOrder.h"
#include <cstdlib>
#include <ctime>
using namespace llvm;

#undef max

namespace {

// In these first two tests all of the origional_uintx values are truncated
// except for 64. We could avoid this, but there's really no point.

TEST(SwapByteOrder, UnsignedRoundTrip) {
  // The point of the bit twiddling of magic is to test with and without bits
  // in every byte.
  uint64_t value = 1;
  for (std::size_t i = 0; i <= sizeof(value); ++i) {
    uint8_t origional_uint8 = static_cast<uint8_t>(value);
    EXPECT_EQ(origional_uint8,
      sys::SwapByteOrder(sys::SwapByteOrder(origional_uint8)));

    uint16_t origional_uint16 = static_cast<uint16_t>(value);
    EXPECT_EQ(origional_uint16,
      sys::SwapByteOrder(sys::SwapByteOrder(origional_uint16)));

    uint32_t origional_uint32 = static_cast<uint32_t>(value);
    EXPECT_EQ(origional_uint32,
      sys::SwapByteOrder(sys::SwapByteOrder(origional_uint32)));

    uint64_t origional_uint64 = static_cast<uint64_t>(value);
    EXPECT_EQ(origional_uint64,
      sys::SwapByteOrder(sys::SwapByteOrder(origional_uint64)));

    value = (value << 8) | 0x55; // binary 0101 0101.
  }
}

TEST(SwapByteOrder, SignedRoundTrip) {
  // The point of the bit twiddling of magic is to test with and without bits
  // in every byte.
  uint64_t value = 1;
  for (std::size_t i = 0; i <= sizeof(value); ++i) {
    int8_t origional_int8 = static_cast<int8_t>(value);
    EXPECT_EQ(origional_int8,
      sys::SwapByteOrder(sys::SwapByteOrder(origional_int8)));

    int16_t origional_int16 = static_cast<int16_t>(value);
    EXPECT_EQ(origional_int16,
      sys::SwapByteOrder(sys::SwapByteOrder(origional_int16)));

    int32_t origional_int32 = static_cast<int32_t>(value);
    EXPECT_EQ(origional_int32,
      sys::SwapByteOrder(sys::SwapByteOrder(origional_int32)));

    int64_t origional_int64 = static_cast<int64_t>(value);
    EXPECT_EQ(origional_int64,
      sys::SwapByteOrder(sys::SwapByteOrder(origional_int64)));

    // Test other sign.
    value *= -1;

    origional_int8 = static_cast<int8_t>(value);
    EXPECT_EQ(origional_int8,
      sys::SwapByteOrder(sys::SwapByteOrder(origional_int8)));

    origional_int16 = static_cast<int16_t>(value);
    EXPECT_EQ(origional_int16,
      sys::SwapByteOrder(sys::SwapByteOrder(origional_int16)));

    origional_int32 = static_cast<int32_t>(value);
    EXPECT_EQ(origional_int32,
      sys::SwapByteOrder(sys::SwapByteOrder(origional_int32)));

    origional_int64 = static_cast<int64_t>(value);
    EXPECT_EQ(origional_int64,
      sys::SwapByteOrder(sys::SwapByteOrder(origional_int64)));

    // Return to normal sign and twiddle.
    value *= -1;
    value = (value << 8) | 0x55; // binary 0101 0101.
  }
}

TEST(SwapByteOrder, uint8_t) {
  EXPECT_EQ(uint8_t(0x11), sys::SwapByteOrder(uint8_t(0x11)));
}

TEST(SwapByteOrder, uint16_t) {
  EXPECT_EQ(uint16_t(0x1122), sys::SwapByteOrder(uint16_t(0x2211)));
}

TEST(SwapByteOrder, uint32_t) {
  EXPECT_EQ(uint32_t(0x11223344), sys::SwapByteOrder(uint32_t(0x44332211)));
}

TEST(SwapByteOrder, uint64_t) {
  EXPECT_EQ(uint64_t(0x1122334455667788ULL),
    sys::SwapByteOrder(uint64_t(0x8877665544332211ULL)));
}

TEST(SwapByteOrder, int8_t) {
  EXPECT_EQ(int8_t(0x11), sys::SwapByteOrder(int8_t(0x11)));
}

TEST(SwapByteOrder, int16_t) {
  EXPECT_EQ(int16_t(0x1122), sys::SwapByteOrder(int16_t(0x2211)));
}

TEST(SwapByteOrder, int32_t) {
  EXPECT_EQ(int32_t(0x11223344), sys::SwapByteOrder(int32_t(0x44332211)));
}

TEST(SwapByteOrder, int64_t) {
  EXPECT_EQ(int64_t(0x1122334455667788LL),
    sys::SwapByteOrder(int64_t(0x8877665544332211LL)));
}

}
