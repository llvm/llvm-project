//===-- GNUstepFormattersTest.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Language/ObjC/GNUstepFormatters.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cstring>

using namespace lldb_private::formatters;

// The small-object payloads below come from libobjc2's tag layout (three low
// bits) and gnustep-base's encodings; the constants are what the compiler and
// runtime actually produce, checked against a live process.

TEST(GNUstepFormattersTest, TinyStringDecodesClangEmittedLiteral) {
  // clang emits @"Hello" for the gnustep-2.x ABI as this constant
  // (CGObjCGNU.cpp: 7-bit characters from the top, 5-bit length, tag 4).
  EXPECT_EQ(GNUstepDecodeTinyString(0x919766cde000002cULL, /*pointer_size=*/8),
            "Hello");
  EXPECT_EQ(GNUstepDecodeTinyString(0x91a4000000000014ULL, /*pointer_size=*/8),
            "Hi");
  EXPECT_EQ(GNUstepDecodeTinyString(0xc3c386cca000002cULL, /*pointer_size=*/8),
            "apple");
  EXPECT_EQ(GNUstepDecodeTinyString(0xc587761dd8400034ULL, /*pointer_size=*/8),
            "banana");
}

TEST(GNUstepFormattersTest, TinyStringEmptyAndLimits) {
  // Zero characters: just the tag.
  EXPECT_EQ(GNUstepDecodeTinyString(0x4, /*pointer_size=*/8), "");
  // Eight characters fill every slot; nine means eight plus a terminator.
  uint64_t eight = 4 | (8ULL << 3);
  for (int i = 0; i < 8; ++i)
    eight |= static_cast<uint64_t>('a' + i) << (57 - 7 * i);
  EXPECT_EQ(GNUstepDecodeTinyString(eight, /*pointer_size=*/8), "abcdefgh");
  uint64_t nine = (eight & ~(0x1fULL << 3)) | (9ULL << 3);
  EXPECT_EQ(GNUstepDecodeTinyString(nine, /*pointer_size=*/8), "abcdefgh");
  // Not a tiny string: wrong tag, or an impossible length.
  EXPECT_FALSE(
      GNUstepDecodeTinyString(0x919766cde000002dULL, /*pointer_size=*/8)
          .has_value());
  EXPECT_FALSE(GNUstepDecodeTinyString(4 | (10ULL << 3), /*pointer_size=*/8)
                   .has_value());
}

TEST(GNUstepFormattersTest, SmallObjectLayoutFollowsThePointerSize) {
  // libobjc2 gives a 32-bit target a single tag bit, so NSSmallInt is the only
  // small class it can express and the payload is shifted by one rather than
  // three (class.h; GNUstepTaggedPointerVendor derives the same widths from
  // the process and the two must agree).
  EXPECT_EQ(GNUstepSmallObjectMask(8), 7u);
  EXPECT_EQ(GNUstepSmallObjectMask(4), 1u);

  // Same value, boxed for each data model.
  EXPECT_EQ(GNUstepDecodeSmallInt((42ULL << 3) | 1, /*pointer_size=*/8), 42);
  EXPECT_EQ(GNUstepDecodeSmallInt((42ULL << 1) | 1, /*pointer_size=*/4), 42);

  // Negative payloads must sign-extend from the target's pointer width, not
  // from 64 bits - a 32-bit -1 is 0xFFFFFFFF, which is positive if widened.
  EXPECT_EQ(GNUstepDecodeSmallInt(0xFFFFFFFFULL, /*pointer_size=*/4), -1);
  // -5 boxed for each data model, decoded back.
  EXPECT_EQ(GNUstepDecodeSmallInt((static_cast<uint64_t>(-5LL) << 3) | 1,
                                  /*pointer_size=*/8),
            -5);
  EXPECT_EQ(GNUstepDecodeSmallInt(
                static_cast<uint32_t>(static_cast<uint64_t>(-5LL) << 1) | 1,
                /*pointer_size=*/4),
            -5);

  // The wider layouts need more bits than a 32-bit pointer has, so a tiny
  // string cannot be encoded there at all.
  EXPECT_FALSE(
      GNUstepDecodeTinyString(0x91a4000000000014ULL, /*pointer_size=*/4)
          .has_value());
}

TEST(GNUstepFormattersTest, SmallIntIsArithmeticallyShifted) {
  // NSSmallInt: value << 3 | 1 (Source/NSNumber.m).
  EXPECT_EQ(GNUstepDecodeSmallInt((42ULL << 3) | 1, /*pointer_size=*/8), 42);
  EXPECT_EQ(GNUstepDecodeSmallInt(0x00000000000f1201ULL, /*pointer_size=*/8),
            123456);
  // Negative values keep their sign through the shift.
  EXPECT_EQ(GNUstepDecodeSmallInt(0xfffffffffffffce9ULL, /*pointer_size=*/8),
            -99);
  EXPECT_EQ(GNUstepDecodeSmallInt((static_cast<uint64_t>(-1LL) << 3) | 1,
                                  /*pointer_size=*/8),
            -1);
}

TEST(GNUstepFormattersTest, SmallDoublesRoundTrip) {
  // Box a double the way boxDouble() does for the repeating (tag 3 / 5) and
  // extended (tag 2) encodings, then check the decoders invert it.
  auto bits_of = [](double d) {
    uint64_t bits;
    std::memcpy(&bits, &d, sizeof(bits));
    return bits;
  };
  // Repeating: the low three mantissa bits are moved up into bits 3-5 and
  // the tag takes their place. 1.5f as boxed by gnustep-base:
  EXPECT_DOUBLE_EQ(GNUstepDecodeSmallRepeatingDouble(0x3ff8000000000005ULL),
                   1.5);
  {
    // A double is boxable as "repeating" when its mantissa bits 3-5 equal
    // its bits 0-2 (boxDouble in Source/NSNumber.m); the box then simply
    // replaces bits 0-2 with the tag. Make 3.14159 satisfy that and check the
    // decoder restores it exactly.
    uint64_t b = bits_of(3.14159);
    const uint64_t low = b & 7;
    b = (b & ~0x38ULL) | (low << 3);
    const uint64_t boxed = (b & ~7ULL) | 3;
    EXPECT_EQ(bits_of(GNUstepDecodeSmallRepeatingDouble(boxed)), b);
  }
  {
    // Extended: the low three mantissa bits are all equal to bit 3.
    const uint64_t b = bits_of(0.1) & ~0xfULL; // clear low nibble
    const uint64_t boxed_zero = b | 2;         // bit 3 = 0 -> low bits 000
    EXPECT_EQ(bits_of(GNUstepDecodeSmallExtendedDouble(boxed_zero)), b);
    const uint64_t boxed_one = b | 8 | 2; // bit 3 = 1 -> low bits 111
    EXPECT_EQ(bits_of(GNUstepDecodeSmallExtendedDouble(boxed_one)), b | 0xf);
  }
}

TEST(GNUstepFormattersTest, SmallDateDecodesReferenceDate) {
  // [NSDate dateWithTimeIntervalSinceReferenceDate: 0] and 1700000000 seconds
  // after 1970 (2023-11-14 22:13:20 UTC = 721692800 seconds after 2001), as
  // observed in a live process. The compressed encoding drops low mantissa
  // bits, so the reference date itself comes back a couple of seconds off -
  // gnustep-base prints the same "00:00:02".
  EXPECT_NEAR(GNUstepDecodeSmallDate(0x0880000000000006ULL), 0.0, 3.0);
  EXPECT_DOUBLE_EQ(GNUstepDecodeSmallDate(0x16ac10a200000006ULL), 721692800.0);
}
