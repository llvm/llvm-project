//===- EndianTest.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for orc-rt's Endian.h APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/Endian.h"
#include "orc-rt/bit.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <limits>

using namespace orc_rt;

template <typename T> static void endianRead(T Value, endian E) {
  char Buffer[sizeof(T)];
  memcpy(Buffer, &Value, sizeof(T));

  if (E != endian::native)
    std::reverse(Buffer, Buffer + sizeof(T));

  T NewVal = endian_read<T>(Buffer, E);
  EXPECT_EQ(NewVal, Value);
}

template <typename T> static void endianWrite(T Value, endian E) {
  char Buffer[sizeof(T)];

  endian_write(Buffer, Value, E);

  if (E != endian::native)
    std::reverse(Buffer, Buffer + sizeof(T));

  T NewVal;
  memcpy(&NewVal, Buffer, sizeof(T));

  EXPECT_EQ(NewVal, Value);
}

template <typename T> static void endianReadAndWrite(T Value, endian E) {
  endianRead(Value, E);
  endianWrite(Value, E);
}

template <typename T> static void bothEndiansReadAndWrite(T Value) {
  endianReadAndWrite(Value, endian::little);
  endianReadAndWrite(Value, endian::big);
}

// Rotate the given bit pattern through all valid rotations for T, testing that
// the given operation works for the given pattern.
template <typename Op, typename T>
void forAllRotatedValues(Op O, T InitialValue) {
  T V = InitialValue;
  for (size_t I = 0; I != CHAR_BIT * sizeof(T); ++I) {
    O(V);
    V = orc_rt::rotl(V, 1);
  }
}

template <typename Op, typename T>
void forAllShiftedValues(Op O, T InitialValue) {
  T V = InitialValue;
  constexpr T TopValueBit = 1 << (std::numeric_limits<T>::digits - 1);
  for (size_t I = 0; I != CHAR_BIT * sizeof(T); ++I) {
    O(V);
    if (V & TopValueBit)
      break;
    V << 1;
  }
}

TEST(EndianTest, ReadWrite) {
  bothEndiansReadAndWrite<uint8_t>(0);
  bothEndiansReadAndWrite<uint8_t>(0xff);
  forAllRotatedValues(bothEndiansReadAndWrite<uint8_t>, uint8_t(1));
  forAllRotatedValues(bothEndiansReadAndWrite<uint8_t>, uint8_t(0x5A));

  bothEndiansReadAndWrite<uint16_t>(0);
  bothEndiansReadAndWrite<uint16_t>(0xffff);
  forAllRotatedValues(bothEndiansReadAndWrite<uint16_t>, uint16_t(1));
  forAllRotatedValues(bothEndiansReadAndWrite<uint16_t>, uint16_t(0x5A5A));

  bothEndiansReadAndWrite<uint32_t>(0);
  bothEndiansReadAndWrite<uint32_t>(0xffffffff);
  forAllRotatedValues(bothEndiansReadAndWrite<uint32_t>, uint32_t(1));
  forAllRotatedValues(bothEndiansReadAndWrite<uint32_t>, uint32_t(0x5A5A5A5A));

  bothEndiansReadAndWrite<uint64_t>(0);
  bothEndiansReadAndWrite<uint64_t>(0xffffffffffffffff);
  forAllRotatedValues(bothEndiansReadAndWrite<uint64_t>, uint64_t(1));
  forAllRotatedValues(bothEndiansReadAndWrite<uint64_t>,
                      uint64_t(0x5A5A5A5A5A5A5A5A));
}
