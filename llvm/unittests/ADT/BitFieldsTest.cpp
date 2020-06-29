//===- llvm/unittests/ADT/BitFieldsTest.cpp - BitFields unit tests --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Bitfields.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(BitfieldsTest, Example) {
  uint8_t Storage = 0;

  // Store and retrieve a single bit as bool.
  using Bool = Bitfield::Element<bool, 0, 1>;
  Bitfield::set<Bool>(Storage, true);
  EXPECT_EQ(Storage, 0b00000001);
  //                          ^
  EXPECT_EQ(Bitfield::get<Bool>(Storage), true);

  // Store and retrieve a 2 bit typed enum.
  // Note: enum underlying type must be unsigned.
  enum class SuitEnum : uint8_t { CLUBS, DIAMONDS, HEARTS, SPADES };
  // Note: enum maximum value needs to be passed in as last parameter.
  using Suit = Bitfield::Element<SuitEnum, 1, 2, SuitEnum::SPADES>;
  Bitfield::set<Suit>(Storage, SuitEnum::HEARTS);
  EXPECT_EQ(Storage, 0b00000101);
  //                        ^^
  EXPECT_EQ(Bitfield::get<Suit>(Storage), SuitEnum::HEARTS);

  // Store and retrieve a 5 bit value as unsigned.
  using Value = Bitfield::Element<unsigned, 3, 5>;
  Bitfield::set<Value>(Storage, 10);
  EXPECT_EQ(Storage, 0b01010101);
  //                   ^^^^^
  EXPECT_EQ(Bitfield::get<Value>(Storage), 10U);

  // Interpret the same 5 bit value as signed.
  using SignedValue = Bitfield::Element<int, 3, 5>;
  Bitfield::set<SignedValue>(Storage, -2);
  EXPECT_EQ(Storage, 0b11110101);
  //                   ^^^^^
  EXPECT_EQ(Bitfield::get<SignedValue>(Storage), -2);

  // Ability to efficiently test if a field is non zero.
  EXPECT_TRUE(Bitfield::test<Value>(Storage));

  // Alter Storage changes value.
  Storage = 0;
  EXPECT_EQ(Bitfield::get<Bool>(Storage), false);
  EXPECT_EQ(Bitfield::get<Suit>(Storage), SuitEnum::CLUBS);
  EXPECT_EQ(Bitfield::get<Value>(Storage), 0U);
  EXPECT_EQ(Bitfield::get<SignedValue>(Storage), 0);

  Storage = 255;
  EXPECT_EQ(Bitfield::get<Bool>(Storage), true);
  EXPECT_EQ(Bitfield::get<Suit>(Storage), SuitEnum::SPADES);
  EXPECT_EQ(Bitfield::get<Value>(Storage), 31U);
  EXPECT_EQ(Bitfield::get<SignedValue>(Storage), -1);
}

TEST(BitfieldsTest, FirstBit) {
  uint8_t Storage = 0;
  using FirstBit = Bitfield::Element<bool, 0, 1>;
  // Set true
  Bitfield::set<FirstBit>(Storage, true);
  EXPECT_EQ(Bitfield::get<FirstBit>(Storage), true);
  EXPECT_EQ(Storage, 0x1ULL);
  // Set false
  Bitfield::set<FirstBit>(Storage, false);
  EXPECT_EQ(Bitfield::get<FirstBit>(Storage), false);
  EXPECT_EQ(Storage, 0x0ULL);
}

TEST(BitfieldsTest, SecondBit) {
  uint8_t Storage = 0;
  using SecondBit = Bitfield::Element<bool, 1, 1>;
  // Set true
  Bitfield::set<SecondBit>(Storage, true);
  EXPECT_EQ(Bitfield::get<SecondBit>(Storage), true);
  EXPECT_EQ(Storage, 0x2ULL);
  // Set false
  Bitfield::set<SecondBit>(Storage, false);
  EXPECT_EQ(Bitfield::get<SecondBit>(Storage), false);
  EXPECT_EQ(Storage, 0x0ULL);
}

TEST(BitfieldsTest, LastBit) {
  uint8_t Storage = 0;
  using LastBit = Bitfield::Element<bool, 7, 1>;
  // Set true
  Bitfield::set<LastBit>(Storage, true);
  EXPECT_EQ(Bitfield::get<LastBit>(Storage), true);
  EXPECT_EQ(Storage, 0x80ULL);
  // Set false
  Bitfield::set<LastBit>(Storage, false);
  EXPECT_EQ(Bitfield::get<LastBit>(Storage), false);
  EXPECT_EQ(Storage, 0x0ULL);
}

TEST(BitfieldsTest, LastBitUint64) {
  uint64_t Storage = 0;
  using LastBit = Bitfield::Element<bool, 63, 1>;
  // Set true
  Bitfield::set<LastBit>(Storage, true);
  EXPECT_EQ(Bitfield::get<LastBit>(Storage), true);
  EXPECT_EQ(Storage, 0x8000000000000000ULL);
  // Set false
  Bitfield::set<LastBit>(Storage, false);
  EXPECT_EQ(Bitfield::get<LastBit>(Storage), false);
  EXPECT_EQ(Storage, 0x0ULL);
}

TEST(BitfieldsTest, Enum) {
  enum Enum : unsigned { Zero = 0, Two = 2, LAST = Two };

  uint8_t Storage = 0;
  using OrderingField = Bitfield::Element<Enum, 1, 2, LAST>;
  EXPECT_EQ(Bitfield::get<OrderingField>(Storage), Zero);
  Bitfield::set<OrderingField>(Storage, Two);
  EXPECT_EQ(Bitfield::get<OrderingField>(Storage), Two);
  EXPECT_EQ(Storage, 0b00000100);
  // value 2 in             ^^
}

TEST(BitfieldsTest, EnumClass) {
  enum class Enum : unsigned { Zero = 0, Two = 2, LAST = Two };

  uint8_t Storage = 0;
  using OrderingField = Bitfield::Element<Enum, 1, 2, Enum::LAST>;
  EXPECT_EQ(Bitfield::get<OrderingField>(Storage), Enum::Zero);
  Bitfield::set<OrderingField>(Storage, Enum::Two);
  EXPECT_EQ(Bitfield::get<OrderingField>(Storage), Enum::Two);
  EXPECT_EQ(Storage, 0b00000100);
  // value 2 in             ^^
}

TEST(BitfieldsTest, OneBitSigned) {
  uint8_t Storage = 0;
  using SignedField = Bitfield::Element<int, 1, 1>;
  EXPECT_EQ(Bitfield::get<SignedField>(Storage), 0);
  EXPECT_EQ(Storage, 0b00000000);
  // value 0 in              ^
  Bitfield::set<SignedField>(Storage, -1);
  EXPECT_EQ(Bitfield::get<SignedField>(Storage), -1);
  EXPECT_EQ(Storage, 0b00000010);
  // value 1 in              ^
}

TEST(BitfieldsTest, TwoBitSigned) {
  uint8_t Storage = 0;
  using SignedField = Bitfield::Element<int, 1, 2>;
  EXPECT_EQ(Bitfield::get<SignedField>(Storage), 0);
  EXPECT_EQ(Storage, 0b00000000);
  // value 0 in             ^^
  Bitfield::set<SignedField>(Storage, 1);
  EXPECT_EQ(Bitfield::get<SignedField>(Storage), 1);
  EXPECT_EQ(Storage, 0b00000010);
  // value 1 in             ^^
  Bitfield::set<SignedField>(Storage, -1);
  EXPECT_EQ(Bitfield::get<SignedField>(Storage), -1);
  EXPECT_EQ(Storage, 0b00000110);
  // value -1 in            ^^
  Bitfield::set<SignedField>(Storage, -2);
  EXPECT_EQ(Bitfield::get<SignedField>(Storage), -2);
  EXPECT_EQ(Storage, 0b00000100);
  // value -2 in            ^^
}

TEST(BitfieldsTest, isOverlapping) {
  //    01234567
  // A: --------
  // B:    ---
  // C:  ---
  // D:     ---
  using A = Bitfield::Element<unsigned, 0, 8>;
  using B = Bitfield::Element<unsigned, 3, 3>;
  using C = Bitfield::Element<unsigned, 1, 3>;
  using D = Bitfield::Element<unsigned, 4, 3>;
  EXPECT_TRUE((Bitfield::isOverlapping<A, B>()));
  EXPECT_TRUE((Bitfield::isOverlapping<A, C>()));
  EXPECT_TRUE((Bitfield::isOverlapping<A, B>()));
  EXPECT_TRUE((Bitfield::isOverlapping<A, D>()));

  EXPECT_TRUE((Bitfield::isOverlapping<B, C>()));
  EXPECT_TRUE((Bitfield::isOverlapping<B, D>()));
  EXPECT_FALSE((Bitfield::isOverlapping<C, D>()));
}

TEST(BitfieldsTest, FullUint64) {
  uint64_t Storage = 0;
  using Value = Bitfield::Element<uint64_t, 0, 64>;
  Bitfield::set<Value>(Storage, -1ULL);
  EXPECT_EQ(Bitfield::get<Value>(Storage), -1ULL);
  Bitfield::set<Value>(Storage, 0ULL);
  EXPECT_EQ(Bitfield::get<Value>(Storage), 0ULL);
}

TEST(BitfieldsTest, FullInt64) {
  uint64_t Storage = 0;
  using Value = Bitfield::Element<int64_t, 0, 64>;
  Bitfield::set<Value>(Storage, -1);
  EXPECT_EQ(Bitfield::get<Value>(Storage), -1);
  Bitfield::set<Value>(Storage, 0);
  EXPECT_EQ(Bitfield::get<Value>(Storage), 0);
}

#ifdef EXPECT_DEBUG_DEATH

TEST(BitfieldsTest, ValueTooBigBool) {
  uint64_t Storage = 0;
  using A = Bitfield::Element<unsigned, 0, 1>;
  Bitfield::set<A>(Storage, true);
  Bitfield::set<A>(Storage, false);
  EXPECT_DEBUG_DEATH(Bitfield::set<A>(Storage, 2), "value is too big");
}

TEST(BitfieldsTest, ValueTooBigInt) {
  uint64_t Storage = 0;
  using A = Bitfield::Element<unsigned, 0, 2>;
  Bitfield::set<A>(Storage, 3);
  EXPECT_DEBUG_DEATH(Bitfield::set<A>(Storage, 4), "value is too big");
  EXPECT_DEBUG_DEATH(Bitfield::set<A>(Storage, -1), "value is too big");
}

TEST(BitfieldsTest, ValueTooBigBounded) {
  uint8_t Storage = 0;
  using A = Bitfield::Element<int, 1, 2>;
  Bitfield::set<A>(Storage, 1);
  Bitfield::set<A>(Storage, 0);
  Bitfield::set<A>(Storage, -1);
  Bitfield::set<A>(Storage, -2);
  EXPECT_DEBUG_DEATH(Bitfield::set<A>(Storage, 2), "value is too big");
  EXPECT_DEBUG_DEATH(Bitfield::set<A>(Storage, -3), "value is too small");
}

#endif

} // namespace
