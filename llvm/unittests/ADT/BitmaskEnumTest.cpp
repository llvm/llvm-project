//===- llvm/unittest/ADT/BitmaskEnumTest.cpp - BitmaskEnum unit tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/BitmaskEnum.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
enum Flags {
  F0 = 0,
  F1 = 1,
  F2 = 2,
  F3 = 4,
  F4 = 8,
  LLVM_MARK_AS_BITMASK_ENUM(F4)
};

static_assert(is_bitmask_enum<Flags>::value != 0);
static_assert(largest_bitmask_enum_bit<Flags>::value == Flags::F4);

enum Flags2 { V0 = 0, V1 = 1, V2 = 2, V3 = 4, V4 = 8 };
} // namespace

namespace llvm {
LLVM_DECLARE_ENUM_AS_BITMASK(Flags2, V4);
}

static_assert(is_bitmask_enum<Flags>::value != 0);
static_assert(largest_bitmask_enum_bit<Flags>::value == Flags::F4);

namespace {
TEST(BitmaskEnumTest, BitwiseOr) {
  Flags f = F1 | F2;
  EXPECT_EQ(3, f);

  f = f | F3;
  EXPECT_EQ(7, f);

  Flags2 f2 = V1 | V2;
  EXPECT_EQ(3, f2);

  f2 = f2 | V3;
  EXPECT_EQ(7, f2);
}

TEST(BitmaskEnumTest, BitwiseOrEquals) {
  Flags f = F1;
  f |= F3;
  EXPECT_EQ(5, f);

  // |= should return a reference to the LHS.
  f = F2;
  (f |= F3) = F1;
  EXPECT_EQ(F1, f);

  Flags2 f2 = V1;
  f2 |= V3;
  EXPECT_EQ(5, f2);

  f2 = V2;
  (f2 |= V3) = V1;
  EXPECT_EQ(V1, f2);
}

TEST(BitmaskEnumTest, BitwiseAnd) {
  Flags f = static_cast<Flags>(3) & F2;
  EXPECT_EQ(F2, f);

  f = (f | F3) & (F1 | F2 | F3);
  EXPECT_EQ(6, f);

  Flags2 f2 = static_cast<Flags2>(3) & V2;
  EXPECT_EQ(V2, f2);

  f2 = (f2 | V3) & (V1 | V2 | V3);
  EXPECT_EQ(6, f2);
}

TEST(BitmaskEnumTest, BitwiseAndEquals) {
  Flags f = F1 | F2 | F3;
  f &= F1 | F2;
  EXPECT_EQ(3, f);

  // &= should return a reference to the LHS.
  (f &= F1) = F3;
  EXPECT_EQ(F3, f);

  Flags2 f2 = V1 | V2 | V3;
  f2 &= V1 | V2;
  EXPECT_EQ(3, f2);

  (f2 &= V1) = V3;
  EXPECT_EQ(V3, f2);
}

TEST(BitmaskEnumTest, BitwiseXor) {
  Flags f = (F1 | F2) ^ (F2 | F3);
  EXPECT_EQ(5, f);

  f = f ^ F1;
  EXPECT_EQ(4, f);

  Flags2 f2 = (V1 | V2) ^ (V2 | V3);
  EXPECT_EQ(5, f2);

  f2 = f2 ^ V1;
  EXPECT_EQ(4, f2);
}

TEST(BitmaskEnumTest, BitwiseXorEquals) {
  Flags f = (F1 | F2);
  f ^= (F2 | F4);
  EXPECT_EQ(9, f);

  // ^= should return a reference to the LHS.
  (f ^= F4) = F3;
  EXPECT_EQ(F3, f);

  Flags2 f2 = (V1 | V2);
  f2 ^= (V2 | V4);
  EXPECT_EQ(9, f2);

  (f2 ^= V4) = V3;
  EXPECT_EQ(V3, f2);
}

TEST(BitmaskEnumTest, BitwiseShift) {
  Flags f = (F1 << F1);
  EXPECT_EQ(f, F2);

  Flags f2 = F1;
  f2 <<= F1;
  EXPECT_EQ(f2, F2);

  Flags f3 = (F1 >> F1);
  EXPECT_EQ(f3, F0);

  Flags f4 = F1;
  f4 >>= F1;
  EXPECT_EQ(f4, F0);
}

TEST(BitmaskEnumTest, ConstantExpression) {
  constexpr Flags f1 = ~F1;
  constexpr Flags f2 = F1 | F2;
  constexpr Flags f3 = F1 & F2;
  constexpr Flags f4 = F1 ^ F2;
  EXPECT_EQ(f1, ~F1);
  EXPECT_EQ(f2, F1 | F2);
  EXPECT_EQ(f3, F1 & F2);
  EXPECT_EQ(f4, F1 ^ F2);

  constexpr Flags2 f21 = ~V1;
  constexpr Flags2 f22 = V1 | V2;
  constexpr Flags2 f23 = V1 & V2;
  constexpr Flags2 f24 = V1 ^ V2;
  EXPECT_EQ(f21, ~V1);
  EXPECT_EQ(f22, V1 | V2);
  EXPECT_EQ(f23, V1 & V2);
  EXPECT_EQ(f24, V1 ^ V2);
}

TEST(BitmaskEnumTest, BitwiseNot) {
  Flags f = ~F1;
  EXPECT_EQ(14, f); // Largest value for f is 15.
  EXPECT_EQ(15, ~F0);

  Flags2 f2 = ~V1;
  EXPECT_EQ(14, f2);
  EXPECT_EQ(15, ~V0);
}

enum class FlagsClass {
  F0 = 0,
  F1 = 1,
  F2 = 2,
  F3 = 4,
  LLVM_MARK_AS_BITMASK_ENUM(F3)
};

TEST(BitmaskEnumTest, ScopedEnum) {
  FlagsClass f = (FlagsClass::F1 & ~FlagsClass::F0) | FlagsClass::F2;
  f |= FlagsClass::F3;
  EXPECT_EQ(7, static_cast<int>(f));
}

struct Container {
  enum Flags { F0 = 0, F1 = 1, F2 = 2, F3 = 4, LLVM_MARK_AS_BITMASK_ENUM(F3) };

  static Flags getFlags() {
    Flags f = F0 | F1;
    f |= F2;
    return f;
  }
};

TEST(BitmaskEnumTest, EnumInStruct) { EXPECT_EQ(3, Container::getFlags()); }

} // namespace

namespace foo {
namespace bar {
namespace {
enum FlagsInNamespace {
  F0 = 0,
  F1 = 1,
  F2 = 2,
  F3 = 4,
  LLVM_MARK_AS_BITMASK_ENUM(F3)
};
} // namespace
} // namespace foo
} // namespace bar

namespace {
TEST(BitmaskEnumTest, EnumInNamespace) {
  foo::bar::FlagsInNamespace f = ~foo::bar::F0 & (foo::bar::F1 | foo::bar::F2);
  f |= foo::bar::F3;
  EXPECT_EQ(7, f);
}
} // namespace
