//===------------------ LaneBitmaskTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/MC/LaneBitmask.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace llvm;

TEST(LaneBitmaskTest, Basic) {
  EXPECT_EQ(LaneBitmask::getAll(), ~LaneBitmask::getNone());
  EXPECT_EQ(LaneBitmask::getNone(), ~LaneBitmask::getAll());
  EXPECT_EQ(LaneBitmask::getLane(0) | LaneBitmask::getLane(1), LaneBitmask(3));
  EXPECT_EQ(LaneBitmask(3) & LaneBitmask::getLane(1), LaneBitmask::getLane(1));

  EXPECT_EQ(LaneBitmask(APInt(128, 42)).getAsAPInt(), APInt(128, 42));
  EXPECT_EQ(LaneBitmask(3).getNumLanes(), 2);
  EXPECT_EQ(LaneBitmask::getLane(0).getHighestLane(), 0);
  EXPECT_EQ(LaneBitmask::getLane(64).getHighestLane(), 64);
  EXPECT_EQ(LaneBitmask::getLane(127).getHighestLane(), 127);

  EXPECT_LT(LaneBitmask::getLane(64), LaneBitmask::getLane(65));
  EXPECT_LT(LaneBitmask::getLane(63), LaneBitmask::getLane(64));
  EXPECT_LT(LaneBitmask::getLane(62), LaneBitmask::getLane(63));
  EXPECT_LT(LaneBitmask::getLane(64), LaneBitmask::getLane(64) | LaneBitmask::getLane(0));

  LaneBitmask X(1);
  X |= LaneBitmask(2);
  EXPECT_EQ(X, LaneBitmask(3));

  LaneBitmask Y(3);
  Y &= LaneBitmask(1);
  EXPECT_EQ(Y, LaneBitmask(1));
}

TEST(LaneBitmaskTest, Print) {
  std::string buffer;
  raw_string_ostream OS(buffer);

  buffer = "";
  OS << PrintLaneMask(LaneBitmask::getAll(), /*FormatAsCLiterals=*/true);
  EXPECT_STREQ(OS.str().data(), "0xFFFFFFFFFFFFFFFF,0xFFFFFFFFFFFFFFFF");

  buffer = "";
  OS << PrintLaneMask(LaneBitmask::getAll(), /*FormatAsCLiterals=*/false);
  EXPECT_STREQ(OS.str().data(), "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");

  buffer = "";
  OS << PrintLaneMask(LaneBitmask::getLane(0), /*FormatAsCLiterals=*/true);
  EXPECT_STREQ(OS.str().data(), "0x0000000000000001");

  buffer = "";
  OS << PrintLaneMask(LaneBitmask::getLane(63), /*FormatAsCLiterals=*/true);
  EXPECT_STREQ(OS.str().data(), "0x8000000000000000");

  buffer = "";
  OS << PrintLaneMask(LaneBitmask::getNone(), /*FormatAsCLiterals=*/true);
  EXPECT_STREQ(OS.str().data(), "0x0000000000000000");

  buffer = "";
  OS << PrintLaneMask(LaneBitmask::getLane(64), /*FormatAsCLiterals=*/true);
  EXPECT_STREQ(OS.str().data(), "0x0000000000000000,0x0000000000000001");
}
