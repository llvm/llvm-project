//===- STLExtrasTest.cpp - Unit tests for STL extras ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

int f(rank<0>) { return 0; }
int f(rank<1>) { return 1; }
int f(rank<2>) { return 2; }
int f(rank<4>) { return 4; }

TEST(STLExtrasTest, Rank) {
  // We shouldn't get ambiguities and should select the overload of the same
  // rank as the argument.
  EXPECT_EQ(0, f(rank<0>()));
  EXPECT_EQ(1, f(rank<1>()));
  EXPECT_EQ(2, f(rank<2>()));

  // This overload is missing so we end up back at 2.
  EXPECT_EQ(2, f(rank<3>()));

  // But going past 3 should work fine.
  EXPECT_EQ(4, f(rank<4>()));

  // And we can even go higher and just fall back to the last overload.
  EXPECT_EQ(4, f(rank<5>()));
  EXPECT_EQ(4, f(rank<6>()));
}

}
