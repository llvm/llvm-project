//===-- FlagsTest.cpp -------------------===---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Utility/Flags.h"

using namespace lldb_private;

enum DummyFlags {
  eFlag0     = 1 << 0,
  eFlag1     = 1 << 1,
  eFlag2     = 1 << 2,
  eAllFlags  = (eFlag0 | eFlag1 | eFlag2)
};

TEST(Flags, GetBitSize) {
  Flags f;
  // Methods like ClearCount depend on this specific value, so we test
  // against it here.
  EXPECT_EQ(32U, f.GetBitSize());
}

TEST(Flags, Reset) {
  Flags f;
  f.Reset(0x3);
  EXPECT_EQ(0x3U, f.Get());
}

TEST(Flags, Clear) {
  Flags f;
  f.Reset(0x3);
  EXPECT_EQ(0x3U, f.Get());

  f.Clear(0x5);
  EXPECT_EQ(0x2U, f.Get());

  f.Clear();
  EXPECT_EQ(0x0U, f.Get());
}

TEST(Flags, AllSet) {
  Flags f;

  EXPECT_FALSE(f.AllSet(eFlag0 | eFlag1));

  f.Set(eFlag0);
  EXPECT_FALSE(f.AllSet(eFlag0 | eFlag1));

  f.Set(eFlag1);
  EXPECT_TRUE(f.AllSet(eFlag0 | eFlag1));

  f.Clear(eFlag1);
  EXPECT_FALSE(f.AllSet(eFlag0 | eFlag1));

  f.Clear(eFlag0);
  EXPECT_FALSE(f.AllSet(eFlag0 | eFlag1));
}

TEST(Flags, AnySet) {
  Flags f;

  EXPECT_FALSE(f.AnySet(eFlag0 | eFlag1));

  f.Set(eFlag0);
  EXPECT_TRUE(f.AnySet(eFlag0 | eFlag1));

  f.Set(eFlag1);
  EXPECT_TRUE(f.AnySet(eFlag0 | eFlag1));

  f.Clear(eFlag1);
  EXPECT_TRUE(f.AnySet(eFlag0 | eFlag1));

  f.Clear(eFlag0);
  EXPECT_FALSE(f.AnySet(eFlag0 | eFlag1));
}

TEST(Flags, Test) {
  Flags f;

  EXPECT_FALSE(f.Test(eFlag0));
  EXPECT_FALSE(f.Test(eFlag1));
  EXPECT_FALSE(f.Test(eFlag2));

  f.Set(eFlag0);
  EXPECT_TRUE(f.Test(eFlag0));
  EXPECT_FALSE(f.Test(eFlag1));
  EXPECT_FALSE(f.Test(eFlag2));

  f.Set(eFlag1);
  EXPECT_TRUE(f.Test(eFlag0));
  EXPECT_TRUE(f.Test(eFlag1));
  EXPECT_FALSE(f.Test(eFlag2));

  f.Clear(eFlag0);
  EXPECT_FALSE(f.Test(eFlag0));
  EXPECT_TRUE(f.Test(eFlag1));
  EXPECT_FALSE(f.Test(eFlag2));

  // FIXME: Should Flags assert on Test(eFlag0 | eFlag1) (more than one bit)?
}

TEST(Flags, AllClear) {
  Flags f;

  EXPECT_TRUE(f.AllClear(eFlag0 | eFlag1));

  f.Set(eFlag0);
  EXPECT_FALSE(f.AllClear(eFlag0 | eFlag1));

  f.Set(eFlag1);
  f.Clear(eFlag0);
  EXPECT_FALSE(f.AllClear(eFlag0 | eFlag1));

  f.Clear(eFlag1);
  EXPECT_TRUE(f.AnyClear(eFlag0 | eFlag1));
}

TEST(Flags, AnyClear) {
  Flags f;
  EXPECT_TRUE(f.AnyClear(eFlag0 | eFlag1));

  f.Set(eFlag0);
  EXPECT_TRUE(f.AnyClear(eFlag0 | eFlag1));

  f.Set(eFlag1);
  f.Set(eFlag0);
  EXPECT_FALSE(f.AnyClear(eFlag0 | eFlag1));

  f.Clear(eFlag1);
  EXPECT_TRUE(f.AnyClear(eFlag0 | eFlag1));

  f.Clear(eFlag0);
  EXPECT_TRUE(f.AnyClear(eFlag0 | eFlag1));
}

TEST(Flags, IsClear) {
  Flags f;

  EXPECT_TRUE(f.IsClear(eFlag0));
  EXPECT_TRUE(f.IsClear(eFlag1));

  f.Set(eFlag0);
  EXPECT_FALSE(f.IsClear(eFlag0));
  EXPECT_TRUE(f.IsClear(eFlag1));

  f.Set(eFlag1);
  EXPECT_FALSE(f.IsClear(eFlag0));
  EXPECT_FALSE(f.IsClear(eFlag1));

  f.Clear(eFlag0);
  EXPECT_TRUE(f.IsClear(eFlag0));
  EXPECT_FALSE(f.IsClear(eFlag1));

  f.Clear(eFlag1);
  EXPECT_TRUE(f.IsClear(eFlag0));
  EXPECT_TRUE(f.IsClear(eFlag1));
}
