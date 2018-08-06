//===-- FlagsTest.cpp -------------------===---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  EXPECT_EQ(2U, f.SetCount());
}

TEST(Flags, Clear) {
  Flags f;
  f.Reset(0x3);
  EXPECT_EQ(2U, f.SetCount());

  f.Clear(0x5);
  EXPECT_EQ(1U, f.SetCount());

  f.Clear();
  EXPECT_EQ(0U, f.SetCount());
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

TEST(Flags, ClearCount) {
  Flags f;
  EXPECT_EQ(32U, f.ClearCount());

  f.Set(eFlag0);
  EXPECT_EQ(31U, f.ClearCount());

  f.Set(eFlag0);
  EXPECT_EQ(31U, f.ClearCount());

  f.Set(eFlag1);
  EXPECT_EQ(30U, f.ClearCount());

  f.Set(eAllFlags);
  EXPECT_EQ(29U, f.ClearCount());
}

TEST(Flags, SetCount) {
  Flags f;
  EXPECT_EQ(0U, f.SetCount());

  f.Set(eFlag0);
  EXPECT_EQ(1U, f.SetCount());

  f.Set(eFlag0);
  EXPECT_EQ(1U, f.SetCount());

  f.Set(eFlag1);
  EXPECT_EQ(2U, f.SetCount());

  f.Set(eAllFlags);
  EXPECT_EQ(3U, f.SetCount());
}
