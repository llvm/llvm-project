//===- llvm/unittest/CodeGen/DwarfStringPoolEntryRefTest.cpp --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/DwarfStringPoolEntry.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Testing/Support/Error.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(DwarfStringPoolEntryRefTest, TestFullEntry) {
  BumpPtrAllocator Allocator;
  StringMapEntry<DwarfStringPoolEntry> *StringEntry1 =
      StringMapEntry<DwarfStringPoolEntry>::create(
          "Key1", Allocator, DwarfStringPoolEntry{nullptr, 0, 0});

  EXPECT_TRUE(StringEntry1->getKey() == "Key1");
  EXPECT_TRUE(StringEntry1->second.Symbol == nullptr);
  EXPECT_TRUE(StringEntry1->second.Offset == 0);
  EXPECT_TRUE(StringEntry1->second.Index == 0);

  DwarfStringPoolEntryRef Ref1(*StringEntry1);
  EXPECT_TRUE(Ref1.getString() == "Key1");
  EXPECT_TRUE(Ref1.getOffset() == 0);
  EXPECT_TRUE(Ref1.getIndex() == 0);

  DwarfStringPoolEntryRef Ref2(*StringEntry1);
  EXPECT_TRUE(Ref2.getString() == "Key1");
  EXPECT_TRUE(Ref2.getOffset() == 0);
  EXPECT_TRUE(Ref2.getIndex() == 0);
  EXPECT_TRUE(Ref1 == Ref2);
  EXPECT_FALSE(Ref1 != Ref2);

  StringMapEntry<DwarfStringPoolEntry> *StringEntry2 =
      StringMapEntry<DwarfStringPoolEntry>::create(
          "Key2", Allocator, DwarfStringPoolEntry{nullptr, 0x1000, 1});
  EXPECT_TRUE(StringEntry2->getKey() == "Key2");
  EXPECT_TRUE(StringEntry2->second.Symbol == nullptr);
  EXPECT_TRUE(StringEntry2->second.Offset == 0x1000);
  EXPECT_TRUE(StringEntry2->second.Index == 1);

  DwarfStringPoolEntryRef Ref3(*StringEntry2);
  EXPECT_TRUE(Ref3.getString() == "Key2");
  EXPECT_TRUE(Ref3.getOffset() == 0x1000);
  EXPECT_TRUE(Ref3.getIndex() == 1);
  EXPECT_TRUE(Ref1 != Ref3);
}

bool isEntryEqual(const DwarfStringPoolEntry &LHS,
                  const DwarfStringPoolEntry &RHS) {
  return LHS.Symbol == RHS.Symbol && LHS.Offset == RHS.Offset &&
         LHS.Index == RHS.Index;
}

TEST(DwarfStringPoolEntryRefTest, TestShortEntry) {
  DwarfStringPoolEntryWithExtString DwarfEntry1 = {{nullptr, 0, 0}, "Key1"};

  DwarfStringPoolEntryRef Ref1(DwarfEntry1);
  EXPECT_TRUE(Ref1.getString() == "Key1");
  EXPECT_TRUE(Ref1.getOffset() == 0);
  EXPECT_TRUE(Ref1.getIndex() == 0);
  EXPECT_TRUE(isEntryEqual(Ref1.getEntry(), DwarfEntry1));

  DwarfStringPoolEntryRef Ref2(DwarfEntry1);
  EXPECT_TRUE(Ref2.getString() == "Key1");
  EXPECT_TRUE(Ref2.getOffset() == 0);
  EXPECT_TRUE(isEntryEqual(Ref2.getEntry(), DwarfEntry1));
  EXPECT_TRUE(Ref1 == Ref2);
  EXPECT_FALSE(Ref1 != Ref2);

  DwarfStringPoolEntryWithExtString DwarfEntry2 = {{nullptr, 0x1000, 1},
                                                   "Key2"};

  DwarfStringPoolEntryRef Ref3(DwarfEntry2);
  EXPECT_TRUE(Ref3.getString() == "Key2");
  EXPECT_TRUE(Ref3.getOffset() == 0x1000);
  EXPECT_TRUE(Ref3.getIndex() == 1);
  EXPECT_TRUE(isEntryEqual(Ref3.getEntry(), DwarfEntry2));
  EXPECT_TRUE(Ref1 != Ref3);
}

TEST(DwarfStringPoolEntryRefTest, CompareFullAndShort) {
  BumpPtrAllocator Allocator;

  DwarfStringPoolEntryWithExtString DwarfEntry1 = {{nullptr, 0, 0}, "Key1"};
  DwarfStringPoolEntryRef Ref1(DwarfEntry1);

  StringMapEntry<DwarfStringPoolEntry> *StringEntry2 =
      StringMapEntry<DwarfStringPoolEntry>::create(
          "Key1", Allocator, DwarfStringPoolEntry{nullptr, 0, 0});
  DwarfStringPoolEntryRef Ref2(*StringEntry2);

  EXPECT_FALSE(Ref1 == Ref2);
}
