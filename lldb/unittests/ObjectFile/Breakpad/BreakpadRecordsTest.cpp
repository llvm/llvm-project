//===-- BreakpadRecordsTest.cpp -------------------------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ObjectFile/Breakpad/BreakpadRecords.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb_private::breakpad;

TEST(Record, classify) {
  EXPECT_EQ(Record::Module, Record::classify("MODULE"));
  EXPECT_EQ(Record::Info, Record::classify("INFO"));
  EXPECT_EQ(Record::File, Record::classify("FILE"));
  EXPECT_EQ(Record::Func, Record::classify("FUNC"));
  EXPECT_EQ(Record::Inline, Record::classify("INLINE"));
  EXPECT_EQ(Record::InlineOrigin, Record::classify("INLINE_ORIGIN"));
  EXPECT_EQ(Record::Public, Record::classify("PUBLIC"));
  EXPECT_EQ(Record::StackCFI, Record::classify("STACK CFI"));
  EXPECT_EQ(Record::StackWin, Record::classify("STACK WIN"));

  // Any obviously incorrect lines will be classified as such.
  EXPECT_EQ(llvm::None, Record::classify("STACK"));
  EXPECT_EQ(llvm::None, Record::classify("STACK CODE_ID"));
  EXPECT_EQ(llvm::None, Record::classify("CODE_ID"));

  // Any line which does not start with a known keyword will be classified as a
  // line record, as those are the only ones that start without a keyword.
  EXPECT_EQ(Record::Line, Record::classify("deadbeef"));
  EXPECT_EQ(Record::Line, Record::classify("12"));
}

TEST(ModuleRecord, parse) {
  EXPECT_EQ(ModuleRecord(llvm::Triple::Linux, llvm::Triple::x86_64,
                         UUID("@ABCDEFGHIJKLMNO", 16)),
            ModuleRecord::parse(
                "MODULE Linux x86_64 404142434445464748494a4b4c4d4e4f0 a.out"));

  EXPECT_EQ(llvm::None, ModuleRecord::parse("MODULE"));
  EXPECT_EQ(llvm::None, ModuleRecord::parse("MODULE Linux"));
  EXPECT_EQ(llvm::None, ModuleRecord::parse("MODULE Linux x86_64"));
  EXPECT_EQ(llvm::None,
            ModuleRecord::parse("MODULE Linux x86_64 deadbeefbaadf00d"));
}

TEST(InfoRecord, parse) {
  EXPECT_EQ(InfoRecord(UUID("@ABCDEFGHIJKLMNO", 16)),
            InfoRecord::parse("INFO CODE_ID 404142434445464748494a4b4c4d4e4f"));
  EXPECT_EQ(InfoRecord(UUID()), InfoRecord::parse("INFO CODE_ID 47 a.exe"));

  EXPECT_EQ(llvm::None, InfoRecord::parse("INFO"));
  EXPECT_EQ(llvm::None, InfoRecord::parse("INFO CODE_ID"));
}

TEST(FileRecord, parse) {
  EXPECT_EQ(FileRecord(47, "foo"), FileRecord::parse("FILE 47 foo"));
  EXPECT_EQ(llvm::None, FileRecord::parse("FILE 47"));
  EXPECT_EQ(llvm::None, FileRecord::parse("FILE"));
  EXPECT_EQ(llvm::None, FileRecord::parse(""));
}

TEST(FuncRecord, parse) {
  EXPECT_EQ(FuncRecord(true, 0x47, 0x7, 0x8, "foo"),
            FuncRecord::parse("FUNC m 47 7 8 foo"));
  EXPECT_EQ(FuncRecord(false, 0x47, 0x7, 0x8, "foo"),
            FuncRecord::parse("FUNC 47 7 8 foo"));

  EXPECT_EQ(llvm::None, FuncRecord::parse("PUBLIC 47 7 8 foo"));
  EXPECT_EQ(llvm::None, FuncRecord::parse("FUNC 47 7 8"));
  EXPECT_EQ(llvm::None, FuncRecord::parse("FUNC 47 7"));
  EXPECT_EQ(llvm::None, FuncRecord::parse("FUNC 47"));
  EXPECT_EQ(llvm::None, FuncRecord::parse("FUNC m"));
  EXPECT_EQ(llvm::None, FuncRecord::parse("FUNC"));
}

TEST(InlineOriginRecord, parse) {
  EXPECT_EQ(InlineOriginRecord(47, "foo"),
            InlineOriginRecord::parse("INLINE_ORIGIN 47 foo"));
  EXPECT_EQ(llvm::None, InlineOriginRecord::parse("INLINE_ORIGIN 47"));
  EXPECT_EQ(llvm::None, InlineOriginRecord::parse("INLINE_ORIGIN"));
  EXPECT_EQ(llvm::None, InlineOriginRecord::parse(""));
}

TEST(InlineRecord, parse) {
  InlineRecord record1 = InlineRecord(0, 1, 2, 3);
  record1.Ranges.emplace_back(4, 5);
  EXPECT_EQ(record1, InlineRecord::parse("INLINE 0 1 2 3 4 5"));
  record1.Ranges.emplace_back(6, 7);
  EXPECT_EQ(record1, InlineRecord::parse("INLINE 0 1 2 3 4 5 6 7"));
  EXPECT_EQ(llvm::None, InlineRecord::parse("INLINE 0 1 2 3"));
  EXPECT_EQ(llvm::None, InlineRecord::parse("INLINE 0 1 2 3 4 5 6"));
  EXPECT_EQ(llvm::None, InlineRecord::parse("INLNIE 0"));
  EXPECT_EQ(llvm::None, InlineRecord::parse(""));
  EXPECT_EQ(llvm::None, InlineRecord::parse("FUNC"));
}

TEST(LineRecord, parse) {
  EXPECT_EQ(LineRecord(0x47, 0x74, 47, 74), LineRecord::parse("47 74 47 74"));
  EXPECT_EQ(llvm::None, LineRecord::parse("47 74 47"));
  EXPECT_EQ(llvm::None, LineRecord::parse("47 74"));
  EXPECT_EQ(llvm::None, LineRecord::parse("47"));
  EXPECT_EQ(llvm::None, LineRecord::parse(""));
  EXPECT_EQ(llvm::None, LineRecord::parse("FUNC"));
}

TEST(PublicRecord, parse) {
  EXPECT_EQ(PublicRecord(true, 0x47, 0x8, "foo"),
            PublicRecord::parse("PUBLIC m 47 8 foo"));
  EXPECT_EQ(PublicRecord(false, 0x47, 0x8, "foo"),
            PublicRecord::parse("PUBLIC 47 8 foo"));

  EXPECT_EQ(llvm::None, PublicRecord::parse("FUNC 47 8 foo"));
  EXPECT_EQ(llvm::None, PublicRecord::parse("PUBLIC 47 8"));
  EXPECT_EQ(llvm::None, PublicRecord::parse("PUBLIC 47"));
  EXPECT_EQ(llvm::None, PublicRecord::parse("PUBLIC m"));
  EXPECT_EQ(llvm::None, PublicRecord::parse("PUBLIC"));
}

TEST(StackCFIRecord, parse) {
  EXPECT_EQ(StackCFIRecord(0x47, 0x8, ".cfa: $esp 4 + $eip: .cfa 4 - ^"),
            StackCFIRecord::parse(
                "STACK CFI INIT 47 8 .cfa: $esp 4 + $eip: .cfa 4 - ^"));

  EXPECT_EQ(StackCFIRecord(0x47, 0x8, ".cfa: $esp 4 +"),
            StackCFIRecord::parse("STACK CFI INIT 47 8 .cfa: $esp 4 +  "));

  EXPECT_EQ(StackCFIRecord(0x47, llvm::None, ".cfa: $esp 4 +"),
            StackCFIRecord::parse("STACK CFI 47 .cfa: $esp 4 +"));

  // The validity of the register value expressions is not checked
  EXPECT_EQ(StackCFIRecord(0x47, 0x8, ".cfa: ^ ^ ^"),
            StackCFIRecord::parse("STACK CFI INIT 47 8 .cfa: ^ ^ ^"));

  EXPECT_EQ(llvm::None, StackCFIRecord::parse("STACK CFI INIT 47"));
  EXPECT_EQ(llvm::None, StackCFIRecord::parse("STACK CFI INIT"));
  EXPECT_EQ(llvm::None, StackCFIRecord::parse("STACK CFI"));
  EXPECT_EQ(llvm::None, StackCFIRecord::parse("STACK"));
  EXPECT_EQ(llvm::None, StackCFIRecord::parse("FILE 47 foo"));
  EXPECT_EQ(llvm::None, StackCFIRecord::parse("42 47"));
}

TEST(StackWinRecord, parse) {
  EXPECT_EQ(
      StackWinRecord(0x47, 0x8, 3, 4, 5, llvm::StringRef("$eip $esp ^ =")),
      StackWinRecord::parse("STACK WIN 4 47 8 1 2 3 4 5 6 1 $eip $esp ^ ="));

  EXPECT_EQ(llvm::None, StackWinRecord::parse(
                            "STACK WIN 0 47 8 1 0 0 0 0 0 1 $eip $esp ^ ="));
  EXPECT_EQ(llvm::None,
            StackWinRecord::parse("STACK WIN 4 47 8 1 0 0 0 0 0 0 1"));
  EXPECT_EQ(llvm::None, StackWinRecord::parse(
                            "STACK WIN 3 47 8 1 0 0 0 0 0 1 $eip $esp ^ ="));
  EXPECT_EQ(llvm::None,
            StackWinRecord::parse("STACK WIN 3 47 8 1 0 0 0 0 0 0 1"));
  EXPECT_EQ(llvm::None, StackWinRecord::parse(
                            "STACK WIN 4 47 8 1 0 0 0 0 1 $eip $esp ^ ="));
  EXPECT_EQ(llvm::None, StackWinRecord::parse("STACK WIN 4 47 8 1 0 0 0 0 0"));
  EXPECT_EQ(llvm::None, StackWinRecord::parse("STACK WIN 4 47 8 1 0 0 0 0"));
  EXPECT_EQ(llvm::None, StackWinRecord::parse("STACK WIN 4 47 8 1 0 0 0"));
  EXPECT_EQ(llvm::None, StackWinRecord::parse("STACK WIN 4 47 8 1 0 0"));
  EXPECT_EQ(llvm::None, StackWinRecord::parse("STACK WIN 4 47 8 1 0"));
  EXPECT_EQ(llvm::None, StackWinRecord::parse("STACK WIN 4 47 8 1"));
  EXPECT_EQ(llvm::None, StackWinRecord::parse("STACK WIN 4 47 8"));
  EXPECT_EQ(llvm::None, StackWinRecord::parse("STACK WIN 4 47"));
  EXPECT_EQ(llvm::None, StackWinRecord::parse("STACK WIN 4"));
  EXPECT_EQ(llvm::None, StackWinRecord::parse("STACK WIN"));
  EXPECT_EQ(llvm::None, StackWinRecord::parse("STACK"));
  EXPECT_EQ(llvm::None, StackWinRecord::parse(""));
  EXPECT_EQ(llvm::None, StackCFIRecord::parse("FILE 47 foo"));
  EXPECT_EQ(llvm::None, StackCFIRecord::parse("42 47"));
}
