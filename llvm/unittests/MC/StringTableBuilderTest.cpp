//===----------- StringTableBuilderTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/StringTableBuilder.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Endian.h"
#include "gtest/gtest.h"
#include <string>

using namespace llvm;

namespace {

TEST(StringTableBuilderTest, BasicELF) {
  StringTableBuilder B(StringTableBuilder::ELF);

  B.add("foo");
  B.add("bar");
  B.add("foobar");

  B.finalize();

  std::string Expected;
  Expected += '\x00';
  Expected += "foobar";
  Expected += '\x00';
  Expected += "foo";
  Expected += '\x00';

  SmallString<64> Data;
  raw_svector_ostream OS(Data);
  B.write(OS);

  EXPECT_EQ(Expected, Data);
  EXPECT_EQ(1U, B.getOffset("foobar"));
  EXPECT_EQ(4U, B.getOffset("bar"));
  EXPECT_EQ(8U, B.getOffset("foo"));
}

TEST(StringTableBuilderTest, BasicWinCOFF) {
  StringTableBuilder B(StringTableBuilder::WinCOFF);

  // Strings must be 9 chars or longer to go in the table.
  B.add("hippopotamus");
  B.add("pygmy hippopotamus");
  B.add("river horse");

  B.finalize();

  // size_field + "pygmy hippopotamus\0" + "river horse\0"
  uint32_t ExpectedSize = 4 + 19 + 12;
  EXPECT_EQ(ExpectedSize, B.getSize());

  std::string Expected;

  ExpectedSize =
      support::endian::byte_swap<uint32_t, support::little>(ExpectedSize);
  Expected.append((const char*)&ExpectedSize, 4);
  Expected += "pygmy hippopotamus";
  Expected += '\x00';
  Expected += "river horse";
  Expected += '\x00';

  SmallString<64> Data;
  raw_svector_ostream OS(Data);
  B.write(OS);

  EXPECT_EQ(Expected, Data);
  EXPECT_EQ(4U, B.getOffset("pygmy hippopotamus"));
  EXPECT_EQ(10U, B.getOffset("hippopotamus"));
  EXPECT_EQ(23U, B.getOffset("river horse"));
}

TEST(StringTableBuilderTest, ELFInOrder) {
  StringTableBuilder B(StringTableBuilder::ELF);
  EXPECT_EQ(1U, B.add("foo"));
  EXPECT_EQ(5U, B.add("bar"));
  EXPECT_EQ(9U, B.add("foobar"));

  B.finalizeInOrder();

  std::string Expected;
  Expected += '\x00';
  Expected += "foo";
  Expected += '\x00';
  Expected += "bar";
  Expected += '\x00';
  Expected += "foobar";
  Expected += '\x00';

  SmallString<64> Data;
  raw_svector_ostream OS(Data);
  B.write(OS);

  EXPECT_EQ(Expected, Data);
  EXPECT_EQ(1U, B.getOffset("foo"));
  EXPECT_EQ(5U, B.getOffset("bar"));
  EXPECT_EQ(9U, B.getOffset("foobar"));
}

TEST(StringTableBuilderTest, MachOInOrder) {
  StringTableBuilder B(StringTableBuilder::MachO);

  B.add("foo");
  B.add("bar");
  B.add("fooba");

  B.finalizeInOrder();

  std::string Expected;
  Expected += '\x00';
  Expected += "foo";
  Expected += '\x00';
  Expected += "bar";
  Expected += '\x00';
  Expected += "fooba";
  Expected += '\x00';

  // Mach-O pads to 4 bytes
  Expected += '\x00';

  SmallString<64> Data;
  raw_svector_ostream OS(Data);
  B.write(OS);

  EXPECT_EQ(Expected, Data);
  EXPECT_EQ(1U, B.getOffset("foo"));
  EXPECT_EQ(5U, B.getOffset("bar"));
  EXPECT_EQ(9U, B.getOffset("fooba"));
}

TEST(StringTableBuilderTest, MachO64InOrder) {
  StringTableBuilder B(StringTableBuilder::MachO64);

  B.add("foo");
  B.add("bar");
  B.add("f");

  B.finalizeInOrder();

  std::string Expected;
  Expected += '\x00';
  Expected += "foo";
  Expected += '\x00';
  Expected += "bar";
  Expected += '\x00';
  Expected += "f";
  Expected += '\x00';

  // 64 bit Mach-O pads to 8 bytes
  Expected += '\x00';
  Expected += '\x00';
  Expected += '\x00';
  Expected += '\x00';
  Expected += '\x00';

  SmallString<64> Data;
  raw_svector_ostream OS(Data);
  B.write(OS);

  EXPECT_EQ(Expected, Data);
  EXPECT_EQ(1U, B.getOffset("foo"));
  EXPECT_EQ(5U, B.getOffset("bar"));
  EXPECT_EQ(9U, B.getOffset("f"));
}

TEST(StringTableBuilderTest, MachOLinkedInOrder) {
  StringTableBuilder B(StringTableBuilder::MachOLinked);

  B.add("foo");
  B.add("bar");
  B.add("foob");

  B.finalizeInOrder();

  std::string Expected;
  Expected += ' ';
  Expected += '\x00';
  Expected += "foo";
  Expected += '\x00';
  Expected += "bar";
  Expected += '\x00';
  Expected += "foob";
  Expected += '\x00';

  // Mach-O pads to 4 bytes
  Expected += '\x00';

  SmallString<64> Data;
  raw_svector_ostream OS(Data);
  B.write(OS);

  EXPECT_EQ(Expected, Data);
  EXPECT_EQ(2U, B.getOffset("foo"));
  EXPECT_EQ(6U, B.getOffset("bar"));
  EXPECT_EQ(10U, B.getOffset("foob"));
}

TEST(StringTableBuilderTest, MachO64LinkedInOrder) {
  StringTableBuilder B(StringTableBuilder::MachO64Linked);

  B.add("foo");
  B.add("ba");
  B.add("f");

  B.finalizeInOrder();

  std::string Expected;
  Expected += ' ';
  Expected += '\x00';
  Expected += "foo";
  Expected += '\x00';
  Expected += "ba";
  Expected += '\x00';
  Expected += "f";
  Expected += '\x00';

  // 64 bit Mach-O pads to 8 bytes
  Expected += '\x00';
  Expected += '\x00';
  Expected += '\x00';
  Expected += '\x00';
  Expected += '\x00';

  SmallString<64> Data;
  raw_svector_ostream OS(Data);
  B.write(OS);

  EXPECT_EQ(Expected, Data);
  EXPECT_EQ(2U, B.getOffset("foo"));
  EXPECT_EQ(6U, B.getOffset("ba"));
  EXPECT_EQ(9U, B.getOffset("f"));
}
}
