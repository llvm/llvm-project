//===- StringTableBuilderTest.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/PDBStringTable.h"
#include "llvm/DebugInfo/PDB/Native/PDBStringTableBuilder.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/BinaryStreamWriter.h"
#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::pdb;
using namespace llvm::support;

TEST(StringTableBuilderTest, Simple) {
  // Create /names table contents.
  PDBStringTableBuilder Builder;

  // This test case is carefully constructed to ensure that at least one
  // string gets bucketed into slot 0, *and* to ensure that at least one
  // has a hash collision at the end of the bucket list so it has to
  // wrap around.
  uint32_t FooID = Builder.insert("foo");
  uint32_t BarID = Builder.insert("bar");
  uint32_t BazID = Builder.insert("baz");
  uint32_t BuzzID = Builder.insert("buzz");
  uint32_t BazzID = Builder.insert("bazz");
  uint32_t BarrID = Builder.insert("barr");

  // Re-inserting the same item should return the same id.
  EXPECT_EQ(FooID, Builder.insert("foo"));
  EXPECT_EQ(BarID, Builder.insert("bar"));
  EXPECT_EQ(BazID, Builder.insert("baz"));
  EXPECT_EQ(BuzzID, Builder.insert("buzz"));
  EXPECT_EQ(BazzID, Builder.insert("bazz"));
  EXPECT_EQ(BarrID, Builder.insert("barr"));

  // Each ID should be distinct.
  std::set<uint32_t> Distinct{FooID, BarID, BazID, BuzzID, BazzID, BarrID};
  EXPECT_EQ(6U, Distinct.size());

  std::vector<uint8_t> Buffer(Builder.calculateSerializedSize());
  MutableBinaryByteStream OutStream(Buffer, llvm::endianness::little);
  BinaryStreamWriter Writer(OutStream);
  EXPECT_THAT_ERROR(Builder.commit(Writer), Succeeded());

  // Reads the contents back.
  BinaryByteStream InStream(Buffer, llvm::endianness::little);
  BinaryStreamReader Reader(InStream);
  PDBStringTable Table;
  EXPECT_THAT_ERROR(Table.reload(Reader), Succeeded());

  EXPECT_EQ(6U, Table.getNameCount());
  EXPECT_EQ(1U, Table.getHashVersion());

  EXPECT_THAT_EXPECTED(Table.getStringForID(FooID), HasValue("foo"));
  EXPECT_THAT_EXPECTED(Table.getStringForID(BarID), HasValue("bar"));
  EXPECT_THAT_EXPECTED(Table.getStringForID(BazID), HasValue("baz"));
  EXPECT_THAT_EXPECTED(Table.getStringForID(BuzzID), HasValue("buzz"));
  EXPECT_THAT_EXPECTED(Table.getStringForID(BazzID), HasValue("bazz"));
  EXPECT_THAT_EXPECTED(Table.getStringForID(BarrID), HasValue("barr"));

  EXPECT_THAT_EXPECTED(Table.getIDForString("foo"), HasValue(FooID));
  EXPECT_THAT_EXPECTED(Table.getIDForString("bar"), HasValue(BarID));
  EXPECT_THAT_EXPECTED(Table.getIDForString("baz"), HasValue(BazID));
  EXPECT_THAT_EXPECTED(Table.getIDForString("buzz"), HasValue(BuzzID));
  EXPECT_THAT_EXPECTED(Table.getIDForString("bazz"), HasValue(BazzID));
  EXPECT_THAT_EXPECTED(Table.getIDForString("barr"), HasValue(BarrID));
}

TEST(StringTableHashTraitsTest, Simple) {
  PDBStringTableBuilder Builder;

  // Create more than 64kiB of dummy entries.
  for (int i = 0; i < 320; ++i) {
    std::string aaaaa = std::string(220, 'a') + std::to_string(i);
    Builder.insert(aaaaa);
  }

  std::string S = "foo.natvis";
  uint32_t Pos = Builder.insert(S);

  EXPECT_GT(Pos, 0xFFFFu);

  StringTableHashTraits Traits(Builder);
  EXPECT_LE(Traits.hashLookupKey(S), 0xFFFFu);
}
