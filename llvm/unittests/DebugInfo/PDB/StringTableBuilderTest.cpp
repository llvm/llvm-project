//===- StringTableBuilderTest.cpp -----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

namespace {
class StringTableBuilderTest : public ::testing::Test {};
}

TEST_F(StringTableBuilderTest, Simple) {
  // Create /names table contents.
  PDBStringTableBuilder Builder;
  EXPECT_EQ(1U, Builder.insert("foo"));
  EXPECT_EQ(5U, Builder.insert("bar"));
  EXPECT_EQ(1U, Builder.insert("foo"));
  EXPECT_EQ(9U, Builder.insert("baz"));

  std::vector<uint8_t> Buffer(Builder.calculateSerializedSize());
  MutableBinaryByteStream OutStream(Buffer, little);
  BinaryStreamWriter Writer(OutStream);
  EXPECT_THAT_ERROR(Builder.commit(Writer), Succeeded());

  // Reads the contents back.
  BinaryByteStream InStream(Buffer, little);
  BinaryStreamReader Reader(InStream);
  PDBStringTable Table;
  EXPECT_THAT_ERROR(Table.reload(Reader), Succeeded());

  EXPECT_EQ(3U, Table.getNameCount());
  EXPECT_EQ(1U, Table.getHashVersion());

  EXPECT_THAT_EXPECTED(Table.getStringForID(1), HasValue("foo"));
  EXPECT_THAT_EXPECTED(Table.getStringForID(5), HasValue("bar"));
  EXPECT_THAT_EXPECTED(Table.getStringForID(9), HasValue("baz"));
  EXPECT_THAT_EXPECTED(Table.getIDForString("foo"), HasValue(1U));
  EXPECT_THAT_EXPECTED(Table.getIDForString("bar"), HasValue(5U));
  EXPECT_THAT_EXPECTED(Table.getIDForString("baz"), HasValue(9U));
}
