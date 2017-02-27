//===- StringTableBuilderTest.cpp -----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ErrorChecking.h"

#include "llvm/DebugInfo/MSF/BinaryByteStream.h"
#include "llvm/DebugInfo/MSF/BinaryStreamReader.h"
#include "llvm/DebugInfo/MSF/BinaryStreamWriter.h"
#include "llvm/DebugInfo/PDB/Native/StringTable.h"
#include "llvm/DebugInfo/PDB/Native/StringTableBuilder.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::pdb;

namespace {
class StringTableBuilderTest : public ::testing::Test {};
}

TEST_F(StringTableBuilderTest, Simple) {
  // Create /names table contents.
  StringTableBuilder Builder;
  EXPECT_EQ(1U, Builder.insert("foo"));
  EXPECT_EQ(5U, Builder.insert("bar"));
  EXPECT_EQ(1U, Builder.insert("foo"));
  EXPECT_EQ(9U, Builder.insert("baz"));

  std::vector<uint8_t> Buffer(Builder.finalize());
  MutableBinaryByteStream OutStream(Buffer);
  BinaryStreamWriter Writer(OutStream);
  EXPECT_NO_ERROR(Builder.commit(Writer));

  // Reads the contents back.
  BinaryByteStream InStream(Buffer);
  BinaryStreamReader Reader(InStream);
  StringTable Table;
  EXPECT_NO_ERROR(Table.load(Reader));

  EXPECT_EQ(3U, Table.getNameCount());
  EXPECT_EQ(1U, Table.getHashVersion());
  EXPECT_EQ("foo", Table.getStringForID(1));
  EXPECT_EQ("bar", Table.getStringForID(5));
  EXPECT_EQ("baz", Table.getStringForID(9));
  EXPECT_EQ(1U, Table.getIDForString("foo"));
  EXPECT_EQ(5U, Table.getIDForString("bar"));
  EXPECT_EQ(9U, Table.getIDForString("baz"));
}
