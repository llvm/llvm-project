//===- unittest/TableGen/ParserEntryPointTest.cpp - Parser tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/TableGen/Parser.h"
#include "llvm/TableGen/Record.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(Parser, SanityTest) {
  // Simple TableGen source file with a single record.
  const char *SimpleTdSource = R"td(
    def Foo {
      string strField = "value";
    }
  )td";

  SourceMgr SrcMgr;
  SrcMgr.AddNewSourceBuffer(
      MemoryBuffer::getMemBuffer(SimpleTdSource, "test_buffer"), SMLoc());

  RecordKeeper Records;
  bool ProcessResult =
      TableGenParseFile(SrcMgr, Records, /*TrackReferenceLocs=*/false);
  EXPECT_FALSE(ProcessResult);

  const Record *Foo = Records.getDef("Foo");
  std::optional<StringRef> Field = Foo->getValueAsOptionalString("strField");
  EXPECT_TRUE(Field.has_value());
  EXPECT_EQ(*Field, "value");
}

// Check that reference locations are only recorded when requested.
TEST(Parser, TrackReferenceLocs) {
  const char *TdSource = R"td(
    class Foo {
      int x = 1;
    }
    def Bar : Foo {
      int y = x;
    }
  )td";

  for (bool Track : {false, true}) {
    SourceMgr SrcMgr;
    SrcMgr.AddNewSourceBuffer(
        MemoryBuffer::getMemBuffer(TdSource, "test_buffer"), SMLoc());

    RecordKeeper Records;
    EXPECT_FALSE(TableGenParseFile(SrcMgr, Records, Track));

    const Record *Foo = Records.getClass("Foo");
    const Record *Bar = Records.getDef("Bar");
    ASSERT_TRUE(Foo && Bar);
    const Init *X = StringInit::get(Records, "x");
    EXPECT_EQ(Foo->getReferenceLocs().size(), Track ? 1u : 0u);
    EXPECT_EQ(Bar->getValueReferenceLocs(X).size(), Track ? 1u : 0u);
  }
}
