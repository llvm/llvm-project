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
  bool ProcessResult = TableGenParseFile(SrcMgr, Records);
  EXPECT_FALSE(ProcessResult);

  Record *Foo = Records.getDef("Foo");
  Optional<StringRef> Field = Foo->getValueAsOptionalString("strField");
  EXPECT_TRUE(Field.hasValue());
  EXPECT_EQ(Field.getValue(), "value");
}
