//===-- EventTest.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Event.h"
#include "lldb/Utility/StreamString.h"
#include "gtest/gtest.h"

using namespace lldb_private;

static std::string to_string(const EventDataBytes &E) {
  StreamString S;
  E.Dump(&S);
  return S.GetString();
}

TEST(EventTest, DumpEventDataBytes) {
  EXPECT_EQ(R"("foo")", to_string(EventDataBytes("foo")));
  EXPECT_EQ("01 02 03", to_string(EventDataBytes("\x01\x02\x03")));
}
