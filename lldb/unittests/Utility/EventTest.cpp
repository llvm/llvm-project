//===-- EventTest.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/Event.h"
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
