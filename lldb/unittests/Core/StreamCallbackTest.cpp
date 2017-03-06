//===-- StreamCallbackTest.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/StreamCallback.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

static char test_baton;
static size_t callback_count = 0;
static void TestCallback(const char *data, void *baton) {
  EXPECT_STREQ("Foobar", data);
  EXPECT_EQ(&test_baton, baton);
  ++callback_count;
}

TEST(StreamCallbackTest, Callback) {
  StreamCallback stream(TestCallback, &test_baton);
  stream << "Foobar";
  EXPECT_EQ(1u, callback_count);
}
