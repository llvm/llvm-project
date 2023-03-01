//===-- DumpRegisterInfoTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/DumpRegisterInfo.h"
#include "lldb/Utility/StreamString.h"
#include "gtest/gtest.h"

using namespace lldb_private;

TEST(DoDumpRegisterInfoTest, MinimumInfo) {
  StreamString strm;
  DoDumpRegisterInfo(strm, "foo", nullptr, 4, {}, {}, {});
  ASSERT_EQ(strm.GetString(), "       Name: foo\n"
                              "       Size: 4 bytes (32 bits)");
}

TEST(DoDumpRegisterInfoTest, AltName) {
  StreamString strm;
  DoDumpRegisterInfo(strm, "foo", "bar", 4, {}, {}, {});
  ASSERT_EQ(strm.GetString(), "       Name: foo (bar)\n"
                              "       Size: 4 bytes (32 bits)");
}

TEST(DoDumpRegisterInfoTest, Invalidates) {
  StreamString strm;
  DoDumpRegisterInfo(strm, "foo", nullptr, 4, {"foo2"}, {}, {});
  ASSERT_EQ(strm.GetString(), "       Name: foo\n"
                              "       Size: 4 bytes (32 bits)\n"
                              "Invalidates: foo2");

  strm.Clear();
  DoDumpRegisterInfo(strm, "foo", nullptr, 4, {"foo2", "foo3", "foo4"}, {}, {});
  ASSERT_EQ(strm.GetString(), "       Name: foo\n"
                              "       Size: 4 bytes (32 bits)\n"
                              "Invalidates: foo2, foo3, foo4");
}

TEST(DoDumpRegisterInfoTest, ReadFrom) {
  StreamString strm;
  DoDumpRegisterInfo(strm, "foo", nullptr, 4, {}, {"foo1"}, {});
  ASSERT_EQ(strm.GetString(), "       Name: foo\n"
                              "       Size: 4 bytes (32 bits)\n"
                              "  Read from: foo1");

  strm.Clear();
  DoDumpRegisterInfo(strm, "foo", nullptr, 4, {}, {"foo1", "foo2", "foo3"}, {});
  ASSERT_EQ(strm.GetString(), "       Name: foo\n"
                              "       Size: 4 bytes (32 bits)\n"
                              "  Read from: foo1, foo2, foo3");
}

TEST(DoDumpRegisterInfoTest, InSets) {
  StreamString strm;
  DoDumpRegisterInfo(strm, "foo", nullptr, 4, {}, {}, {{"set1", 101}});
  ASSERT_EQ(strm.GetString(), "       Name: foo\n"
                              "       Size: 4 bytes (32 bits)\n"
                              "    In sets: set1 (index 101)");

  strm.Clear();
  DoDumpRegisterInfo(strm, "foo", nullptr, 4, {}, {},
                     {{"set1", 0}, {"set2", 1}, {"set3", 2}});
  ASSERT_EQ(strm.GetString(),
            "       Name: foo\n"
            "       Size: 4 bytes (32 bits)\n"
            "    In sets: set1 (index 0), set2 (index 1), set3 (index 2)");
}

TEST(DoDumpRegisterInfoTest, MaxInfo) {
  StreamString strm;
  DoDumpRegisterInfo(strm, "foo", nullptr, 4, {"foo2", "foo3"},
                     {"foo3", "foo4"}, {{"set1", 1}, {"set2", 2}});
  ASSERT_EQ(strm.GetString(), "       Name: foo\n"
                              "       Size: 4 bytes (32 bits)\n"
                              "Invalidates: foo2, foo3\n"
                              "  Read from: foo3, foo4\n"
                              "    In sets: set1 (index 1), set2 (index 2)");
}
