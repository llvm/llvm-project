//===-- DumpRegisterInfoTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/DumpRegisterInfo.h"
#include "lldb/Target/RegisterFlags.h"
#include "lldb/Utility/StreamString.h"
#include "gtest/gtest.h"

using namespace lldb_private;

TEST(DoDumpRegisterInfoTest, MinimumInfo) {
  StreamString strm;
  DoDumpRegisterInfo(strm, "foo", nullptr, 4, {}, {}, {}, nullptr, 0);
  ASSERT_EQ(strm.GetString(), "       Name: foo\n"
                              "       Size: 4 bytes (32 bits)");
}

TEST(DoDumpRegisterInfoTest, AltName) {
  StreamString strm;
  DoDumpRegisterInfo(strm, "foo", "bar", 4, {}, {}, {}, nullptr, 0);
  ASSERT_EQ(strm.GetString(), "       Name: foo (bar)\n"
                              "       Size: 4 bytes (32 bits)");
}

TEST(DoDumpRegisterInfoTest, Invalidates) {
  StreamString strm;
  DoDumpRegisterInfo(strm, "foo", nullptr, 4, {"foo2"}, {}, {}, nullptr, 0);
  ASSERT_EQ(strm.GetString(), "       Name: foo\n"
                              "       Size: 4 bytes (32 bits)\n"
                              "Invalidates: foo2");

  strm.Clear();
  DoDumpRegisterInfo(strm, "foo", nullptr, 4, {"foo2", "foo3", "foo4"}, {}, {},
                     nullptr, 0);
  ASSERT_EQ(strm.GetString(), "       Name: foo\n"
                              "       Size: 4 bytes (32 bits)\n"
                              "Invalidates: foo2, foo3, foo4");
}

TEST(DoDumpRegisterInfoTest, ReadFrom) {
  StreamString strm;
  DoDumpRegisterInfo(strm, "foo", nullptr, 4, {}, {"foo1"}, {}, nullptr, 0);
  ASSERT_EQ(strm.GetString(), "       Name: foo\n"
                              "       Size: 4 bytes (32 bits)\n"
                              "  Read from: foo1");

  strm.Clear();
  DoDumpRegisterInfo(strm, "foo", nullptr, 4, {}, {"foo1", "foo2", "foo3"}, {},
                     nullptr, 0);
  ASSERT_EQ(strm.GetString(), "       Name: foo\n"
                              "       Size: 4 bytes (32 bits)\n"
                              "  Read from: foo1, foo2, foo3");
}

TEST(DoDumpRegisterInfoTest, InSets) {
  StreamString strm;
  DoDumpRegisterInfo(strm, "foo", nullptr, 4, {}, {}, {{"set1", 101}}, nullptr,
                     0);
  ASSERT_EQ(strm.GetString(), "       Name: foo\n"
                              "       Size: 4 bytes (32 bits)\n"
                              "    In sets: set1 (index 101)");

  strm.Clear();
  DoDumpRegisterInfo(strm, "foo", nullptr, 4, {}, {},
                     {{"set1", 0}, {"set2", 1}, {"set3", 2}}, nullptr, 0);
  ASSERT_EQ(strm.GetString(),
            "       Name: foo\n"
            "       Size: 4 bytes (32 bits)\n"
            "    In sets: set1 (index 0), set2 (index 1), set3 (index 2)");
}

TEST(DoDumpRegisterInfoTest, MaxInfo) {
  StreamString strm;
  DoDumpRegisterInfo(strm, "foo", nullptr, 4, {"foo2", "foo3"},
                     {"foo3", "foo4"}, {{"set1", 1}, {"set2", 2}}, nullptr, 0);
  ASSERT_EQ(strm.GetString(), "       Name: foo\n"
                              "       Size: 4 bytes (32 bits)\n"
                              "Invalidates: foo2, foo3\n"
                              "  Read from: foo3, foo4\n"
                              "    In sets: set1 (index 1), set2 (index 2)");
}

TEST(DoDumpRegisterInfoTest, FieldsTable) {
  // This is thoroughly tested in RegisterFlags itself, only checking the
  // integration here.
  StreamString strm;
  RegisterFlags flags(
      "", 4,
      {RegisterFlags::Field("A", 24, 31), RegisterFlags::Field("B", 16, 23),
       RegisterFlags::Field("C", 8, 15), RegisterFlags::Field("D", 0, 7)});

  DoDumpRegisterInfo(strm, "foo", nullptr, 4, {}, {}, {}, &flags, 100);
  ASSERT_EQ(strm.GetString(), "       Name: foo\n"
                              "       Size: 4 bytes (32 bits)\n"
                              "\n"
                              "| 31-24 | 23-16 | 15-8 | 7-0 |\n"
                              "|-------|-------|------|-----|\n"
                              "|   A   |   B   |  C   |  D  |");
}

TEST(DoDumpRegisterInfoTest, Enumerators) {
  StreamString strm;

  FieldEnum enum_one("enum_one", {{0, "an_enumerator"}});
  FieldEnum enum_two("enum_two",
                     {{1, "another_enumerator"}, {2, "another_enumerator_2"}});

  RegisterFlags flags("", 4,
                      {RegisterFlags::Field("A", 24, 31, &enum_one),
                       RegisterFlags::Field("B", 16, 23),
                       RegisterFlags::Field("C", 8, 15, &enum_two)});

  DoDumpRegisterInfo(strm, "abc", nullptr, 4, {}, {}, {}, &flags, 100);
  ASSERT_EQ(strm.GetString(),
            "       Name: abc\n"
            "       Size: 4 bytes (32 bits)\n"
            "\n"
            "| 31-24 | 23-16 | 15-8 | 7-0 |\n"
            "|-------|-------|------|-----|\n"
            "|   A   |   B   |  C   |     |\n"
            "\n"
            "A: 0 = an_enumerator\n"
            "\n"
            "C: 1 = another_enumerator, 2 = another_enumerator_2");
}
