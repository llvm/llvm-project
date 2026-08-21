//===- llvm/unittest/Support/DebugLogTest.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/DebugLog.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <string>
using namespace llvm;
using testing::Eq;
using testing::HasSubstr;

#ifndef NDEBUG
TEST(DebugLogTest, Basic) {
  llvm::DebugFlag = true;
  static const char *DT[] = {"A", "B"};

  // Clear debug types.
  setCurrentDebugTypes(DT, 0);
  {
    std::string str;
    raw_string_ostream os(str);
    LDGB_STREAM_LEVEL_AND_TYPE(os, "", 0) << "NoType";
    EXPECT_TRUE(StringRef(os.str()).starts_with('['));
    EXPECT_TRUE(StringRef(os.str()).ends_with("NoType\n"));
  }

  setCurrentDebugTypes(DT, 2);
  {
    std::string str;
    raw_string_ostream os(str);
    LDGB_STREAM_LEVEL_AND_TYPE(os, 0, "A") << "A";
    LDGB_STREAM_LEVEL_AND_TYPE(os, "B", 0) << "B";
    EXPECT_TRUE(StringRef(os.str()).starts_with('['));
    EXPECT_THAT(os.str(), AllOf(HasSubstr("A\n"), HasSubstr("B\n")));
  }

  setCurrentDebugType("A");
  {
    std::string str;
    raw_string_ostream os(str);
    // Just check that the macro doesn't result in dangling else.
    if (true)
      LDGB_STREAM_LEVEL_AND_TYPE(os, 0, "A") << "A";
    else
      LDGB_STREAM_LEVEL_AND_TYPE(os, 0, "A") << "B";
    LDGB_STREAM_LEVEL_AND_TYPE(os, 0, "B") << "B";
    EXPECT_THAT(os.str(), AllOf(HasSubstr("A\n"), Not(HasSubstr("B\n"))));

    int count = 0;
    auto inc = [&]() { return ++count; };
    EXPECT_THAT(count, Eq(0));
    LDGB_STREAM_LEVEL_AND_TYPE(os, 0, "A") << inc();
    EXPECT_THAT(count, Eq(1));
    LDGB_STREAM_LEVEL_AND_TYPE(os, 0, "B") << inc();
    EXPECT_THAT(count, Eq(1));
  }
}

TEST(DebugLogTest, BasicWithLevel) {
  llvm::DebugFlag = true;
  // We expect A to be always printed, B to be printed only when level is 1 or
  // below, and C to be printed only when level is 0 or below.
  static const char *DT[] = {"A", "B:1", "C:"};

  setCurrentDebugTypes(DT, sizeof(DT) / sizeof(DT[0]));
  std::string str;
  raw_string_ostream os(str);
  for (auto type : {"A", "B", "C", "D"})
    for (int level : llvm::seq<int>(0, 4))
      LDBG_STREAM_LEVEL_TYPE_FILE_AND_LINE(os, level, type, type, level)
          << level;
  EXPECT_EQ(os.str(), "[A:0 0] 0\n[A:1 1] 1\n[A:2 2] 2\n[A:3 3] 3\n[B:0 0] "
                      "0\n[B:1 1] 1\n[C:0 0] 0\n");
}

TEST(DebugLogTest, NegativeLevel) {
  llvm::DebugFlag = true;
  // Test the special behavior when all the levels are 0.
  // In this case we expect all the debug types to be printed.
  static const char *DT[] = {"A:"};

  setCurrentDebugTypes(DT, sizeof(DT) / sizeof(DT[0]));
  std::string str;
  raw_string_ostream os(str);
  for (auto type : {"A", "B"})
    for (int level : llvm::seq<int>(0, 2))
      LDBG_STREAM_LEVEL_TYPE_FILE_AND_LINE(
          os, level, type, (std::string(type) + ".cpp").c_str(), level)
          << level;
  EXPECT_EQ(os.str(), "[A A.cpp:0 0] 0\n[B B.cpp:0 0] 0\n[B B.cpp:1 1] 1\n");
}

TEST(DebugLogTest, StreamPrefix) {
  llvm::DebugFlag = true;
  static const char *DT[] = {"A", "B"};
  setCurrentDebugTypes(DT, 2);

  std::string str;
  raw_string_ostream os(str);
  std::string expected = "PrefixA 1\nPrefixA 2\nPrefixA \nPrefixB "
                         "3\nPrefixB 4\nPrefixA 5";
  {
    llvm::impl::raw_ldbg_ostream ldbg_osB("PrefixB ", os);
    llvm::impl::raw_ldbg_ostream ldbg_osA("PrefixA ", os);
    ldbg_osA << "1\n2";
    ldbg_osA << "\n\n";
    ldbg_osB << "3\n4\n";
    ldbg_osA << "5";
    EXPECT_EQ(os.str(), expected);
  }
  EXPECT_EQ(os.str(), expected);
}

TEST(DebugLogTest, DestructorPrefix) {
  llvm::DebugFlag = true;
  std::string str;
  raw_string_ostream os(str);
  {
    llvm::impl::raw_ldbg_ostream ldbg_osB("PrefixB ", os);
  }
  // After destructors, nothing should have been printed.
  EXPECT_EQ(os.str(), "");
}

TEST(DebugLogTest, LDBG_MACROS) {
  llvm::DebugFlag = true;
  static const char *DT[] = {"A:3", "B:2"};
  setCurrentDebugTypes(DT, sizeof(DT) / sizeof(DT[0]));
  std::string Str;
  raw_string_ostream DebugOs(Str);
  std::string StrExpected;
  raw_string_ostream ExpectedOs(StrExpected);
#undef LDBG_STREAM
#define LDBG_STREAM DebugOs
#define DEBUG_TYPE "A"
  LDBG() << "Hello, world!";
  ExpectedOs << "[A " << __LLVM_FILE_NAME__ << ":" << (__LINE__ - 1)
             << " 1] Hello, world!\n";
  EXPECT_EQ(DebugOs.str(), ExpectedOs.str());
  Str.clear();
  StrExpected.clear();

  // Test with a level, no type.
  LDBG(2) << "Hello, world!";
  ExpectedOs << "[A " << __LLVM_FILE_NAME__ << ":" << (__LINE__ - 1)
             << " 2] Hello, world!\n";
  EXPECT_EQ(DebugOs.str(), ExpectedOs.str());
  Str.clear();
  StrExpected.clear();

// Now check when we don't use DEBUG_TYPE, the file name is implicitly used
// instead.
#undef DEBUG_TYPE

  // Repeat the tests above, they won't match since the debug types defined
  // above don't match the file name.
  LDBG() << "Hello, world!";
  EXPECT_EQ(DebugOs.str(), "");
  Str.clear();
  StrExpected.clear();

  // Test with a level, no type.
  LDBG(2) << "Hello, world!";
  EXPECT_EQ(DebugOs.str(), "");
  Str.clear();
  StrExpected.clear();

  // Now enable the debug types that match the file name.
  auto fileNameAndLevel = std::string(__LLVM_FILE_NAME__) + ":3";
  static const char *DT2[] = {fileNameAndLevel.c_str(), "B:2"};
  setCurrentDebugTypes(DT2, sizeof(DT2) / sizeof(DT2[0]));

  // Repeat the tests above, they should match now.

  LDBG() << "Hello, world!";
  ExpectedOs << "[" << __LLVM_FILE_NAME__ << ":" << (__LINE__ - 1)
             << " 1] Hello, world!\n";
  EXPECT_EQ(DebugOs.str(), ExpectedOs.str());
  Str.clear();
  StrExpected.clear();

  // Test with a level, no type.
  LDBG(2) << "Hello, world!";
  ExpectedOs << "[" << __LLVM_FILE_NAME__ << ":" << (__LINE__ - 1)
             << " 2] Hello, world!\n";
  EXPECT_EQ(DebugOs.str(), ExpectedOs.str());
  Str.clear();
  StrExpected.clear();

  // Test with a type
  LDBG("B") << "Hello, world!";
  ExpectedOs << "[B " << __LLVM_FILE_NAME__ << ":" << (__LINE__ - 1)
             << " 1] Hello, world!\n";
  EXPECT_EQ(DebugOs.str(), ExpectedOs.str());
  Str.clear();
  StrExpected.clear();

  // Test with a type and a level
  LDBG("B", 2) << "Hello, world!";
  ExpectedOs << "[B " << __LLVM_FILE_NAME__ << ":" << (__LINE__ - 1)
             << " 2] Hello, world!\n";
  EXPECT_EQ(DebugOs.str(), ExpectedOs.str());
  Str.clear();
  StrExpected.clear();

  // Test with a type not enabled.
  LDBG("C", 1) << "Hello, world!";
  EXPECT_EQ(DebugOs.str(), "");

  // Test with a level not enabled.
  LDBG("B", 3) << "Hello, world!";
  EXPECT_EQ(DebugOs.str(), "");
  LDBG(__LLVM_FILE_NAME__, 4) << "Hello, world!";
  EXPECT_EQ(DebugOs.str(), "");
}

TEST(DebugLogTest, LDBG_OS_MACROS) {
  llvm::DebugFlag = true;
  static const char *DT[] = {"A:3", "B:2"};
  setCurrentDebugTypes(DT, sizeof(DT) / sizeof(DT[0]));
  std::string Str;
  raw_string_ostream DebugOs(Str);
  std::string StrExpected;
  raw_string_ostream ExpectedOs(StrExpected);
#undef LDBG_STREAM
#define LDBG_STREAM DebugOs
#define DEBUG_TYPE "A"
  LDBG_OS([](raw_ostream &Os) { Os << "Hello, world!"; });
  ExpectedOs << "[A " << __LLVM_FILE_NAME__ << ":" << (__LINE__ - 1)
             << " 1] Hello, world!\n";
  EXPECT_EQ(DebugOs.str(), ExpectedOs.str());
  Str.clear();
  StrExpected.clear();

  // Test with a level, no type.
  LDBG_OS(2, [](raw_ostream &Os) { Os << "Hello, world!"; });
  ExpectedOs << "[A " << __LLVM_FILE_NAME__ << ":" << (__LINE__ - 1)
             << " 2] Hello, world!\n";
  EXPECT_EQ(DebugOs.str(), ExpectedOs.str());
  Str.clear();
  StrExpected.clear();

// Now check when we don't use DEBUG_TYPE, the file name is implicitly used
// instead.
#undef DEBUG_TYPE

  // Repeat the tests above, they won't match since the debug types defined
  // above don't match the file name.
  LDBG_OS([](raw_ostream &Os) { Os << "Hello, world!"; });
  EXPECT_EQ(DebugOs.str(), "");
  Str.clear();
  StrExpected.clear();

  // Test with a level, no type.
  LDBG_OS(2, [](raw_ostream &Os) { Os << "Hello, world!"; });
  EXPECT_EQ(DebugOs.str(), "");
  Str.clear();
  StrExpected.clear();

  // Now enable the debug types that match the file name.
  auto fileNameAndLevel = std::string(__LLVM_FILE_NAME__) + ":3";
  static const char *DT2[] = {fileNameAndLevel.c_str(), "B:2"};
  setCurrentDebugTypes(DT2, sizeof(DT2) / sizeof(DT2[0]));

  // Repeat the tests above, they should match now.
  LDBG_OS([](raw_ostream &Os) { Os << "Hello, world!"; });
  ExpectedOs << "[" << __LLVM_FILE_NAME__ << ":" << (__LINE__ - 1)
             << " 1] Hello, world!\n";
  EXPECT_EQ(DebugOs.str(), ExpectedOs.str());
  Str.clear();
  StrExpected.clear();

  // Test with a level, no type.
  LDBG_OS(2, [](raw_ostream &Os) { Os << "Hello, world!"; });
  ExpectedOs << "[" << __LLVM_FILE_NAME__ << ":" << (__LINE__ - 1)
             << " 2] Hello, world!\n";
  EXPECT_EQ(DebugOs.str(), ExpectedOs.str());
  Str.clear();
  StrExpected.clear();

  // Test with a type.
  LDBG_OS("B", [](raw_ostream &Os) { Os << "Hello, world!"; });
  ExpectedOs << "[B " << __LLVM_FILE_NAME__ << ":" << (__LINE__ - 1)
             << " 1] Hello, world!\n";
  EXPECT_EQ(DebugOs.str(), ExpectedOs.str());
  Str.clear();
  StrExpected.clear();

  // Test with a type and a level
  LDBG_OS("B", 2, [](raw_ostream &Os) { Os << "Hello, world!"; });
  ExpectedOs << "[B " << __LLVM_FILE_NAME__ << ":" << (__LINE__ - 1)
             << " 2] Hello, world!\n";
  EXPECT_EQ(DebugOs.str(), ExpectedOs.str());
  Str.clear();
  StrExpected.clear();

  // Test with a type not enabled.
  LDBG_OS("C", 1, [](raw_ostream &Os) { Os << "Hello, world!"; });
  EXPECT_EQ(DebugOs.str(), "");

  // Test with a level not enabled.
  LDBG_OS("B", 3, [](raw_ostream &Os) { Os << "Hello, world!"; });
  EXPECT_EQ(DebugOs.str(), "");
}

#else
TEST(DebugLogTest, Basic) {
  // LDBG should be compiled out in NDEBUG, so just check it compiles and has
  // no effect.
  llvm::DebugFlag = true;
  static const char *DT[] = {"A"};
  setCurrentDebugTypes(DT, 0);
  int count = 0;
  auto inc = [&]() { return ++count; };
  EXPECT_THAT(count, Eq(0));
  LDBG() << inc();
  EXPECT_THAT(count, Eq(0));
}
#endif
